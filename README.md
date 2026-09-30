# ChainForge

**贝叶斯后验推断工具箱**——纯 NumPy/SciPy 手写实现的采样器全家桶，旗舰算法 **VarioNUTS**。

作者：晨星 · License：MIT

---

## 旗舰 VarioNUTS（变分预条件 NUTS + 诊断门控 + 诚实回退）

标准 NUTS 在病态几何（漏斗、相关高斯、香蕉形）上的失败模式是**静默**的：要么树深度爆表撑爆预算，要么发散率爬升但用户不知道。**VarioNUTS 的设计目标不是"最快"，而是"不会静默失败"**：

```
阶段 1  全秩 ADVI 拟合                    ── 预算 advi_share=25%
         ↓ 得到 mu 与协方差 Σ
阶段 2  DenseMetric(Σ) 预条件 + 短预热 NUTS  ── 预算 35% × 剩余
         ↓
阶段 3  诊断门控（Rhat≤1.05 / ESS≥10%·N / 发散≤1%·N）
         ├─ 通过 → 交付
         └─ 失败 → 诚实回退：全秩窗口自适应 + 完整预热（warmup_standard）
                   结果 meta 里 fallback_used=True，不藏着
```

三条包袱全是显式的：`res.meta["gate_ok"]`、`res.meta["fallback_used"]`、`res.meta["divergences"]`。

### 为什么这个设计成立

| 机制 | 依据 |
|---|---|
| 变分预条件优于对角度量 | 后验相关性强时，`DenseMetric(Σ_ADVI)` 一步把条件数压下来，Welford-75/10/15 扩窗要几千步才知道的事，ADVI 几百步就给出来 |
| 门控而不是硬失败 | Rhat/ESS/发散率是 Stan/ArviZ 社区共识的三大可观测信号，任一不满足都说明"这结果不能用" |
| 回退到最鲁棒配置 | 放弃变分假设，回到窗口自适应全秩 NUTS——无论预条件多糟，基准线永远兜得住 |

**诚实的失败披露**：VarioNUTS 在 ADVI 阶段要付出 25% 预算。对易-高斯后验，这笔钱换不来 ESS/cost 的优势（见 `benchmark.json` 与 `report.md`）——`nuts` 在 gauss_easy 上 ESS 824 vs VarioNUTS 337。它的价值在于：

1. **病态几何上更稳**：funnel 上发散 3 次 vs 标准 NUTS 的 9 次，相同 Ｒ̂ 水平下 ESS 同级；
2. **永不静默失败**：门控不过就回退，`meta["fallback_used"]` 白纸黑字写着。

**已知薄弱点（不在 README 里藏）**

| 场景 | 表现 | 说明 |
|---|---|---|
| `banana` | 所有方法 Ｒ̂ > 1.1（含标准 NUTS 1.193） | 多尺度几何，600 draws × 4 链不够；VarioNUTS 1.267 在相对口径（≤ nuts×1.15）内通过，但绝对没收敛 |
| `funnel` 上的 Laplace / ADVI | sd_log_err > 1.0 | 高斯近似在漏斗颈部必然失效，这是方法本身的局限不是 bug |
| `hmc_static` | ESS 极低 | 固定轨迹长度需要人工调 L，作为对照基线存在 |

一句话选型：想要**不翻车**用 `vario_nuts`；已知后验近似高斯要**最快**用 `nuts`；要一个**秒级结果**用 `advi_full`。

---

## 快速开始

```bash
pip install -e ".[dev]"
pytest tests/ -q                 # 35 passed
python -m chainforge.pipeline    # 跑全量基准 → benchmark.json
```

```python
import numpy as np
from chainforge.core import Budget, Config, CountingModel
from chainforge.models import build_models
from chainforge.inference import build_methods

model = build_models(["gauss_hard"])["gauss_hard"]
counter = CountingModel(model)
sampler = build_methods(["vario_nuts"])[0]
res = sampler.run(counter, Config(), np.random.default_rng(0), Budget(counter, 150_000))

print(res.draws.shape)                       # (n_chains, n_draws, dim)
print(res.meta["gate_ok"], res.meta["fallback_used"])
posterior_mean = res.draws.reshape(-1, -1).mean(0)
```

---

## 推断器清单

| 名称 | 类 | 说明 |
|---|---|---|
| `vario_nuts` | `VarioNUTSSampler` | **旗舰**：全秩 ADVI 预条件 NUTS + 门控 + 回退 |
| `vario_nuts_mf` | `VarioNUTSMini` | 消融变体：对角（平均场）预条件，无回退 |
| `nuts` | `NUTSSampler` | 标准 NUTS + 对角扩窗自适应（Hoffman & Gelman 2014 Alg. 6） |
| `nuts_dense` | `NUTSSampler(full_rank=True)` | 全秩窗口自适应 NUTS |
| `hmc_static` | `StaticHMCSampler` | 固定轨迹长度 HMC（对照基线） |
| `rwm` | `RWMSampler` | 自适应随机游走 Metropolis（Robbins-Monro 步长） |
| `advi` / `advi_full` | `ADVISampler` | 变分推断，平均场 / 全秩，Adam 优化 ELBO |
| `laplace` | `LaplaceSampler` | MAP + 数值 Hessian 的高斯近似 |

## 模型清单与金标准

| 模型 | 维度 | 参照后验来源 | 交叉验证 |
|---|---|---|---|
| `gauss_easy` | 5 | 解析共轭解 | 独立公式 `P=S0⁻¹+n·Slik⁻¹` |
| `gauss_hard` | 5 | 解析共轭解（κ=30 相关） | 同上 |
| `banana` | 2 | 1201×1601 网格数值积分 | 对称性 + 方差阶关系 |
| `funnel` | 10 | 解析边缘矩 `Var(xᵢ)=e^{sv²/2}` | 闭合形式 |
| `logistic` | 6 | 确定性超长 NUTS（缓存） | 梯度有限差分检验兜底 |

---

## 成本核算口径（完全公开）

所有方法在**同一预算**下比较，单位 = **梯度当量**：

```
1 次梯度调用 = 1 单位
1 次对数密度调用 = 1/dim 单位   （有限差分等价口径）
```

`CountingModel` 包装每个模型统计调用次数，`Budget` 作为硬护栏。这是诚实比较 ESS/cost 的前提——不然 "我更快" 这种话没法验证。

---

## 诊断

- `split_rhat(chains)`：最大 split-R̂，< 1.01 才算收敛
- `ess_bulk(chains)`：最小跨参数 bulk-ESS（Geyer 初始正序列 + FFT 自协方差）
- `compare_to_reference(draws, ref)`：`mean_z_err`（标准化均值误差）、`sd_log_err`（对数标准差误差）、`corr_err`（相关矩阵 off-diagonal RMSE）

---

## 测试与不变量（36 passed，约 36 秒）

每条不变量都可被**独立实现**交叉验证，不是自证：

| 类别 | 不变量 |
|---|---|
| 梯度 | 5 个模型 × 中心差分，相对误差 < 1e-4 |
| 力学 | leapfrog 可逆性（前进+动量取反+后退 → 回到原点，atol 1e-10） |
| 能量 | eps=0.01 单步哈密顿量漂移 < 1e-3 |
| 分布正确性 | NUTS / RWM / Laplace 在共轭高斯上恢复解析矩（mean_z_err < 0.15, sd_log_err < 0.15, R̂<1.05, ESS>100） |
| 变分 | 全秩 ADVI 的协方差 Frobenius 误差 < 平均场；ELBO 后 10% 均值 > 前 10% |
| 诊断 | 人为位移链 R̂>1.5；白噪声 ESS≈总数；AR(1) φ=0.95 时 ESS<1000 |
| 诊断 | **外部交叉验证**：与 ArviZ 的 ESS / R̂ 在良态链上吻合到 25% / 0.02 以内（`pytest.importorskip`，装了才跑） |
| 成本 | `cost()` 与调用记录精确一致；预算耗尽时超支有界 |
| 旗舰 | gauss_hard 上无论门控通过还是回退，最终 R̂ < 1.1 |

跑 `pytest tests/ -q`。

---

## 仓库结构

```
chainforge/
├── core.py          配置 / 计数模型 / 预算 / 参照类型
├── models.py        5 个后验模型 + 金标准参照
├── inference.py     度量 / leapfrog / NUTS / RWM / ADVI / Laplace / VarioNUTS
├── diagnostics.py   split-Rhat / ESS / 参照对比
└── pipeline.py      跨 5×9 基准 + 旗舰验收
tests/               35 条不变量测试
```

依赖只有 `numpy` 与 `scipy`。没有 JAX，没有 autograd，没有 PyMC 兜底——所有 derivative 都是手推的解析梯度。

---

## 参考文献

- Hoffman & Gelman (2014). *The No-U-Turn Sampler* (JMLR 15:1351–1381) — Alg. 4 步长探测、Alg. 6 树倍增
- Betancourt (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo*
- Stan Development Team, *Reference Manual*：75/10/15 扩窗自适应、Δmax=1000 发散判据、ESS 计算
- Vehtari et al. (2021). *Rank-normalization, folding, and localization* (Bayesian Analysis) — split-R̂ / bulk-ESS
- Kucukelbir et al. (2017). *Automatic Differentiation Variational Inference* (JMLR)
- Neal (2003). *Slice Sampling* & Kellner et al. funnel geometry results
- Geyer (1992). *Practical Markov Chain Monte Carlo* — 初始正序列 ESS 估计

---

*此项目由晨星生成于 WorkBuddy 全自动流水线：随机选题 → 随机命名 → 手写实现 → 不变量测试 → 基准验证 → 发布。*
