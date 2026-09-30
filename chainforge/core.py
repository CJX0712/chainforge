"""核心类型：配置 / 错误 / 计数模型 / 预算 / 参照后验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class InferenceError(RuntimeError):
    """推断失败（预算耗尽 / 数值发散 / 收敛门控失败且回退关闭）。"""


@dataclass
class Config:
    """全局配置。cost_budget 单位 = 梯度当量。

    成本约定（完全公开，写入 docs/architecture.md）：
    1 次梯度调用 = 1 单位；1 次对数密度调用 = 1/dim 单位（有限差分等价口径）。
    """

    # ---- 采样通用 ----
    n_chains: int = 4
    max_draws_per_chain: int = 600
    warmup_standard: int = 500
    warmup_vario: int = 150
    max_treedepth: int = 10
    target_accept: float = 0.9
    delta_max: float = 1000.0  # NUTS 发散阈值（Stan 口径）
    # ---- VarioNUTS ----
    advi_share: float = 0.25  # 预算中分给 ADVI 预条件化的比例
    vario_advi_steps: int = 2500
    vario_advi_mc: int = 6
    vario_advi_lr: float = 0.08
    rhat_gate: float = 1.05
    ess_gate: float = 0.10  # 相对总 draw 数的最小 ESS 比例
    divergence_gate: float = 0.01  # 相对总 draw 数的最大发散比例
    allow_fallback: bool = True
    # ---- ADVI 独立方法 ----
    advi_steps: int = 4000
    advi_lr: float = 0.05
    advi_mc_samples: int = 8
    advi_init_scale: float = 0.1
    # ---- RWM ----
    rwm_step: float = 0.3
    # ---- 资源 ----
    cost_budget: int = 150_000
    seed: int = 2026


@dataclass(frozen=True)
class ReferencePosterior:
    """金标准参照：解析解 / 数值积分 / 长链确定性 NUTS。"""

    mean: np.ndarray
    sd: np.ndarray
    cov: np.ndarray
    source: str = "analytic"


class ModelProtocol(Protocol):
    """模型接口：只需 log_prob 与 log_prob_grad。"""

    name: str
    dim: int
    param_names: tuple[str, ...]

    def log_prob(self, z: np.ndarray) -> float: ...

    def log_prob_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]: ...


def _finite_or_neginf(value: float) -> float:
    """非有限 / NaN 的对数密度统一折叠为 -inf（HMC 会自动拒绝）。"""
    v = float(value)
    return v if np.isfinite(v) else -np.inf


class CountingModel:
    """包装模型，统计 logp / grad 调用次数并换算为梯度当量成本。"""

    def __init__(self, inner: ModelProtocol) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = int(inner.dim)
        self.param_names = tuple(inner.param_names)
        self.n_logp = 0
        self.n_grad = 0

    def log_prob(self, z: np.ndarray) -> float:
        self.n_logp += 1
        return _finite_or_neginf(self.inner.log_prob(np.asarray(z, dtype=float)))

    def log_prob_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        self.n_grad += 1
        try:
            lp, g = self.inner.log_prob_grad(np.asarray(z, dtype=float))
        except (OverflowError, ValueError, FloatingPointError):
            # 建议点落在数值悬崖外（如 funnel 颈部以下、banana x²溢出）
            lp, g = -np.inf, np.zeros(self.dim)
        return _finite_or_neginf(float(lp)), np.nan_to_num(
            np.asarray(g, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )

    def cost(self) -> float:
        """梯度当量成本。"""
        dim = max(self.dim, 1)
        return float(self.n_grad + self.n_logp / dim)

    def reset(self) -> None:
        self.n_logp = 0
        self.n_grad = 0


@dataclass
class Budget:
    """成本预算护栏。耗尽后 sampler 必须停止（或抛 InferenceError）。"""

    model: CountingModel
    limit: float
    start: float = field(default_factory=lambda: 0.0)

    def __post_init__(self) -> None:
        self.start = self.model.cost()

    def spent(self) -> float:
        return self.model.cost() - self.start

    def remaining(self) -> float:
        return self.limit - self.spent()

    def exhausted(self) -> bool:
        return self.remaining() <= 0


@dataclass
class InferenceResult:
    """一次推断运行的产出。draws 形状 (n_chains, n_draws, dim)。"""

    method: str
    model: str
    draws: np.ndarray
    divergences: int
    n_warmup: int
    wall_time: float
    meta: dict[str, Any]

    @property
    def total_draws(self) -> int:
        return int(self.draws.shape[0] * self.draws.shape[1])

    @property
    def cost(self) -> float:
        return float(self.meta.get("cost", float("nan")))


def make_result(
    model: CountingModel,
    method: str,
    draws: np.ndarray,
    t0: float,
    *,
    divergences: int = 0,
    n_warmup: int = 0,
    **meta: Any,
) -> InferenceResult:
    import time

    meta.setdefault("cost", model.cost())
    return InferenceResult(
        method=method,
        model=model.name,
        draws=np.asarray(draws, dtype=float),
        divergences=int(divergences),
        n_warmup=int(n_warmup),
        wall_time=time.perf_counter() - t0,
        meta=meta,
    )
