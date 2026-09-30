"""ChainForge 测试套件：每条不变量都可被独立实现交叉验证。

金标准：
- 解析解（共轭高斯）vs 采样
- 2D 网格积分（banana）vs 采样
- 解析边缘矩（funnel）vs 采样
- 有限差分 vs 解析梯度
- leapfrog 可逆性（体积保持 → 确定性反向）
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from chainforge.core import Budget, Config, CountingModel, InferenceError
from chainforge.diagnostics import (
    compare_to_reference,
    ess_bulk,
    split_rhat,
    summarize_draws,
)
from chainforge.inference import (
    ADVISampler,
    LaplaceSampler,
    Metric,
    NUTSSampler,
    RWMSampler,
    VarioNUTSMini,
    VarioNUTSSampler,
    build_methods,
    find_reasonable_epsilon,
    fit_advi,
    hamiltonian,
    leapfrog,
    nuts_transition,
    unit_metric,
)
from chainforge.models import MODEL_BUILDERS, build_models
from chainforge.pipeline import run_one, verify_flagship


def fast_cfg(**kw) -> Config:
    base = {
        "n_chains": 2,
        "max_draws_per_chain": 300,
        "warmup_standard": 200,
        "warmup_vario": 80,
        "cost_budget": 60_000,
        "vario_advi_steps": 600,
        "vario_advi_mc": 4,
    }
    base.update(kw)
    return Config(**base)


ALL_MODELS = ["gauss_easy", "gauss_hard", "banana", "funnel", "logistic"]


# ---------------------------------------------------------------------------
# 模型：梯度检验（金标准：中心差分）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ALL_MODELS)
def test_gradient_check(key):
    m = build_models([key])[key]
    rng = np.random.default_rng(3)
    for _ in range(3):
        z = rng.normal(0, 0.5, size=m.dim)
        _, g = m.log_prob_grad(z)
        h = 1e-6
        gnum = np.zeros(m.dim)
        for i in range(m.dim):
            zp, zm = z.copy(), z.copy()
            zp[i] += h
            zm[i] -= h
            gnum[i] = (m.log_prob(zp) - m.log_prob(zm)) / (2 * h)
        rel = np.max(np.abs(g - gnum) / (np.abs(gnum) + 1e-8))
        assert rel < 1e-4, f"{key}: grad rel err {rel}"


def test_conjugate_reference_analytic():
    """参照后验 = 解析共轭解（独立公式交叉验证）。"""
    m = build_models(["gauss_easy"])["gauss_easy"]
    ref = m.reference_posterior()
    n = m.y.shape[0]
    prec = np.linalg.inv(m.S0) + n * np.linalg.inv(m.S_lik)
    cov = np.linalg.inv(prec)
    mean = cov @ (np.linalg.solve(m.S0, m.m0) + n * (np.linalg.inv(m.S_lik) @ m.y.mean(0)))
    assert np.allclose(ref.cov, cov, rtol=1e-10)
    assert np.allclose(ref.mean, mean, rtol=1e-10)


def test_banana_reference_quadrature_symmetry():
    """banana 参照：z0 边缘对称 → mean_z0 ≈ 0（积分正确性）。"""
    ref = build_models(["banana"])["banana"].reference_posterior()
    assert abs(ref.mean[0]) < 0.05
    assert ref.sd[1] > ref.sd[0] * 1.5  # y 比 x 宽


def test_funnel_reference_marginal_moments():
    """funnel 参照：Var(x_i) = E[e^v] = e^{sv²/2}（解析）。"""
    m = build_models(["funnel"])["funnel"]
    ref = m.reference_posterior()
    expected_sd = math.exp(m.sv**2 / 4.0)
    assert np.allclose(ref.sd, expected_sd, rtol=1e-10)


# ---------------------------------------------------------------------------
# HMC 力学：可逆性 / 能量守恒 / 度量一致性
# ---------------------------------------------------------------------------
def test_leapfrog_reversibility():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    met = Metric(cov=m.reference_posterior().cov)
    rng = np.random.default_rng(5)
    z = rng.normal(size=m.dim)
    lp, g = c.log_prob_grad(z)
    r = met.sample_momentum(rng)
    eps = 0.01
    z1, r1, _, g1 = leapfrog(c, met, z, r, g, eps)
    z0, r0, lp0, _ = leapfrog(c, met, z1, -r1, g1, eps)
    assert np.allclose(z0, z, atol=1e-10)
    assert np.allclose(r0, -r, atol=1e-10)
    assert abs(lp0 - lp) < 1e-10


def test_leapfrog_energy_conservation_small_step():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    met = unit_metric(m.dim)
    rng = np.random.default_rng(6)
    z = rng.normal(size=m.dim)
    lp, g = c.log_prob_grad(z)
    r = met.sample_momentum(rng)
    h0 = hamiltonian(lp, met, r)
    _, r1, lp1, _ = leapfrog(c, met, z, r, g, 0.01)
    assert abs(h0 - hamiltonian(lp1, met, r1)) < 1e-3


def test_metric_velocity_kinetic_consistency():
    rng = np.random.default_rng(7)
    cov = rng.normal(size=(4, 4))
    cov = cov @ cov.T + 4 * np.eye(4)
    met = Metric(cov=cov)
    r = rng.normal(size=4)
    v = met.velocity(r)
    assert np.allclose(v, np.linalg.solve(cov, r), rtol=1e-10)
    assert abs(met.kinetic_energy(r) - 0.5 * r @ np.linalg.solve(cov, r)) < 1e-12


def test_metric_rejects_non_pd():
    with pytest.raises(InferenceError):
        Metric(diag=np.array([1.0, -2.0]))
    with pytest.raises(InferenceError):
        Metric(cov=np.array([[1.0, 2.0], [2.0, 1.0]]))  # 特征值 -1


def test_find_reasonable_epsilon_positive():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    z = np.zeros(m.dim)
    lp, g = c.log_prob_grad(z)
    for target in (0.6, 0.8, 0.95):
        eps = find_reasonable_epsilon(
            c, unit_metric(m.dim), np.random.default_rng(8), z, lp, g, target=target
        )
        assert 1e-8 < eps < 1e6


def test_nuts_transition_accept_stat_bounds():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = fast_cfg()
    z = np.zeros(m.dim)
    lp, g = c.log_prob_grad(z)
    z1, _, _, astat, ndiv = nuts_transition(
        c, unit_metric(m.dim), cfg, np.random.default_rng(9), z, lp, g, 0.2
    )
    assert 0.0 <= astat <= 1.0
    assert ndiv >= 0
    assert np.all(np.isfinite(z1))


# ---------------------------------------------------------------------------
# 共轭高斯：NUTS / RWM / Laplace / ADVI vs 解析解
# ---------------------------------------------------------------------------
def _moment_errors(draws, key):
    m = build_models([key])[key]
    ref = m.reference_posterior()
    return compare_to_reference(draws, ref), ref


def test_nuts_recovers_conjugate_posterior():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = Config(n_chains=4, max_draws_per_chain=800, warmup_standard=500, cost_budget=120_000)
    res = NUTSSampler().run(c, cfg, np.random.default_rng(42), Budget(c, cfg.cost_budget))
    sc, _ref = _moment_errors(res.draws, "gauss_easy")
    rep = summarize_draws(res.draws)
    assert sc.mean_z_err < 0.15
    assert sc.sd_log_err < 0.15
    assert rep["rhat_max"] < 1.05
    assert rep["ess_bulk_min"] > 100


def test_rwm_recovers_mean():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = Config(n_chains=2, max_draws_per_chain=3000, warmup_standard=1000, cost_budget=200_000)
    res = RWMSampler().run(c, cfg, np.random.default_rng(10), Budget(c, cfg.cost_budget))
    sc, _ = _moment_errors(res.draws, "gauss_easy")
    assert sc.mean_z_err < 0.25


def test_laplace_exact_on_gaussian():
    """Laplace 在高斯后验上应精确恢复 mean 与 cov。"""
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = fast_cfg()
    res = LaplaceSampler().run(c, cfg, np.random.default_rng(11), Budget(c, cfg.cost_budget))
    sc, _ref = _moment_errors(res.draws, "gauss_easy")
    assert sc.mean_z_err < 0.10
    assert sc.sd_log_err < 0.10
    assert sc.corr_err < 0.10


def test_advi_elbo_improves():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    fit = fit_advi(c, np.random.default_rng(12), steps=1500, mc_samples=8, lr=0.05, full_rank=True)
    hist = fit.elbo_history
    k = max(len(hist) // 10, 1)
    assert np.mean(hist[-k:]) > np.mean(hist[:k])  # ELBO 上升
    assert len(hist) == 1500


def test_advi_fullrank_beats_diagonal_on_correlated():
    """全秩 ADVI 在相关高斯上的协方差 Frobenius 误差 < 平均场。"""
    m = build_models(["gauss_hard"])["gauss_hard"]
    ref = m.reference_posterior()
    errs = {}
    for fr in (False, True):
        c = CountingModel(m)
        fit = fit_advi(
            c,
            np.random.default_rng(13),
            steps=2500,
            mc_samples=6,
            lr=0.08,
            full_rank=fr,
        )
        errs[fr] = np.linalg.norm(fit.cov - ref.cov) / np.linalg.norm(ref.cov)
    assert errs[True] < errs[False]


# ---------------------------------------------------------------------------
# NUTS 不变量：Detailed balance 代理（两种子同分布）+ 发散率
# ---------------------------------------------------------------------------
def test_nuts_two_seeds_agree():
    m = build_models(["banana"])["banana"]
    ref = m.reference_posterior()
    cfg = Config(
        n_chains=2,
        max_draws_per_chain=1000,
        warmup_standard=500,
        target_accept=0.95,
        cost_budget=200_000,
    )
    means = []
    for seed in (100, 200):
        c = CountingModel(m)
        res = NUTSSampler().run(c, cfg, np.random.default_rng(seed), Budget(c, cfg.cost_budget))
        means.append(res.draws.reshape(-1, 2).mean(axis=0))
    # 两个独立运行给出一致的均值（单位：参照 sd）
    diff = np.abs(means[0] - means[1]) / ref.sd
    assert np.max(diff) < 0.5


def test_funnel_divergences_expected_but_bounded():
    """居中 funnel：NUTS 会发散（诚实行为），但发散率 < 20%。"""
    m = build_models(["funnel"])["funnel"]
    c = CountingModel(m)
    cfg = Config(n_chains=2, max_draws_per_chain=400, warmup_standard=300, cost_budget=120_000)
    res = NUTSSampler().run(c, cfg, np.random.default_rng(14), Budget(c, cfg.cost_budget))
    assert res.divergences > 0  # 居中漏斗必然发散（文献结论）
    assert res.divergences < 0.2 * res.total_draws


# ---------------------------------------------------------------------------
# 诊断不变量
# ---------------------------------------------------------------------------
def test_split_rhat_detects_nonconvergence():
    rng = np.random.default_rng(15)
    good = rng.normal(0, 1, size=(4, 500, 2))
    bad = good.copy()
    bad[0] += 5.0  # 链间位移
    assert split_rhat(good) < 1.02
    assert split_rhat(bad) > 1.5


def test_ess_bounds_and_white_noise():
    rng = np.random.default_rng(16)
    white = rng.normal(0, 1, size=(4, 1000, 1))
    e = ess_bulk(white)
    assert 200 < e <= 4 * 1000  # 白噪声 ESS≈总数，且不超过


def test_ess_low_for_ar1():
    rng = np.random.default_rng(17)
    x = np.zeros((1, 5000, 1))
    for t in range(1, 5000):
        x[0, t] = 0.95 * x[0, t - 1] + rng.normal()
    assert ess_bulk(x) < 1000  # AR(1) φ=0.95 → ESS ≈ N·(1-φ)/(1+φ) ≈ 128


def test_ess_matches_arviz_if_available():
    """外部交叉验证：与 ArviZ 的 ESS / R̂ 对照（良态链 25% 容差）。

    ArviZ 采用 rank-normalize + folding 口径，链病态时两者会分道扬镳——
    这种差异本身是对的，因此只对白噪声 / AR(1) 这类良态情况做对照。
    """
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(40)
    white = rng.normal(size=(4, 1000, 2))
    ar1 = np.zeros((4, 3000, 2))
    for c in range(4):
        for t in range(1, 3000):
            ar1[c, t] = 0.8 * ar1[c, t - 1] + rng.normal(size=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for chains, label in ((white, "white"), (ar1, "ar1")):
            ds = az.convert_to_dataset(chains)
            az_ess = float(np.min(np.asarray(az.ess(ds).to_array()).reshape(-1)))
            az_rhat = float(np.max(np.asarray(az.rhat(ds).to_array()).reshape(-1)))
            assert abs(ess_bulk(chains) / az_ess - 1.0) < 0.25, f"{label}: ESS 与 ArviZ 偏差过大"
            assert abs(split_rhat(chains) - az_rhat) < 0.02, f"{label}: R̂ 与 ArviZ 偏差过大"


def test_compare_perfect_draws_zero_error():
    m = build_models(["gauss_easy"])["gauss_easy"]
    ref = m.reference_posterior()
    rng = np.random.default_rng(18)
    draws = rng.multivariate_normal(ref.mean, ref.cov, size=4000)[None, ...]
    sc = compare_to_reference(draws, ref)
    assert sc.mean_z_err < 0.15
    assert sc.sd_log_err < 0.05


# ---------------------------------------------------------------------------
# 成本核算与预算护栏
# ---------------------------------------------------------------------------
def test_cost_accounting_exact():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    rng = np.random.default_rng(19)
    z = rng.normal(size=m.dim)
    c.log_prob(z)
    c.log_prob_grad(z)
    c.log_prob_grad(z)
    expected = 2 + 1 / m.dim
    assert abs(c.cost() - expected) < 1e-12


def test_budget_guards_cost():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = fast_cfg(cost_budget=5_000)
    with pytest.raises(InferenceError):
        NUTSSampler(n_chains=1, warmup=200).run(c, cfg, np.random.default_rng(20), Budget(c, 1_000))
    assert c.cost() < 3_000  # 超支有限


def test_vario_budget_respected():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = fast_cfg(cost_budget=20_000)
    VarioNUTSSampler(allow_fallback=True).run(c, cfg, np.random.default_rng(21), Budget(c, 20_000))
    assert c.cost() <= 22_000  # 少量超支允许（单步粒度）


# ---------------------------------------------------------------------------
# 旗舰 VarioNUTS：鲁棒性 + 回退 + 门控
# ---------------------------------------------------------------------------
def test_vario_nuts_gate_or_fallback_on_hard():
    """gauss_hard 上无论门控通过或回退，最终 rhat 必须 ≤ 1.1。"""
    m = build_models(["gauss_hard"])["gauss_hard"]
    c = CountingModel(m)
    cfg = Config(
        n_chains=4,
        max_draws_per_chain=500,
        warmup_standard=400,
        warmup_vario=120,
        cost_budget=120_000,
    )
    res = VarioNUTSSampler().run(c, cfg, np.random.default_rng(22), Budget(c, cfg.cost_budget))
    rep = summarize_draws(res.draws)
    assert rep["rhat_max"] < 1.1
    assert res.meta["gate_ok"] or res.meta["fallback_used"]


def test_vario_nuts_mf_matches_nuts_on_easy():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = Config(
        n_chains=2,
        max_draws_per_chain=400,
        warmup_standard=300,
        warmup_vario=100,
        cost_budget=80_000,
    )
    res = VarioNUTSMini().run(c, cfg, np.random.default_rng(23), Budget(c, cfg.cost_budget))
    sc, _ = _moment_errors(res.draws, "gauss_easy")
    assert sc.mean_z_err < 0.2
    assert sc.sd_log_err < 0.2


def test_advi_sampler_draws_from_q():
    m = build_models(["gauss_easy"])["gauss_easy"]
    c = CountingModel(m)
    cfg = fast_cfg()
    res = ADVISampler(name="advi_full", full_rank=True).run(
        c, cfg, np.random.default_rng(24), Budget(c, cfg.cost_budget)
    )
    assert res.draws.shape == (1, cfg.max_draws_per_chain, m.dim)
    assert np.all(np.isfinite(res.draws))


# ---------------------------------------------------------------------------
# 注册表与流水线
# ---------------------------------------------------------------------------
def test_registry_complete():
    from chainforge.inference import METHOD_BUILDERS

    expected = {
        "vario_nuts",
        "vario_nuts_mf",
        "nuts",
        "nuts_dense",
        "hmc_static",
        "rwm",
        "advi",
        "advi_full",
        "laplace",
    }
    assert set(METHOD_BUILDERS) == expected
    assert set(MODEL_BUILDERS) == set(ALL_MODELS)
    methods = build_methods(["nuts", "rwm"])
    assert [s.name for s in methods] == ["nuts", "rwm"]


def test_run_one_produces_row():
    row = run_one("gauss_easy", "nuts", fast_cfg(), seed=25)
    assert row["status"] == "ok"
    assert row["mean_z_err"] < 0.3


def test_verify_flagship_runs():
    rows = [
        run_one("gauss_easy", "vario_nuts", fast_cfg(), seed=26),
        run_one("gauss_easy", "nuts", fast_cfg(), seed=26),
    ]
    # 不强断言通过（小预算下），只验证可执行且字段齐全
    assert all(r["status"] == "ok" for r in rows)
    assert isinstance(verify_flagship(rows), bool)


def test_config_defaults_sane():
    cfg = Config()
    assert cfg.n_chains >= 2
    assert cfg.cost_budget >= 10_000
    assert 0 < cfg.advi_share < 1
    assert cfg.rhat_gate >= 1.0
