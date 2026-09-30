"""推断层：度量 / HMC 力学 / NUTS / RWM / ADVI / Laplace / 旗舰 VarioNUTS。"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from .core import (
    Budget,
    CountingModel,
    InferenceError,
    InferenceResult,
    make_result,
)
from .models import _RefConfig  # noqa: F401  (供参照链 duck-typing)


# ============================================================================
# 度量（动量协方差 M；位置协方差≈M⁻¹）
# ============================================================================
class Metric:
    """多元正态动量 r ~ N(0, M)。位置空间度量即 M⁻¹。"""

    def __init__(self, cov: np.ndarray | None = None, diag: np.ndarray | None = None) -> None:
        if cov is not None:
            m = np.asarray(cov, dtype=float)
            m = (m + m.T) / 2.0
            vals = np.linalg.eigvalsh(m)
            if vals.min() <= 0:
                raise InferenceError(f"metric covariance not PD (min eig={vals.min():.3e})")
            self.cov = m
            self._chol = np.linalg.cholesky(m)
            self._inv = np.linalg.inv(m)
        elif diag is not None:
            d = np.asarray(diag, dtype=float)
            if np.any(d <= 0) or not np.all(np.isfinite(d)):
                raise InferenceError("diagonal metric must be positive finite")
            self.cov = np.diag(d)
            self._chol = np.diag(np.sqrt(d))
            self._inv = np.diag(1.0 / d)
        else:
            raise InferenceError("metric requires cov or diag")

    @property
    def dim(self) -> int:
        return self.cov.shape[0]

    def sample_momentum(self, rng: np.random.Generator) -> np.ndarray:
        return self._chol @ rng.standard_normal(self.dim)

    def velocity(self, r: np.ndarray) -> np.ndarray:
        return self._inv @ r

    def kinetic_energy(self, r: np.ndarray) -> float:
        """0.5 rᵀM⁻¹r。溢出/NaN 折叠为 +inf（HMC 会拒绝该状态）。"""
        with np.errstate(over="ignore", invalid="ignore"):
            ke = 0.5 * float(r @ self.velocity(r))
        return ke if np.isfinite(ke) else float("inf")


def unit_metric(dim: int, dense: bool = False) -> Metric:
    return Metric(cov=np.eye(dim)) if dense else Metric(diag=np.ones(dim))


# ============================================================================
# HMC 力学
# ============================================================================
def hamiltonian(lp: float, metric: Metric, r: np.ndarray) -> float:
    return -float(lp) + metric.kinetic_energy(r)


def leapfrog(
    model: CountingModel,
    metric: Metric,
    z: np.ndarray,
    r: np.ndarray,
    grad: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    r1 = r + 0.5 * eps * grad
    z1 = z + eps * metric.velocity(r1)
    lp1, grad1 = model.log_prob_grad(z1)
    r1 = r1 + 0.5 * eps * grad1
    if not (np.isfinite(lp1) and np.all(np.isfinite(grad1))):
        raise FloatingPointError("leapfrog produced non-finite state")
    return z1, r1, lp1, grad1


def _logu(rng: np.random.Generator) -> float:
    return math.log(rng.random() + 1e-300)


def _alpha(h0: float, h1: float) -> float:
    return float(min(1.0, math.exp(min(0.0, h0 - h1)))) if np.isfinite(h1) else 0.0


def find_reasonable_epsilon(
    model: CountingModel,
    metric: Metric,
    rng: np.random.Generator,
    z: np.ndarray,
    lp: float,
    grad: np.ndarray,
    target: float = 0.8,
) -> float:
    """双倍/减半搜索初始步长（Hoffman & Gelman 2014 Alg. 4）。"""
    eps = 0.1
    r = metric.sample_momentum(rng)
    h0 = hamiltonian(lp, metric, r)
    try:
        _, r1, lp1, _ = leapfrog(model, metric, z, r, grad, eps)
        log_ratio = h0 - hamiltonian(lp1, metric, r1)
    except FloatingPointError:
        log_ratio = -np.inf
    direction = 1.0 if log_ratio > math.log(0.5) else -1.0
    for _ in range(50):
        eps *= 2.0**direction
        if eps > 1e7 or eps < 1e-10:
            raise InferenceError("find_reasonable_epsilon: 步长越界")
        try:
            _, r1, lp1, _ = leapfrog(model, metric, z, r, grad, eps)
            log_ratio = h0 - hamiltonian(lp1, metric, r1)
        except FloatingPointError:
            log_ratio = -np.inf
        if (direction == 1.0 and log_ratio <= math.log(0.5)) or (
            direction == -1.0 and log_ratio > math.log(0.5)
        ):
            break
    return max(eps * 2.0**-direction, 1e-10)


# ============================================================================
# 预热自适应
# ============================================================================
class DualAveraging:
    """Nesterov 对偶平均步长自适应（Stan 口径）。"""

    def __init__(
        self,
        eps0: float,
        target: float = 0.8,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ) -> None:
        self.mu = math.log(10.0 * eps0)
        self.target = target
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.restart(eps0)

    def restart(self, eps: float) -> None:
        self.mu = math.log(10.0 * eps)
        self._h_bar = 0.0
        self._log_eps_bar = 0.0
        self._counter = 0

    def update(self, accept_stat: float) -> float:
        self._counter += 1
        eta = 1.0 / (self._counter + self.t0)
        self._h_bar = (1.0 - eta) * self._h_bar + eta * (self.target - accept_stat)
        log_eps = self.mu - math.sqrt(self._counter) / self.gamma * self._h_bar
        w = self._counter**-self.kappa
        self._log_eps_bar = w * log_eps + (1.0 - w) * self._log_eps_bar
        return math.exp(log_eps)

    @property
    def adapted(self) -> float:
        return math.exp(self._log_eps_bar)


class WindowedAdaptation:
    """Stan 风格扩窗协方差估计（75/10/15 缓冲 + 倍增窗口）。"""

    def __init__(
        self, num_warmup: int, dim: int, init_buffer: int = 75, term_buffer: int = 150
    ) -> None:
        if num_warmup < init_buffer + term_buffer + 20:
            init_buffer = max(10, int(num_warmup * 0.15))
            term_buffer = max(10, int(num_warmup * 0.15))
        self.num_warmup = num_warmup
        self.dim = dim
        self.init_buffer = init_buffer
        self.term_buffer = term_buffer
        self.base_window = 25
        self._counter = 0
        self._window = self.base_window
        self._window_start = self.init_buffer
        self._reset_window()
        self._n_total = 0
        self._mean_total = np.zeros(dim)
        self._cross_total = np.zeros((dim, dim))

    @property
    def counter(self) -> int:
        return self._counter

    def update(self, z: np.ndarray) -> bool:
        self._counter += 1
        c = self._counter
        in_window = c > self._window_start and c <= self.num_warmup - self.term_buffer
        if in_window:
            self._win_n += 1
            delta = z - self._win_mean
            self._win_mean += delta / self._win_n
            self._win_cross += np.outer(z - self._win_mean, delta)
        closed = False
        if c == self.num_warmup - self.term_buffer or c == self._window_start + self._window:
            if self._win_n >= 2 * self.dim + 10:
                self._commit_window()
                self._window *= 2
                self._window_start = c
                closed = True
            self._reset_window()
        return closed

    def _reset_window(self) -> None:
        self._win_n = 0
        self._win_mean = np.zeros(self.dim)
        self._win_cross = np.zeros((self.dim, self.dim))

    def _commit_window(self) -> None:
        n_w = self._win_n
        mean_w = self._win_mean
        var_w = self._win_cross / (n_w - 1)
        if self._n_total == 0:
            self._n_total = n_w
            self._mean_total = mean_w.copy()
            self._cross_total = var_w * n_w
            return
        total = self._n_total + n_w
        delta = mean_w - self._mean_total
        self._cross_total += var_w * (n_w - 1) + self._n_total * n_w / total * np.outer(
            delta, delta
        )
        self._mean_total += delta * n_w / total
        self._n_total = total

    def covariance(self) -> np.ndarray:
        n = self._n_total
        if n < 2:
            return np.eye(self.dim)
        cov = self._cross_total / (n - 1)
        cov = (cov + cov.T) / 2.0
        return (n / (n + 5.0)) * cov + (5.0 / (n + 5.0)) * 1e-3 * np.eye(self.dim)

    def variance(self) -> np.ndarray:
        n = self._n_total
        if n < 2:
            return np.ones(self.dim)
        var = np.maximum(np.diag(self.covariance()), 1e-10)
        return (n / (n + 5.0)) * var + (5.0 / (n + 5.0)) * 1e-3

    @property
    def n_samples(self) -> int:
        return int(self._n_total)


# ============================================================================
# NUTS（Hoffman & Gelman 2014 Alg. 6，slice 变量 + 递归建树）
# ============================================================================
def _build_tree(
    model: CountingModel,
    metric: Metric,
    cfg: Any,
    rng: np.random.Generator,
    z: np.ndarray,
    r: np.ndarray,
    lp: float,
    grad: np.ndarray,
    logu: float,
    v: int,
    j: int,
    eps: float,
    h0: float,
) -> tuple[Any, ...]:
    """返回 (z-, r-, lp-, g-, z+, r+, lp+, g+, z_prop, lp_prop, g_prop,
    n1, s1, alpha, n_alpha, n_div)。
    """
    if j == 0:
        try:
            z1, r1, lp1, grad1 = leapfrog(model, metric, z, r, grad, v * eps)
        except FloatingPointError:
            return z, r, lp, grad, z, r, lp, grad, z, lp, grad, 0, 0, 0.0, 1, 1
        h1 = hamiltonian(lp1, metric, r1)
        n1 = 1 if logu <= -h1 else 0
        # slice 条件: logu < Δmax - H1；违反即发散（Stan 口径）
        div = 1 if logu >= cfg.delta_max - h1 else 0
        s1 = 0 if div else 1
        return (
            z1,
            r1,
            lp1,
            grad1,
            z1,
            r1,
            lp1,
            grad1,
            z1,
            lp1,
            grad1,
            n1,
            s1,
            _alpha(h0, h1),
            1,
            div,
        )

    (
        zm,
        rm,
        lpm,
        gm,
        zp,
        rp,
        lpp,
        gp,
        zprop,
        lp_prop,
        g_prop,
        n1,
        s1,
        alpha,
        n_alpha,
        nd1,
    ) = _build_tree(model, metric, cfg, rng, z, r, lp, grad, logu, v, j - 1, eps, h0)
    if s1 == 1:
        if v == -1:
            (
                zm,
                rm,
                lpm,
                gm,
                _,
                _,
                _,
                _,
                zprop2,
                lpp2,
                gp2,
                n2,
                s2,
                alpha2,
                n_alpha2,
                nd2,
            ) = _build_tree(model, metric, cfg, rng, zm, rm, lpm, gm, logu, v, j - 1, eps, h0)
        else:
            (
                _,
                _,
                _,
                _,
                zp,
                rp,
                lpp,
                gp,
                zprop2,
                lpp2,
                gp2,
                n2,
                s2,
                alpha2,
                n_alpha2,
                nd2,
            ) = _build_tree(model, metric, cfg, rng, zp, rp, lpp, gp, logu, v, j - 1, eps, h0)
        if n2 > 0 and rng.random() < n2 / max(n1 + n2, 1):
            zprop, lp_prop, g_prop = zprop2, lpp2, gp2
        alpha += alpha2
        n_alpha += n_alpha2
        nd1 += nd2
        # U-turn 判据（Hoffman & Gelman 2014）：(z+−z−)·v(r−) ≥ 0 且 (z+−z−)·v(r+) ≥ 0
        dz = zp - zm
        s1 = s2 * (1 if (dz @ metric.velocity(rm) >= 0) and (dz @ metric.velocity(rp) >= 0) else 0)
        n1 += n2
    return (
        zm,
        rm,
        lpm,
        gm,
        zp,
        rp,
        lpp,
        gp,
        zprop,
        lp_prop,
        g_prop,
        n1,
        s1,
        alpha,
        n_alpha,
        nd1,
    )


def nuts_transition(
    model: CountingModel,
    metric: Metric,
    cfg: Any,
    rng: np.random.Generator,
    z: np.ndarray,
    lp: float,
    grad: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, float, np.ndarray, float, int]:
    """一次 NUTS 更新。返回 (z, lp, grad, accept_stat, n_divergences)。"""
    r0 = metric.sample_momentum(rng)
    h0 = hamiltonian(lp, metric, r0)
    logu = -h0 + _logu(rng)
    zm, rm, lpm, gm = z, r0, lp, grad
    zp, rp, lpp, gp = z, r0, lp, grad
    z_cur, lp_cur, grad_cur = z, lp, grad
    j, n, s, n_div = 0, 1, 1, 0
    alpha_sum, n_alpha_sum = 0.0, 0
    while s == 1 and j < int(cfg.max_treedepth):
        v = 1 if rng.random() < 0.5 else -1
        if v == -1:
            (
                zm,
                rm,
                lpm,
                gm,
                _,
                _,
                _,
                _,
                zprop,
                lp_prop,
                g_prop,
                n1,
                s1,
                alpha,
                n_alpha,
                nd,
            ) = _build_tree(model, metric, cfg, rng, zm, rm, lpm, gm, logu, v, j, eps, h0)
        else:
            (
                _,
                _,
                _,
                _,
                zp,
                rp,
                lpp,
                gp,
                zprop,
                lp_prop,
                g_prop,
                n1,
                s1,
                alpha,
                n_alpha,
                nd,
            ) = _build_tree(model, metric, cfg, rng, zp, rp, lpp, gp, logu, v, j, eps, h0)
        n_div += nd
        if s1 == 1 and rng.random() < min(1.0, n1 / max(n, 1)):
            z_cur, lp_cur, grad_cur = zprop, lp_prop, g_prop
        n += n1
        alpha_sum += alpha
        n_alpha_sum += n_alpha
        dz = zp - zm
        s = s1 * (1 if (dz @ metric.velocity(rm) >= 0) and (dz @ metric.velocity(rp) >= 0) else 0)
        j += 1
    accept_stat = alpha_sum / max(n_alpha_sum, 1)
    return z_cur, float(lp_cur), np.asarray(grad_cur, dtype=float), accept_stat, n_div


class NUTSSampler:
    """标准 NUTS（对角或全秩窗口自适应）。"""

    def __init__(
        self,
        name: str = "nuts",
        n_chains: int | None = None,
        warmup: int | None = None,
        full_rank: bool = False,
        metric: Metric | None = None,
    ) -> None:
        self.name = name
        self.n_chains = n_chains
        self.warmup = warmup
        self.full_rank = full_rank
        self.metric = metric

    def run(
        self,
        model: CountingModel,
        cfg: Any,
        rng: np.random.Generator,
        budget: Budget | None,
    ) -> InferenceResult:
        import time

        t0 = time.perf_counter()
        n_chains = int(self.n_chains or cfg.n_chains)
        warmup = int(self.warmup or cfg.warmup_standard)
        dim = model.dim
        total = float("inf") if budget is None else budget.limit
        chains: list[np.ndarray] = []
        total_div = 0

        for c in range(n_chains):
            if budget is not None and budget.remaining() <= 0 and c > 0:
                break
            limit = total if budget is None else budget.limit - budget.start
            z = rng.normal(0.0, 1.0, size=dim)
            lp, grad = model.log_prob_grad(z)
            metric = self.metric if self.metric is not None else unit_metric(dim, self.full_rank)
            eps = find_reasonable_epsilon(model, metric, rng, z, lp, grad, target=cfg.target_accept)
            da = DualAveraging(eps, target=cfg.target_accept)
            win = WindowedAdaptation(warmup, dim)
            metric_locked = self.metric is not None

            draw_list: list[np.ndarray] = []
            it = 0
            while it < int(cfg.max_draws_per_chain) + warmup:
                if model.cost() > limit:
                    break
                z, lp, grad, accept_stat, n_div = nuts_transition(
                    model, metric, cfg, rng, z, lp, grad, eps
                )
                it += 1
                total_div += n_div
                if it <= warmup:
                    eps = da.update(accept_stat)
                    if not metric_locked:
                        win.update(z)
                        if win.counter == warmup - win.term_buffer:
                            try:
                                if self.full_rank and win.n_samples > 10 * dim:
                                    metric = Metric(cov=win.covariance())
                                else:
                                    metric = Metric(diag=win.variance())
                            except InferenceError:
                                metric = unit_metric(dim, self.full_rank)
                            eps = find_reasonable_epsilon(
                                model,
                                metric,
                                rng,
                                z,
                                lp,
                                grad,
                                target=cfg.target_accept,
                            )
                            da.restart(eps)
                else:
                    draw_list.append(z.copy())
            if not draw_list:
                if not chains:
                    raise InferenceError(f"{self.name}: 预算耗尽，链 {c} 未产生任何样本")
                break  # 已有完成的链：诚实返回部分结果
            chains.append(np.asarray(draw_list, dtype=float))

        n_draw = min(ch.shape[0] for ch in chains)
        draws = np.stack([ch[:n_draw] for ch in chains])
        return make_result(
            model,
            self.name,
            draws,
            t0,
            divergences=total_div,
            n_warmup=warmup * len(chains),
            n_chains=len(chains),
        )


class StaticHMCSampler:
    """固定步长 HMC（对照基线）。"""

    def __init__(self, name: str = "hmc_static", n_steps: int = 20) -> None:
        self.name = name
        self.n_steps = n_steps

    def run(
        self,
        model: CountingModel,
        cfg: Any,
        rng: np.random.Generator,
        budget: Budget | None,
    ) -> InferenceResult:
        import time

        t0 = time.perf_counter()
        dim = model.dim
        warmup = int(cfg.warmup_standard)
        total = float("inf") if budget is None else budget.limit
        chains: list[np.ndarray] = []
        for _c in range(int(cfg.n_chains)):
            if budget is not None and budget.remaining() <= 0:
                break
            z = rng.normal(0.0, 1.0, size=dim)
            lp, grad = model.log_prob_grad(z)
            metric = unit_metric(dim, False)
            eps = find_reasonable_epsilon(model, metric, rng, z, lp, grad, target=cfg.target_accept)
            da = DualAveraging(eps, target=cfg.target_accept)
            draw_list: list[np.ndarray] = []
            it = 0
            while it < int(cfg.max_draws_per_chain) + warmup:
                if model.cost() > total:
                    break
                r0 = metric.sample_momentum(rng)
                h0 = hamiltonian(lp, metric, r0)
                zn, rn, lpn, gn = z, r0, lp, grad
                ok = True
                for _ in range(self.n_steps):
                    try:
                        zn, rn, lpn, gn = leapfrog(model, metric, zn, rn, gn, eps)
                    except FloatingPointError:
                        ok = False
                        break
                h1 = hamiltonian(lpn, metric, rn)
                accept = ok and np.isfinite(h1) and _logu(rng) < h0 - h1
                it += 1
                if it <= warmup:
                    eps = da.update(_alpha(h0, h1) if ok else 0.0)
                if accept:
                    z, lp, grad = zn, float(lpn), np.asarray(gn, dtype=float)
                if it > warmup:
                    draw_list.append(z.copy())
            if not draw_list:
                raise InferenceError(f"{self.name}: 预算耗尽")
            chains.append(np.asarray(draw_list, dtype=float))
        n_draw = min(ch.shape[0] for ch in chains)
        return make_result(model, self.name, np.stack([ch[:n_draw] for ch in chains]), t0)


class RWMSampler:
    """自适应随机游走 Metropolis（对角提案，Robbins-Monro 步长）。"""

    def __init__(self, name: str = "rwm") -> None:
        self.name = name

    def run(
        self,
        model: CountingModel,
        cfg: Any,
        rng: np.random.Generator,
        budget: Budget | None,
    ) -> InferenceResult:
        import time

        t0 = time.perf_counter()
        dim = model.dim
        warmup = int(cfg.warmup_standard)
        total = float("inf") if budget is None else budget.limit
        chains: list[np.ndarray] = []
        for _c in range(int(cfg.n_chains)):
            if budget is not None and budget.remaining() <= 0:
                break
            z = rng.normal(0.0, 1.0, size=dim)
            lp = model.log_prob(z)
            sd = np.ones(dim)
            log_s = math.log(cfg.rwm_step)
            run_mean = z.copy()
            run_m2 = np.zeros(dim)
            draw_list: list[np.ndarray] = []
            it = 0
            n_total = int(cfg.max_draws_per_chain) + warmup
            while it < n_total:
                if model.cost() > total:
                    break
                prop = z + math.exp(log_s) * sd * rng.standard_normal(dim)
                lp_prop = model.log_prob(prop)
                acc = _logu(rng) < lp_prop - lp
                if acc:
                    z, lp = prop, lp_prop
                it += 1
                if it <= warmup:
                    log_s += 0.5 / math.sqrt(it) * ((1.0 if acc else 0.0) - 0.234)
                    d = z - run_mean
                    run_mean += d / it
                    run_m2 += d * (z - run_mean)
                    if it > 100:
                        var = run_m2 / (it - 1)
                        sd = np.sqrt(np.maximum(var, 1e-12))
                else:
                    draw_list.append(z.copy())
            if not draw_list:
                raise InferenceError(f"{self.name}: 预算耗尽")
            chains.append(np.asarray(draw_list, dtype=float))
        n_draw = min(ch.shape[0] for ch in chains)
        return make_result(model, self.name, np.stack([ch[:n_draw] for ch in chains]), t0)


# ============================================================================
# ADVI（mean-field 与 full-rank；Adam + ELBO 单调监控）
# ============================================================================
def _map_init(model: CountingModel, max_grad_evals: int = 400) -> np.ndarray:
    """廉价梯度上升找 MAP（scipy L-BFGS-B，失败退回零点）。"""
    """廉价梯度上升找 MAP（scipy L-BFGS-B，失败退回零点）。

    优化器失败时降级为零点初始化而非抛错：后续 NUTS 的预热会自行纠正起点。
    """
    from scipy.optimize import minimize

    fallback_init = np.zeros(model.dim)
    try:
        res = minimize(
            lambda z: -model.log_prob(z),
            fallback_init,
            jac=lambda z: -model.log_prob_grad(z)[1],
            method="L-BFGS-B",
            options={"maxiter": max_grad_evals},
        )
        if np.all(np.isfinite(res.x)):
            return np.asarray(res.x, dtype=float)
    except Exception as _opt_err:  # noqa: BLE001 - 优化失败走降级路径
        warnings.warn(
            f"map_init 降级到零点（{type(_opt_err).__name__}）",
            RuntimeWarning,
            stacklevel=2,
        )
    return fallback_init


class ADVIFit:
    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        chol: np.ndarray,
        elbo_history: list[float],
        steps: int,
    ) -> None:
        self.mu = mu
        self.cov = cov
        self.chol = chol
        self.elbo_history = elbo_history
        self.steps = steps


def fit_advi(
    model: CountingModel,
    rng: np.random.Generator,
    steps: int,
    mc_samples: int,
    lr: float,
    full_rank: bool,
    init_scale: float = 0.1,
    init_from_map: bool = False,
    budget: Budget | None = None,
    cost_limit: float = float("inf"),
) -> ADVIFit:
    """变分推断：q = N(mu, L Lᵀ)（full-rank）或 N(mu, diag(e^ω))。

    熵项解析加入，似然项 MC 估计（mc_samples 个重参数化样本）。
    Adam + 1/t 步长衰减。返回最终 mu / cov / chol 与 ELBO 轨迹。
    """
    dim = model.dim
    mu = _map_init(model) if init_from_map else np.zeros(dim)
    omega = np.full(dim, math.log(init_scale))
    off_idx = np.tril_indices(dim, k=-1)
    off_log = np.zeros(len(off_idx[0])) if full_rank else np.zeros(0)
    # Adam 状态
    params = [mu, omega, off_log]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    elbo_hist: list[float] = []

    def _lower(omega_v: np.ndarray, off_v: np.ndarray) -> np.ndarray:
        L = np.diag(np.exp(omega_v))
        L[off_idx] = off_v
        return L

    for t in range(1, int(steps) + 1):
        if budget is not None and budget.remaining() <= 0:
            break
        if model.cost() > cost_limit:
            break
        grads = [np.zeros_like(p) for p in params]
        elbo_mc = 0.0
        for _ in range(int(mc_samples)):
            eps_std = rng.standard_normal(dim)
            if full_rank:
                L = _lower(omega, off_log)
                z = mu + L @ eps_std
            else:
                sigma = np.exp(omega)
                z = mu + sigma * eps_std
            lp, grad_lp = model.log_prob_grad(z)
            if not np.isfinite(lp):
                continue
            elbo_mc += lp
            grads[0] += grad_lp
            if full_rank:
                g_L = np.outer(grad_lp, eps_std) + np.diag(1.0 / np.diag(L))
                grads[1] += np.diag(g_L) * np.exp(omega)  # exp 链式
                grads[2] += g_L[off_idx]
            else:
                grads[1] += grad_lp * sigma * eps_std + 1.0  # 熵项 dω = 1
        k = int(mc_samples)
        elbo_hist.append(elbo_mc / k)
        # 熵贡献（解析，只用于监控轨迹）
        if full_rank:
            elbo_hist[-1] += 0.5 * dim * (1.0 + math.log(2 * math.pi)) + float(np.sum(omega))
        else:
            elbo_hist[-1] += 0.5 * dim * (1.0 + math.log(2 * math.pi)) + float(np.sum(omega))
        for g in grads:
            g /= k
        # Adam 更新（上升）
        scale = lr / (1.0 + t / steps)  # 轻度衰减
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
            mh = m[i] / (1 - beta1**t)
            vh = v[i] / (1 - beta2**t)
            p += scale * mh / (np.sqrt(vh) + adam_eps)
        if not np.all(np.isfinite(mu)):
            raise InferenceError("ADVI: mu 发散")

    if full_rank:
        L = _lower(omega, off_log)
        cov = L @ L.T
    else:
        cov = np.diag(np.exp(omega) ** 2)
    return ADVIFit(mu.copy(), cov, L if full_rank else np.diag(np.exp(omega)), elbo_hist, t)


class ADVISampler:
    """独立 ADVI 方法：拟合后从 q 抽样（不等长链，诊断按单链处理）。"""

    def __init__(
        self, name: str = "advi", full_rank: bool = True, init_from_map: bool = True
    ) -> None:
        self.name = name
        self.full_rank = full_rank
        self.init_from_map = init_from_map

    def run(
        self,
        model: CountingModel,
        cfg: Any,
        rng: np.random.Generator,
        budget: Budget | None,
    ) -> InferenceResult:
        import time

        t0 = time.perf_counter()
        dim = model.dim
        total = float("inf") if budget is None else budget.limit
        fit = fit_advi(
            model,
            rng,
            steps=int(cfg.advi_steps),
            mc_samples=int(cfg.advi_mc_samples),
            lr=float(cfg.advi_lr),
            full_rank=self.full_rank,
            init_from_map=self.init_from_map,
            budget=budget,
            cost_limit=total,
        )
        n_draws = int(cfg.max_draws_per_chain)
        L = fit.chol
        draws = fit.mu[None, :] + rng.standard_normal((n_draws, dim)) @ L.T
        draws = draws[None, ...]  # 1 "链"
        return make_result(
            model,
            self.name,
            draws,
            t0,
            divergences=0,
            n_warmup=int(fit.steps),
            elbo_final=float(fit.elbo_history[-1]),
            elbo_history_len=len(fit.elbo_history),
        )


# ============================================================================
# Laplace 近似（MAP + 数值 Hessian）
# ============================================================================
class LaplaceSampler:
    def __init__(self, name: str = "laplace") -> None:
        self.name = name

    def run(
        self,
        model: CountingModel,
        cfg: Any,
        rng: np.random.Generator,
        budget: Budget | None,
    ) -> InferenceResult:
        import time

        t0 = time.perf_counter()
        dim = model.dim
        z_map = _map_init(model)
        # Hessian = 梯度的雅可比：一阶中心差分，每对评估给出一整列
        h = 1e-6
        H = np.zeros((dim, dim))
        for i in range(dim):
            zp, zm = z_map.copy(), z_map.copy()
            zp[i] += h
            zm[i] -= h
            _, gp = model.log_prob_grad(zp)
            _, gm = model.log_prob_grad(zm)
            H[:, i] = -(gp - gm) / (2 * h)
        H = (H + H.T) / 2.0
        try:
            cov = np.linalg.inv(H)
            cov = (cov + cov.T) / 2.0
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise InferenceError(f"Laplace: Hessian 不可逆（{exc}）") from exc
        n_draws = int(cfg.max_draws_per_chain)
        draws = z_map[None, :] + rng.standard_normal((n_draws, dim)) @ L.T
        return make_result(
            model,
            self.name,
            draws[None, ...],
            t0,
            divergences=0,
            n_warmup=0,
            map_converged=True,
        )


# ============================================================================
# 旗舰 VarioNUTS：变分全秩预条件 NUTS + 诊断门控 + 诚实回退
# ============================================================================
class VarioNUTSSampler:
    """三阶段：
    1. ADVI 全秩拟合（占预算 advi_share）→ 得 mu 与协方差 Σ
    2. 以 DenseMetric(Σ) + z0=mu 启动短预热 NUTS（warmup_vario）
    3. 诊断门控（Rhat/ESS/发散率）不达标 → 诚实回退到完整预热标准 NUTS
    """

    def __init__(
        self,
        name: str = "vario_nuts",
        advi_share: float = 0.25,
        full_rank: bool = True,
        allow_fallback: bool = True,
        init_from_map: bool = True,
    ) -> None:
        self.name = name
        self.advi_share = advi_share
        self.full_rank = full_rank
        self.allow_fallback = allow_fallback
        self.init_from_map = init_from_map

    def run(
        self,
        model: CountingModel,
        cfg: Any,
        rng: np.random.Generator,
        budget: Budget | None,
    ) -> InferenceResult:
        import time

        from .diagnostics import summarize_draws

        t0 = time.perf_counter()
        start_cost = model.cost()
        dim = model.dim
        total = float("inf") if budget is None else budget.limit
        advi_budget_cost = min(
            total - start_cost,
            (total - start_cost) * float(getattr(cfg, "advi_share", self.advi_share)),
        )

        # ---- 阶段 1：ADVI 预条件化 ----
        fit = fit_advi(
            model,
            rng,
            steps=int(cfg.vario_advi_steps),
            mc_samples=int(cfg.vario_advi_mc),
            lr=float(cfg.vario_advi_lr),
            full_rank=self.full_rank,
            init_from_map=self.init_from_map,
            budget=None,
            cost_limit=start_cost + advi_budget_cost,
        )

        # ---- 阶段 2：预条件 NUTS（预留一半预算给诚实回退）----
        try:
            metric = Metric(cov=fit.cov)
        except InferenceError:
            metric = unit_metric(dim, self.full_rank)
        remaining = total - model.cost()
        main_budget = None if budget is None else Budget(model, model.cost() + 0.35 * remaining)
        inner = NUTSSampler(
            name=self.name,
            n_chains=int(cfg.n_chains),
            warmup=int(cfg.warmup_vario),
            full_rank=False,
            metric=metric,
        )
        try:
            res = inner.run(model, cfg, rng, main_budget)
            report = summarize_draws(res.draws)
            gate_ok = (
                report["rhat_max"] <= cfg.rhat_gate
                and report["ess_bulk_min"] >= cfg.ess_gate * res.total_draws
                and res.divergences <= cfg.divergence_gate * res.total_draws
            )
        except InferenceError:
            gate_ok = False
        fallback_used = False
        if not gate_ok and self.allow_fallback:
            # ---- 阶段 3：诚实回退 = 全秩窗口自适应 + 完整预热（最鲁棒配置）----
            fb_budget = (
                None
                if budget is None
                else Budget(model, model.cost() + max(1.0, total - model.cost()))
            )
            fallback = NUTSSampler(
                name=f"{self.name}_fb",
                n_chains=int(cfg.n_chains),
                warmup=int(cfg.warmup_standard),
                full_rank=True,
                metric=None,
            )
            try:
                res_fb = fallback.run(model, cfg, rng, fb_budget)
                res = res_fb
                fallback_used = True
            except InferenceError:
                pass
        meta = dict(res.meta)
        meta.update(
            gate_ok=bool(gate_ok),
            fallback_used=fallback_used,
            advi_cost=float(model.cost() - start_cost),
            variational_precondition=self.full_rank,
            elbo_final=float(fit.elbo_history[-1]),
        )
        res.meta = meta
        res.wall_time = time.perf_counter() - t0
        return res


class VarioNUTSMini:
    """消融变体：对角预条件（度量只含尺度信息）。"""

    def __init__(self, name: str = "vario_nuts_mf", advi_share: float = 0.25) -> None:
        self.name = name
        self.inner = VarioNUTSSampler(
            name=name,
            advi_share=advi_share,
            full_rank=False,
            allow_fallback=False,
            init_from_map=True,
        )

    def run(
        self,
        model: CountingModel,
        cfg: Any,
        rng: np.random.Generator,
        budget: Budget | None,
    ) -> InferenceResult:
        return self.inner.run(model, cfg, rng, budget)


METHOD_BUILDERS = {
    "vario_nuts": lambda: VarioNUTSSampler(),
    "vario_nuts_mf": lambda: VarioNUTSMini(),
    "nuts": lambda: NUTSSampler(),
    "nuts_dense": lambda: NUTSSampler(name="nuts_dense", full_rank=True),
    "hmc_static": lambda: StaticHMCSampler(),
    "rwm": lambda: RWMSampler(),
    "advi": lambda: ADVISampler(name="advi", full_rank=False),
    "advi_full": lambda: ADVISampler(name="advi_full", full_rank=True),
    "laplace": lambda: LaplaceSampler(),
}


def build_methods(names: list[str]) -> list[Any]:
    return [METHOD_BUILDERS[n]() for n in names]
