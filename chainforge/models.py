"""模型库：5 个后验，每个都带可交叉验证的金标准参照。

参照来源（诚实标注）：
- ConjugateGaussian: 解析解（共轭高斯）
- Banana: 2D 网格数值积分
- Funnel: 解析边缘矩（mean=0, Var=e^{sv²/2}）
- BayesianLogisticRegression: 确定性超长 NUTS（4 链 × 4000 draws，缓存）
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .core import Config, CountingModel, ReferencePosterior


class _Base:
    name: str
    dim: int
    param_names: tuple[str, ...]

    def log_prob(self, z: np.ndarray) -> float:
        raise NotImplementedError

    def log_prob_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    def reference_posterior(self, cfg: Config | None = None) -> ReferencePosterior:
        raise NotImplementedError


class ConjugateGaussian(_Base):
    """z ~ N(m0, S0)；y_i ~ N(z, S_lik)，i=1..n。共轭 → 解析后验。"""

    def __init__(self, dim: int = 5, n: int = 20, kappa: float = 1.0, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.name = "gauss_easy" if kappa <= 1.0 else "gauss_hard"
        self.dim = dim
        self.param_names = tuple(f"z{i}" for i in range(dim))
        self.m0 = rng.normal(0.0, 1.0, size=dim)
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        vals = np.linspace(0.5, 2.0, dim) * (kappa ** (np.arange(dim) / max(dim - 1, 1)))
        self.S0 = (q * vals) @ q.T
        self.S0 = (self.S0 + self.S0.T) / 2.0
        # 似然协方差：正交基 + 显式条件数
        q2, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        vl = np.geomspace(0.4, 0.4 * max(kappa, 10.0), dim)
        self.S_lik = (q2 * vl) @ q2.T
        self.S_lik = (self.S_lik + self.S_lik.T) / 2.0
        self.y = rng.normal(0.0, 1.0, size=(n, dim)) + self.m0

    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.y.shape[0]
        prec = np.linalg.inv(self.S0) + n * np.linalg.inv(self.S_lik)
        cov = np.linalg.inv(prec)
        mean = cov @ (
            np.linalg.solve(self.S0, self.m0)
            + n * (np.linalg.inv(self.S_lik) @ self.y.mean(axis=0))
        )
        return mean, cov

    def log_prob(self, z: np.ndarray) -> float:
        d0 = z - self.m0
        r = self.y - z
        p0 = -0.5 * d0 @ np.linalg.solve(self.S0, d0)
        pl = -0.5 * float(np.einsum("ij,jk,ik->", r, np.linalg.inv(self.S_lik), r))
        return float(p0 + pl)

    def log_prob_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        p0inv = np.linalg.inv(self.S0)
        plinv = np.linalg.inv(self.S_lik)
        d0 = z - self.m0
        r = self.y - z
        lp = -0.5 * d0 @ p0inv @ d0 - 0.5 * float(np.einsum("ij,jk,ik->", r, plinv, r))
        g = -(p0inv @ d0) + plinv @ r.sum(axis=0)
        return float(lp), g

    def reference_posterior(self, cfg: Config | None = None) -> ReferencePosterior:
        mean, cov = self._posterior()
        return ReferencePosterior(mean, np.sqrt(np.diag(cov)), cov, source="analytic")


class Banana(_Base):
    """z0 ~ N(0, s²)；z1 | z0 ~ N(z0² - bend, 1)。2D 网格积分参照。"""

    def __init__(self, s: float = 2.0, bend: float = 1.0) -> None:
        self.name = "banana"
        self.dim = 2
        self.param_names = ("z0", "z1")
        self.s = s
        self.bend = bend

    def log_prob(self, z: np.ndarray) -> float:
        x, y = float(z[0]), float(z[1])
        return float(-0.5 * (x / self.s) ** 2 - 0.5 * (y - x * x + self.bend) ** 2)

    def log_prob_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        x, y = float(z[0]), float(z[1])
        lp = self.log_prob(z)
        gx = -x / self.s**2 + 2.0 * x * (y - x * x + self.bend)
        gy = -(y - x * x + self.bend)
        return lp, np.array([gx, gy])

    def reference_posterior(self, cfg: Config | None = None) -> ReferencePosterior:
        xs = np.linspace(-5.0, 5.0, 1201)
        ys = np.linspace(-4.0, 12.0, 1601)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        logp = -0.5 * (X / self.s) ** 2 - 0.5 * (Y - X * X + self.bend) ** 2
        w = np.exp(logp - logp.max())
        w /= w.sum()
        mean_x = float((w * X).sum())
        mean_y = float((w * Y).sum())
        var_x = float((w * (X - mean_x) ** 2).sum())
        var_y = float((w * (Y - mean_y) ** 2).sum())
        cov_xy = float((w * (X - mean_x) * (Y - mean_y)).sum())
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        return ReferencePosterior(
            np.array([mean_x, mean_y]), np.sqrt(np.diag(cov)), cov, source="quadrature"
        )


class Funnel(_Base):
    """v ~ N(0, sv²)；x_i | v ~ N(0, e^v)，i=1..k。 Neal 漏斗（居中参数化，故意的）。

    边缘矩解析：E[x_i]=0，Var(x_i)=E[e^v]=e^{sv²/2}，Cov(x_i,x_j)=0。
    """

    def __init__(self, k: int = 9, sv: float = 1.0) -> None:
        self.name = "funnel"
        self.dim = k + 1
        self.param_names = ("v", *tuple(f"x{i}" for i in range(k)))
        self.k = k
        self.sv = sv

    def log_prob(self, z: np.ndarray) -> float:
        v, xs = float(z[0]), np.asarray(z[1:])
        w = math.exp(-min(v, 30.0))  # e^{-v}，防下溢溢出
        return float(-0.5 * (v / self.sv) ** 2 - 0.5 * w * float(xs @ xs) - v * self.k / 2.0)

    def log_prob_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        v, xs = float(z[0]), np.asarray(z[1:])
        w = math.exp(-min(v, 30.0))
        lp = self.log_prob(z)
        gv = -v / self.sv**2 + 0.5 * w * float(xs @ xs) - self.k / 2.0
        gxs = -w * xs
        return float(lp), np.concatenate([[gv], gxs])

    def reference_posterior(self, cfg: Config | None = None) -> ReferencePosterior:
        sd = float(np.exp(self.sv**2 / 4.0))
        cov = np.eye(self.dim) * sd**2
        return ReferencePosterior(
            np.zeros(self.dim), np.full(self.dim, sd), cov, source="marginal-moments"
        )


class BayesianLogisticRegression(_Base):
    """w ~ N(0, 2²I)；y ~ Bernoulli(σ(Xw))。参照 = 确定性超长 NUTS（缓存）。"""

    def __init__(self, dim: int = 6, n: int = 60, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        self.name = "logistic"
        self.dim = dim
        self.param_names = tuple(f"w{i}" for i in range(dim))
        self.X = rng.normal(0.0, 1.0, size=(n, dim))
        self.X = np.concatenate([np.ones((n, 1)), self.X[:, 1:]], axis=1)
        self.w_true = rng.normal(0.0, 1.5, size=dim)
        self.y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(self.X @ self.w_true)))).astype(float)

    def log_prob(self, z: np.ndarray) -> float:
        logits = self.X @ z
        b = np.logaddexp(0.0, logits)
        ll = float(np.sum(self.y * logits) - np.sum(b))
        return float(ll - 0.5 * float(z @ z) / 4.0)

    def log_prob_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        with np.errstate(over="ignore"):
            p = 1.0 / (1.0 + np.exp(-(self.X @ z)))
        g = self.X.T @ (self.y - p) - z / 4.0
        return self.log_prob(z), g

    def reference_posterior(self, cfg: Config | None = None) -> ReferencePosterior:
        return _logistic_reference()


@lru_cache(maxsize=1)
def _logistic_reference() -> ReferencePosterior:
    """确定性长链 NUTS 作为金标准（与本库实现交叉，见 tests 的梯度检验兜底）。"""
    from .inference import NUTSSampler  # 延迟导入避免环

    model = BayesianLogisticRegression()
    counter = CountingModel(model)
    sampler = NUTSSampler(n_chains=4, warmup=1200, full_rank=True)
    res = sampler.run(
        counter,
        _RefConfig(),
        np.random.default_rng(99),
        budget=None,
    )
    draws = res.draws.reshape(-1, model.dim)
    mean = draws.mean(axis=0)
    cov = np.cov(draws.T)
    return ReferencePosterior(mean, np.sqrt(np.diag(cov)), cov, source="long-nuts")


class _RefConfig:
    """参照链专用配置（不受全局 cost_budget 影响）。"""

    n_chains = 4
    max_draws_per_chain = 3000
    warmup_standard = 1200
    warmup_vario = 300
    max_treedepth = 10
    target_accept = 0.9
    delta_max = 1000.0
    seed = 99


MODEL_BUILDERS = {
    "gauss_easy": lambda: ConjugateGaussian(dim=5, kappa=1.0),
    "gauss_hard": lambda: ConjugateGaussian(dim=5, kappa=30.0),
    "banana": lambda: Banana(),
    "funnel": lambda: Funnel(),
    "logistic": lambda: BayesianLogisticRegression(),
}


def build_models(keys: list[str]) -> dict[str, _Base]:
    return {k: MODEL_BUILDERS[k]() for k in keys}
