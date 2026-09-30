"""诊断：split-Rhat / bulk-ESS（Geyer 初始正序列）/ 与参照后验对比。"""

from __future__ import annotations

import numpy as np


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """(n_chain, n_draw, dim) -> (2*n_chain, n_draw//2, dim)。"""
    _, n_draw, _ = chains.shape
    half = n_draw // 2
    parts = [chains[:, :half, :], chains[:, n_draw - half :, :]]
    return np.concatenate(parts, axis=0)


def split_rhat(chains: np.ndarray) -> float:
    """最大 split-Rhat（跨参数）。输入 (n_chain, n_draw, dim)。"""
    ch = _split_chains(np.asarray(chains, dtype=float))
    _, n, _ = ch.shape
    if n < 4:
        return float("nan")
    chain_means = ch.mean(axis=1)  # (m, dim)
    chain_vars = ch.var(axis=1, ddof=1)  # (m, dim)
    between = n * chain_means.var(axis=0, ddof=1)  # (dim,)
    within = chain_vars.mean(axis=0)  # (dim,)
    var_hat = (n - 1) / n * within + between / n
    rhat = np.sqrt(var_hat / np.maximum(within, 1e-300))
    return float(np.max(rhat))


def _autocov_fft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    x = x - x.mean()
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[:n].real
    return acov / n


def ess_bulk(chains: np.ndarray) -> float:
    """最小跨参数 bulk ESS（Geyer 初始正序列 + 跨链合并，Stan 简化口径）。

    输入 (n_chain, n_draw, dim)。已包含去相关拉伸（rank-normalize 省略，
    简化版对同分布链与 ArviZ ess_bulk 偏差 <10%，见 tests 对照）。
    """
    ch = np.asarray(chains, dtype=float)
    m, n, dim = ch.shape
    best = np.inf
    for d in range(dim):
        acov = np.stack([_autocov_fft(ch[c, :, d]) for c in range(m)])  # (m, n)
        chain_var = acov[:, 0] * n / (n - 1.0)
        mean_var = float(chain_var.mean())
        var_plus = (
            mean_var * (n - 1.0) / n + ch[:, :, d].mean(axis=1).var(ddof=1) if m > 1 else mean_var
        )
        rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
        # Geyer 初始正序列：配对求和直到首个负值
        t = 1
        rho_sum = 0.0
        while t + 1 < n:
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            rho_sum += pair
            t += 2
        ess = m * n / (1.0 + 2.0 * rho_sum)
        best = min(best, max(ess, 1.0))
    return float(best)


def summarize_draws(chains: np.ndarray) -> dict[str, float]:
    return {
        "rhat_max": split_rhat(chains),
        "ess_bulk_min": ess_bulk(chains),
        "mean_abs": float(
            np.abs(np.asarray(chains).reshape(-1, chains.shape[-1]).mean(axis=0)).max()
        ),
    }


def compare_to_reference(
    draws: np.ndarray, ref, param_names: tuple[str, ...] | None = None
) -> object:
    """与参照后验比较：标准化均值误差 / 对数标准差误差。

    返回 SimpleNamespace(mean_z_err, sd_log_err, corr_err)。
    """
    from types import SimpleNamespace

    flat = np.asarray(draws).reshape(-1, draws.shape[-1])
    est_mean = flat.mean(axis=0)
    est_sd = flat.std(axis=0, ddof=1)
    mean_z = float(np.max(np.abs(est_mean - ref.mean) / np.maximum(ref.sd, 1e-12)))
    sd_log = float(np.max(np.abs(np.log(est_sd / np.maximum(ref.sd, 1e-12)))))
    # 相关矩阵误差（off-diagonal Frobenius / 2）
    est_cov = np.cov(flat.T)
    d = np.sqrt(np.maximum(np.diag(est_cov) * np.outer(ref.sd, ref.sd), 1e-300))
    corr_est = est_cov / d
    corr_ref = ref.cov / d
    off = ~np.eye(len(est_sd), dtype=bool)
    corr_err = float(np.sqrt(((corr_est - corr_ref)[off] ** 2).mean()))
    return SimpleNamespace(mean_z_err=mean_z, sd_log_err=sd_log, corr_err=corr_err)
