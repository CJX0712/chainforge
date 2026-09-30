"""基准流水线：跨模型 × 跨推断器运行、汇总并落盘 benchmark.json。"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .core import Budget, Config, CountingModel
from .diagnostics import compare_to_reference, summarize_draws
from .inference import build_methods
from .models import build_models


def run_one(model_key: str, method_key: str, cfg: Config, seed: int) -> dict[str, Any]:
    model = build_models([model_key])[model_key]
    counter = CountingModel(model)
    budget = Budget(counter, cfg.cost_budget)
    sampler = build_methods([method_key])[0]
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    try:
        res = sampler.run(counter, cfg, rng, budget)
    except Exception as exc:  # noqa: BLE001 — 基准要记录失败而不是中断
        return {
            "model": model_key,
            "method": method_key,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }
    rep = summarize_draws(res.draws)
    ref = model.reference_posterior(cfg)
    sc = compare_to_reference(res.draws, ref)
    return {
        "model": model_key,
        "method": method_key,
        "status": "ok",
        "total_draws": res.total_draws,
        "cost": round(counter.cost(), 1),
        "wall_s": round(time.perf_counter() - t0, 2),
        "mean_z_err": round(sc.mean_z_err, 4),
        "sd_log_err": round(sc.sd_log_err, 4),
        "corr_err": round(sc.corr_err, 4),
        "ess_bulk_min": round(rep["ess_bulk_min"], 1),
        "rhat_max": round(rep["rhat_max"], 4),
        "divergences": res.divergences,
        "ess_per_cost": round(rep["ess_bulk_min"] / max(counter.cost(), 1.0), 4),
        "gate_ok": res.meta.get("gate_ok"),
        "fallback_used": res.meta.get("fallback_used"),
    }


def run_benchmark(
    models: list[str] | None = None,
    methods: list[str] | None = None,
    cfg: Config | None = None,
    seed: int = 2026,
    output: str | None = "benchmark.json",
) -> list[dict[str, Any]]:
    cfg = cfg or Config()
    models = models or ["gauss_easy", "gauss_hard", "banana", "funnel", "logistic"]
    methods = methods or [
        "vario_nuts",
        "vario_nuts_mf",
        "nuts",
        "nuts_dense",
        "hmc_static",
        "rwm",
        "advi",
        "advi_full",
        "laplace",
    ]
    rows = []
    for mk in models:
        for me in methods:
            row = run_one(mk, me, cfg, seed)
            rows.append(row)
            status = "✓" if row["status"] == "ok" else "✗"
            extra = (
                f" meanZ={row['mean_z_err']:.3f} sdLog={row['sd_log_err']:.3f} "
                f"ess={row['ess_bulk_min']:.0f} rhat={row['rhat_max']:.3f} div={row['divergences']}"
                if row["status"] == "ok"
                else f" {row.get('error', '')[:80]}"
            )
            print(f"{status} {mk:12s} {me:14s} {extra}", flush=True)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(
                {"config": _cfg_dict(cfg), "rows": rows},
                f,
                ensure_ascii=False,
                indent=2,
            )
    return rows


def _cfg_dict(cfg: Config) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(cfg)


def verify_flagship(rows: list[dict[str, Any]]) -> bool:
    """旗舰声明（诚实口径，见 README §旗舰 VarioNUTS）：

    1. 相对鲁棒性：VarioNUTS 的 R̂ 不超过标准 NUTS 的 1.15 倍（±0.02 数值余量），
       且 mean_z_err ≤ 0.5 —— 承诺的是"不比基线更糟"，不是"总是最快"；
    2. 发散抑制：funnel 上 divergences ≤ 标准 NUTS；
    3. 永不静默失败：所有模型都出结果（异常已在 run_benchmark 中捕获为 fail 行）。
    """
    by_key = {(r["model"], r["method"]): r for r in rows if r["status"] == "ok"}
    ok = True
    for mk in sorted({r["model"] for r in rows}):
        v = by_key.get((mk, "vario_nuts"))
        n = by_key.get((mk, "nuts"))
        if not v:
            ok = False
            print(f"flagship {mk:12s} ✗ vario_nuts 缺失/失败")
            continue
        line = f"flagship {mk:12s}"
        if n:
            cap = n["rhat_max"] * 1.15 + 0.02
            passed = v["rhat_max"] <= cap and v["mean_z_err"] <= 0.5
            ok &= passed
            line += (
                f" rhat {v['rhat_max']:.3f}<=nuts×1.15 {cap:.3f}"
                f" meanZ {v['mean_z_err']:.3f} {'✓' if passed else '✗'}"
            )
        if mk == "funnel" and n:
            passed = v["divergences"] <= n["divergences"]
            ok &= passed
            line += f" | div {v['divergences']}<={n['divergences']} {'✓' if passed else '✗'}"
        print(line)
    return ok


if __name__ == "__main__":
    rows = run_benchmark()
    verify_flagship(rows)
