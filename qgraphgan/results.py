"""Aggregation of per-run metrics into mean ± std over seeds."""

import os

import numpy as np

from .utils import log


def summarize(rows, output_dir):
    """Prints mean ± std per (dataset, method) and writes every raw row to CSV."""
    methods = sorted({r["method"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        log(f"\n=== {ds}: mean ± std over seeds ===")
        for m in methods:
            rr = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            if not rr:
                continue

            def ms(key):
                vals = np.array([r.get(key, np.nan) for r in rr], dtype=float)
                return np.nanmean(vals), np.nanstd(vals)

            aucm, aucs = ms("auc")
            apm, aps = ms("ap")
            p5, _ = ms("p@5")
            p10, _ = ms("p@10")
            p20, _ = ms("p@20")
            log(f"{m:32s} AUC={aucm:.4f}±{aucs:.4f} AP={apm:.4f}±{aps:.4f} P@5={p5:.4f} P@10={p10:.4f} P@20={p20:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "summary_results.csv")
    with open(csv_path, "w") as f:
        keys = sorted(rows[0].keys()) if rows else []
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    log(f"\nSaved CSV: {csv_path}")
    return csv_path
