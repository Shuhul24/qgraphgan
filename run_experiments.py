"""Reproduce the paper's experiments: every method x every dataset x every seed.

    python run_experiments.py                                   # full paper protocol
    python run_experiments.py --datasets karate_full --seeds 7  # a single run
"""

import argparse

from qgraphgan import Config, run_experiment
from qgraphgan.results import summarize
from qgraphgan.utils import log

HEADLINE_METHODS = ["Hybrid QGraphGAN Gen post", "Hybrid QGraphGAN Gen pre", "GCN-only dot product"]


def parse_args():
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=d.datasets,
                   choices=["synthetic_sbm", "karate_full", "graphgan_ca_grqc_subgraph"])
    p.add_argument("--seeds", nargs="+", type=int, default=d.seeds)
    p.add_argument("--epochs", type=int, default=d.n_epochs)
    p.add_argument("--output-dir", default=d.output_dir)
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--no-deepwalk", action="store_true")
    p.add_argument("--no-classical-bfs-gan", action="store_true")
    p.add_argument("--no-qcbm", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        datasets=args.datasets, seeds=args.seeds, n_epochs=args.epochs,
        output_dir=args.output_dir, data_dir=args.data_dir,
        run_deepwalk=not args.no_deepwalk,
        run_classical_bfs_gan=not args.no_classical_bfs_gan,
        run_qcbm=not args.no_qcbm,
    )
    all_rows = []
    for ds in cfg.datasets:
        for seed in cfg.seeds:
            log(f"\n{'=' * 72}\nRunning {ds}, seed={seed}\n{'=' * 72}")
            try:
                rows = run_experiment(ds, seed, cfg)
                all_rows.extend(rows)
                for r in rows:
                    if r["method"] in HEADLINE_METHODS:
                        log(f"{r['method']:32s} AUC={r['auc']:.4f} AP={r['ap']:.4f}")
            except Exception as e:
                log(f"FAILED {ds} seed={seed}: {repr(e)}")
    if all_rows:
        summarize(all_rows, cfg.output_dir)


if __name__ == "__main__":
    main()
