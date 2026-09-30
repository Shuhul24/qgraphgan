"""Experiment configuration.

The defaults below are the exact values used to produce the results reported
in the MusiML @ ICML 2026 workshop paper. Change them only for new experiments.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- experiment protocol ---
    datasets: List[str] = field(
        default_factory=lambda: ["synthetic_sbm", "karate_full", "graphgan_ca_grqc_subgraph"]
    )
    seeds: List[int] = field(default_factory=lambda: [7, 42, 123])
    subgraph_size: int = 32          # nodes in the SBM graph and the CA-GrQc subgraph
    test_ratio: float = 0.25         # fraction of edges held out for link prediction

    # --- architecture ---
    emb_dim: int = 16                # node embedding dimension
    gcn_hidden: int = 32             # GraphSAGE hidden width
    n_gen_layers: int = 3            # StronglyEntanglingLayers per local circuit

    # --- adversarial training ---
    n_epochs: int = 200
    n_reinforce: int = 8             # REINFORCE samples per center node
    k_fake_d: int = 4                # negatives per center for the discriminator
    ent_reg: float = 0.005           # generator entropy bonus
    reward_clip: float = 5.0         # rewards clipped to [-reward_clip, 0]
    gp_lambda: float = 0.5           # embedding gradient-penalty weight
    lr_g: float = 0.02
    lr_d: float = 0.01
    clip_norm: float = 1.0

    # --- baselines ---
    run_deepwalk: bool = True
    run_classical_bfs_gan: bool = True
    run_qcbm: bool = True

    # --- I/O ---
    output_dir: str = "outputs"
    data_dir: str = "data"
    log_every: int = 20
