# QGraphGAN

**QGraphGAN: Quantum Graph Softmax for Adversarial Link Prediction via BFS-Tree Variational Circuits**

Shoiab Shafi, Shuhul Handoo. *The 6th Muslims in ML (MusIML) Workshop at ICML 2026.*

This repository contains the code for the paper. QGraphGAN is a hybrid quantum-classical version of
[GraphGAN](https://arxiv.org/abs/1711.08267) (Wang et al., 2018).

GraphGAN generates a node's neighbours by walking down a BFS tree rooted at that node, and it scores each
branch with a classical softmax. QGraphGAN keeps this tree-structured *graph softmax* but scores each branch
with a small **variational quantum circuit**. Its discriminator is the classical dot-product discriminator
from GraphGAN, so any difference in results comes from the quantum generator.

<p align="center"><em>center node → BFS tree → at each tree node, a local circuit picks STOP or a child → generated neighbour</em></p>

## Method

| Component | Design |
|---|---|
| **Generator** | Quantum Graph Softmax. `G(v | c)` is factorised along the BFS tree rooted at `c`. At every tree node `u`, a local circuit on `⌈log₂(k_max + 1)⌉` qubits outputs a distribution over `[STOP, child₁, …, child_k]`. The circuit has Hadamards, then `L = 3` blocks, each with RY/RZ angle re-uploading followed by a StronglyEntanglingLayer. The angles come from `[e_c ; e_u]` through a softsign-bounded linear projector. |
| **STOP action** | Stopping at any non-root node selects that node. Without it, a walk must end at a leaf, and nodes deep in the tree get exponentially small probability. STOP is masked at the root. |
| **Encoders** | Two independent 2-layer GraphSAGE-mean encoders, one for the generator and one for the discriminator. The discriminator's encoder uses spectral normalisation. |
| **Discriminator** | `D(c, v) = σ(e_cᵀ e_v)`, as in GraphGAN. Its negatives are a mix of generator samples, random non-neighbours and distance-2 non-neighbours. |
| **Training** | REINFORCE with a mean-reward baseline. The reward is `log D`, clipped to `[-5, 0]`. Other terms: an entropy bonus, an embedding gradient penalty on D, and cosine LR schedules. |
| **Link score** | `s(i, j) = sqrt(G(j | i) · G(i | j))`, computed exactly by summing the probabilities of every path to the node. |

## Repository structure

```
qgraphgan/
├── config.py        # Config dataclass — defaults are the paper's hyperparameters
├── data.py          # datasets, connectivity-preserving edge split, adjacency, negative samplers
├── bfs.py           # BFS-tree construction and action-space sizing
├── models.py        # GraphSAGEEncoder, AngleProjector, LocalBranchCircuit, QuantumGraphSoftmax
├── losses.py        # discriminator loss + gradient penalty, REINFORCE generator loss
├── baselines.py     # heuristics, GCN-only, DeepWalk, classical BFS-GAN, conditional QCBM
├── evaluation.py    # test-pair scoring, generator/discriminator diagnostics, heatmaps
├── metrics.py       # AUC, AP, Precision@K, node-classification F1
├── results.py       # mean ± std aggregation and CSV export
└── train.py         # run_experiment(): data → baselines → adversarial training → metrics
run_experiments.py   # entry point (all datasets × seeds)
scripts/run_slurm.sh # SLURM job script
```

## Installation

```bash
conda env create -f environment.yml
conda activate qenv
```

or `pip install -r requirements.txt` (Python 3.11). Everything runs on CPU in float64. The circuits are
simulated exactly with PennyLane's `default.qubit` and differentiated by backpropagation.

## Usage

```bash
# Full paper protocol: 3 datasets × 3 seeds (7, 42, 123), all baselines
python run_experiments.py

# A single run
python run_experiments.py --datasets karate_full --seeds 7

# Only QGraphGAN and the cheap baselines
python run_experiments.py --no-deepwalk --no-classical-bfs-gan --no-qcbm

# On a SLURM cluster
sbatch scripts/run_slurm.sh
```

Outputs are written to `outputs/`:

- `summary_results.csv`: one row per (dataset, seed, method), with AUC, AP, P@5/10/20 and the diagnostics
  (generator neighbour/non-neighbour probability ratio, discriminator real/non-edge gap, node-classification F1,
  parameter count, number of qubits).
- `<dataset>_seed<seed>/matrices.png`: heatmaps of the training adjacency, `G(v | c)` and `D(c, v)`, saved for
  the first seed only.

The mean ± std table over seeds is printed to stdout.

## Datasets

| Name | Graph | Source |
|---|---|---|
| `karate_full` | Zachary's Karate Club, 34 nodes, 78 edges | NetworkX |
| `synthetic_sbm` | 2-block SBM, 32 nodes, `p_in = 0.65`, `p_out = 0.12` | generated per seed |
| `graphgan_ca_grqc_subgraph` | connected 32-node subgraph of CA-GrQc | downloaded from the [GraphGAN repo](https://github.com/hwwang55/GraphGAN) into `data/` on first use |

25% of edges are held out for testing, and removing them never disconnects the training graph. Test negatives
are non-edges of the full graph, sampled 1:1 with the positives.

## Hyperparameters

The paper's values are the defaults in [`qgraphgan/config.py`](qgraphgan/config.py):

| | |
|---|---|
| Embedding / hidden dim | 16 / 32 |
| Circuit layers `L` | 3 |
| Epochs | 200 |
| REINFORCE samples per center | 8 |
| Discriminator negatives per center | 4 |
| Entropy weight | 0.005 |
| Gradient-penalty weight `λ_GP` | 0.5 |
| LR (G / D) | 0.02 / 0.01, cosine-annealed |
| Gradient clip | 1.0 |

## Results

Link prediction on the test sets, reported as mean ± std over 3 seeds (from the paper):

| Method | Karate AUC | SBM AUC | CA-GrQc AUC* |
|---|---|---|---|
| Common Neighbors | 0.704 ± 0.083 | 0.785 ± 0.042 | 0.904 ± 0.056 |
| Jaccard | 0.649 ± 0.059 | 0.800 ± 0.037 | 0.879 ± 0.068 |
| Adamic–Adar | 0.730 ± 0.075 | 0.778 ± 0.042 | 0.906 ± 0.058 |
| Graph Distance | 0.748 ± 0.068 | 0.647 ± 0.067 | 0.869 ± 0.068 |
| DeepWalk | 0.741 ± 0.030 | 0.811 ± 0.045 | 0.876 ± 0.020 |
| Classical BFS-GAN | 0.502 ± 0.042 | 0.539 ± 0.017 | 0.458 ± 0.068 |
| Conditional QCBM | 0.726 ± 0.071 | 0.669 ± 0.090 | 0.818 ± 0.098 |
| GCN-only dot product | 0.567 ± 0.016 | 0.742 ± 0.057 | 0.782 ± 0.131 |
| **QGraphGAN** | **0.724 ± 0.010** | **0.614 ± 0.075** | **0.778 ± 0.144** |

\*CA-GrQc is **indicative only**. A 32-node subgraph leaves very few positive test edges, so its variance
across seeds is large.

The **classical BFS-GAN** uses the same BFS tree, STOP action and REINFORCE training, with a classical softmax
in place of each circuit. QGraphGAN beats it on all three graphs. On Karate Club, QGraphGAN is the most stable
method across seeds. On the clean synthetic SBM, simple structural heuristics are clearly better.

## Notes and limitations

- The graphs are deliberately small (≤ 34 nodes). Exact state-vector simulation and exact path summation are
  computed in Python loops, and one full run of the protocol takes hours on CPU.
- A method's results match the paper exactly only when run with the same package versions and the same
  sequence of random calls: dataset → split → baselines in the listed order → QGraphGAN. Disabling a baseline
  changes the random stream seen by the methods after it.
- Node-classification F1 is only computed when a graph has labels and at least 20 nodes, and it is not part of
  the paper's claims.

## Citation

```bibtex
@inproceedings{shafi2026qgraphgan,
  title={QGraphGAN: Quantum Graph Softmax for Adversarial Link Prediction via BFS-Tree Variational Circuits},
  author={Shafi, Shoiab and Handoo, Shuhul},
  booktitle={The 6th Muslims in ML (MusIML) Workshop at ICML 2026}
}
```

## Acknowledgements

The problem setup, BFS-tree graph softmax and CA-GrQc split follow
[GraphGAN](https://github.com/hwwang55/GraphGAN) (Wang et al., AAAI 2018).
