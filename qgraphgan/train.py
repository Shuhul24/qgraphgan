"""Single (dataset, seed) experiment: baselines, then adversarial training of QGraphGAN."""

import os
import random
import time
import warnings

import torch

from .baselines import (classical_bfs_gan, deepwalk, gcn_only_baseline, heuristic_baselines,
                        qcbm_baseline)
from .bfs import build_all_bfs, max_branching_actions
from .config import Config
from .data import adj_tensor, load_dataset, norm_adj, shortest_distances, split_edges
from .evaluation import disc_gap, eval_disc, eval_gen, gen_neighbor_stats, plot_matrices
from .losses import disc_loss, gen_loss
from .metrics import metric_row, node_f1, y_for_edges
from .models import AngleProjector, GraphSAGEEncoder, LocalBranchCircuit, QuantumGraphSoftmax
from .utils import log, set_seed


def train_qgraphgan(generator, gen_gcn, disc_gcn, projector, gen_wts, train_G, A_train, A_nbr, cfg, tag):
    """Alternating D / G updates. Returns total training time in seconds."""
    N = A_train.shape[0]
    g_params = list(gen_gcn.parameters()) + list(projector.parameters()) + [gen_wts]
    d_params = list(disc_gcn.parameters())
    optG = torch.optim.Adam(g_params, lr=cfg.lr_g)
    optD = torch.optim.Adam(d_params, lr=cfg.lr_d)
    schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=cfg.n_epochs, eta_min=1e-4)
    schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=cfg.n_epochs, eta_min=1e-5)

    all_centers = list(range(N))
    t0 = time.time()
    for ep in range(1, cfg.n_epochs + 1):
        centers = random.sample(all_centers, min(N, max(8, N // 2)))

        # discriminator step (generator frozen)
        optD.zero_grad()
        de = disc_gcn(A_nbr)
        ge = gen_gcn(A_nbr).detach()
        ld = disc_loss(de, ge, projector, gen_wts.detach(), centers, generator, train_G, A_train, cfg)
        ld.backward()
        torch.nn.utils.clip_grad_norm_(d_params, cfg.clip_norm)
        optD.step()
        schD.step()

        # generator step (discriminator frozen, rewards detached)
        optG.zero_grad()
        ge = gen_gcn(A_nbr)
        de = disc_gcn(A_nbr).detach()
        lg = gen_loss(ge, de, projector, gen_wts, centers, generator, cfg)
        lg.backward()
        torch.nn.utils.clip_grad_norm_(g_params, cfg.clip_norm)
        optG.step()
        schG.step()

        if ep % cfg.log_every == 0 or ep == 1:
            log(f"{tag} epoch {ep:03d}: LD={float(ld.detach()):.4f}, LG={float(lg.detach()):.4f}, time={time.time()-t0:.0f}s")
    return time.time() - t0, sum(p.numel() for p in g_params + d_params)


def run_experiment(dataset: str, seed: int, cfg: Config):
    """Runs every method on one (dataset, seed) and returns a list of metric rows."""
    set_seed(seed)
    tag = f"{dataset}_seed{seed}"
    run_dir = os.path.join(cfg.output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    # ---- data ----
    G_full, labels = load_dataset(dataset, seed, cfg.subgraph_size, cfg.data_dir)
    N = G_full.number_of_nodes()
    train_G, test_pos, test_neg = split_edges(G_full, cfg.test_ratio, seed)
    A_train = adj_tensor(train_G)
    A_nbr, _ = norm_adj(A_train)
    dist = shortest_distances(train_G)
    y_true = y_for_edges(test_pos, test_neg)
    if len(test_pos) < 10:
        warnings.warn(f"{tag}: only {len(test_pos)} positive test edges; metrics are noisy.")

    # ---- BFS trees and the local circuit sized to the largest branching factor ----
    bfs_trees = build_all_bfs(train_G)
    circuit = LocalBranchCircuit(max_branching_actions(bfs_trees, N), cfg.n_gen_layers)
    generator = QuantumGraphSoftmax(bfs_trees, circuit, N, A_train)

    # ---- baselines ----
    results = heuristic_baselines(train_G, dist, test_pos, test_neg, y_true)
    results.append(metric_row("GCN-only dot product", y_true,
                              gcn_only_baseline(train_G, A_train, A_nbr, test_pos, test_neg, cfg)))
    dw_f1 = float("nan")
    if cfg.run_deepwalk:
        scores, dw_emb = deepwalk(train_G, test_pos, test_neg, N, seed)
        results.append(metric_row("DeepWalk", y_true, scores))
        dw_f1 = node_f1(dw_emb, labels, seed)
    if cfg.run_classical_bfs_gan:
        scores = classical_bfs_gan(A_train, bfs_trees, test_pos, test_neg, N, seed)
        results.append(metric_row("Classical BFS-GAN", y_true, scores))
    if cfg.run_qcbm:
        scores = qcbm_baseline(A_train, A_nbr, bfs_trees, circuit, test_pos, test_neg, seed, cfg)
        results.append(metric_row("Conditional QCBM", y_true, scores))

    # ---- QGraphGAN: quantum generator + classical dot-product discriminator ----
    gen_gcn = GraphSAGEEncoder(N, cfg.emb_dim, cfg.gcn_hidden, use_spectral_norm=False)
    disc_gcn = GraphSAGEEncoder(N, cfg.emb_dim, cfg.gcn_hidden, use_spectral_norm=True)
    projector = AngleProjector(2 * cfg.emb_dim, circuit.n_angles)
    gen_wts = circuit.init_weights()

    with torch.no_grad():
        pre_ge = gen_gcn(A_nbr)
        pre_de = disc_gcn(A_nbr)
        pre_gen_scores = eval_gen(generator, pre_ge, projector, gen_wts, test_pos, test_neg)
        pre_disc_scores = eval_disc(pre_de, test_pos, test_neg)
    results.append(metric_row("Hybrid QGraphGAN Gen pre", y_true, pre_gen_scores))
    results.append(metric_row("Hybrid QGraphGAN Disc pre", y_true, pre_disc_scores))

    train_time, n_params = train_qgraphgan(generator, gen_gcn, disc_gcn, projector, gen_wts,
                                           train_G, A_train, A_nbr, cfg, tag)

    ge = gen_gcn(A_nbr).detach()
    de = disc_gcn(A_nbr).detach()
    gen_scores = eval_gen(generator, ge, projector, gen_wts, test_pos, test_neg)
    results.append(metric_row("Hybrid QGraphGAN Gen post", y_true, gen_scores))

    # ---- diagnostics ----
    gt, gn, gratio = gen_neighbor_stats(generator, ge, projector, gen_wts, A_train)
    dr, dn, dgap = disc_gap(de, A_train)
    log(f"  [Diag] D(real)={dr:.4f}, D(non)={dn:.4f}, gap={dgap:.4f}")
    q_f1 = node_f1(ge.numpy(), labels, seed)
    if seed == cfg.seeds[0]:
        plot_matrices(generator, ge, projector, gen_wts, de, A_train, os.path.join(run_dir, "matrices.png"))

    extra = {
        "dataset": dataset, "seed": seed, "N": N, "train_edges": train_G.number_of_edges(),
        "test_pos": len(test_pos), "time_sec": train_time,
        "gen_true_mean": gt, "gen_non_mean": gn, "gen_ratio": gratio,
        "disc_real_mean": dr, "disc_non_mean": dn, "disc_gap": dgap,
        "q_gen_f1": q_f1, "deepwalk_f1": dw_f1,
        "params_q": n_params,
        "local_qubits": circuit.n_qubits,
    }
    for r in results:
        r.update(extra)
    return results
