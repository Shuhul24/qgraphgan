"""Baselines: structural heuristics, GCN-only, DeepWalk, classical BFS-GAN, and
the conditional QCBM ablation (same quantum generator, no adversary)."""

import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from .data import edges_of, random_non_neighbor, train_neighbors
from .evaluation import eval_disc, eval_gen
from .metrics import metric_row
from .models import AngleProjector, GraphSAGEEncoder, QuantumGraphSoftmax, dot_disc
from .utils import set_seed


def heuristic_baselines(train_G, dist, pos, neg, y_true):
    pairs = pos + neg
    return [
        metric_row("Common Neighbors", y_true, [len(list(nx.common_neighbors(train_G, i, j))) for i, j in pairs]),
        metric_row("Jaccard", y_true, [s for _, _, s in nx.jaccard_coefficient(train_G, pairs)]),
        metric_row("Adamic-Adar", y_true, [s for _, _, s in nx.adamic_adar_index(train_G, pairs)]),
        metric_row("Graph Distance", y_true, [1.0 / (1.0 + dist[i, j].item()) for i, j in pairs]),
    ]


def gcn_only_baseline(train_G, A_train, A_nbr, pos, neg, cfg, n_iters=80, lr=0.02):
    """Same GraphSAGE encoder + dot-product scorer, trained supervised with BCE.

    Isolates the encoder's contribution from the quantum generator.
    """
    N = A_train.shape[0]
    model = GraphSAGEEncoder(N, cfg.emb_dim, cfg.gcn_hidden, use_spectral_norm=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pairs_pos = edges_of(train_G)
    for _ in range(n_iters):
        emb = model(A_nbr)
        batch_pos = random.sample(pairs_pos, min(len(pairs_pos), max(8, N)))
        loss_terms = []
        for i, j in batch_pos:
            loss_terms.append(-torch.log(dot_disc(i, j, emb) + 1e-8))
            k = random_non_neighbor(A_train, i)
            loss_terms.append(-torch.log(1 - dot_disc(i, k, emb) + 1e-8))
        loss = torch.stack(loss_terms).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
    emb = model(A_nbr).detach()
    return eval_disc(emb, pos, neg)


def deepwalk(G, pos, neg, N, seed, dim=16, walk_len=20, walks_per_node=8, window=5, neg_k=5, epochs=35):
    """Skip-gram with negative sampling on uniform random walks; cosine similarity scores."""
    set_seed(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(walks_per_node):
        for s in nodes:
            w = [s]
            for _ in range(walk_len - 1):
                nb = list(G.neighbors(w[-1]))
                if not nb:
                    break
                w.append(random.choice(nb))
            walks.append(w)
    emb = nn.Embedding(N, dim, dtype=torch.float64)
    ctx = nn.Embedding(N, dim, dtype=torch.float64)
    nn.init.normal_(emb.weight, std=0.1)
    nn.init.zeros_(ctx.weight)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ctx.parameters()), lr=0.025)
    for _ in range(epochs):
        random.shuffle(walks)
        for w in walks:
            total = torch.tensor(0.0, dtype=torch.float64)
            count = 0
            for pi, c in enumerate(w):
                for qi in range(max(0, pi - window), min(len(w), pi + window + 1)):
                    if pi == qi:
                        continue
                    u = w[qi]
                    ps = (emb(torch.tensor(c)) * ctx(torch.tensor(u))).sum()
                    loss = -torch.log(torch.sigmoid(ps) + 1e-8)
                    neg_sum = torch.tensor(0.0, dtype=torch.float64)
                    for _ in range(neg_k):
                        nv = random.choice(nodes)
                        ns = (emb(torch.tensor(c)) * ctx(torch.tensor(nv))).sum()
                        neg_sum = neg_sum - torch.log(torch.sigmoid(-ns) + 1e-8)
                    loss = loss + neg_sum / neg_k
                    total = total + loss
                    count += 1
            if count:
                opt.zero_grad()
                (total / count).backward()
                opt.step()
    E = emb.weight.detach().numpy()
    scores = [float(np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]) + 1e-12)) for i, j in pos + neg]
    return scores, E


def classical_bfs_gan(A_train, bfs_trees, pos, neg, N, seed, dim=16, epochs=80):
    """Classical counterpart: same BFS tree, STOP action and REINFORCE, but each
    branch is scored by a softmax over embedding dot products instead of a circuit."""
    set_seed(seed)
    ge = nn.Embedding(N, dim, dtype=torch.float64)
    de = nn.Embedding(N, dim, dtype=torch.float64)
    nn.init.normal_(ge.weight, std=0.1)
    nn.init.normal_(de.weight, std=0.1)
    optG = torch.optim.Adam(ge.parameters(), lr=0.01)
    optD = torch.optim.Adam(de.parameters(), lr=0.01)

    def score_stop(c, cur):
        return torch.dot(ge(torch.tensor(c)), ge(torch.tensor(cur)))

    def action_logprobs(c, cur):
        kids = bfs_trees[c][1][cur]
        logits = [score_stop(c, cur)] if cur != c else []
        logits += [torch.dot(ge(torch.tensor(c)), ge(torch.tensor(k))) for k in kids]
        return torch.log_softmax(torch.stack(logits), dim=0), kids

    def sample(c):
        cur = c
        lp = torch.tensor(0.0, dtype=torch.float64)
        while True:
            kids = bfs_trees[c][1][cur]
            if not kids:
                break
            lps, kids = action_logprobs(c, cur)
            with torch.no_grad():
                a = int(torch.multinomial(torch.exp(lps).detach(), 1).item())
            lp = lp + lps[a]
            if cur != c and a == 0:
                break
            cur = kids[a - 1] if cur != c else kids[a]
        return cur, lp

    nodes = list(range(N))
    for _ in range(epochs):
        random.shuffle(nodes)
        for c in nodes:
            nbrs = train_neighbors(A_train, c)
            if not nbrs:
                continue
            optD.zero_grad()
            r = random.choice(nbrs)
            f, _ = sample(c)
            dr = torch.sigmoid(torch.dot(de(torch.tensor(c)), de(torch.tensor(r))))
            df = torch.sigmoid(torch.dot(de(torch.tensor(c)), de(torch.tensor(f))))
            ld = -torch.log(dr + 1e-8) - torch.log(1 - df + 1e-8)
            ld.backward()
            optD.step()
            optG.zero_grad()
            losses, rewards = [], []
            for _ in range(5):
                sv, lp = sample(c)
                with torch.no_grad():
                    rew = torch.log(torch.sigmoid(torch.dot(de(torch.tensor(c)), de(torch.tensor(sv)))) + 1e-8)
                losses.append((rew, lp))
                rewards.append(rew)
            base = torch.stack(rewards).mean().detach()
            lg = torch.stack([-(r - base).detach() * lp for r, lp in losses]).mean()
            lg.backward()
            optG.step()
    E = de.weight.detach().numpy()
    return [float(np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]) + 1e-12)) for i, j in pos + neg]


def qcbm_baseline(A_train, A_nbr, bfs_trees, circuit, pos, neg, seed, cfg, n_iters=60, lr=0.02):
    """Conditional QCBM ablation: the identical quantum graph-softmax generator,
    fit by maximum likelihood to the training neighbour distribution (no adversary)."""
    set_seed(seed)
    N = A_train.shape[0]
    enc = GraphSAGEEncoder(N, cfg.emb_dim, cfg.gcn_hidden, use_spectral_norm=False)
    proj = AngleProjector(2 * cfg.emb_dim, circuit.n_angles)
    wts = circuit.init_weights()
    params = list(enc.parameters()) + list(proj.parameters()) + [wts]
    opt = torch.optim.Adam(params, lr=lr)
    generator = QuantumGraphSoftmax(bfs_trees, circuit, N, A_train)

    targets = []
    for c in range(N):
        t = torch.zeros(N, dtype=torch.float64)
        for u in train_neighbors(A_train, c):
            t[u] = 1.0
        if t.sum() > 0:
            t = t / t.sum()
        targets.append(t)

    for _ in range(n_iters):
        emb = enc(A_nbr)
        terms = []
        for c in range(N):
            if targets[c].sum() > 0:
                p = generator.probs(c, emb, proj, wts)
                terms.append(-torch.sum(targets[c] * torch.log(p + 1e-12)))
        loss = torch.stack(terms).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    emb = enc(A_nbr).detach()
    return eval_gen(generator, emb, proj, wts, pos, neg)
