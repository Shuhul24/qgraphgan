"""Scoring test pairs and training diagnostics."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import dot_disc
from .utils import savefig


def eval_gen(generator, emb, projector, weights, pos, neg):
    """Symmetric generator score for a pair: sqrt(G(j | i) * G(i | j))."""
    scores = []
    with torch.no_grad():
        for i, j in pos + neg:
            pi = generator.probs(i, emb, projector, weights).numpy()
            pj = generator.probs(j, emb, projector, weights).numpy()
            scores.append(float(math.sqrt(max(float(pi[j] * pj[i]), 0.0))))
    return scores


def eval_disc(disc_emb, pos, neg):
    with torch.no_grad():
        return [float(dot_disc(i, j, disc_emb)) for i, j in pos + neg]


def gen_neighbor_stats(generator, emb, projector, weights, A_train):
    """Mean generator probability on training neighbours vs. non-neighbours, and their ratio."""
    N = A_train.shape[0]
    tp, np_ = [], []
    with torch.no_grad():
        for c in range(N):
            p = generator.probs(c, emb, projector, weights).numpy()
            for v in range(N):
                if v == c:
                    continue
                (tp if A_train[c, v].item() > 0 else np_).append(p[v])
    return float(np.mean(tp)), float(np.mean(np_)), float(np.mean(tp) / (np.mean(np_) + 1e-12))


def disc_gap(disc_emb, A_train):
    """Mean discriminator score on training edges vs. non-edges, and their gap."""
    N = A_train.shape[0]
    rs, ns = [], []
    with torch.no_grad():
        for i in range(N):
            for j in range(i + 1, N):
                s = float(dot_disc(i, j, disc_emb))
                (rs if A_train[i, j].item() > 0 else ns).append(s)
    return float(np.mean(rs)), float(np.mean(ns)), float(np.mean(rs) - np.mean(ns))


def plot_matrices(generator, gen_emb, projector, weights, disc_emb, A_train, path):
    """Side-by-side heatmaps: training adjacency, G(v | c), and D(c, v)."""
    N = A_train.shape[0]
    with torch.no_grad():
        Gmat = np.vstack([generator.probs(c, gen_emb, projector, weights).numpy() for c in range(N)])
        Dmat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    Dmat[i, j] = float(dot_disc(i, j, disc_emb))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, mat, title in zip(axes, [A_train.numpy(), Gmat, Dmat], ["train adjacency", "generator", "discriminator"]):
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    savefig(path)
