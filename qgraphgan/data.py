"""Datasets, edge splits, adjacency utilities and negative samplers."""

import os
import random
import urllib.request

import networkx as nx
import numpy as np
import torch

CA_GRQC_URL = (
    "https://raw.githubusercontent.com/hwwang55/GraphGAN/master/data/link_prediction/CA-GrQc_train.txt"
)


# ------------------------- datasets -------------------------
def relabel(G: nx.Graph) -> nx.Graph:
    return nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes())})


def make_sbm(n=32, p_in=0.65, p_out=0.12, seed=7):
    """Two-block stochastic block model with community labels."""
    n1 = n // 2
    n2 = n - n1
    G = nx.Graph(nx.stochastic_block_model([n1, n2], [[p_in, p_out], [p_out, p_in]], seed=seed))
    G.remove_edges_from(nx.selfloop_edges(G))
    return relabel(G), np.array([0] * n1 + [1] * n2)


def make_karate():
    """Zachary's Karate Club (34 nodes) with the two-faction labels."""
    G = relabel(nx.karate_club_graph())
    y = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(G.number_of_nodes())])
    return G, y


def download_ca_grqc(data_dir: str) -> nx.Graph:
    """CA-GrQc collaboration network, as distributed with the original GraphGAN code."""
    os.makedirs(data_dir, exist_ok=True)
    local = os.path.join(data_dir, "CA-GrQc_train.txt")
    if not os.path.exists(local):
        urllib.request.urlretrieve(CA_GRQC_URL, local)
    G = nx.Graph()
    with open(local) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                u, v = int(p[0]), int(p[1])
                if u != v:
                    G.add_edge(u, v)
    return G


def sample_connected_subgraph(G: nx.Graph, k=32, seed=7) -> nx.Graph:
    """Random frontier expansion until a connected k-node subgraph is found."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    for _ in range(300):
        start = rng.choice(nodes)
        chosen = {start}
        frontier = set(G.neighbors(start))
        while len(chosen) < k and frontier:
            nxt = rng.choice(list(frontier))
            chosen.add(nxt)
            frontier.update(G.neighbors(nxt))
            frontier -= chosen
        if len(chosen) == k:
            H = G.subgraph(chosen).copy()
            if nx.is_connected(H):
                return relabel(H)
    cc = max(nx.connected_components(G), key=len)
    H0 = G.subgraph(cc).copy()
    root = rng.choice(list(H0.nodes()))
    nodes_k = list(nx.bfs_tree(H0, root).nodes())[:k]
    return relabel(H0.subgraph(nodes_k).copy())


def load_dataset(name: str, seed: int, subgraph_size: int, data_dir: str):
    """Returns (graph, node_labels_or_None)."""
    if name == "synthetic_sbm":
        return make_sbm(subgraph_size, seed=seed)
    if name == "karate_full":
        return make_karate()
    if name == "graphgan_ca_grqc_subgraph":
        return sample_connected_subgraph(download_ca_grqc(data_dir), subgraph_size, seed), None
    raise ValueError(f"Unknown dataset: {name}")


# ------------------------- edge split -------------------------
def edges_of(G: nx.Graph):
    return [(min(u, v), max(u, v)) for u, v in G.edges() if u != v]


def split_edges(G: nx.Graph, test_ratio=0.25, seed=7):
    """Hold out test edges while keeping the training graph connected.

    Returns (train_graph, test_positive_edges, test_negative_pairs); negatives
    are sampled from non-edges of the full graph, balanced 1:1 with positives.
    """
    rng = random.Random(seed)
    edges = edges_of(G)
    rng.shuffle(edges)
    n_test = max(1, int(len(edges) * test_ratio))
    train_G = G.copy()
    test_pos = []
    for e in edges:
        if len(test_pos) >= n_test:
            break
        train_G.remove_edge(*e)
        if nx.is_connected(train_G):
            test_pos.append(e)
        else:
            train_G.add_edge(*e)
    if not test_pos:
        raise RuntimeError("Could not remove any edge while keeping graph connected.")
    all_pairs = [(i, j) for i in range(G.number_of_nodes()) for j in range(i + 1, G.number_of_nodes())]
    full_edges = set(edges_of(G))
    non_edges = [p for p in all_pairs if p not in full_edges]
    rng.shuffle(non_edges)
    return train_G, test_pos, non_edges[:len(test_pos)]


# ------------------------- adjacency -------------------------
def adj_tensor(G: nx.Graph) -> torch.Tensor:
    A = torch.zeros((G.number_of_nodes(), G.number_of_nodes()), dtype=torch.float64)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def norm_adj(A: torch.Tensor):
    """Returns (row-normalised neighbour-mean operator, symmetric-normalised A + I)."""
    deg = A.sum(1).clamp(min=1.0)
    A_nbr = torch.diag(1.0 / deg) @ A
    A_hat = A + torch.eye(A.shape[0], dtype=A.dtype)
    d = A_hat.sum(1)
    D = torch.diag(torch.pow(d + 1e-12, -0.5))
    return A_nbr, D @ A_hat @ D


def shortest_distances(G: nx.Graph) -> torch.Tensor:
    N = G.number_of_nodes()
    D = torch.full((N, N), 1e6, dtype=torch.float64)
    for i in range(N):
        D[i, i] = 0.0
    for i, dd in dict(nx.all_pairs_shortest_path_length(G)).items():
        for j, d in dd.items():
            D[i, j] = float(d)
    return D


# ------------------------- samplers -------------------------
def train_neighbors(A: torch.Tensor, c: int):
    return [j for j in range(A.shape[0]) if j != c and A[c, j].item() > 0]


def random_non_neighbor(A: torch.Tensor, c: int):
    pool = [j for j in range(A.shape[0]) if j != c and A[c, j].item() == 0]
    return random.choice(pool) if pool else random.choice([j for j in range(A.shape[0]) if j != c])


def distance_two_non_neighbor(G: nx.Graph, A: torch.Tensor, c: int):
    """Hard negative: a node two hops from c that is not its neighbour."""
    cand = []
    for u in G.neighbors(c):
        for v in G.neighbors(u):
            if v != c and A[c, v].item() == 0:
                cand.append(v)
    return random.choice(cand) if cand else random_non_neighbor(A, c)
