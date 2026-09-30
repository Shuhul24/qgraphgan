"""Model components: GraphSAGE encoders, the local variational circuit, and the
Quantum Graph Softmax generator with a STOP action."""

import math

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .data import random_non_neighbor


# ------------------------- classical components -------------------------
class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE-mean encoder over learnable node features.

    Self and neighbour-mean features are concatenated (no self-loop in the
    neighbour operator, so the node is not double-counted).
    """

    def __init__(self, n_nodes, emb_dim, hidden_dim, use_spectral_norm=False):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, emb_dim, dtype=torch.float64)
        nn.init.normal_(self.emb.weight, std=0.1)
        W1 = nn.Linear(2 * emb_dim, hidden_dim, dtype=torch.float64)
        W2 = nn.Linear(2 * hidden_dim, emb_dim, dtype=torch.float64)
        self.W1 = spectral_norm(W1) if use_spectral_norm else W1
        self.W2 = spectral_norm(W2) if use_spectral_norm else W2
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=torch.float64)
        self.ln2 = nn.LayerNorm(emb_dim, dtype=torch.float64)

    def forward(self, A_nbr):
        x = self.emb.weight
        h1 = self.ln1(F.relu(self.W1(torch.cat([x, A_nbr @ x], dim=1))))
        h2 = self.ln2(self.W2(torch.cat([h1, A_nbr @ h1], dim=1)))
        return h2


class AngleProjector(nn.Module):
    """Linear map to rotation angles, squashed into (-pi, pi) by a softsign."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, dtype=torch.float64)

    def forward(self, x):
        z = self.proj(x)
        return math.pi * z / (1.0 + z.abs())


def dot_disc(c, v, emb):
    """GraphGAN dot-product discriminator: D(c, v) = sigmoid(e_c . e_v)."""
    return torch.sigmoid(torch.dot(emb[c], emb[v]))


# ------------------------- quantum components -------------------------
class LocalBranchCircuit:
    """Variational circuit that scores the branching actions at one BFS-tree node.

    Uses n = ceil(log2(max_actions)) qubits: Hadamards, then per layer data
    re-uploading of the angles (RY, RZ per qubit) followed by one
    StronglyEntanglingLayer. Returns the 2^n computational-basis probabilities.
    """

    def __init__(self, max_actions: int, n_layers: int):
        self.n_qubits = max(1, math.ceil(math.log2(max_actions)))
        self.n_outcomes = 2 ** self.n_qubits
        self.n_angles = 2 * self.n_qubits
        self.n_layers = n_layers
        self.wires = list(range(self.n_qubits))
        self.weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=self.n_qubits)
        dev = qml.device("default.qubit", wires=self.wires)
        self.qnode = qml.QNode(self._circuit, dev, interface="torch", diff_method="backprop")

    def _circuit(self, angles, weights):
        for w in self.wires:
            qml.Hadamard(wires=w)
        for li in range(self.n_layers):
            for i, w in enumerate(self.wires):
                qml.RY(angles[2 * i], wires=w)
                qml.RZ(angles[2 * i + 1], wires=w)
            qml.StronglyEntanglingLayers(weights[li:li + 1], wires=self.wires)
        return qml.probs(wires=self.wires)

    def init_weights(self, scale=0.01) -> nn.Parameter:
        return nn.Parameter(scale * torch.randn(self.weight_shape, dtype=torch.float64))

    def __call__(self, angles, weights):
        return self.qnode(angles, weights)


class QuantumGraphSoftmax:
    """Quantum Graph Softmax: G(v | c) factorised over the BFS tree rooted at c.

    At each tree node the local circuit gives a distribution over
    [STOP, child_1, ..., child_k]. STOP selects the current node, which avoids
    exponential suppression of deep nodes. STOP is masked at the root.

    The trainable pieces (embeddings, angle projector, circuit weights) are
    passed to each call so they can be detached independently during training.
    """

    def __init__(self, bfs_trees, circuit: LocalBranchCircuit, n_nodes: int, A_train: torch.Tensor):
        self.bfs = bfs_trees
        self.circuit = circuit
        self.n_nodes = n_nodes
        self.A_train = A_train

    def action_probs(self, center, cur, emb, projector, weights, allow_stop=True, eps=1e-12):
        """Distribution over [STOP, children of `cur`] in the tree rooted at `center`."""
        _, children, _ = self.bfs[center]
        kids = children[cur]
        angles = projector(torch.cat([emb[center], emb[cur]]))
        raw = self.circuit(angles, weights)
        n_actions = len(kids) + 1
        mask = torch.zeros(self.circuit.n_outcomes, dtype=torch.float64)
        mask[:n_actions] = 1.0
        if not allow_stop:
            mask[0] = 0.0
        probs = raw * mask
        return probs[:n_actions] / (probs[:n_actions].sum() + eps), kids

    def probs(self, center, emb, projector, weights, eps=1e-12):
        """Exact G(. | center) over all nodes, by summing probabilities of all root-to-node paths."""
        probs = torch.zeros(self.n_nodes, dtype=torch.float64)
        reach = {center: torch.tensor(1.0, dtype=torch.float64)}
        q = [center]
        while q:
            cur = q.pop(0)
            _, children, _ = self.bfs[center]
            kids = children[cur]
            if not kids:
                if cur != center:
                    probs[cur] = probs[cur] + reach[cur]
                continue
            ap, kids = self.action_probs(center, cur, emb, projector, weights, allow_stop=(cur != center))
            if cur != center:
                probs[cur] = probs[cur] + reach[cur] * ap[0]
            offset = 1
            for idx, kid in enumerate(kids):
                p = ap[idx + offset]
                reach[kid] = reach.get(kid, torch.tensor(0.0, dtype=torch.float64)) + reach[cur] * p
                q.append(kid)
        probs[center] = 0.0
        return probs / (probs.sum() + eps)

    def sample(self, center, emb, projector, weights, eps=1e-12):
        """Draw one node by walking down the tree; returns (node, log-probability of the walk)."""
        cur = center
        logp = torch.tensor(0.0, dtype=torch.float64)
        while True:
            _, children, _ = self.bfs[center]
            kids = children[cur]
            if not kids:
                break
            ap, kids = self.action_probs(center, cur, emb, projector, weights, allow_stop=(cur != center))
            with torch.no_grad():
                sp = ap.detach().clone()
                if cur == center:
                    sp[0] = 0.0
                sp = sp / (sp.sum() + eps)
                a = int(torch.multinomial(sp.clamp_min(eps), 1).item())
                if cur == center and a == 0:
                    a = int(torch.argmax(sp).item())
            logp = logp + torch.log(ap[a] + eps)
            if a == 0:
                break
            cur = kids[a - 1]
        if cur == center:
            kids = self.bfs[center][1][center]
            cur = kids[0] if kids else random_non_neighbor(self.A_train, center)
        return cur, logp
