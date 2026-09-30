"""BFS-tree construction used to factorise the generator distribution."""

from collections import defaultdict

import networkx as nx


def build_bfs(G: nx.Graph, root: int):
    """Returns (parent, children, depth) dictionaries of the BFS tree rooted at `root`."""
    parent = {root: None}
    children = defaultdict(list)
    depth = {root: 0}
    q = [root]
    seen = {root}
    while q:
        cur = q.pop(0)
        for nb in G.neighbors(cur):
            if nb not in seen:
                seen.add(nb)
                parent[nb] = cur
                children[cur].append(nb)
                depth[nb] = depth[cur] + 1
                q.append(nb)
    return parent, children, depth


def build_all_bfs(G: nx.Graph):
    """One BFS tree per node, keyed by root."""
    return {r: build_bfs(G, r) for r in range(G.number_of_nodes())}


def max_branching_actions(bfs_trees, n_nodes: int) -> int:
    """Largest action space over all trees: children + 1 STOP action (minimum 2)."""
    max_actions = 2
    for r in range(n_nodes):
        _, ch, _ = bfs_trees[r]
        for u in range(n_nodes):
            max_actions = max(max_actions, len(ch[u]) + 1)
    return max_actions


def path_from_root(parent, target):
    p = []
    cur = target
    while cur is not None:
        p.append(cur)
        cur = parent[cur]
    return list(reversed(p))
