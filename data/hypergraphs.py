"""固定超边数的 ER、SF 和 WS 三元简单超图。"""

from math import comb

import numpy as np


def _er_hyperedges(num_nodes, num_edges, rng):
    """沿用原 ER 生成规则：均匀抽取不重复的三节点组合。"""
    hyperedges = set()
    while len(hyperedges) < num_edges:
        hyperedges.add(
            tuple(sorted(rng.choice(num_nodes, 3, replace=False).tolist()))
        )
    return hyperedges


def _sf_hyperedges(num_nodes, num_edges, rng, *, alpha):
    """直接使用 Hypergraphx 的单层无标度生成器，不是 BA 增长模型。"""
    from hypergraphx.generation.scale_free import scale_free_hypergraph

    hypergraph = scale_free_hypergraph(
        num_nodes=num_nodes,
        edges_by_size={3: num_edges},
        alpha_by_size=alpha,
        seed=int(rng.integers(0, 2**32)),
        enforce_unique_edges=True,
    )
    return {tuple(sorted(edge)) for edge in hypergraph.get_edges()}


def _ws_hyperedges(num_nodes, num_edges, rng, *, rewire_prob):
    """环形三元超边加随机重连；始终保持 E 条不同的三元超边。

    直接使用 xgi.ring_lattice，设置 d=3、k=2*(E/N)、l=0。
    按 XGI WS 的规则保留每条边编号最小的节点，重采样另外两个节点。
    重连处排除重复成员和重复超边，以符合当前三元简单超图的输入要求。
    https://github.com/xgi-org/xgi/blob/main/xgi/generators/lattice.py
    https://xgi.readthedocs.io/en/stable/_modules/xgi/generators/random.html#watts_strogatz_hypergraph
    """
    import xgi

    if not 0 <= rewire_prob <= 1:
        raise ValueError("WS_REWIRE_PROB 必须在 [0,1] 内")
    edges_per_anchor, remainder = divmod(num_edges, num_nodes)
    if remainder or not 1 <= edges_per_anchor <= (num_nodes - 2) // 2:
        raise ValueError("WS 要求 E=N*q，且 1 <= q <= (N-2)//2")

    lattice = xgi.ring_lattice(
        n=num_nodes, d=3, k=2 * edges_per_anchor, l=0
    )
    lattice_edges = [
        tuple(sorted(lattice.edges.members(edge_id))) for edge_id in lattice.edges
    ]
    hyperedges = set(lattice_edges)
    for edge in lattice_edges:
        if rng.random() >= rewire_prob:
            continue
        anchor = edge[0]
        # 先移除旧边；允许重新抽到它，稠密情况下也存在合法候选。
        hyperedges.remove(edge)
        while True:
            neighbors = rng.choice(num_nodes - 1, 2, replace=False)
            neighbors += neighbors >= anchor
            candidate = tuple(sorted([anchor, *neighbors.tolist()]))
            if candidate not in hyperedges:
                hyperedges.add(candidate)
                break
    return hyperedges


_GENERATORS = {
    "ER": _er_hyperedges,
    "SF": _sf_hyperedges,
    "WS": _ws_hyperedges,
}


def generate_hyperedges(network_type, num_nodes, num_edges, rng, **parameters):
    """返回 [E,3] 的节点编号数组；节点范围为 0..N-1。"""
    if num_nodes < 3 or not 1 <= num_edges <= comb(num_nodes, 3):
        raise ValueError("三元超图要求 N >= 3，且 1 <= E <= C(N,3)")
    hyperedges = _GENERATORS[network_type](
        num_nodes, num_edges, rng, **parameters
    )
    return np.asarray(sorted(hyperedges), dtype=np.int64)
