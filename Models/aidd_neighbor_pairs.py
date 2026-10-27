"""AIDD 的逐节点流程：每个目标节点对应一行候选邻居对。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aidd_hypergraph import NN1, NN2, NN3


class GumbelNeighborPairGenerator(nn.Module):
    """
    N 行候选邻居对 logits，每行形状为 [C(N-1, 2), 2]。
    第 0 类表示有边，第 1 类表示无边。
    每行单独作为 Parameter，Adam 只更新当前参与损失的行。
    """

    def __init__(
        self, num_nodes, temperature=1.0, init_edge_prob=0.001, init_std=0.1
    ):
        super().__init__()
        if not 0 < init_edge_prob < 1:
            raise ValueError("init_edge_prob 必须在 (0,1) 内")
        self.N = num_nodes
        self.num_pairs = (num_nodes - 1) * (num_nodes - 2) // 2
        self.temperature = temperature
        base_logits = torch.tensor(
            [math.log(init_edge_prob), math.log1p(-init_edge_prob)]
        )
        self.gen_matrix = nn.ParameterList([
            nn.Parameter(
                base_logits.expand(self.num_pairs, 2).clone()
                + init_std * torch.randn(self.num_pairs, 2)
            )
            for _ in range(num_nodes)
        ])

    def sample_adj_row(self, target_i, hard=False):
        """只采样目标节点 i 的一行邻居对，返回 [C(N-1,2)]。"""
        return F.gumbel_softmax(
            self.gen_matrix[target_i], tau=self.temperature, hard=hard, dim=-1
        )[:, 0]

    def row_probabilities(self, target_i):
        """当前行的 Bernoulli 有边概率，用于期望超度惩罚。"""
        return self.gen_matrix[target_i].softmax(dim=-1)[:, 0]

    def edge_probabilities(self):
        """按原行顺序返回 [N,Q] 概率矩阵，用于结构评价和保存。"""
        return torch.stack([self.row_probabilities(i) for i in range(self.N)])


class AIDDNeighborPairDynamics(nn.Module):
    """
    对目标 i：NN1([x_j, x_k, x_i]) -> 邻居对门控加权求和
              -> NN2 -> 拼接自身状态 -> NN3 -> log_softmax。
    NN1 输入 [x_j(t), x_k(t), x_i(t)]，三个值都是当前时刻的 0/1/2。
    """

    def __init__(self, num_nodes, feature_dim=128, num_states=3, pair_chunk_size=4096):
        super().__init__()
        if num_nodes < 3:
            raise ValueError("三元超图至少需要 3 个节点")
        self.N = num_nodes
        self.num_states = num_states
        self.F_dim = feature_dim
        self.pair_chunk_size = pair_chunk_size

        self.nn1 = NN1(3, feature_dim)
        self.nn2 = NN2(feature_dim, feature_dim, feature_dim)
        self.nn3 = NN3(1 + feature_dim, num_states)

        # local_pairs 在 N-1 个局部编号上枚举；编号 >= i 的位置加 1，排除节点 i。
        # 例如 N=4：第 0 行为 (1,2),(1,3),(2,3)；第 1 行为 (0,2),(0,3),(2,3)。
        local_pairs = torch.combinations(torch.arange(num_nodes - 1), r=2)
        target = torch.arange(num_nodes).reshape(num_nodes, 1, 1)
        neighbor_pairs = local_pairs.unsqueeze(0) + (local_pairs.unsqueeze(0) >= target)
        self.register_buffer("neighbor_pairs", neighbor_pairs)
        self.num_pairs = local_pairs.shape[0]

        # NN1 只有 K^3 种离散输入。先编码这些输入，再按状态组合累加门控，
        # 与对每个候选邻居对显式计算 NN1 后求和等价，避免 [B,Q,F] 大张量。
        states = torch.arange(num_states)
        own, first, second = torch.meshgrid(states, states, states, indexing="ij")
        state_triples = torch.stack(
            (torch.minimum(first, second), torch.maximum(first, second), own), dim=-1
        )
        self.register_buffer("state_inputs", state_triples.float().reshape(-1, 3))

    def forward(self, x, adj_row, target_i):
        """x: [B,N]；adj_row: [Q]；仅预测节点 target_i，返回 [B,K]。"""
        batch_size = x.shape[0]
        k = self.num_states
        states = x.long()
        pairs = self.neighbor_pairs[target_i]
        group_weight = adj_row.new_zeros(batch_size, k * k)

        for start in range(0, self.num_pairs, self.pair_chunk_size):
            stop = min(start + self.pair_chunk_size, self.num_pairs)
            current_pairs = pairs[start:stop]
            first = states[:, current_pairs[:, 0]]
            second = states[:, current_pairs[:, 1]]
            pair_state = torch.minimum(first, second) * k + torch.maximum(first, second)
            weights = adj_row[start:stop].unsqueeze(0).expand(batch_size, -1)
            group_weight = group_weight.scatter_add(1, pair_state, weights)

        message_table = self.nn1(self.state_inputs).reshape(k, k * k, self.F_dim)
        messages = message_table[states[:, target_i]]
        neighbor_sum = torch.sum(group_weight.unsqueeze(-1) * messages, dim=1)
        h2 = self.nn2(neighbor_sum)
        own_state = x[:, target_i].unsqueeze(-1)
        return self.nn3(torch.cat((own_state, h2), dim=-1)).log_softmax(dim=-1)
