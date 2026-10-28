"""连续状态的三元邻居组预测器；沿用 AIDD 的 NN1 -> NN2 -> NN3。"""

import torch
import torch.nn as nn

from .layers import NN1, NN2, NN3


class AIDDContinuousNeighborPairDynamics(nn.Module):
    def __init__(
        self, num_nodes, state_dim, delta_scale, feature_dim=128,
        pair_chunk_size=4096,
    ):
        super().__init__()
        self.N = num_nodes
        self.state_dim = state_dim
        self.F_dim = feature_dim
        self.pair_chunk_size = pair_chunk_size
        self.nn1 = NN1(3 * state_dim, feature_dim)
        self.nn2 = NN2(feature_dim, feature_dim, feature_dim)
        self.nn3 = NN3(state_dim + feature_dim, state_dim)
        self.register_buffer("delta_scale", torch.as_tensor(delta_scale, dtype=torch.float32))

        local_pairs = torch.combinations(torch.arange(num_nodes - 1), r=2)
        targets = torch.arange(num_nodes).reshape(num_nodes, 1, 1)
        pairs = local_pairs.unsqueeze(0) + (local_pairs.unsqueeze(0) >= targets)
        self.register_buffer("neighbor_pairs", pairs)
        self.num_pairs = local_pairs.shape[0]

    def forward(self, x, adj_row, target_i):
        own = x[:, target_i]
        pairs = self.neighbor_pairs[target_i]
        neighbor_sum = x.new_zeros(x.shape[0], self.F_dim)
        for start in range(0, self.num_pairs, self.pair_chunk_size):
            stop = min(start + self.pair_chunk_size, self.num_pairs)
            selected = pairs[start:stop]
            first, second = x[:, selected[:, 0]], x[:, selected[:, 1]]
            target = own.unsqueeze(1).expand(-1, stop - start, -1)
            # 两种排列共用 NN1 并取平均，确保交换两个邻居不改变消息。
            message = 0.5 * (
                self.nn1(torch.cat((first, second, target), dim=-1))
                + self.nn1(torch.cat((second, first, target), dim=-1))
            )
            neighbor_sum = neighbor_sum + (
                message * adj_row[start:stop].reshape(1, -1, 1)
            ).sum(dim=1)
        hidden = self.nn2(neighbor_sum)
        delta = self.nn3(torch.cat((own, hidden), dim=-1))
        return own + delta * self.delta_scale

    def prediction_loss(self, prediction, target, reduction="mean"):
        error = ((prediction - target) / self.delta_scale).square()
        return {"mean": torch.mean, "sum": torch.sum}[reduction](error)
