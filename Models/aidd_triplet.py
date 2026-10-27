"""AIDD 式三元超图：联合邻居消息、直接超边门控和共享的三层网络。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aidd_hypergraph import NN1, NN2, NN3


class AIDDTripletModel(nn.Module):
    """
    对目标 i：
      message_i = sum_{e={i,j,k}} gate[e] * NN1([x_j, x_k, x_i])
      logits_i = NN3([NN2(message_i), x_i])

    每个无序三元组只维护一份门控，三个目标方向共用它。
    两个邻居的状态排序后编码，保证交换邻居不改变消息。
    """

    def __init__(
        self,
        num_nodes,
        feature_dim=128,
        num_classes=3,
        init_edge_prob=0.001,
        gumbel_hard=False,
        candidate_hyperedges=None,
        device="cpu",
    ):
        super().__init__()
        if num_nodes < 3:
            raise ValueError("三元超图至少需要 3 个节点")
        if not 0 < init_edge_prob < 1:
            raise ValueError("init_edge_prob 必须在 (0, 1) 内")
        self.N = num_nodes
        self.F_dim = feature_dim
        self.num_states = num_classes
        self.gumbel_hard = gumbel_hard

        # 与 AIDD 一样，全体目标节点共用 NN1、NN2、NN3。
        # NN1 的输入由单个邻居 + 目标，扩展为两个共同邻居 + 目标。
        self.nn1 = NN1(input_dim=3, hidden_dim=feature_dim)
        self.nn2 = NN2(feature_dim, feature_dim, feature_dim)
        self.nn3 = NN3(feature_dim + 1, num_classes)

        # 联合学习时枚举全部候选，不读取真实结构；固定结构对照才传入真实三元组。
        if candidate_hyperedges is None:
            candidates = torch.combinations(torch.arange(num_nodes), r=3)
        else:
            candidates = torch.as_tensor(candidate_hyperedges, dtype=torch.long)
            candidates = candidates.sort(dim=1).values
        self.register_buffer("hyperedges", candidates.to(device))
        self.num_candidates = len(candidates)

        base = torch.tensor(
            [math.log1p(-init_edge_prob), math.log(init_edge_prob)],
            dtype=torch.float32,
            device=device,
        )
        self.edge_logits = nn.Parameter(base.expand(self.num_candidates, 2).clone())

        # 对离散状态只计算 K^3 个 NN1 输入，与逐候选组编码严格等价。
        states = torch.arange(num_classes, dtype=torch.float32)
        target, first, second = torch.meshgrid(states, states, states, indexing="ij")
        state_inputs = torch.stack(
            (torch.minimum(first, second), torch.maximum(first, second), target),
            dim=-1,
        ).reshape(-1, 3)
        self.register_buffer("state_inputs", state_inputs.to(device))
        self.to(device)

    def dynamics_parameters(self):
        for module in (self.nn1, self.nn2, self.nn3):
            yield from module.parameters()

    def edge_probabilities(self):
        # Bernoulli 有边概率，用于结构评价；不等于软 Gumbel 门控的期望。
        return self.edge_logits.softmax(dim=-1)[:, 1]

    def sample_edge_weights(self, temperature=1.0):
        """训练和 Monte Carlo 评价共用同一门控采样。"""
        return F.gumbel_softmax(
            self.edge_logits, tau=temperature, hard=self.gumbel_hard, dim=-1
        )[:, 1]

    def forward_with_structure(self, x, edge_weight):
        """
        x: [batch, N]，SIR 状态为 0/1/2。
        edge_weight: [M]，控制每个候选三元组是否传递消息。

        按目标节点和邻居状态组合累加权重，不创建 [batch, M, F] 特征。
        scatter_add 保留 gate 的梯度；每条超边向三个节点分别发送消息。
        """
        batch_size = x.shape[0]
        k = self.num_states
        pair_count = k * k
        state_index = x.long()
        group_weight = edge_weight.new_zeros(batch_size, self.N * pair_count)
        weights = edge_weight.unsqueeze(0).expand(batch_size, -1)

        for target_slot, first_slot, second_slot in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
            target_nodes = self.hyperedges[:, target_slot]
            first = state_index[:, self.hyperedges[:, first_slot]]
            second = state_index[:, self.hyperedges[:, second_slot]]
            pair_state = torch.minimum(first, second) * k + torch.maximum(first, second)
            destination = target_nodes.unsqueeze(0) * pair_count + pair_state
            group_weight = group_weight.scatter_add(1, destination, weights)

        group_weight = group_weight.reshape(batch_size, self.N, pair_count)
        message_table = self.nn1(self.state_inputs).reshape(k, pair_count, self.F_dim)
        messages = message_table[state_index]  # [batch, N, K^2, F]
        neighbor_sum = torch.einsum("bnq,bnqf->bnf", group_weight, messages)
        h2 = self.nn2(neighbor_sum)
        return self.nn3(torch.cat((h2, x.unsqueeze(-1)), dim=-1))

    def forward(self, x, temperature=1.0):
        edge_prob = self.edge_probabilities()
        sampled = self.sample_edge_weights(temperature)
        logits = self.forward_with_structure(x, sampled)
        return logits, sampled, edge_prob
