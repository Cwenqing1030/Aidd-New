"""AIDD 超图动力学模型，不包含数据读取、训练循环或评价代码。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NN1(nn.Module):
    # 原始 AIDD：只提取候选节点 j 与目标节点 i 的成对特征。
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class NN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.e2n = nn.Linear(input_dim, hidden_dim)
        self.n2n = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.e2n(x))
        x = F.relu(self.n2n(x))
        return x


class NN3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.output(x)


class HyperedgeFusionMLP(nn.Module):
    """将按固定超边编号拼接的 E 个 F 维特征映射回 F 维。"""

    def __init__(self, num_hyperedges, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_hyperedges * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# NxE 代理 incidence matrix
# A_logits[i,e,:] 是节点 i 是否属于代理超边 e 的二分类 logits。
# 采样按“节点行”组织：每一行对应一个节点对 E 条代理超边的成员关系。
# ============================================================
def init_A_logits(N, E, init_prob, device):
    if not (0.0 < init_prob < 1.0):
        raise ValueError(f"init_prob 必须在 (0,1)，当前为 {init_prob}")

    no_member_logit = float(np.log(1.0 - init_prob))
    member_logit = float(np.log(init_prob))
    base = torch.tensor(
        [no_member_logit, member_logit], dtype=torch.float32, device=device
    )
    # 独立随机扰动使每个 (i,e) 都随机初始化；不读取 B_true 的任何结构信息。
    initial_logits = base.view(1, 1, 2) + 0.01 * torch.randn(
        N, E, 2, dtype=torch.float32, device=device
    )
    A_logits = nn.Parameter(initial_logits)
    return A_logits


def sample_gumbel_incidence_rowwise(A_logits, temperature):
    """
    A_logits: [N,E,2]
    return:
      A_sampled: [N,E]
      A_prob:    [N,E]
      two_class: [N,E,2]
    """
    if A_logits.ndim != 3 or A_logits.shape[-1] != 2:
        raise ValueError(f"A_logits 应为 [N,E,2]，实际 {tuple(A_logits.shape)}")

    N, E, _ = A_logits.shape

    A_prob = torch.softmax(A_logits, dim=-1)[..., 1]  # [N,E]

    # 前向严格为 0/1；反向通过 straight-through Gumbel-Softmax 更新 A_logits。
    two_class = F.gumbel_softmax(
        A_logits, tau=temperature, hard=True, dim=-1
    )  # [N,E,2]
    A_sampled = two_class[..., 1]  # [N,E]

    return A_sampled, A_prob, two_class


# ============================================================
# 对单个目标节点 i：先经 NN1 提取 pairwise 特征，再按每条代理超边聚合。
# ============================================================
def aggregate_hyperedge_messages(
    nn1,
    X_batch,
    A_sampled,
    target_i,
    shape_trace=None,
):
    """
    X_batch:   [batch,N]
    A_sampled: [N,E]
    return:    [batch,E,F]，保留每条超边的固定槽位。
    """
    batch_size, N = X_batch.shape
    N_A, E = A_sampled.shape
    if N_A != N or not (0 <= target_i < N):
        raise ValueError(
            f"维度或目标节点错误: X N={N}, "
            f"A shape={tuple(A_sampled.shape)}, i={target_i}"
        )
    source_state = X_batch.unsqueeze(-1)  # [batch,N,1]
    target_state = X_batch[:, target_i].view(batch_size, 1, 1).expand(-1, N, -1)
    pair_input_i = torch.cat((source_state, target_state), dim=-1)  # [batch,N,2]
    h1_i = nn1(pair_input_i)  # [batch,N,F]

    # A_exclude_i[i,e]=0，确保 h_ii 从所有超边信息中排除。
    A_exclude_i = A_sampled.clone()
    A_exclude_i[target_i, :] = 0.0
    single_edge_column = A_exclude_i[:, 0]  # [N]

    # 对每个 e: [N]^T @ [batch,N,F] -> [batch,F]。
    # einsum 同时计算 E 条边，结果仍保留 [batch,E,F]，不提前对边求和。
    all_edge_features = torch.einsum("ne,bnf->bef", A_exclude_i, h1_i)
    all_edge_features = all_edge_features * A_sampled[target_i].view(1, E, 1)
    single_edge_feature = all_edge_features[:, 0, :]  # [batch,F]

    if shape_trace is not None:
        shape_trace.update(
            pair_input_i=tuple(pair_input_i.shape),
            h1_i=tuple(h1_i.shape),
            single_edge_column=tuple(single_edge_column.shape),
            single_edge_feature=tuple(single_edge_feature.shape),
            all_edge_features=tuple(all_edge_features.shape),
        )
    return all_edge_features


class AIDDHypergraphModel(nn.Module):
    """成对编码、超边融合和可学习关联矩阵组成的动力学模型。"""

    def __init__(
        self,
        num_nodes,
        num_hyperedges,
        feature_dim=128,
        num_classes=2,
        init_prob=0.02,
        device="cpu",
    ):
        super().__init__()
        self.N = num_nodes
        self.E = num_hyperedges
        self.F_dim = feature_dim

        self.nn1 = NN1(input_dim=2, hidden_dim=feature_dim)
        self.fusion_mlp = HyperedgeFusionMLP(
            num_hyperedges=num_hyperedges, feature_dim=feature_dim
        )
        self.nn2 = NN2(
            input_dim=feature_dim, hidden_dim=feature_dim, output_dim=feature_dim
        )
        self.nn3 = NN3(input_dim=feature_dim + 1, output_dim=num_classes)

        # 沿用原脚本的初始化顺序与结构参数随机数所在设备。
        self.A_logits = init_A_logits(num_nodes, num_hyperedges, init_prob, device)
        self.to(device)

    def dynamics_parameters(self):
        """动力学网络参数，供训练脚本设置与结构参数不同的学习率。"""
        for module in (self.nn1, self.fusion_mlp, self.nn2, self.nn3):
            yield from module.parameters()

    def forward_with_incidence(self, X_batch, A_weight, shape_trace=None):
        """
        训练时 A_weight=A_sampled；最终动力学评价时使用确定性的 A_prob_final。
        每次只构造一个目标节点的 [batch,E,F]，不创建 [batch,N,E,F]。
        """
        batch_size = X_batch.shape[0]

        predictions = []
        for target_i in range(self.N):
            trace_i = shape_trace if target_i == 0 else None
            all_edge_features = aggregate_hyperedge_messages(
                self.nn1, X_batch, A_weight, target_i, shape_trace=trace_i
            )  # [batch,E,F]
            concat_edge_features = all_edge_features.reshape(batch_size, self.E * self.F_dim)
            edge_fused_i = self.fusion_mlp(concat_edge_features)
            h2_i = self.nn2(edge_fused_i)
            nn3_input = torch.cat(
                (h2_i, X_batch[:, target_i].unsqueeze(-1)), dim=-1
            )
            prediction_i = self.nn3(nn3_input)

            if trace_i is not None:
                trace_i.update(
                    concat_edge_features=tuple(concat_edge_features.shape),
                    fusion_MLP_output=tuple(edge_fused_i.shape),
                    NN2_output=tuple(h2_i.shape),
                    NN3_input=tuple(nn3_input.shape),
                    prediction_i=tuple(prediction_i.shape),
                )
            predictions.append(prediction_i)

        prediction = torch.stack(predictions, dim=1)
        return prediction

    def forward(self, X_batch, temperature=1.0, shape_trace=None):
        A_sampled, A_prob, _ = sample_gumbel_incidence_rowwise(
            self.A_logits, temperature
        )

        prediction = self.forward_with_incidence(X_batch, A_sampled, shape_trace=shape_trace)
        return prediction, A_sampled, A_prob
