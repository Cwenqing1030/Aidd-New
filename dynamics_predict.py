import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ================
# 第一个神经网络 NN1
# ================
# AIDD pairwise message passing：
# h_ij^(1) = NN^(1)([x_j^t, x_i^t])
# i = target node，j = source node
# 每条候选消息只看 (x_j, x_i)，NN1 在所有节点对之间共享
# 输入 [B,target,source,2]  ->  输出 [B,target,source,F]
#   dim0 B      = batch（同一时刻的多个时间样本）
#   dim1 target = 目标节点 i
#   dim2 source = 候选邻居节点 j
#   dim3        = 2（[x_j, x_i]）或 F（32维消息特征）
class NN1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # 输入维度固定为2（[x_j, x_i]），与节点数n无关
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return x


# ================
# 第二个神经网络 NN2
# ================
# 输入 [B,target,F]（邻居聚合后的消息和）  ->  输出 [B,target,F]
class NN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.e2n = nn.Linear(input_dim, hidden_dim)
        self.n2n = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.e2n(x))
        x = F.relu(self.n2n(x))

        return x


# ================
# 第三个神经网络 NN3
# ================
# 输入 [B,target,F+1]（[h2, x_i]）  ->  输出 [B,target,3]（0=S,1=I,2=R 的logits）
class NN3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.output(x)
        return x


# ==========
# 读取数据
# ==========
def load_cond_data(data_dir, dynamics_type):
    data_dir = Path(data_dir)
    if not data_dir.exists() and data_dir == Path("CoND-main/data/BA_N200_m2"):
        data_dir = Path("data/BA_N200_m2")

    network_name = data_dir.name
    edge_or_adjacency = np.loadtxt(data_dir / f"{network_name}.txt", dtype=np.int64)
    Xt = np.loadtxt(
        data_dir / f"{network_name}_{dynamics_type}_x.txt", dtype=np.float32
    )
    Yt = np.loadtxt(
        data_dir / f"{network_name}_{dynamics_type}_y.txt", dtype=np.float32
    ).astype(np.int64)

    n = Xt.shape[1]
    if edge_or_adjacency.shape == (n, n):
        A_true = edge_or_adjacency
    else:
        A_true = np.zeros((n, n), dtype=np.int64)
        source = edge_or_adjacency[:, 0]
        target = edge_or_adjacency[:, 1]
        A_true[source, target] = 1
        A_true[target, source] = 1

    if Yt.shape != Xt.shape:
        raise ValueError(f"x/y shape mismatch: {Xt.shape} and {Yt.shape}")

    return Xt, Yt, A_true


# ================================
# 无向 Gumbel-Softmax 邻接采样
# ================================
# edge_logits: [num_upper, 2]
#   dim0 = num_upper = n*(n-1)/2（每条唯一无向候选边 i<j 只维护一份参数）
#   dim1 = 2 个类别：
#     edge_logits[:, 0] = no-edge logit
#     edge_logits[:, 1] = edge logit
#
# 返回：
#   A_sampled             [N,N]     采样后的对称邻接矩阵（训练用）
#   sampled_edge_gate     [num_upper] 采样后的边门控（soft时为[0,1]实数，hard时为0/1）
#   edge_prob             [num_upper] 确定性边概率（softmax[:, 1]，用于稀疏正则与日志）
#   edge_sample_two_class [num_upper,2] 完整二分类Gumbel-Softmax样本
#
# 要求：
#   每条无向边只采样一次；上下三角使用同一个采样结果；
#   A_sampled 严格对称；对角线严格为零；
#   Gumbel-Softmax 的梯度回传到 edge_logits。
def sample_gumbel_adjacency(edge_logits, tri_i, tri_j, n, device, temperature, hard):
    # 1. 确定性边概率（期望）：softmax 的第二个分量 = 边的概率
    edge_prob = torch.softmax(edge_logits, dim=-1)[:, 1]  # [num_upper]

    # 2. 二分类 Gumbel-Softmax 采样（训练路径，可导）
    edge_sample_two_class = F.gumbel_softmax(
        edge_logits,
        tau=temperature,
        hard=hard,
        dim=-1,
    )  # [num_upper, 2]

    # 3. 取出"是边"的分量作为门控
    sampled_edge_gate = edge_sample_two_class[:, 1]  # [num_upper]

    # 4. 同一个采样结果镜像填充上下三角，对角线恒为0
    #    out-of-place index_put，保证梯度能经两个方向都回到 edge_logits
    A_sampled = torch.zeros(n, n, device=device)
    A_sampled = A_sampled.index_put((tri_i, tri_j), sampled_edge_gate)
    A_sampled = A_sampled.index_put((tri_j, tri_i), sampled_edge_gate)

    return A_sampled, sampled_edge_gate, edge_prob, edge_sample_two_class


# ================================
# 结构恢复评价（确定性edge_prob vs A_true，严格上三角）
# ================================
# 注意：这里不是验证集——A_true是整张网络的真实结构，
# 动力学数据全部来自同一张网络并全部参与训练。
# 衡量的是"学到的结构 vs 真实结构"的吻合度。
def evaluate_structure(edge_prob, A_true_np, threshold, tri_i, tri_j, n):
    with torch.no_grad():
        A_hat = torch.zeros(n, n, device=edge_prob.device)
        A_hat[tri_i, tri_j] = edge_prob
        A_hat[tri_j, tri_i] = edge_prob
        A_rec = (A_hat >= threshold).float()
        A_hat_np = A_hat.cpu().numpy()
        A_rec_np = A_rec.cpu().numpy().astype(np.int64)

    upper = np.triu_indices(n, k=1)
    A_true_eval = A_true_np[upper]
    A_hat_eval = A_hat_np[upper]
    A_rec_eval = A_rec_np[upper]

    roc_auc = roc_auc_score(A_true_eval, A_hat_eval)
    pr_auc = average_precision_score(A_true_eval, A_hat_eval)
    precision = precision_score(A_true_eval, A_rec_eval, zero_division=0)
    recall = recall_score(A_true_eval, A_rec_eval, zero_division=0)
    f1 = f1_score(A_true_eval, A_rec_eval, zero_division=0)
    accuracy = accuracy_score(A_true_eval, A_rec_eval)
    shd = int(np.sum(A_true_eval != A_rec_eval))

    return roc_auc, pr_auc, precision, recall, f1, accuracy, shd


# ================================
# 一次性 shape trace（--trace_shapes）
# 调试用，训练时暂时注释掉；需要时取消注释并加 --trace_shapes
# ================================
# def trace_tensor(name, tensor, max_values=6):
#     req = tensor.requires_grad
#     t = tensor.detach()
#     flat = t.reshape(-1)
#     vals = flat[:max_values].tolist()
#
#     if t.numel() > 0:
#         mn, mx = t.min().item(), t.max().item()
#         if t.dtype.is_floating_point:
#             mean = t.mean().item()
#         else:
#             mean = t.float().mean().item()
#     else:
#         mn = mx = mean = float("nan")
#
#     if t.dtype.is_floating_point:
#         vstr = ", ".join(f"{v:.4f}" for v in vals)
#     else:
#         vstr = ", ".join(str(int(v)) for v in vals)
#
#     print(
#         f"[trace] {name} | shape={tuple(t.shape)} dtype={t.dtype} "
#         f"device={t.device} requires_grad={req} "
#         f"min={mn:.4f} max={mx:.4f} mean={mean:.4f} | first={vstr}"
#     )


# ===============
# CoND数据路径参数
# ===============
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="CoND-main/data/BA_N200_m2")
parser.add_argument("--dynamics_type", default="SIR")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lambda_sparse", type=float, default=0.01)
parser.add_argument("--init_edge_prob", type=float, default=0.05)
parser.add_argument("--lr_dyn", type=float, default=0.001)
parser.add_argument("--lr_adj", type=float, default=0.004)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--tau_start", type=float, default=1.0)
parser.add_argument("--tau_end", type=float, default=0.5)
parser.add_argument("--gumbel_hard", action="store_true", default=False)
parser.add_argument("--trace_shapes", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ===================
# 读取CoND网络和动力学样本
# ===================
Xt, Yt, A_true = load_cond_data(args.data_dir, args.dynamics_type)
# A_true只在训练结束后的结构恢复评价中使用
print("==============================")
print(f"开始恢复 CoND 网络 {Path(args.data_dir).name} ({args.dynamics_type})")
print(f"数据形状: x={Xt.shape}, y={Yt.shape}, A_true={A_true.shape}")
print("==============================")


# =========
# 创建 NN1
# =========
# 隐藏维度 F
F_dim = 32
n = Xt.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 输入为节点对拼接 [x_j, x_i]，维度2与n无关
nn1 = NN1(input_dim=2, hidden_dim=F_dim).to(device)


# ==================================
# 创建无向结构参数（二分类 logits，严格上三角）
# ==================================
# 每条唯一无向候选边 i<j 维护 2 个 logits：
#   edge_logits[:, 0] = no-edge logit
#   edge_logits[:, 1] = edge logit
# 初始化保证 softmax(edge_logits, dim=-1)[:, 1] == init_edge_prob
num_upper = n * (n - 1) // 2
if not (0.0 < args.init_edge_prob < 1.0):
    raise ValueError(f"--init_edge_prob 必须在 (0,1) 内，当前为 {args.init_edge_prob}")

p = args.init_edge_prob
no_edge_logit = float(np.log(1.0 - p))
edge_logit = float(np.log(p))
init_logits = (
    torch.tensor([no_edge_logit, edge_logit], device=device)
    .reshape(1, 2)
    .repeat(num_upper, 1)
)
edge_logits = nn.Parameter(init_logits)  # [num_upper, 2]

with torch.no_grad():
    init_check = torch.softmax(edge_logits, dim=-1)[:, 1].mean().item()
print(
    f"结构参数初始化: edge_logits={tuple(edge_logits.shape)}, "
    f"softmax(edge_logits)[:,1].mean()={init_check:.6f} (应≈{args.init_edge_prob})"
)

# 上三角索引缓存（i < j）
tri_i, tri_j = torch.triu_indices(n, n, offset=1, device=device)


# =========
# 创建 NN2
# =========
# NN1最终输出的特征维度为F
# A_sampled筛选以后仍然得到1×F
# 所以NN2：1×F -> 1×F
nn2_input_dim = F_dim
nn2_hidden_dim = F_dim
nn2_output_dim = F_dim

nn2 = NN2(
    input_dim=nn2_input_dim, hidden_dim=nn2_hidden_dim, output_dim=nn2_output_dim
).to(device)


# =========
# 创建 NN3
# =========
# NN2输出：h2 = 1×F
# 当前节点状态：x_i^t = 1×1
# 拼接以后：1×(F+1)
nn3_input_dim = F_dim + 1

# SIR三个类别：
# 0 -> S
# 1 -> I
# 2 -> R
nn3_output_dim = 3

nn3 = NN3(input_dim=nn3_input_dim, output_dim=nn3_output_dim).to(device)


# ===========
# 定义损失函数
# ===========
criterion = nn.CrossEntropyLoss()


# ===========
# 定义优化器（动力学网络与结构参数分开的学习率）
# ===========
optimizer = torch.optim.Adam(
    [
        {
            "params": list(nn1.parameters())
            + list(nn2.parameters())
            + list(nn3.parameters()),
            "lr": args.lr_dyn,
        },
        {"params": [edge_logits], "lr": args.lr_adj},
    ]
)


# ===================
# Gumbel 温度退火参数校验
# ===================
if args.tau_start <= 0 or args.tau_end <= 0:
    raise ValueError(
        f"--tau_start / --tau_end 必须为正数，当前 tau_start={args.tau_start}, "
        f"tau_end={args.tau_end}"
    )


# ====
# 训练
# ====
num_epochs = 100
Xt_tensor = torch.from_numpy(Xt).to(device)
Yt_tensor = torch.from_numpy(Yt).to(device)

# A_true转numpy，供每个epoch的结构评价和最终评价使用
A_true_np = A_true.astype(np.int64)

# =============
# 第一层：epoch
# =============
# trace_done = False
for epoch in range(num_epochs):
    # 平滑指数退火：epoch=0 时 temperature == tau_start，最后一轮 == tau_end
    progress = epoch / max(num_epochs - 1, 1)
    temperature = args.tau_start * (args.tau_end / args.tau_start) ** progress

    epoch_pred = 0.0
    epoch_sparse = 0.0
    epoch_total = 0.0
    epoch_gate_mean = 0.0
    sample_count = 0
    num_batches = 0

    # 每个batch同时训练多个时间样本和全部目标节点
    for start in range(0, Xt_tensor.shape[0], args.batch_size):
        # X_batch: [B,N]   B=batch内时间样本数，N=节点数，值∈{0,1,2}
        # Y_batch: [B,N]   B=batch内时间样本数，N=节点数，值∈{0,1,2}，dtype=long
        X_batch = Xt_tensor[start : start + args.batch_size]
        Y_batch = Yt_tensor[start : start + args.batch_size]

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------
        # 邻接采样（Gumbel-Softmax）
        # ----------------------------------------
        # edge_logits: [num_upper,2] -> edge_prob: [num_upper]
        #             -> edge_sample_two_class: [num_upper,2]
        #             -> sampled_edge_gate: [num_upper]
        #             -> A_sampled: [N,N]（严格对称、对角全0）
        (
            A_sampled,
            sampled_edge_gate,
            edge_prob,
            edge_sample_two_class,
        ) = sample_gumbel_adjacency(
            edge_logits,
            tri_i,
            tri_j,
            n,
            device,
            temperature,
            args.gumbel_hard,
        )

        # ----------------------------------------
        # 构造所有节点对的 pairwise 输入
        # ----------------------------------------
        # x_batch: [B,N,1]  （X_batch.unsqueeze(-1)，每节点1维状态）
        x_batch = X_batch.unsqueeze(-1)

        # source_state: [B,target,source,1]  位置(b,i,j)放 x_j（候选源节点状态）
        source_state = x_batch.unsqueeze(1).expand(
            X_batch.shape[0], n, n, 1
        )

        # target_state: [B,target,source,1]  位置(b,i,j)放 x_i（目标节点状态）
        target_state = x_batch.unsqueeze(2).expand(
            X_batch.shape[0], n, n, 1
        )

        # pair_input: [B,target,source,2]  cat([x_j, x_i], dim=-1)
        # pair_input[b,i,j] == [X_batch[b,j], X_batch[b,i]]
        pair_input = torch.cat((source_state, target_state), dim=-1)

        # h1: [B,target,source,F]  每条消息h1[b,i,j]只依赖x_i、x_j和NN1共享参数
        h1 = nn1(pair_input)

        # neighbor_sum: [B,target,F]
        # einsum("st,btsf->btf")：A_sampled[source,target] 乘 h1[:,target,source,:] 按source求和
        neighbor_sum = torch.einsum("st,btsf->btf", A_sampled, h1)

        # h2: [B,target,F]  NN2处理后的聚合邻居特征
        h2 = nn2(neighbor_sum)

        # nn3_input: [B,target,F+1]  cat([h2, x_i], dim=-1)
        nn3_input = torch.cat((h2, X_batch.unsqueeze(-1)), dim=-1)

        # prediction: [B,target,3]  NN3输出的logits（0=S,1=I,2=R）
        prediction = nn3(nn3_input)

        # prediction_loss: 标量  CrossEntropyLoss（reshape成 [-1,3] vs [-1]）
        prediction_loss = criterion(
            prediction.reshape(-1, nn3_output_dim), Y_batch.reshape(-1)
        )

        # sparse_loss: 标量  使用确定性期望边概率，避免Gumbel采样抖动
        # edge_prob: [num_upper]（每条唯一上三角边只计一次）
        sparse_loss = edge_prob.mean()

        # total_loss: 标量 = prediction_loss + lambda_sparse * sparse_loss
        total_loss = prediction_loss + args.lambda_sparse * sparse_loss

        # -------------------------------------------------
        # 一次性 shape trace（仅第一个epoch的第一个batch）
        # 调试用，训练时暂时注释掉；需要时取消注释并加 --trace_shapes
        # -------------------------------------------------
        # if args.trace_shapes and not trace_done:
        #     trace_tensor("1. X_batch", X_batch)
        #     trace_tensor("2. x_batch", x_batch)
        #     trace_tensor("3. source_state", source_state)
        #     trace_tensor("4. target_state", target_state)
        #     trace_tensor("5. pair_input", pair_input)
        #     trace_tensor("6. h1", h1)
        #     trace_tensor("7. edge_logits", edge_logits)
        #     trace_tensor("8. edge_prob", edge_prob)
        #     trace_tensor("9. edge_sample_two_class", edge_sample_two_class)
        #     trace_tensor("10. sampled_edge_gate", sampled_edge_gate)
        #     trace_tensor("11. A_sampled", A_sampled)
        #     trace_tensor("12. neighbor_sum", neighbor_sum)
        #     trace_tensor("13. h2", h2)
        #     trace_tensor("14. nn3_input", nn3_input)
        #     trace_tensor("15. prediction", prediction)
        #     trace_tensor("16. Y_batch", Y_batch)
        #     trace_tensor("17. prediction_loss", prediction_loss)
        #     trace_tensor("18. sparse_loss", sparse_loss)
        #     trace_tensor("19. total_loss", total_loss)

        total_loss.backward()

        # 反向传播后：确认梯度确实经Gumbel-Softmax回到结构参数（调试用）
        # if args.trace_shapes and not trace_done:
        #     g = edge_logits.grad
        #     print(
        #         f"[trace] edge_logits.grad | shape={tuple(g.shape)} "
        #         f"min={g.min().item():.6f} max={g.max().item():.6f} "
        #         f"mean={g.mean().item():.6f} norm={g.norm().item():.6f}"
        #     )
        #     p0 = list(nn1.parameters())[0]
        #     print(f"[trace] NN1 第一个参数梯度 norm = {p0.grad.norm().item():.6f}")
        #     trace_done = True

        optimizer.step()

        batch_sample_count = Y_batch.numel()
        epoch_pred += prediction_loss.detach().item() * batch_sample_count
        epoch_sparse += sparse_loss.detach().item() * batch_sample_count
        epoch_total += total_loss.detach().item() * batch_sample_count
        epoch_gate_mean += sampled_edge_gate.detach().mean().item()
        sample_count += batch_sample_count
        num_batches += 1

    # -------------------
    # 每个epoch的日志
    # -------------------
    with torch.no_grad():
        current_edge_prob = torch.softmax(edge_logits, dim=-1)[:, 1]
        mean_edge_probability = current_edge_prob.mean().item()
        thresholded_edge_count = (
            current_edge_prob >= args.threshold
        ).sum().item()
        # 每个epoch的结构恢复评价（确定性edge_prob vs A_true，严格上三角）
        roc_auc, pr_auc, precision, recall, f1, accuracy, shd = evaluate_structure(
            current_edge_prob, A_true_np, args.threshold, tri_i, tri_j, n
        )

    mean_sampled_gate = epoch_gate_mean / max(num_batches, 1)

    print(
        f"CoND Epoch {epoch + 1}/{num_epochs} | "
        f"temperature = {temperature:.4f} | hard = {args.gumbel_hard} | "
        f"pred_loss = {epoch_pred / sample_count:.6f} | "
        f"sparse_loss = {epoch_sparse / sample_count:.6f} | "
        f"total_loss = {epoch_total / sample_count:.6f} | "
        f"mean_edge_probability = {mean_edge_probability:.4f} | "
        f"mean_sampled_gate = {mean_sampled_gate:.4f} | "
        f"edges = {thresholded_edge_count}"
    )
    print(
        f"  eval: ROC-AUC = {roc_auc:.4f} | PR-AUC = {pr_auc:.4f} | "
        f"Precision = {precision:.4f} | Recall = {recall:.4f} | "
        f"F1 = {f1:.4f} | Accuracy = {accuracy:.4f} | SHD = {shd}"
    )


# =================================
# 训练结束：确定性评价（不再采样）
# =================================
# 最终边概率只用 softmax 期望，不包含任何Gumbel随机性，
# 因此同一模型重复评价结果完全一致。
with torch.no_grad():
    # edge_prob_final: [num_upper]
    edge_prob_final = torch.softmax(edge_logits, dim=-1)[:, 1]

    # A_hat_final: [N,N] 对称的确定性邻接概率矩阵（对角线恒为0）
    A_hat_final = torch.zeros(n, n, device=device)
    A_hat_final[tri_i, tri_j] = edge_prob_final
    A_hat_final[tri_j, tri_i] = edge_prob_final

# A_recovered: [N,N] 对称的0/1邻接矩阵（阈值来自命令行参数，默认0.5）
A_recovered = (A_hat_final >= args.threshold).float()

A_hat_np = A_hat_final.cpu().numpy()
A_recovered_np = A_recovered.cpu().numpy().astype(np.int64)

# 无向图：边数 = 对称矩阵非零元素数的一半
predicted_edges = int(A_recovered.sum().item() / 2)


# ==========
# 输出结果
# ==========
print()
print("最终邻接概率矩阵 A_hat_final（对称、确定性）：")
print(A_hat_final)
print("恢复后的0/1邻接矩阵 A_recovered（对称）：")
print(A_recovered)
print(f"预测边数 = {predicted_edges}")


# =========================
# 只取邻接矩阵严格上三角部分并计算评价指标
# =========================
upper_indices = np.triu_indices(n, k=1)

print()
print(f"严格上三角索引数量: {len(upper_indices[0])}")

roc_auc, pr_auc, precision, recall, f1, accuracy, shd = evaluate_structure(
    edge_prob_final, A_true_np, args.threshold, tri_i, tri_j, n
)


# =========================
# 输出评价指标
# =========================
print()
print("==============================")
print("网络结构恢复评价结果")
print("==============================")
print(f"ROC-AUC = {roc_auc:.6f}")
print(f"PR-AUC = {pr_auc:.6f}")
print(f"Precision = {precision:.6f}")
print(f"Recall = {recall:.6f}")
print(f"F1 = {f1:.6f}")
print(f"Accuracy = {accuracy:.6f}")
print(f"SHD = {shd}")
