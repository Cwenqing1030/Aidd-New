import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# AIDD 三层网络：保留原始主体
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
# 读取超图数据：真实结构为 B_true[N,E]
# ============================================================
def read_hyperedge_list(path: Path):
    """每行一条超边，节点编号以空格分隔。支持不同大小的超边。"""
    hyperedges = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                nodes = [int(v) for v in line.split()]
            except ValueError as exc:
                raise ValueError(
                    f"{path} 第 {line_no} 行包含非整数节点编号: {line}"
                ) from exc
            if len(nodes) != len(set(nodes)):
                raise ValueError(
                    f"{path} 第 {line_no} 行同一超边中存在重复节点: {nodes}"
                )
            hyperedges.append(nodes)

    if not hyperedges:
        raise ValueError(f"超图文件为空: {path}")
    return hyperedges


def resolve_xy_paths(data_dir: Path, network_name: str, dynamics_type: str):
    """兼容原 CoND 命名和 hypergraph_N200_x/y 命名。"""
    x_candidates = []
    y_candidates = []
    if dynamics_type:
        x_candidates.append(data_dir / f"{network_name}_{dynamics_type}_x.txt")
        y_candidates.append(data_dir / f"{network_name}_{dynamics_type}_y.txt")
    x_candidates.append(data_dir / f"{network_name}_x.txt")
    y_candidates.append(data_dir / f"{network_name}_y.txt")

    x_path = next((p for p in x_candidates if p.exists()), None)
    y_path = next((p for p in y_candidates if p.exists()), None)
    if x_path is None or y_path is None:
        tried = "\n".join(str(p) for p in x_candidates + y_candidates)
        raise FileNotFoundError(f"找不到 X/Y 文件，尝试过:\n{tried}")
    return x_path, y_path


def load_hypergraph_data(data_dir, network_name=None, dynamics_type=""):
    data_dir = Path(data_dir)
    if network_name is None:
        network_name = data_dir.name

    topology_path = data_dir / f"{network_name}.txt"
    if not topology_path.exists():
        raise FileNotFoundError(f"找不到超图结构文件: {topology_path}")

    x_path, y_path = resolve_xy_paths(data_dir, network_name, dynamics_type)

    Xt = np.atleast_2d(np.loadtxt(x_path, dtype=np.float32))
    Yt = np.atleast_2d(np.loadtxt(y_path, dtype=np.float32))
    if Xt.shape != Yt.shape:
        raise ValueError(f"X/Y shape 不一致: X={Xt.shape}, Y={Yt.shape}")

    N = Xt.shape[1]
    hyperedges = read_hyperedge_list(topology_path)
    E = len(hyperedges)

    # 自动识别 0-based / 1-based 编号
    all_nodes = [node for edge in hyperedges for node in edge]
    min_node, max_node = min(all_nodes), max(all_nodes)
    if min_node >= 1 and max_node == N:
        hyperedges = [[v - 1 for v in edge] for edge in hyperedges]
        all_nodes = [node for edge in hyperedges for node in edge]
        min_node, max_node = min(all_nodes), max(all_nodes)

    if min_node < 0 or max_node >= N:
        raise ValueError(
            f"超图节点编号应在 [0,{N-1}] 或 [1,{N}]，实际范围 [{min_node},{max_node}]"
        )

    # B_true[i,e]=1 表示节点 i 属于第 e 条超边
    B_true = np.zeros((N, E), dtype=np.int64)
    for e, nodes in enumerate(hyperedges):
        B_true[nodes, e] = 1

    hyperedge_sizes = B_true.sum(axis=0)

    return Xt, Yt, B_true, hyperedge_sizes, topology_path, x_path, y_path


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


# ============================================================
# TaHiP 论文评价指标
# 1) 动力学预测：MSE / MAE（论文 Eq.18 / Eq.19）
# 2) 结构相关：
#    HD  = true hyperdegree = sum_e B_true[i,e]
#    PHD = predicted hyperdegree = sum_e B_hat[i,e]（论文 Eq.22）
#    HOH = Higher-Order H-index
#    HOP = Higher-Order PageRank
#    对 HD/PHD/HOH/HOP 两两计算 Pearson 相关系数
#
# 注意：
# - 当前 AIDD+Gumbel 版本的可学习代理结构是 A_prob∈[0,1]，
#   因此用 A_prob 作为本实现中的 surrogate matrix B_hat 来计算 PHD。
# - 下面构造 NxN 的 AH 只用于最终结构指标 HOH/HOP 的离线评价，
#   不参与模型前向传播，也不会把模型结构重新改回 NxN。
# ============================================================
def safe_pearson(x, y, eps=1e-12):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"Pearson 输入长度不一致: {x.size} vs {y.size}")
    x_std = x.std()
    y_std = y.std()
    if x_std < eps or y_std < eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_prediction_metrics(y_true, y_pred):
    """
    TaHiP 论文 Eq.(18)(19):
      MSE = mean((Y_true - Y_pred)^2)
      MAE = mean(abs(Y_true - Y_pred))
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"预测评价 shape 不一致: true={y_true.shape}, pred={y_pred.shape}"
        )
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return mse, mae


def hypergraph_adjacency_for_metrics(B_true_np):
    """
    论文 Appendix A:
      A_H = B B^T - Lambda_H
    其中 Lambda_H = diag(hyperdegree)。

    这里只用于 HOH/HOP 评价，不进入模型训练。
    """
    B = np.asarray(B_true_np, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError(f"B_true 应为二维 [N,E]，实际 {B.shape}")
    hd = B.sum(axis=1)
    A_H = B @ B.T
    A_H[np.diag_indices_from(A_H)] -= hd
    # 数值安全
    A_H[np.abs(A_H) < 1e-12] = 0.0
    return A_H


def compute_hd(B_true_np):
    """HD: 真实超度 d_H(i)=sum_e B[i,e]."""
    B = np.asarray(B_true_np, dtype=np.float64)
    return B.sum(axis=1)


def compute_phd(B_hat_np):
    """
    PHD: 论文 Eq.(22)
      d_hat_H(i) = sum_e B_hat[i,e]
    """
    B_hat = np.asarray(B_hat_np, dtype=np.float64)
    return B_hat.sum(axis=1)


def compute_hoh(B_true_np):
    """
    HOH: Higher-Order H-index
    节点 i 的邻居定义为：与 i 至少共享一条超边的节点。
    HOH(i) 是最大的 H，使得 i 至少有 H 个邻居的 hyperdegree >= H。
    """
    B = np.asarray(B_true_np, dtype=np.float64)
    N, E = B.shape
    hd = compute_hd(B)

    # 只作为结构评价构造共享超边关系
    A_H = hypergraph_adjacency_for_metrics(B)
    neighbor_mask = A_H > 0

    hoh = np.zeros(N, dtype=np.float64)
    for i in range(N):
        neighbor_degrees = hd[neighbor_mask[i]]
        if neighbor_degrees.size == 0:
            hoh[i] = 0.0
            continue
        vals = np.sort(neighbor_degrees)[::-1]
        ranks = np.arange(1, vals.size + 1)
        valid = ranks[vals >= ranks]
        hoh[i] = float(valid.max()) if valid.size else 0.0
    return hoh


def compute_hop(
    B_true_np,
    damping=0.85,
    tol=1e-12,
    max_iter=10000,
):
    """
    HOP: Higher-Order PageRank，依据论文 Definition 6。
    先由 incidence matrix 得到 A_H，再构造转移矩阵 P；
    使用带 damping 的 PageRank 迭代求稳态。

    damping 在论文中记为 s∈(0,1)；这里默认 0.85，并提供命令行参数修改。
    """
    if not (0.0 < damping < 1.0):
        raise ValueError(f"HOP damping 必须在 (0,1)，当前 {damping}")

    A_H = hypergraph_adjacency_for_metrics(B_true_np)
    N = A_H.shape[0]

    row_sum = A_H.sum(axis=1)
    P = np.zeros_like(A_H, dtype=np.float64)

    nonzero = row_sum > 0
    P[nonzero] = A_H[nonzero] / row_sum[nonzero, None]

    # dangling node: 用均匀转移补齐，避免概率质量丢失
    P[~nonzero] = 1.0 / N

    pr = np.full(N, 1.0 / N, dtype=np.float64)
    teleport = np.full(N, (1.0 - damping) / N, dtype=np.float64)

    for _ in range(max_iter):
        new_pr = damping * (P.T @ pr) + teleport
        if np.linalg.norm(new_pr - pr, ord=1) < tol:
            pr = new_pr
            break
        pr = new_pr

    s = pr.sum()
    if s > 0:
        pr = pr / s
    return pr


def compute_tahip_structure_metrics(
    B_true_np,
    B_hat_np,
    hop_damping=0.85,
    hop_tol=1e-12,
    hop_max_iter=10000,
):
    """
    计算论文 Section IV / Fig.6 对应的：
      HD, PHD, HOH, HOP
    以及四者两两 Pearson 相关系数。
    """
    B_true_np = np.asarray(B_true_np)
    B_hat_np = np.asarray(B_hat_np)

    if B_true_np.shape != B_hat_np.shape:
        raise ValueError(
            f"B shape 不一致: true={B_true_np.shape}, surrogate={B_hat_np.shape}"
        )

    hd = compute_hd(B_true_np)
    phd = compute_phd(B_hat_np)
    hoh = compute_hoh(B_true_np)
    hop = compute_hop(
        B_true_np,
        damping=hop_damping,
        tol=hop_tol,
        max_iter=hop_max_iter,
    )

    metrics = {
        "HD": hd,
        "PHD": phd,
        "HOH": hoh,
        "HOP": hop,
    }

    names = ["HD", "PHD", "HOH", "HOP"]
    pearson = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pearson[f"{a}-{b}"] = safe_pearson(metrics[a], metrics[b])

    return metrics, pearson


# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/cwq/AiddNew/generated_data/SI_dataset/")
parser.add_argument("--network_name", default="hypergraph_N200")
parser.add_argument("--dynamics_type", nargs="?", const="SI", default="SI")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lambda_sparse", type=float, default=0.01)
parser.add_argument("--init_A_prob", "--init_B_prob", dest="init_A_prob", type=float, default=0.02)
parser.add_argument("--lr_dyn", type=float, default=0.001)
parser.add_argument("--lr_B", type=float, default=0.004)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--tau_start", type=float, default=1.0)
parser.add_argument("--tau_end", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--F_dim", type=int, default=128)
parser.add_argument(
    "--task",
    choices=["auto", "classification", "regression"],
    default="auto",
)
parser.add_argument("--check_only", action="store_true", default=False)
parser.add_argument("--hop_damping", type=float, default=0.85)
parser.add_argument("--hop_tol", type=float, default=1e-12)
parser.add_argument("--hop_max_iter", type=int, default=10000)
parser.add_argument("--output_dir", default="outputs_NxE_SI")
args = parser.parse_args()
if not args.dynamics_type:
    args.dynamics_type = "SI"
if args.network_name:
    dataset_dir = Path(args.data_dir) / args.network_name
    if dataset_dir.is_dir():
        args.data_dir = str(dataset_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# ============================================================
# 数据与基础维度
# ============================================================
Xt, Yt, B_true, hyperedge_sizes, topology_path, x_path, y_path = (
    load_hypergraph_data(
        args.data_dir,
        network_name=args.network_name,
        dynamics_type=args.dynamics_type,
    )
)

T, N = Xt.shape
E = B_true.shape[1]

# 自动判断离散分类 / 连续回归
if args.task == "auto":
    x_int = np.allclose(Xt, np.rint(Xt), atol=1e-6)
    y_int = np.allclose(Yt, np.rint(Yt), atol=1e-6)
    if x_int and y_int:
        unique_values = np.unique(
            np.concatenate((Xt.reshape(-1), Yt.reshape(-1)))
        )
        task = "classification" if unique_values.size <= 20 else "regression"
    else:
        task = "regression"
else:
    task = args.task

if task != "classification":
    raise ValueError("当前SI任务为二分类")
if not np.all(np.isin(np.unique(np.concatenate((Xt.ravel(), Yt.ravel()))), [0, 1])):
    raise ValueError("SI状态只能为0或1")
init_A_prob = float(args.init_A_prob)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 76)
print("AiddNew NxE hypergraph incidence learning")
print(f"device                : {device}")
print(f"X shape               : {Xt.shape} = [T,N]")
print(f"Y shape               : {Yt.shape} = [T,N]")
print(f"B_true shape          : {B_true.shape} = [N,E]")
print(f"N                     : {N}")
print(f"E                     : {E}")
print(f"task                  : {task}")
print(f"init_A_prob           : {init_A_prob:.6f}")
print("=" * 76)


# ============================================================
# 模型
# ============================================================
F_dim = args.F_dim
nn1 = NN1(input_dim=2, hidden_dim=F_dim).to(device)
fusion_mlp = HyperedgeFusionMLP(num_hyperedges=E, feature_dim=F_dim).to(device)
nn2 = NN2(input_dim=F_dim, hidden_dim=F_dim, output_dim=F_dim).to(device)

nn3_output_dim = 2
criterion = nn.CrossEntropyLoss()

nn3 = NN3(input_dim=F_dim + 1, output_dim=nn3_output_dim).to(device)

# A_logits 不使用 B_true 的结构或密度信息。
A_logits = init_A_logits(N, E, init_A_prob, device)

with torch.no_grad():
    A_prob_init = torch.softmax(A_logits, dim=-1)[..., 1]
print(
    f"A_logits init         : {tuple(A_logits.shape)} = [N,E,2]\n"
    f"A_prob init           : {tuple(A_prob_init.shape)} = [N,E], "
    f"mean={A_prob_init.mean().item():.6f}"
)

optimizer = torch.optim.Adam(
    [
        {
            "params": list(nn1.parameters())
            + list(fusion_mlp.parameters())
            + list(nn2.parameters())
            + list(nn3.parameters()),
            "lr": args.lr_dyn,
        },
        {"params": [A_logits], "lr": args.lr_B},
    ]
)

if args.tau_start <= 0 or args.tau_end <= 0:
    raise ValueError("tau_start / tau_end 必须为正数")

Xt_tensor = torch.from_numpy(Xt).to(device=device, dtype=torch.float32)
Yt_tensor = torch.from_numpy(np.rint(Yt).astype(np.int64)).to(device)


# ============================================================
# 统一前向
# ============================================================
def forward_with_incidence(X_batch, A_weight, shape_trace=None):
    """
    训练时 A_weight=A_sampled；最终动力学评价时使用确定性的 A_prob_final。
    每次只构造一个目标节点的 [batch,E,F]，不创建 [batch,N,E,F]。
    """
    batch_size = X_batch.shape[0]

    predictions = []
    for target_i in range(N):
        trace_i = shape_trace if target_i == 0 else None
        all_edge_features = aggregate_hyperedge_messages(
            nn1, X_batch, A_weight, target_i, shape_trace=trace_i
        )  # [batch,E,F]
        concat_edge_features = all_edge_features.reshape(batch_size, E * F_dim)
        edge_fused_i = fusion_mlp(concat_edge_features)
        h2_i = nn2(edge_fused_i)
        nn3_input = torch.cat(
            (h2_i, X_batch[:, target_i].unsqueeze(-1)), dim=-1
        )
        prediction_i = nn3(nn3_input)

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


def forward_batch(X_batch, temperature, shape_trace=None):
    A_sampled, A_prob, _ = sample_gumbel_incidence_rowwise(
        A_logits, temperature
    )

    prediction = forward_with_incidence(X_batch, A_sampled, shape_trace=shape_trace)
    return prediction, A_sampled, A_prob


# ============================================================
# 正式训练前：先跑一小批做维度检查
# ============================================================
check_bs = min(args.batch_size, T)
X_check = Xt_tensor[:check_bs]
Y_check = Yt_tensor[:check_bs]

with torch.no_grad():
    shape_trace = {}
    pred_check, A_sampled_check, A_prob_check = forward_batch(
        X_check, args.tau_start, shape_trace=shape_trace
    )

print("\n[训练前维度检查]")
expected_shapes = {
    "X_batch": (check_bs, N),
    "B_true": (N, E),
    "A_logits": (N, E, 2),
    "A_prob": (N, E),
    "A_sampled": (N, E),
    "pair_input_i": (check_bs, N, 2),
    "h1_i": (check_bs, N, F_dim),
    "single_edge_column": (N,),
    "single_edge_feature": (check_bs, F_dim),
    "all_edge_features": (check_bs, E, F_dim),
    "concat_edge_features": (check_bs, E * F_dim),
    "fusion_MLP_output": (check_bs, F_dim),
    "NN2_output": (check_bs, F_dim),
    "NN3_input": (check_bs, F_dim + 1),
    "prediction_i": (check_bs, 2),
    "prediction": (check_bs, N, 2),
    "CrossEntropy_input": (check_bs * N, 2),
    "Y_labels": (check_bs * N,),
}
actual_shapes = {
    "X_batch": tuple(X_check.shape),
    "B_true": tuple(B_true.shape),
    "A_logits": tuple(A_logits.shape),
    "A_prob": tuple(A_prob_check.shape),
    "A_sampled": tuple(A_sampled_check.shape),
    **shape_trace,
    "prediction": tuple(pred_check.shape),
    "CrossEntropy_input": tuple(pred_check.reshape(-1, 2).shape),
    "Y_labels": tuple(Y_check.reshape(-1).shape),
}
for name, expected in expected_shapes.items():
    actual = actual_shapes.get(name)
    print(f"{name:24s}: {actual}，期望 {expected}")
    assert actual == expected, f"{name} 维度错误: {actual} != {expected}"
assert torch.all((A_sampled_check == 0) | (A_sampled_check == 1))
print("=> 逐层维度和 hard 0/1 incidence 检查通过。")

if args.check_only:
    print("check_only=True：检查结束，不进入训练。")
    raise SystemExit(0)


# ============================================================
# 训练
# ============================================================
for epoch in range(args.epochs):
    progress = epoch / max(args.epochs - 1, 1)
    temperature = args.tau_start * (
        args.tau_end / args.tau_start
    ) ** progress

    epoch_pred = 0.0
    epoch_sparse = 0.0
    epoch_total = 0.0
    epoch_mae = 0.0
    epoch_A_sample_mean = 0.0
    sample_count = 0
    num_batches = 0

    for start in range(0, T, args.batch_size):
        X_batch = Xt_tensor[start : start + args.batch_size]
        Y_batch = Yt_tensor[start : start + args.batch_size]
        optimizer.zero_grad(set_to_none=True)

        prediction, A_sampled, A_prob = forward_batch(
            X_batch, temperature
        )

        prediction_loss = criterion(
            prediction.reshape(-1, 2), Y_batch.reshape(-1)
        )

        # 保留原有稀疏正则；真实 B_true 不参与训练目标。
        sparse_loss = A_prob.mean()
        total_loss = (
            prediction_loss + args.lambda_sparse * sparse_loss
        )

        # TaHiP 论文同时报告 MSE 和 MAE。
        batch_pred_value = prediction.argmax(dim=-1).to(torch.float32)
        batch_true_value = Y_batch.to(torch.float32)
        batch_mae = torch.mean(torch.abs(batch_pred_value - batch_true_value))

        total_loss.backward()
        optimizer.step()

        batch_sample_count = Y_batch.numel()
        epoch_pred += prediction_loss.detach().item() * batch_sample_count
        epoch_sparse += sparse_loss.detach().item() * batch_sample_count
        epoch_total += total_loss.detach().item() * batch_sample_count
        epoch_mae += batch_mae.detach().item() * batch_sample_count
        epoch_A_sample_mean += A_sampled.detach().mean().item()
        sample_count += batch_sample_count
        num_batches += 1

    with torch.no_grad():
        current_A_prob = torch.softmax(A_logits, dim=-1)[..., 1]
        mean_A_probability = current_A_prob.mean().item()
        thresholded_memberships = int(
            (current_A_prob >= args.threshold).sum().item()
        )

    mean_A_sample = epoch_A_sample_mean / max(num_batches, 1)

    print(
        f"Epoch {epoch + 1}/{args.epochs} | "
        f"temperature={temperature:.4f} | "
        f"CrossEntropy={epoch_pred / sample_count:.8e} | "
        f"MAE={epoch_mae / sample_count:.8e} | "
        f"sparse_loss={epoch_sparse / sample_count:.6f} | "
        f"total_loss={epoch_total / sample_count:.8e} | "
        f"mean_A_prob={mean_A_probability:.5f} | "
        f"mean_A_sample={mean_A_sample:.5f} | "
        f"memberships={thresholded_memberships}"
    )


# ============================================================
# 最终输出 + TaHiP 论文评价
# ============================================================
with torch.no_grad():
    A_prob_final = torch.softmax(A_logits, dim=-1)[..., 1]
    A_recovered = (A_prob_final >= args.threshold).to(torch.int64)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(
    output_dir / "B_prob_final.txt",
    A_prob_final.cpu().numpy(),
    fmt="%.8f",
)
np.savetxt(
    output_dir / "B_recovered.txt",
    A_recovered.cpu().numpy(),
    fmt="%d",
)

# ------------------------------------------------------------
# 1) TaHiP 动力学评价：MSE / MAE
# 最终评价不再做 Gumbel 随机采样，直接使用确定性的 A_prob_final。
# ------------------------------------------------------------
pred_all = []
with torch.no_grad():
    for start in range(0, T, args.batch_size):
        X_batch = Xt_tensor[start : start + args.batch_size]
        pred = forward_with_incidence(X_batch, A_prob_final)

        pred_value = pred.argmax(dim=-1).to(torch.float32)

        pred_all.append(pred_value.cpu())

Y_pred_all = torch.cat(pred_all, dim=0).numpy()
Y_true_eval = Yt_tensor.to(torch.float32).cpu().numpy()

tahip_mse, tahip_mae = compute_prediction_metrics(
    Y_true_eval,
    Y_pred_all,
)

np.savetxt(
    output_dir / "Y_pred_final.txt",
    Y_pred_all,
    fmt="%.10g",
)

# ------------------------------------------------------------
# 2) TaHiP 结构指标：HD / PHD / HOH / HOP + Pearson
# ------------------------------------------------------------
B_hat_np = A_prob_final.detach().cpu().numpy()
B_true_np = B_true.astype(np.int64)

structure_metrics, pearson = compute_tahip_structure_metrics(
    B_true_np,
    B_hat_np,
    hop_damping=args.hop_damping,
    hop_tol=args.hop_tol,
    hop_max_iter=args.hop_max_iter,
)

HD = structure_metrics["HD"]
PHD = structure_metrics["PHD"]
HOH = structure_metrics["HOH"]
HOP = structure_metrics["HOP"]

# 保存每个节点的 4 个结构指标
node_metrics = np.column_stack(
    [
        np.arange(N, dtype=np.int64),
        HD,
        PHD,
        HOH,
        HOP,
    ]
)
np.savetxt(
    output_dir / "TaHiP_node_metrics.txt",
    node_metrics,
    fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g"],
    header="node HD PHD HOH HOP",
    comments="",
)

# 保存 Pearson 相关矩阵，顺序与论文 Fig.6 一致
metric_names = ["HD", "PHD", "HOH", "HOP"]
metric_matrix = np.column_stack([HD, PHD, HOH, HOP])
pearson_matrix = np.corrcoef(metric_matrix, rowvar=False)

np.savetxt(
    output_dir / "TaHiP_pearson_matrix.txt",
    pearson_matrix,
    fmt="%.8f",
    header="rows/cols: HD PHD HOH HOP",
    comments="",
)

torch.save(
    {
        "nn1": nn1.state_dict(),
        "fusion_mlp": fusion_mlp.state_dict(),
        "nn2": nn2.state_dict(),
        "nn3": nn3.state_dict(),
        "A_logits": A_logits.detach().cpu(),
        "A_prob_final": A_prob_final.detach().cpu(),
        "N": N,
        "E": E,
        "F_dim": F_dim,
        "M": E * F_dim,
        "task": task,
        "TaHiP_MSE": tahip_mse,
        "TaHiP_MAE": tahip_mae,
        "Pearson": pearson,
    },
    output_dir / "model_NxE.pt",
)

print("\n" + "=" * 76)
print("训练结束：TaHiP 论文评价指标")
print("=" * 76)
print("[Dynamics Prediction]")
print(f"MSE = {tahip_mse:.10e}")
print(f"MAE = {tahip_mae:.10e}")

print("\n[Surrogate Matrix / Hypergraph Structure]")
print(f"A_prob_final shape = {tuple(A_prob_final.shape)} = [N,E]")
print(f"B_true shape       = {B_true_np.shape} = [N,E]")
print(f"HD  shape          = {HD.shape}")
print(f"PHD shape          = {PHD.shape}")
print(f"HOH shape          = {HOH.shape}")
print(f"HOP shape          = {HOP.shape}")

print("\n[Pearson correlations: HD / PHD / HOH / HOP]")
for key, value in pearson.items():
    print(f"{key:8s} = {value:.6f}")

print("\nPearson matrix order = [HD, PHD, HOH, HOP]")
print(pearson_matrix)

print("\n[saved]")
print(output_dir / "B_prob_final.txt")
print(output_dir / "B_recovered.txt")
print(output_dir / "Y_pred_final.txt")
print(output_dir / "TaHiP_node_metrics.txt")
print(output_dir / "TaHiP_pearson_matrix.txt")
print(output_dir / "model_NxE.pt")
print("=" * 76)
