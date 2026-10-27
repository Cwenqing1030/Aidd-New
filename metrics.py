"""动力学误差、节点结构指标和 Pearson 相关性。"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
    """SIR 的 MSE/MAE 按 0/1/2 状态编码计算，属于分类标签误差。"""
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
    return A_H


def compute_hd(B_true_np):
    """HD: 真实超度 d_H(i)=sum_e B[i,e]."""
    B = np.asarray(B_true_np, dtype=np.float64)
    return B.sum(axis=1)


def compute_hoh(B_true_np):
    """
    HOH: Higher-Order H-index
    节点 i 的邻居定义为：与 i 至少共享一条超边的节点。
    HOH(i) 是最大的 H，使得 i 至少有 H 个邻居的 hyperdegree >= H。
    """
    B = np.asarray(B_true_np, dtype=np.float64)
    N = B.shape[0]
    hd = compute_hd(B)

    # 只作为结构评价构造共享超边关系
    A_H = hypergraph_adjacency_for_metrics(B)
    neighbor_mask = A_H > 0

    hoh = np.zeros(N, dtype=np.float64)
    for i in range(N):
        neighbor_degrees = hd[neighbor_mask[i]]
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

    damping 默认取 0.85。
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


def compute_tahip_metrics_from_phd(
    B_true_np, predicted_hyperdegree,
    hop_damping=0.85, hop_tol=1e-12, hop_max_iter=10000,
):
    """由真实拓扑和预测超度计算节点指标；无需构造稠密候选关联矩阵。"""
    metrics = {
        "HD": compute_hd(B_true_np),
        "PHD": np.asarray(predicted_hyperdegree, dtype=np.float64),
        "HOH": compute_hoh(B_true_np),
        "HOP": compute_hop(
            B_true_np, damping=hop_damping, tol=hop_tol, max_iter=hop_max_iter
        ),
    }
    names = ["HD", "PHD", "HOH", "HOP"]
    pearson_matrix = np.empty((len(names), len(names)), dtype=np.float64)
    for i, a in enumerate(names):
        for j in range(i, len(names)):
            value = safe_pearson(metrics[a], metrics[names[j]])
            pearson_matrix[i, j] = pearson_matrix[j, i] = value
    pearson = {
        f"{a}-{b}": float(pearson_matrix[i, j])
        for i, a in enumerate(names) for j, b in enumerate(names) if i < j
    }
    return metrics, pearson, pearson_matrix


def incidence_to_triplets(incidence):
    """将每列恰好三个成员的关联矩阵转为 [E, 3] 超边列表。"""
    incidence = np.asarray(incidence)
    if not np.all(incidence.sum(axis=0) == 3):
        raise ValueError("当前模型要求每条超边恰好包含三个不同节点")
    return np.nonzero(incidence.T)[1].reshape(-1, 3)


def triplet_neighbor_row_indices(num_nodes, hyperedges):
    """输入 i<j<k 的三元组，返回三个 (目标行, 邻居对列) 位置。"""
    edges = np.asarray(hyperedges, dtype=np.int64).reshape(-1, 3)
    targets = edges
    first = edges[:, [1, 0, 0]]
    second = edges[:, [2, 2, 1]]
    # 每行排除了自己的节点：全局编号 > target 的邻居减 1，得到局部编号。
    first = first - (first > targets)
    second = second - (second > targets)
    m = num_nodes - 1
    columns = first * (2 * m - first - 1) // 2 + second - first - 1
    return targets, columns


def triplets_to_neighbor_rows(num_nodes, hyperedges):
    """构造固定无向超图的行矩阵，仅供真实结构对照和训练后评价使用。"""
    q = (num_nodes - 1) * (num_nodes - 2) // 2
    rows = np.zeros((num_nodes, q), dtype=np.float32)
    targets, columns = triplet_neighbor_row_indices(num_nodes, hyperedges)
    rows[targets, columns] = 1.0
    return rows


def _binary_structure_metrics(labels, selected):
    """一组选边结果对应的 TP/FP/FN、Precision、Recall 和 F1。"""
    true_count = int(labels.sum())
    selected_count = int(selected.sum())
    tp = int(np.count_nonzero(labels & selected))
    fp = selected_count - tp
    fn = true_count - tp
    return {
        "Recovered_edges": selected_count,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": tp / max(1, selected_count),
        "Recall": tp / max(1, true_count),
        "F1": 2 * tp / max(1, 2 * tp + fp + fn),
        "SHD": fp + fn,
        "True_edges": true_count,
    }


def compute_neighbor_pair_structure_metrics(b_true, neighbor_pairs, row_prob):
    """三方向概率取均值，只评价 Top-E、0.5 和 0.1 三种结构。"""
    threshold = 0.5
    n = b_true.shape[0]
    pairs = np.asarray(neighbor_pairs, dtype=np.int64)
    row_prob = np.asarray(row_prob, dtype=np.float64)

    # i 是三元组的最小编号时收集一次候选，避免重复计算同一条无向超边。
    candidates = []
    for target in range(n - 2):
        selected = pairs[target][pairs[target, :, 0] > target]
        candidates.append(np.column_stack((np.full(len(selected), target), selected)))
    candidates = np.concatenate(candidates)
    targets, columns = triplet_neighbor_row_indices(n, candidates)
    directional_prob = row_prob[targets, columns]
    probability = directional_prob.mean(axis=1)

    true_rows = triplets_to_neighbor_rows(n, incidence_to_triplets(b_true)).astype(bool)
    labels = true_rows[targets[:, 0], columns[:, 0]]
    recovered_rows = row_prob >= threshold
    tp = int(np.count_nonzero(true_rows & recovered_rows))
    fp = int(np.count_nonzero(~true_rows & recovered_rows))
    fn = int(np.count_nonzero(true_rows & ~recovered_rows))
    directional_binary = directional_prob >= threshold
    row_degrees = row_prob.sum(axis=1)
    row_metrics = {
        "ROC_AUC": float(roc_auc_score(true_rows.ravel(), row_prob.ravel())),
        "PR_AUC": float(average_precision_score(true_rows.ravel(), row_prob.ravel())),
        "Precision": tp / max(1, tp + fp),
        "Recall": tp / max(1, tp + fn),
        "F1": 2 * tp / max(1, 2 * tp + fp + fn),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "True_row_entries": int(true_rows.sum()),
        "Recovered_row_entries": int(recovered_rows.sum()),
        "Expected_row_entries": float(row_prob.sum()),
        "Direction_disagreement": float(np.mean(
            np.any(directional_binary != directional_binary[:, :1], axis=1)
        )),
        "HD_row_degree_Pearson": safe_pearson(compute_hd(b_true), row_degrees),
    }
    # Top-E 只使用真实超边数量作为预算，选边顺序由模型概率决定。
    top_e = np.zeros(len(probability), dtype=bool)
    top_e[np.argsort(-probability, kind="stable")[:int(labels.sum())]] = True
    selections = {
        "top_e": top_e,
        "threshold_0_5": probability >= 0.5,
        "threshold_0_1": probability >= 0.1,
    }
    reconstructions = {
        rule: {
            "recovered": selected,
            "metrics": _binary_structure_metrics(labels, selected),
        }
        for rule, selected in selections.items()
    }
    primary = reconstructions["threshold_0_5"]
    edge_metrics = {
        "ROC_AUC": float(roc_auc_score(labels, probability)),
        "PR_AUC": float(average_precision_score(labels, probability)),
        **primary["metrics"],
        "Expected_edges": float(probability.sum()),
    }

    phd = np.zeros(n, dtype=np.float64)
    for slot in range(3):
        np.add.at(phd, candidates[:, slot], probability)
    node_metrics, pearson, pearson_matrix = compute_tahip_metrics_from_phd(b_true, phd)
    return {
        "hyperedges": candidates,
        "probabilities": probability,
        "directional_probabilities": directional_prob,
        "recovered": primary["recovered"],
        "node_metrics": node_metrics,
        "Pearson": pearson,
        "Pearson_matrix": pearson_matrix,
        "edge_metrics": edge_metrics,
        "row_metrics": row_metrics,
        "row_expected_degrees": row_degrees,
        "reconstructions": reconstructions,
    }


def compute_sir_metrics(x, y_true, y_pred, y_prob, ce):
    susceptible = x == 0
    infection_true = y_true[susceptible] == 1
    infection_prob = y_prob[..., 1][susceptible]
    predicted_infection = y_pred[susceptible] == 1
    tp = int(np.count_nonzero(infection_true & predicted_infection))
    mse, mae = compute_prediction_metrics(y_true, y_pred)
    infection_auc = float("nan")
    if infection_true.any() and (~infection_true).any():
        infection_auc = float(roc_auc_score(infection_true, infection_prob))
    return {
        "CE": ce,
        "MSE": mse,
        "MAE": mae,
        "Accuracy": float(np.mean(y_true == y_pred)),
        "S_infection_Brier": float(np.mean((infection_prob - infection_true) ** 2)),
        "S_infection_AUC": infection_auc,
        "S_to_I_recall": tp / max(1, int(infection_true.sum())),
        "S_to_I_precision": tp / max(1, int(predicted_infection.sum())),
        "Persistence_accuracy": float(np.mean(x == y_true)),
    }


def compute_sir_infection_groups(x, y_true, y_prob, b_true):
    """按真实激活超边数分组；仅用于训练完成后的诊断。"""
    active_count = np.zeros_like(x, dtype=np.int64)
    infected = x == 1
    for nodes in incidence_to_triplets(b_true):
        for target in nodes:
            others = nodes[nodes != target]
            active_count[:, target] += np.all(infected[:, others], axis=1)
    susceptible = x == 0
    rows = []
    for count in np.unique(active_count[susceptible]):
        mask = susceptible & (active_count == count)
        rows.append([
            int(count), int(mask.sum()),
            float(np.mean(y_true[mask] == 1)),
            float(np.mean(y_prob[..., 1][mask])),
        ])
    return np.asarray(rows)


def print_sir_infection_groups(rows):
    print("\n[Susceptible nodes | activated true hyperedges]")
    print("active_edges  count  actual_infection_rate  mean_predicted_infection_prob")
    for count, samples, actual, predicted in rows:
        print(f"{int(count):12d}  {int(samples):5d}  {actual:21.8f}  {predicted:29.8f}")
