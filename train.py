"""AIDD 逐节点训练：目标节点 × 候选邻居对，SIR 状态保留 0/1/2。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Models import AIDDNeighborPairDynamics, GumbelNeighborPairGenerator
from metrics import (
    compute_neighbor_pair_structure_metrics,
    compute_sir_infection_groups,
    compute_sir_metrics,
    incidence_to_triplets,
    print_sir_infection_groups,
    triplets_to_neighbor_rows,
)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(root / "data" / "hypergraph_N50"))
    parser.add_argument("--network_name", default="hypergraph_N50")
    parser.add_argument("--output_dir", default=str(root / "outputs_SIR_aidd_rows"))
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--F_dim", type=int, default=128)
    parser.add_argument("--pair_chunk_size", type=int, default=4096)
    parser.add_argument("--lr_dyn", type=float, default=0.001)
    parser.add_argument("--lr_structure", type=float, default=0.004)
    parser.add_argument("--lambda_sparse", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--init_edge_prob", type=float, default=0.001)
    parser.add_argument("--init_std", type=float, default=0.1)
    parser.add_argument("--gumbel_hard", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1
            and args.train_ratio + args.val_ratio < 1):
        parser.error("train_ratio、val_ratio 必须为正，且两者之和小于 1")
    if min(args.epochs, args.batch_size, args.F_dim, args.pair_chunk_size) < 1:
        parser.error("epochs、batch_size、F_dim、pair_chunk_size 必须为正整数")
    if args.lambda_sparse < 0:
        parser.error("lambda_sparse 必须非负")
    if not 0 < args.init_edge_prob < 1:
        parser.error("init_edge_prob 必须在 (0,1) 内")
    if args.temperature <= 0 or args.init_std <= 0:
        parser.error("temperature 和 init_std 必须为正")
    return args


def load_data(data_dir, network_name):
    data_dir = Path(data_dir)
    x = np.loadtxt(
        data_dir / f"{network_name}_SIR_x.txt", dtype=np.float32, ndmin=2
    )
    y = np.loadtxt(
        data_dir / f"{network_name}_SIR_y.txt", dtype=np.int64, ndmin=2
    )
    sample_index = np.loadtxt(
        data_dir / f"{network_name}_SIR_index.txt", dtype=np.int64, ndmin=2
    )
    hyperedges = np.loadtxt(
        data_dir / f"{network_name}.txt", dtype=np.int64, ndmin=2
    )
    if x.shape != y.shape or sample_index.shape != (len(x), 2):
        raise ValueError("X、Y 必须同形，index 必须有对应的轨迹编号和时间步两列")
    if not (np.isin(x, [0, 1, 2]).all() and np.isin(y, [0, 1, 2]).all()):
        raise ValueError("SIR 状态必须是 0=S、1=I、2=R")

    # 同一轮中相邻的状态对必须连续，跨轮重置不作为动力学转移。
    same_trajectory = sample_index[1:, 0] == sample_index[:-1, 0]
    if not np.array_equal(x[1:][same_trajectory], y[:-1][same_trajectory]):
        raise ValueError("轮内 X/Y 不连续，请检查同步的数据和 index 是否对应")

    n, e = x.shape[1], len(hyperedges)
    b_true = np.zeros((n, e), dtype=np.int64)
    b_true[hyperedges, np.arange(e)[:, None]] = 1
    return x, y, sample_index, b_true


def split_trajectories(sample_index, train_ratio, val_ratio, seed):
    trajectory_ids = np.unique(sample_index[:, 0])
    trajectory_ids = np.random.default_rng(seed).permutation(trajectory_ids)
    n_train = int(len(trajectory_ids) * train_ratio)
    n_val = int(len(trajectory_ids) * val_ratio)
    if min(n_train, n_val, len(trajectory_ids) - n_train - n_val) < 1:
        raise ValueError("轨迹数不足以划分出非空的训练、验证、测试集")
    groups = {
        "train": trajectory_ids[:n_train],
        "val": trajectory_ids[n_train:n_train + n_val],
        "test": trajectory_ids[n_train + n_val:],
    }
    rows = {
        name: np.flatnonzero(np.isin(sample_index[:, 0], ids))
        for name, ids in groups.items()
    }
    return rows, groups


def make_loader(x, y, rows, batch_size, shuffle=False, generator=None):
    dataset = TensorDataset(
        torch.from_numpy(x[rows]), torch.from_numpy(y[rows])
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


@torch.no_grad()
def evaluate(generator, dynamics, loader, device, *, hard=False, fixed_rows=None, seed=42):
    """沿用逐节点采样评价；fixed_rows 用于固定真实结构或导出的二值超图。"""
    dynamics.eval()
    if generator is not None:
        generator.eval()
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    true_batches, pred_batches, prob_batches = [], [], []
    total_ce = 0.0
    # 固定评价噪声，并恢复随机数状态，使评价不改变后续训练的随机采样。
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            probabilities = x_batch.new_empty(
                x_batch.shape[0], dynamics.N, dynamics.num_states
            )
            for target_i in range(dynamics.N):
                if fixed_rows is None:
                    adj_row = generator.sample_adj_row(target_i, hard=hard)
                else:
                    adj_row = fixed_rows[target_i]
                log_probability = dynamics(x_batch, adj_row, target_i)
                total_ce += F.nll_loss(
                    log_probability, y_batch[:, target_i], reduction="sum"
                ).item()
                probabilities[:, target_i] = log_probability.exp()
            true_batches.append(y_batch.cpu().numpy())
            pred_batches.append(probabilities.argmax(dim=-1).cpu().numpy())
            prob_batches.append(probabilities.cpu().numpy())
    y_true = np.concatenate(true_batches)
    return (
        total_ce / y_true.size, y_true,
        np.concatenate(pred_batches), np.concatenate(prob_batches),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, sample_index, b_true = load_data(args.data_dir, args.network_name)
    incidence_to_triplets(b_true)
    rows, groups = split_trajectories(
        sample_index, args.train_ratio, args.val_ratio, args.seed
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "split_indices.npz",
        train_rows=rows["train"], val_rows=rows["val"], test_rows=rows["test"],
        train_trajectories=groups["train"], val_trajectories=groups["val"],
        test_trajectories=groups["test"],
    )
    data_generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        x, y, rows["train"], args.batch_size, shuffle=True, generator=data_generator
    )
    val_loader = make_loader(x, y, rows["val"], args.batch_size)
    test_loader = make_loader(x, y, rows["test"], args.batch_size)
    n = x.shape[1]
    # 先初始化动力学网络，固定结构对照可用同一个 seed 得到相同的初始权重。
    dynamics = AIDDNeighborPairDynamics(
        n, feature_dim=args.F_dim, pair_chunk_size=args.pair_chunk_size
    ).to(device)
    generator = GumbelNeighborPairGenerator(
        n, temperature=args.temperature,
        init_edge_prob=args.init_edge_prob, init_std=args.init_std
    ).to(device)
    optimizer_dyn = torch.optim.Adam(dynamics.parameters(), lr=args.lr_dyn)
    optimizer_structure = torch.optim.Adam(generator.parameters(), lr=args.lr_structure)

    print(f"script: {Path(__file__).resolve()}")
    print(f"data: {Path(args.data_dir).resolve()}")
    print(f"output: {output_dir.resolve()}")
    print(f"device: {device}; neighbor matrix={n} x {generator.num_pairs}")
    print("states: 0=S, 1=I, 2=R; NN1 input=[x_j(t), x_k(t), x_i(t)]")
    print("structure: independent target rows; class 0=edge, class 1=no-edge")
    print("structure evaluation: Top-E, threshold=0.5, threshold=0.1; primary export=0.5")
    print(
        f"AIDD settings: batch={args.batch_size}; temperature={args.temperature}; "
        f"hard={args.gumbel_hard}; base_edge_probability={args.init_edge_prob}; "
        f"logit_noise_std={args.init_std}"
    )
    print(f"base expected row degree: {generator.num_pairs * args.init_edge_prob:.4f}")
    print(f"prediction updates per target row per epoch: {len(train_loader)}")
    print("updates: sample one row -> CE + sparsity -> backward -> update active row and dynamics")
    print(
        f"sparsity: {args.lambda_sparse} * expected degree of the active row; "
        "prediction and sparsity share one backward pass"
    )
    for name in ("train", "val", "test"):
        print(f"{name}: {len(groups[name])} trajectories, {len(rows[name])} X/Y pairs")

    best_val_ce = float("inf")
    best_checkpoint = None
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        csv.writer(history_file).writerow([
            "epoch", "train_CE", "train_sparse", "train_total_loss", "val_CE",
            "expected_row_entries", "recovered_row_entries", "updates_per_row",
        ])
    for epoch in range(args.epochs):
        generator.train()
        dynamics.train()
        ce_sum, degree_sum, sample_count = 0.0, 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # 每行的预测损失和期望超度惩罚一起反传，只更新当前行。
            for target_i in range(n):
                optimizer_structure.zero_grad(set_to_none=True)
                optimizer_dyn.zero_grad(set_to_none=True)
                adj_row = generator.sample_adj_row(target_i, hard=args.gumbel_hard)
                log_probability = dynamics(x_batch, adj_row, target_i)
                prediction_loss = F.nll_loss(log_probability, y_batch[:, target_i])
                expected_degree = generator.row_probabilities(target_i).sum()
                # 对所有目标取平均后，稀疏项对应平均期望超度，而非整张矩阵之和。
                sparse_loss = args.lambda_sparse * expected_degree
                loss = prediction_loss + sparse_loss
                loss.backward()
                optimizer_structure.step()
                optimizer_dyn.step()

                ce_sum += prediction_loss.detach().item() * x_batch.shape[0]
                degree_sum += expected_degree.detach().item() * x_batch.shape[0]
                sample_count += x_batch.shape[0]

        val_ce, _, _, _ = evaluate(
            generator, dynamics, val_loader, device,
            hard=args.gumbel_hard, seed=args.seed,
        )
        with torch.no_grad():
            row_probability = generator.edge_probabilities()
            expected_row_entries = row_probability.sum().item()
            recovered_row_entries = int((row_probability >= 0.5).sum().item())
        train_ce = ce_sum / sample_count
        train_sparse = args.lambda_sparse * degree_sum / sample_count
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            csv.writer(history_file).writerow([
                epoch + 1, train_ce, train_sparse, train_ce + train_sparse, val_ce,
                expected_row_entries, recovered_row_entries, len(train_loader),
            ])
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_CE={train_ce:.8f} | train_sparse={train_sparse:.8f} | "
            f"val_CE={val_ce:.8f} | expected_row_entries={expected_row_entries:.2f} | "
            f"recovered_row_entries={recovered_row_entries}"
        )
        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_checkpoint = {
                "generator_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in generator.state_dict().items()
                },
                "dynamics_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in dynamics.state_dict().items()
                },
                "architecture": "aidd_neighbor_pairs",
                "structure_mode": "independent_target_rows",
                "training_objective": "node_CE + lambda_sparse * expected_row_degree",
                "structure_parameters": "separate_parameter_per_target_row",
                "args": vars(args).copy(),
                "N": n, "num_pairs": generator.num_pairs,
                "F_dim": args.F_dim, "num_classes": 3,
                "temperature": generator.temperature,
                "best_epoch": epoch + 1, "val_CE": val_ce,
            }
            torch.save(best_checkpoint, checkpoint_path)

    if best_checkpoint is None:
        raise RuntimeError("未得到有效的验证结果，请检查训练损失")
    generator.load_state_dict(best_checkpoint["generator_state_dict"])
    generator.temperature = best_checkpoint["temperature"]
    dynamics.load_state_dict(best_checkpoint["dynamics_state_dict"])
    test_ce, y_true, y_pred, y_prob = evaluate(
        generator, dynamics, test_loader, device,
        hard=args.gumbel_hard, seed=args.seed,
    )
    test_metrics = compute_sir_metrics(
        x[rows["test"]], y_true, y_pred, y_prob, test_ce
    )
    neighbor_pairs = dynamics.neighbor_pairs.detach().cpu().numpy()
    row_probability = generator.edge_probabilities().detach().cpu().numpy()
    structure = compute_neighbor_pair_structure_metrics(
        b_true, neighbor_pairs, row_probability
    )
    rule_labels = {
        "top_e": "Top-E",
        "threshold_0_5": "0.5",
        "threshold_0_1": "0.1",
    }
    export_suffixes = {
        "top_e": "_top_e",
        "threshold_0_5": "",
        "threshold_0_1": "_threshold_0_1",
    }
    recovered_metrics_by_rule = {}
    structure_lines = ["method recovered TP FP FN Precision Recall F1"]
    for rule, result in structure["reconstructions"].items():
        recovered_hyperedges = structure["hyperedges"][result["recovered"]]
        recovered_rows = triplets_to_neighbor_rows(n, recovered_hyperedges)
        recovered_ce, recovered_true, recovered_pred, recovered_prob = evaluate(
            None, dynamics, test_loader, device,
            fixed_rows=torch.from_numpy(recovered_rows).to(device), seed=args.seed,
        )
        recovered_metrics_by_rule[rule] = compute_sir_metrics(
            x[rows["test"]], recovered_true, recovered_pred, recovered_prob, recovered_ce
        )
        suffix = export_suffixes[rule]
        np.savetxt(
            output_dir / f"recovered_hyperedges{suffix}.txt", recovered_hyperedges, fmt="%d"
        )
        np.save(output_dir / f"recovered_neighbor_matrix{suffix}.npy", recovered_rows)
        np.savetxt(
            output_dir / f"Y_pred_recovered{suffix}.txt", recovered_pred, fmt="%d"
        )
        np.save(output_dir / f"Y_prob_recovered{suffix}.npy", recovered_prob)
        values = result["metrics"]
        structure_lines.append(
            f"{rule_labels[rule]} {values['Recovered_edges']} {values['TP']} "
            f"{values['FP']} {values['FN']} {values['Precision']:.8f} "
            f"{values['Recall']:.8f} {values['F1']:.8f}"
        )
    infection_groups = compute_sir_infection_groups(
        x[rows["test"]], y_true, y_prob, b_true
    )
    node_metrics, pearson = structure["node_metrics"], structure["Pearson"]
    metric_names = ["HD", "PHD", "HOH", "HOP"]
    metric_matrix = np.column_stack([node_metrics[name] for name in metric_names])
    pearson_matrix = np.corrcoef(metric_matrix, rowvar=False)

    # 行矩阵原样保存。三个独立方向可能不同，不能把原始行阈值矩阵当作无向超图。
    np.savez_compressed(
        output_dir / "neighbor_pair_matrix.npz",
        neighbor_pairs=neighbor_pairs.astype(np.int32),
        probabilities=row_probability,
        row_thresholded=(row_probability >= 0.5).astype(np.uint8),
        row_threshold=np.float32(0.5),
    )
    # 无向超边分数取三个方向的概率均值，同时保留三个原始方向分数。
    np.savez_compressed(
        output_dir / "hyperedge_probabilities.npz",
        hyperedges=structure["hyperedges"].astype(np.int32),
        probabilities=structure["probabilities"].astype(np.float32),
        directional_probabilities=structure["directional_probabilities"].astype(np.float32),
    )
    np.savetxt(output_dir / "Y_true_test.txt", y_true, fmt="%d")
    np.savetxt(output_dir / "Y_pred_final.txt", y_pred, fmt="%d")
    np.save(output_dir / "Y_prob_test.npy", y_prob)
    np.savetxt(
        output_dir / "TaHiP_node_metrics.txt",
        np.column_stack([np.arange(n), metric_matrix]),
        fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g"],
        header="node HD PHD HOH HOP", comments="",
    )
    np.savetxt(
        output_dir / "row_expected_degrees.txt",
        np.column_stack([np.arange(n), structure["row_expected_degrees"]]),
        fmt=["%d", "%.10g"], header="node row_expected_degree", comments="",
    )
    np.savetxt(
        output_dir / "TaHiP_pearson_matrix.txt", pearson_matrix,
        fmt="%.8f", header="rows/cols: HD PHD HOH HOP", comments="",
    )
    np.savetxt(
        output_dir / "infection_condition_metrics.txt", infection_groups,
        fmt=["%d", "%d", "%.8f", "%.8f"],
        header="active_edges count actual_infection_rate mean_predicted_infection_prob",
        comments="",
    )
    (output_dir / "structure_threshold_metrics.txt").write_text(
        "\n".join(structure_lines) + "\n", encoding="utf-8"
    )
    best_checkpoint["test_metrics"] = test_metrics
    best_checkpoint["recovered_structure_rule"] = "threshold_0_5"
    best_checkpoint["recovered_structure_test_metrics"] = recovered_metrics_by_rule["threshold_0_5"]
    best_checkpoint["recovered_structure_test_metrics_by_rule"] = recovered_metrics_by_rule
    best_checkpoint["structure_metrics_by_rule"] = {
        rule: result["metrics"] for rule, result in structure["reconstructions"].items()
    }
    best_checkpoint["structure_metrics"] = structure["edge_metrics"]
    best_checkpoint["row_structure_metrics"] = structure["row_metrics"]
    best_checkpoint["Pearson"] = pearson
    torch.save(best_checkpoint, checkpoint_path)

    print(f"\nBest epoch: {best_checkpoint['best_epoch']}; val_CE={best_val_ce:.8f}")
    print("\n[Test dynamics | sampled target rows]")
    for name, value in test_metrics.items():
        print(f"{name:26s} = {value:.8f}")
    print("\n[Undirected triplet reconstruction | Top-E / 0.5 / 0.1]")
    print(
        f"E={structure['edge_metrics']['True_edges']}; "
        f"ROC_AUC={structure['edge_metrics']['ROC_AUC']:.8f}; "
        f"PR_AUC={structure['edge_metrics']['PR_AUC']:.8f}"
    )
    print("\n".join(structure_lines))
    for rule, values in recovered_metrics_by_rule.items():
        print(f"\n[Test dynamics | binary hypergraph | {rule_labels[rule]}]")
        for name, value in values.items():
            print(f"{name:26s} = {value:.8f}")
    print("\n[Structure indicators | mean / min / max | undirected mean probabilities]")
    for name in metric_names:
        values = node_metrics[name]
        print(f"{name:4s}: mean={values.mean():.6f}, min={values.min():.6f}, max={values.max():.6f}")
    print("\n[Pearson correlations]")
    for name, value in pearson.items():
        print(f"{name:8s} = {value:.6f}")
    print("\nPearson matrix order: HD PHD HOH HOP")
    print(pearson_matrix)
    print_sir_infection_groups(infection_groups)
    print(f"\nSaved results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
