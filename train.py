"""共享无向三元超边参数的 AIDD 逐节点训练，SIR 状态保留 0/1/2。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import swanlab
import torch
import torch.nn.functional as F
import config
import swanlab_config
from data.dataset import load_data, make_loader, split_trajectories

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(config.DATA_DIR))
    parser.add_argument("--network_name", default=None)
    parser.add_argument(
        "--dynamics_name", default=config.DYNAMICS_NAME, choices=("SIR",),
        help="动力学名称；当前训练入口支持 SIR",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--device", default=config.DEVICE,
        help="auto 自动选择；或指定 cpu、cuda、cuda:0、cuda:1 等设备",
    )
    parser.add_argument("--train_ratio", type=float, default=config.TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=config.VAL_RATIO)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--F_dim", type=int, default=config.FEATURE_DIM)
    parser.add_argument("--pair_chunk_size", type=int, default=config.PAIR_CHUNK_SIZE)
    parser.add_argument("--lr_dyn", type=float, default=config.LR_DYN)
    parser.add_argument("--lr_structure", type=float, default=config.LR_STRUCTURE)
    parser.add_argument("--lambda_sparse", type=float, default=config.LAMBDA_SPARSE)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--init_edge_prob", type=float, default=config.INIT_EDGE_PROB)
    parser.add_argument("--init_std", type=float, default=config.INIT_STD)
    parser.add_argument("--gumbel_hard", action="store_true", default=config.GUMBEL_HARD)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    if args.network_name is None:
        args.network_name = Path(args.data_dir).name
    if args.output_dir is None:
        args.output_dir = str(
            config.OUTPUT_ROOT / args.network_name
            / f"{args.dynamics_name}_train_seed{args.seed}"
        )
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1
            and args.train_ratio + args.val_ratio < 1):
        parser.error("train_ratio、val_ratio 必须为正，且两者之和小于 1")
    if min(args.epochs, args.batch_size, args.F_dim, args.pair_chunk_size) < 1:
        parser.error("epochs、batch_size、F_dim、pair_chunk_size 必须为正整数")
    if args.lambda_sparse < 0:
        parser.error("lambda_sparse 必须非负")
    if not 0 < args.init_edge_prob < 1:
        parser.error("init_edge_prob 必须在 (0,1) 内")
    if args.temperature <= 0 or args.init_std < 0:
        parser.error("temperature 必须为正，init_std 必须非负")
    return args


@torch.no_grad()
def evaluate(generator, dynamics, loader, device, *, hard=False, fixed_rows=None, seed=42):
    """逐节点预测共享采样超图；fixed_rows 用于固定真实结构或导出的二值超图。"""
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
            sampled_rows = (
                generator.sample_adj_matrix(hard=hard)
                if fixed_rows is None else fixed_rows
            )
            for target_i in range(dynamics.N):
                adj_row = sampled_rows[target_i]
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
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        # 同时设置当前 CUDA 设备，使默认 CUDA 操作也使用指定的卡。
        torch.cuda.set_device(device.index if device.index is not None else 0)
        device = torch.device("cuda", torch.cuda.current_device())
    x, y, sample_index, b_true = load_data(
        args.data_dir, args.network_name, args.dynamics_name
    )
    incidence_to_triplets(b_true)
    rows, groups = split_trajectories(
        sample_index, args.train_ratio, args.val_ratio, args.seed
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = x.shape[1]
    run = swanlab.init(
        project=swanlab_config.PROJECT,
        workspace=swanlab_config.WORKSPACE,
        experiment_name=swanlab_config.EXPERIMENT_NAME or (
            f"{args.network_name}_{args.dynamics_name}_train_seed{args.seed}"
        ),
        mode=swanlab_config.MODE,
        logdir=str(output_dir / swanlab_config.LOG_DIR_NAME),
        config={
            **vars(args),
            "resolved_device": str(device),
            "N": n,
            "E": b_true.shape[1],
            "num_samples": len(x),
            "num_pairs_per_node": (n - 1) * (n - 2) // 2,
            "num_hyperedge_candidates": n * (n - 1) * (n - 2) // 6,
            "structure_mode": "shared_undirected_triplets",
            "structure_optimizer": "SparseAdam",
            "evaluation_sampling": "one_shared_hypergraph_per_batch",
            "train_trajectories": len(groups["train"]),
            "val_trajectories": len(groups["val"]),
            "test_trajectories": len(groups["test"]),
            "train_samples": len(rows["train"]),
            "val_samples": len(rows["val"]),
            "test_samples": len(rows["test"]),
            "state_encoding": {"S": 0, "I": 1, "R": 2},
            "structure_rules": ["top_e", "threshold_0_5", "threshold_0_1"],
        },
    )
    for prefix in ("train", "val", "structure"):
        swanlab.define_metric(f"{prefix}/*", x_axis="epoch")
    swanlab.define_metric("best/*", x_axis="best_epoch", hidden=True)
    swanlab.define_metric("best/summary", hidden=False)
    swanlab.define_metric("epoch", hidden=True)
    swanlab.define_metric("best_epoch", hidden=True)

    # 在记录器初始化之后设置训练随机种子。
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
    # 先初始化动力学网络，固定结构对照可用同一个 seed 得到相同的初始权重。
    dynamics = AIDDNeighborPairDynamics(
        n, feature_dim=args.F_dim, pair_chunk_size=args.pair_chunk_size
    ).to(device)
    generator = GumbelNeighborPairGenerator(
        n, temperature=args.temperature,
        init_edge_prob=args.init_edge_prob, init_std=args.init_std
    ).to(device)
    optimizer_dyn = torch.optim.Adam(dynamics.parameters(), lr=args.lr_dyn)
    optimizer_structure = torch.optim.SparseAdam(
        generator.parameters(), lr=args.lr_structure
    )
    neighbor_pairs = dynamics.neighbor_pairs.detach().cpu().numpy()

    print(f"script: {Path(__file__).resolve()}")
    print(f"data: {Path(args.data_dir).resolve()}")
    print(f"output: {output_dir.resolve()}")
    print(f"network: {args.network_name}; dynamics: {args.dynamics_name}")
    print(f"device: {device}; neighbor matrix={n} x {generator.num_pairs}")
    print("states: 0=S, 1=I, 2=R; NN1 input=[x_j(t), x_k(t), x_i(t)]")
    print("structure: shared undirected triplets; class 0=edge, class 1=no-edge")
    print(f"unique structure candidates: {generator.num_hyperedges}")
    print("structure evaluation: Top-E, threshold=0.5, threshold=0.1; primary export=0.5")
    print(
        f"AIDD settings: batch={args.batch_size}; temperature={args.temperature}; "
        f"hard={args.gumbel_hard}; base_edge_probability={args.init_edge_prob}; "
        f"logit_noise_std={args.init_std}"
    )
    print(f"base expected row degree: {generator.num_pairs * args.init_edge_prob:.4f}")
    print(f"prediction updates per target row per epoch: {len(train_loader)}")
    print("updates: sample target row -> CE + sparsity -> update incident shared edges and dynamics")
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
            # 逐节点反传；三个成员的损失共同更新同一份无向超边参数。
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

        val_ce, val_true, val_pred, val_prob = evaluate(
            generator, dynamics, val_loader, device,
            hard=args.gumbel_hard, seed=args.seed,
        )
        val_metrics = compute_sir_metrics(
            x[rows["val"]], val_true, val_pred, val_prob, val_ce
        )
        with torch.no_grad():
            row_probability = generator.edge_probabilities()
            expected_row_entries = row_probability.sum().item()
            recovered_row_entries = int((row_probability >= 0.5).sum().item())
        epoch_structure = compute_neighbor_pair_structure_metrics(
            b_true, neighbor_pairs, row_probability.cpu().numpy()
        )
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
                "swanlab_run_id": run.id,
                "architecture": "aidd_neighbor_pairs",
                "structure_mode": "shared_undirected_triplets",
                "training_objective": "node_CE + lambda_sparse * expected_row_degree",
                "structure_parameters": "one_logit_pair_per_undirected_triplet",
                "structure_optimizer": "SparseAdam",
                "evaluation_sampling": "one_shared_hypergraph_per_batch",
                "args": vars(args).copy(),
                "N": n, "num_pairs": generator.num_pairs,
                "num_hyperedge_candidates": generator.num_hyperedges,
                "F_dim": args.F_dim, "num_classes": 3,
                "temperature": generator.temperature,
                "best_epoch": epoch + 1, "val_CE": val_ce,
            }
            torch.save(best_checkpoint, checkpoint_path)

        # 一个 epoch 汇总一次；真实结构指标只作监控，选模仍只依据 val_CE。
        epoch_metrics = {
            "epoch": epoch + 1,
            "train/CE": train_ce,
            "train/sparsity": train_sparse,
            "train/total_loss": train_ce + train_sparse,
            **{f"val/{name}": value for name, value in val_metrics.items()},
            "val/best_CE": best_val_ce,
            "structure/ROC_AUC": epoch_structure["edge_metrics"]["ROC_AUC"],
            "structure/PR_AUC": epoch_structure["edge_metrics"]["PR_AUC"],
            "structure/Expected_edges": epoch_structure["edge_metrics"]["Expected_edges"],
            "structure/expected_row_entries": expected_row_entries,
            "structure/recovered_row_entries_0_5": recovered_row_entries,
            "structure/HD_PHD_Pearson": epoch_structure["Pearson"]["HD-PHD"],
            "structure/Triplet_view_disagreement": epoch_structure["row_metrics"]["Direction_disagreement"],
        }
        for rule, result in epoch_structure["reconstructions"].items():
            for name in ("Precision", "Recall", "F1", "Recovered_edges"):
                epoch_metrics[f"structure/{rule}/{name}"] = result["metrics"][name]
        # 常量序列的 Pearson、单类别的 AUC 可能无定义，不将其写成 0。
        run.log(
            {name: value for name, value in epoch_metrics.items() if np.isfinite(value)},
            step=epoch + 1,
        )
        del epoch_structure

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
    pearson_matrix = structure["Pearson_matrix"]

    # 行矩阵是共享无向超边参数的展开，同一超边的三个位置取值一致。
    np.savez_compressed(
        output_dir / "neighbor_pair_matrix.npz",
        neighbor_pairs=neighbor_pairs.astype(np.int32),
        probabilities=row_probability,
        row_thresholded=(row_probability >= 0.5).astype(np.uint8),
        row_threshold=np.float32(0.5),
    )
    # 保留原导出布局；同一超边的三个节点视角现在具有相同概率。
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
    print("MSE/MAE use state codes: 0=S, 1=I, 2=R.")
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

    # 恢复最佳权重后记录一次；保留对比用数值，并用文本汇总最终结果。
    final_metrics = {
        "best_epoch": best_checkpoint["best_epoch"],
        "best/val_CE": best_val_ce,
        **{f"best/test/sampled_rows/{name}": value for name, value in test_metrics.items()},
        **{f"best/structure/{name}": value for name, value in structure["edge_metrics"].items()},
        "best/structure/Triplet_view_disagreement": structure["row_metrics"]["Direction_disagreement"],
        **{f"best/Pearson/{name}": value for name, value in pearson.items()},
    }
    for rule, values in recovered_metrics_by_rule.items():
        for name, value in values.items():
            final_metrics[f"best/test/{rule}/{name}"] = value
    for rule, result in structure["reconstructions"].items():
        for name, value in result["metrics"].items():
            final_metrics[f"best/structure/{rule}/{name}"] = value
    summary_lines = [
        f"Network: {args.network_name}",
        f"Dynamics: {args.dynamics_name}",
        f"Best epoch: {best_checkpoint['best_epoch']}",
    ]
    summary_sections = {}
    for name, value in final_metrics.items():
        if not name.startswith("best/"):
            continue
        section, _, metric_name = name[len("best/"):].rpartition("/")
        value_text = f"{value:.8g}" if np.isfinite(value) else "undefined"
        summary_sections.setdefault(section or "selection", []).append(
            f"{metric_name:28s} = {value_text}"
        )
    for section, lines in summary_sections.items():
        summary_lines.extend(["", f"[{section}]", *lines])
    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "best_summary.txt").write_text(summary_text, encoding="utf-8")

    final_payload = {
        name: value for name, value in final_metrics.items() if np.isfinite(value)
    }
    final_payload["best/summary"] = swanlab.Text(
        summary_text, caption=f"Best epoch {best_checkpoint['best_epoch']}"
    )
    run.log(final_payload, step=args.epochs + 1)
    swanlab.finish()


if __name__ == "__main__":
    main()
