"""生成 Voter、Kuramoto、Spring 的三元超图 X/Y；默认每轮 5 步、1000 轮。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from data.extra_dynamics import CHANNELS, RULES, SIMULATORS
from data.hypergraphs import generate_hyperedges


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics_name", "--dynamics", choices=tuple(SIMULATORS), default="Voter")
    parser.add_argument("--network_type", choices=("ER", "SF", "WS"), default=config.NETWORK_TYPE)
    parser.add_argument("--N", type=int, default=config.N)
    parser.add_argument("--E", type=int, default=config.E)
    parser.add_argument("--alpha", type=float, default=config.SF_ALPHA)
    parser.add_argument("--rewire_prob", type=float, default=config.WS_REWIRE_PROB)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--data_root", type=Path, default=config.ROOT / "data")
    parser.add_argument("--trajectories", type=int, default=config.NUM_TRAJECTORIES)
    parser.add_argument("--steps", type=int, default=config.STEPS_PER_TRAJECTORY)
    parser.add_argument("--simulation_batch_size", type=int, default=config.GENERATION_BATCH_SIZE)
    args = parser.parse_args()
    if min(args.trajectories, args.steps, args.simulation_batch_size) < 1:
        parser.error("trajectories、steps、simulation_batch_size 必须为正整数")
    return args


def main():
    args = parse_args()
    network_parameters = {
        "ER": {}, "SF": {"alpha": args.alpha},
        "WS": {"rewire_prob": args.rewire_prob},
    }[args.network_type]
    suffix = {"ER": "", "SF": f"_alpha{args.alpha:g}", "WS": f"_p{args.rewire_prob:g}"}[args.network_type]
    network_name = f"{args.network_type}3_N{args.N}_E{args.E}{suffix}_seed{args.seed}"
    output_dir = args.data_root / network_name
    network_path = output_dir / f"{network_name}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    if network_path.exists():
        hyperedges = np.loadtxt(network_path, dtype=np.int64, ndmin=2)
        if hyperedges.shape != (args.E, 3):
            raise ValueError(f"已保存网络的形状 {hyperedges.shape} 与指定的 E={args.E} 不一致")
    else:
        hyperedges = generate_hyperedges(
            args.network_type, args.N, args.E,
            np.random.default_rng(args.seed), **network_parameters,
        )
        np.savetxt(network_path, hyperedges, fmt="%d")

    parameters = config.DYNAMICS_PARAMETERS[args.dynamics_name].copy()
    # 拓扑与轨迹使用独立随机数；是否读取已有网络不会改变轨迹初始化。
    dynamics_seed = args.seed + 1
    rng = np.random.default_rng(dynamics_seed)
    x_batches, y_batches, index_batches = [], [], []
    for start in range(0, args.trajectories, args.simulation_batch_size):
        stop = min(start + args.simulation_batch_size, args.trajectories)
        states = SIMULATORS[args.dynamics_name](
            hyperedges, args.N, stop - start, args.steps, rng, **parameters,
        )
        x_batches.append(states[:, :-1].reshape(-1, *states.shape[2:]))
        y_batches.append(states[:, 1:].reshape(-1, *states.shape[2:]))
        index_batches.append(np.column_stack((
            np.repeat(np.arange(start, stop), args.steps),
            np.tile(np.arange(args.steps), stop - start),
        )))
    x, y = np.concatenate(x_batches), np.concatenate(y_batches)
    sample_index = np.concatenate(index_batches)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise FloatingPointError("轨迹出现 NaN/Inf，请减小积分步长后重新生成")

    prefix = f"{network_name}_{args.dynamics_name}"
    if args.dynamics_name == "Voter":
        np.savetxt(output_dir / f"{prefix}_x.txt", x, fmt="%d")
        np.savetxt(output_dir / f"{prefix}_y.txt", y, fmt="%d")
        np.savetxt(output_dir / f"{prefix}_index.txt", sample_index, fmt="%d")
    else:
        np.savez_compressed(output_dir / f"{prefix}_xy.npz", x=x, y=y, index=sample_index)
    metadata = {
        "network_name": network_name, "network_type": args.network_type,
        "network_parameters": network_parameters,
        "dynamics_name": args.dynamics_name, "N": args.N, "E": args.E,
        "hyperedge_size": 3, "mean_hyperdegree": 3 * args.E / args.N,
        "num_trajectories": args.trajectories, "steps_per_trajectory": args.steps,
        "num_samples": len(x), "channels": CHANNELS[args.dynamics_name],
        "parameters": parameters, "rule": RULES[args.dynamics_name],
        "seed": args.seed, "dynamics_seed": dynamics_seed,
        "simulation_batch_size": args.simulation_batch_size,
        "sample_definition": "X=s(t), Y=s(t+1); independent fixed-length trajectories",
        "sample_index_columns": ["trajectory_id", "time_step"],
        "sample_index_base": 0,
        "source_scope": "RESCENE state/initialization conventions; hypergraph interactions adapted in data/extra_dynamics.py",
    }
    if "internal_dt" in parameters:
        metadata["integrator"] = "RK4"
        metadata["observation_dt"] = parameters["internal_dt"] * parameters["microsteps"]
    (output_dir / f"{prefix}_parameters.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"saved: {output_dir}")
    print(f"network: {network_name}; dynamics: {args.dynamics_name}")
    print(f"trajectories: {args.trajectories}; steps per trajectory: {args.steps}; X: {x.shape}; Y: {y.shape}")
    print(f"channels: {CHANNELS[args.dynamics_name]}; parameters: {parameters}")


if __name__ == "__main__":
    main()
