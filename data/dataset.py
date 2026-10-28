"""动力学状态对读取、按完整轨迹划分以及 DataLoader。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


DYNAMICS_SPECS = {
    "SIR": {
        "kind": "discrete", "num_states": 3, "state_dim": 1,
        "state_encoding": {"S": 0, "I": 1, "R": 2},
    },
    "Voter": {
        "kind": "discrete", "num_states": 2, "state_dim": 1,
        "state_encoding": {"opinion_0": 0, "opinion_1": 1},
    },
    "Kuramoto": {
        "kind": "continuous", "num_states": None, "state_dim": 2,
        "state_encoding": ["sin_phase", "cos_phase"],
    },
    "Spring": {
        "kind": "continuous", "num_states": None, "state_dim": 4,
        "state_encoding": ["x", "y", "vx", "vy"],
    },
}


def load_data(data_dir, network_name, dynamics_name):
    data_dir = Path(data_dir)
    dynamics_prefix = f"{network_name}_{dynamics_name}"
    spec = DYNAMICS_SPECS[dynamics_name]
    if spec["kind"] == "discrete":
        x = np.loadtxt(
            data_dir / f"{dynamics_prefix}_x.txt", dtype=np.float32, ndmin=2
        )
        y = np.loadtxt(
            data_dir / f"{dynamics_prefix}_y.txt", dtype=np.int64, ndmin=2
        )
        sample_index = np.loadtxt(
            data_dir / f"{dynamics_prefix}_index.txt", dtype=np.int64, ndmin=2
        )
        allowed = np.arange(spec["num_states"])
        if not (np.isin(x, allowed).all() and np.isin(y, allowed).all()):
            raise ValueError(f"{dynamics_name} 状态必须属于 {allowed.tolist()}")
    else:
        with np.load(data_dir / f"{dynamics_prefix}_xy.npz") as arrays:
            x = arrays["x"].astype(np.float32)
            y = arrays["y"].astype(np.float32)
            sample_index = arrays["index"].astype(np.int64)
        if x.ndim != 3 or x.shape[-1] != spec["state_dim"]:
            raise ValueError(f"{dynamics_name} 需要 [samples,N,{spec['state_dim']}] 状态")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("连续轨迹包含 NaN/Inf，请检查生成参数")

    if x.shape != y.shape or sample_index.shape != (len(x), 2):
        raise ValueError("X、Y 必须同形，index 必须有对应的轨迹编号和时间步两列")
    same_trajectory = sample_index[1:, 0] == sample_index[:-1, 0]
    if not np.array_equal(x[1:][same_trajectory], y[:-1][same_trajectory]):
        raise ValueError("轮内 X/Y 不连续，请检查同步的数据和 index 是否对应")

    hyperedges = np.loadtxt(
        data_dir / f"{network_name}.txt", dtype=np.int64, ndmin=2
    )
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
