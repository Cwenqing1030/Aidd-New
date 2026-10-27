"""SIR 状态对读取、按轨迹划分以及 DataLoader。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir, network_name, dynamics_name):
    data_dir = Path(data_dir)
    dynamics_prefix = f"{network_name}_{dynamics_name}"
    x = np.loadtxt(
        data_dir / f"{dynamics_prefix}_x.txt", dtype=np.float32, ndmin=2
    )
    y = np.loadtxt(
        data_dir / f"{dynamics_prefix}_y.txt", dtype=np.int64, ndmin=2
    )
    sample_index = np.loadtxt(
        data_dir / f"{dynamics_prefix}_index.txt", dtype=np.int64, ndmin=2
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
