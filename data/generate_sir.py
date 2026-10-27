"""在固定三元超图上生成固定长度的 SIR 轨迹。0=S，1=I，2=R。

每行 X=s(t)、Y=s(t+1)。index 两列为从 0 开始的轨迹编号、时间步。
同一轨迹内 Y[k]=X[k+1]；划分训练集和测试集时按轨迹编号分组。
"""

from pathlib import Path
import json

import numpy as np


N = 50
E = 400
NUM_TRAJECTORIES = 1000
STEPS_PER_TRAJECTORY = 5
INITIAL_INFECTED = N // 5  # 初始感染比例保持 20%
BETA = 0.5
GAMMA = 0.5
SEED = 42


def generate_hypergraph(rng):
    hyperedges = set()
    while len(hyperedges) < E:
        hyperedges.add(tuple(sorted(rng.choice(N, 3, replace=False).tolist())))

    hyperedges = np.array(sorted(hyperedges), dtype=np.int64)
    incidence = np.zeros((N, E), dtype=np.int64)
    incidence[hyperedges, np.arange(E)[:, None]] = 1
    return hyperedges, incidence


def sir_step(state, incidence, beta, gamma, rng):
    # 易感节点所在超边的另外两个成员均感染时，该超边才会激活。
    infected = state == 1
    infected_per_edge = infected.astype(np.int64) @ incidence
    active_edges = infected_per_edge == 2
    active_count = incidence @ active_edges.astype(np.int64)
    infection_probability = 1 - (1 - beta) ** active_count

    # 两种转移均读取本步原状态；新感染节点不会在同一步恢复。
    newly_infected = (state == 0) & (rng.random(len(state)) < infection_probability)
    newly_recovered = infected & (rng.random(len(state)) < gamma)
    next_state = state.copy()
    next_state[newly_infected] = 1
    next_state[newly_recovered] = 2
    return next_state


def generate_data():
    rng = np.random.default_rng(SEED)
    hyperedges, incidence = generate_hypergraph(rng)
    x, y, sample_index = [], [], []

    for trajectory_id in range(NUM_TRAJECTORIES):
        state = np.zeros(N, dtype=np.int64)
        state[rng.choice(N, INITIAL_INFECTED, replace=False)] = 1

        # 每轮固定步数；感染者提前清零后，剩余步数继续保存不变的状态。
        for time_step in range(STEPS_PER_TRAJECTORY):
            next_state = sir_step(state, incidence, BETA, GAMMA, rng)
            x.append(state.copy())
            y.append(next_state.copy())
            sample_index.append((trajectory_id, time_step))
            state = next_state

    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    sample_index = np.asarray(sample_index, dtype=np.int64)

    prefix = f"hypergraph_N{N}"
    output_dir = Path(__file__).resolve().parent / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / f"{prefix}.txt", hyperedges, fmt="%d")
    np.savetxt(output_dir / f"{prefix}_SIR_x.txt", x, fmt="%d")
    np.savetxt(output_dir / f"{prefix}_SIR_y.txt", y, fmt="%d")
    np.savetxt(output_dir / f"{prefix}_SIR_index.txt", sample_index, fmt="%d")

    parameters = {
        "N": N, "E": E, "hyperedge_size": 3,
        "num_trajectories": NUM_TRAJECTORIES, "num_samples": len(x),
        "steps_per_trajectory": STEPS_PER_TRAJECTORY,
        "initial_infected": INITIAL_INFECTED, "beta": BETA, "gamma": GAMMA,
        "seed": SEED,
        "sample_definition": "X[row]=s(t), Y[row]=s(t+1); save every step of each fixed-length trajectory",
        "trajectory_end": "fixed horizon; continue unchanged if infections reach zero",
        "sample_index_columns": ["trajectory_id", "time_step"],
        "sample_index_base": 0,
        "infection_rule": "both other members infected; p=1-(1-beta)^active_incident_edges",
        "update": "synchronous; R absorbing",
        "state_encoding": {"S": 0, "I": 1, "R": 2},
        "reference": "https://doi.org/10.1063/5.0201557",
        "reference_scope": "beta and gamma only; topology, initialization and trajectory sampling are our choices",
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_dir}")
    print(f"hyperedges: {hyperedges.shape}; incidence: {incidence.shape}")
    print(f"trajectories: {NUM_TRAJECTORIES}; steps per trajectory: {STEPS_PER_TRAJECTORY}; X: {x.shape}; Y: {y.shape}")
    print("index columns: trajectory_id, time_step (zero-based); states: 0=S, 1=I, 2=R")


if __name__ == "__main__":
    generate_data()
