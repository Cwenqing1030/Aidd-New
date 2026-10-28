"""三元超图上的 Voter、Kuramoto 和非线性三体 Spring。

参考 RESCENE 的状态形式和采样间隔；相互作用改为同一条三元边内的共同作用。
这里不使用超图的两两投影作为生成器。
"""

import numpy as np


def _rk4_step(state, derivative, dt):
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_voter(hyperedges, num_nodes, count, steps, rng, *, initial_probability):
    """同步的超边约束 q=2 Voter：另外两人一致才跟随，否则保持。

    每个节点每步均匀选择一条自己的真实超边；孤立节点保持原状态。
    采用 q-voter 的一致意见规则，epsilon=0；采样面板限制为同一条超边。
    参考：https://doi.org/10.1103/PhysRevE.80.041129
    """
    neighbors = [[] for _ in range(num_nodes)]
    for a, b, c in hyperedges:
        neighbors[a].append((b, c))
        neighbors[b].append((a, c))
        neighbors[c].append((a, b))
    neighbors = [np.asarray(pairs, dtype=np.int64).reshape(-1, 2) for pairs in neighbors]
    state = (rng.random((count, num_nodes)) < initial_probability).astype(np.int64)
    states = np.empty((count, steps + 1, num_nodes), dtype=np.int64)
    states[:, 0] = state
    samples = np.arange(count)
    for time in range(steps):
        next_state = state.copy()
        for node, pairs in enumerate(neighbors):
            if len(pairs) == 0:
                continue
            selected = pairs[rng.integers(len(pairs), size=count)]
            first = state[samples, selected[:, 0]]
            second = state[samples, selected[:, 1]]
            next_state[:, node] = np.where(first == second, first, state[:, node])
        states[:, time + 1] = next_state
        state = next_state
    return states


def simulate_kuramoto(
    hyperedges, num_nodes, count, steps, rng, *, coupling, frequency,
    internal_dt, microsteps,
):
    """d theta_i/dt = omega + K sum_{e={i,j,k}} sin(theta_j+theta_k-2theta_i).

    所有节点共用一个固有频率，匹配当前共享的动力学网络。
    同时观测 sin(theta)、cos(theta)，避免只看 sin(theta) 无法确定完整相位。
    与 RESCENE 的两两 Kuramoto、单通道 sin_phase 观测有所不同。
    """
    theta = rng.uniform(-np.pi, np.pi, size=(count, num_nodes))
    states = np.empty((count, steps + 1, num_nodes, 2), dtype=np.float32)
    states[:, 0] = np.stack((np.sin(theta), np.cos(theta)), axis=-1)

    def derivative(phase):
        edge_phase = phase[:, hyperedges]
        edge_effect = np.sin(edge_phase.sum(axis=-1, keepdims=True) - 3 * edge_phase)
        effects = np.zeros_like(phase)
        for member in range(3):
            np.add.at(effects.T, hyperedges[:, member], edge_effect[:, :, member].T)
        return frequency + coupling * effects

    for time in range(steps):
        for _ in range(microsteps):
            theta = _rk4_step(theta, derivative, internal_dt)
        states[:, time + 1] = np.stack((np.sin(theta), np.cos(theta)), axis=-1)
    return states


def simulate_spring(
    hyperedges, num_nodes, count, steps, rng, *, strength, location_std,
    velocity_norm, internal_dt, microsteps,
):
    """非线性三体弹性模型；每个节点观测 [x,y,vx,vy]，质量统一为 1。

    对一条边 e，质心为 c_e，q_e=sum_{i in e} ||r_i-c_e||^2。
    势能 U_e=k*(q_e/2+q_e^2/4)，因此 F_i,e=-k*(1+q_e)*(r_i-c_e)。
    q_e 使三个成员共同决定弹性强度；去掉它就会退化为两两投影上的线性力。
    这是本项目定义的三体扩展，非 RESCENE 原版线性 Spring。
    使用无边界、无外力的弹性系统，不截断力或位置。
    """
    position = rng.normal(0.0, location_std, size=(count, num_nodes, 2))
    velocity = rng.normal(size=(count, num_nodes, 2))
    velocity *= velocity_norm / np.linalg.norm(velocity, axis=-1, keepdims=True)
    state = np.concatenate((position, velocity), axis=-1)
    states = np.empty((count, steps + 1, num_nodes, 4), dtype=np.float32)
    states[:, 0] = state

    def derivative(current):
        edge_position = current[:, hyperedges, :2]
        offsets = edge_position - edge_position.mean(axis=2, keepdims=True)
        spread = np.sum(offsets ** 2, axis=(2, 3))
        edge_force = -strength * (1 + spread[:, :, None, None]) * offsets
        acceleration = np.zeros((count, num_nodes, 2), dtype=np.float64)
        for member in range(3):
            np.add.at(
                acceleration.transpose(1, 0, 2), hyperedges[:, member],
                edge_force[:, :, member].transpose(1, 0, 2),
            )
        return np.concatenate((current[:, :, 2:], acceleration), axis=-1)

    for time in range(steps):
        for _ in range(microsteps):
            state = _rk4_step(state, derivative, internal_dt)
        states[:, time + 1] = state
    return states


SIMULATORS = {
    "Voter": simulate_voter,
    "Kuramoto": simulate_kuramoto,
    "Spring": simulate_spring,
}

CHANNELS = {
    "Voter": ["opinion"],
    "Kuramoto": ["sin_phase", "cos_phase"],
    "Spring": ["x", "y", "vx", "vy"],
}

RULES = {
    "Voter": "synchronous hyperedge-panel q=2 voter; unanimous pair copies, otherwise stays; epsilon=0",
    "Kuramoto": "theta_dot_i=omega+K*sum_{e={i,j,k}}sin(theta_j+theta_k-2*theta_i); shared omega",
    "Spring": "q_e=sum_i||r_i-centroid_e||^2; U_e=k*(q_e/2+q_e^2/4); force_i,e=-k*(1+q_e)*(r_i-centroid_e)",
}
