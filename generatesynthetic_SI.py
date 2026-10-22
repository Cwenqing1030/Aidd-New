"""在固定的三元超图上生成 SI 一步预测数据。0=S，1=I，无恢复。"""

import argparse
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    """固定拓扑、初始感染节点和传播抽样的随机性。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_incidence_matrix(N: int, E: int, k: int, device: torch.device) -> torch.Tensor:
    """生成 N×E 的真实关联矩阵；每列是一条互不重复的 k 元超边。"""
    Btrue = torch.zeros((N, E), dtype=torch.float32, device=device)
    used = set()
    e = 0
    while e < E:
        edge = tuple(sorted(torch.randperm(N, device=device)[:k].tolist()))
        if edge in used:
            continue
        used.add(edge)
        Btrue[list(edge), e] = 1.0
        e += 1
    return Btrue


def initialize_si_state(N: int, init_infected: int, device: torch.device) -> torch.Tensor:
    """每条短轨迹随机选出 init_infected 个 I 节点，其余节点为 S。"""
    x = torch.zeros(N, dtype=torch.long, device=device)
    x[torch.randperm(N, device=device)[:init_infected]] = 1
    return x


@torch.no_grad()
def hypergraph_si_step(x: torch.Tensor, network: torch.Tensor, beta: float) -> torch.Tensor:
    """与感染节点共享超边的 S 节点，每步只进行一次概率为 beta 的感染判断。"""
    infected = (x == 1).float()
    infected_per_edge = network.T @ infected
    has_infected_neighbor = (network @ (infected_per_edge > 0).float()) > 0
    # 先标记含感染节点的超边，再用 [N,E] 乘 [E] 找到这些超边的成员；>0 表示至少共享一条超边。

    next_state = x.clone()
    # 复制当前状态作为下一时刻暂存区；本步全部判断仍只读取 x，保证同步更新。
    susceptible = (x == 0) & has_infected_neighbor
    newly_infected = susceptible & (torch.rand(x.shape[0], device=x.device) < beta)
    # 每个节点本步抽取一个随机数；只有符合条件的 S 节点且随机数小于 beta 才感染一次。
    next_state[newly_infected] = 1
    return next_state


@torch.no_grad()
def generate_si_pairs(
    network: torch.Tensor, num_pairs: int, episode_length: int, beta: float, init_infected: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """先生成每条短轨迹的完整状态序列，再分别错位得到 X 和 Y。"""
    N = network.shape[0]
    xs, ys, episode_rows = [], [], []
    total_pairs = 0
    episode_id = 0
    while total_pairs < num_pairs:
        states = [initialize_si_state(N, init_infected, network.device)]
        for _ in range(min(episode_length, num_pairs - total_pairs)):
            if bool(torch.all(states[-1] == 1)):
                break
            states.append(hypergraph_si_step(states[-1], network, beta))
        states = torch.stack(states)
        X_episode = states[:-1]
        Y_episode = states[1:]
        # 除第一状态外都作为标签 Y，得到 S(1) 到 S(T)；不额外进行传播。
        xs.append(X_episode)
        ys.append(Y_episode)
        episode_rows.extend((episode_id, step) for step in range(X_episode.shape[0]))
        total_pairs += X_episode.shape[0]
        episode_id += 1
    return (
        torch.cat(xs),
        torch.cat(ys),
        torch.tensor(episode_rows, dtype=torch.long, device=network.device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成固定超图上的 SI 一步预测数据")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N, E, k = 200, 400, 3
    num_pairs, episode_length = 500, 12
    init_infected, beta = 50, 0.10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    set_seed(args.seed)
    Btrue = generate_incidence_matrix(N, E, k, device)
    X, Y, episode_info = generate_si_pairs(
        Btrue, num_pairs, episode_length, beta, init_infected
    )

    # 将关联矩阵转成超边列表，以便按一条超边一行的格式保存。
    Btrue_cpu = Btrue.cpu()
    edges = [
        tuple(torch.nonzero(Btrue_cpu[:, e], as_tuple=False).flatten().tolist())
        for e in range(E)
    ]
    output_dir = Path(__file__).resolve().parent / "generated_data" / "SI_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "hypergraph_N200"
    structure_path = output_dir / f"{prefix}.txt"
    x_path = output_dir / f"{prefix}_SI_x.txt"
    y_path = output_dir / f"{prefix}_SI_y.txt"
    episode_path = output_dir / f"{prefix}_SI_episodes.txt"

    with structure_path.open("w", encoding="utf-8") as f:
        for edge in edges:
            f.write(" ".join(map(str, edge)) + "\n")
    for path, data in ((x_path, X), (y_path, Y), (episode_path, episode_info)):
        with path.open("w", encoding="utf-8") as f:
            for row in data.cpu().tolist():
                f.write(" ".join(map(str, row)) + "\n")

    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
