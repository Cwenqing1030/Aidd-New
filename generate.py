import networkx as nx
import numpy as np

# =======
# 实验配置
# =======
# 随机种子
SIR_SEED = 2000
GRAPH_SEED = 2000
B_SEED = 3000

NUM_NODES = 100
NUM_STEPS = 1000
RESET_INTERVAL = 100

NUM_GRAPHS = 50

NUM_ER = 17
NUM_WS = 17
NUM_BA = 16

# SIR 参数
BETA = 0.3
# 表示感染概率，一个处于 S 状态的节点受到感染者影响后，有多大可能从：S→I
GAMMA = 0.1
# 表示恢复概率，一个处于 I 状态的节点，有多大可能从：I→R

# 网络生成参数
P_ER = 0.3
K_WS = 4
P_WS = 0.3
M_BA = 2
P_B = 0.3

np.random.seed(SIR_SEED)
graph_seed = GRAPH_SEED

network_plan = [("ER", NUM_ER), ("WS", NUM_WS), ("BA", NUM_BA)]

all_A = []
all_Xt = []
graph_types = []

S = 0
I = 1
R = 2

for network_type, network_num in network_plan:
    for i in range(network_num):
        if network_type == "ER":
            G = nx.erdos_renyi_graph(
                n=NUM_NODES, p=P_ER, seed=graph_seed, directed=False
            )

        elif network_type == "WS":
            G = nx.watts_strogatz_graph(n=NUM_NODES, k=K_WS, p=P_WS, seed=graph_seed)

        elif network_type == "BA":
            G = nx.barabasi_albert_graph(n=NUM_NODES, m=M_BA, seed=graph_seed)

        else:
            raise ValueError(f"未知的网络类型: {network_type}")

        A = nx.to_numpy_array(G, dtype=np.int64)

        graph_seed += 1

        # 当前网络自己的初始状态
        x0 = np.zeros(NUM_NODES, dtype=np.int64)
        x0[0] = I

        # 当前网络自己的时间状态矩阵
        Xt = np.zeros((NUM_STEPS, NUM_NODES), dtype=np.int64)
        Xt[0] = x0

        # 当前这一张网络进行SIR时间迭代
        for t in range(NUM_STEPS - 1):

            # 每100个时间步重新初始化一次节点状态
            if (t + 1) % RESET_INTERVAL == 0:
                next_state = np.zeros(NUM_NODES, dtype=np.int64)
                next_state[0] = I
                Xt[t + 1] = next_state
                continue

            current_state = Xt[t]
            next_state = current_state.copy()

            for node in range(NUM_NODES):
                if current_state[node] == S:
                    infected_count = 0

                    for neighbor in range(NUM_NODES):
                        if A[node, neighbor] == 1:
                            if current_state[neighbor] == I:
                                infected_count += 1

                    infection_prob = 1 - ((1 - BETA) ** infected_count)

                    if np.random.rand() < infection_prob:
                        next_state[node] = I

                elif current_state[node] == I:
                    if np.random.rand() < GAMMA:
                        next_state[node] = R

            Xt[t + 1] = next_state

        # 当前这一张网络的1000个时刻全部生成完
        # 再保存整张图的数据
        all_A.append(A.copy())
        all_Xt.append(Xt.copy())
        graph_types.append(network_type)

        # 输出当前这一张网络的数据
        print()
        print("========================================")
        print(f"{network_type}_{i + 1:02d}")
        print("========================================")

        print("邻接矩阵 A：")
        print(A)

        print("时间状态序列 Xt：")

        for t in range(RESET_INTERVAL // 2,NUM_STEPS,RESET_INTERVAL):
            print(f"t={t}时刻的网络状态：",Xt[t])


# ==========================
# 生成用于网络结构恢复的邻接矩阵 B
# ==========================

structure_seed = B_SEED
structure_rng = np.random.default_rng(structure_seed)

p_structure = P_B

B_upper = np.triu(
    (structure_rng.random((NUM_NODES, NUM_NODES)) < p_structure).astype(int), k=1
)

B = B_upper + B_upper.T

print()
print("========================================")
print("邻接矩阵 B")
print("========================================")

print("B的形状：", B.shape)
print("B：")
print(B)

# =========================
# 保存所有生成的数据
# =========================

all_A_array = np.stack(all_A)
all_Xt_array = np.stack(all_Xt)
graph_types_array = np.array(graph_types)

np.savez_compressed(
    "data/generated_data.npz",
    all_A=all_A_array,
    all_Xt=all_Xt_array,
    graph_types=graph_types_array,
    B=B,
)

print()
print("========================================")
print("数据保存完成")
