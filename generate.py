import numpy as np
import networkx as nx
np.random.seed(2000)
num_nodes = 100
num_steps = 20

num_graphs = 50

num_er = 17
num_ws = 17
num_ba = 16

p = 0.3
seed=2000

# WS网络参数
k_ws = 4
p_ws = 0.3

# BA网络参数
m_ba = 2

network_plan = [
    ("ER", num_er),
    ("WS", num_ws),
    ("BA", num_ba)
]

all_A = []
all_Xt = []
graph_types = []

S = 0
I = 1
R = 2

beta = 0.3
# 表示感染概率，一个处于 S 状态的节点受到感染者影响后，有多大可能从：S→I
gamma = 0.1
# 表示恢复概率，一个处于 I 状态的节点，有多大可能从：I→R

for network_type, network_num in network_plan:

    for i in range(network_num):

        if network_type == "ER":

            G = nx.erdos_renyi_graph(
                n=num_nodes,
                p=p,
                seed=seed,
                directed=False
            )

        elif network_type == "WS":

            G = nx.watts_strogatz_graph(
                n=num_nodes,
                k=k_ws,
                p=p_ws,
                seed=seed
            )

        elif network_type == "BA":

            G = nx.barabasi_albert_graph(
                n=num_nodes,
                m=m_ba,
                seed=seed
            )

        A = nx.to_numpy_array(G, dtype=int)

        seed += 1

        # 当前网络自己的初始状态
        x0 = np.zeros(num_nodes, dtype=np.int64)
        x0[0] = I

        # 当前网络自己的时间状态矩阵
        Xt = np.zeros((num_steps, num_nodes), dtype=np.int64)
        Xt[0] = x0

        # 当前这一张ER网络进行SIR时间迭代
        for t in range(num_steps - 1):
            current_state = Xt[t]
            next_state = current_state.copy()
            for node in range(num_nodes):
                if current_state[node] == S:
                    infected_count = 0
                    for neighbor in range(num_nodes):
                        if A[node, neighbor] == 1:
                            if current_state[neighbor] == I:
                                infected_count += 1
                    infection_prob =1-((1-beta) ** infected_count)

                    if np.random.rand() < infection_prob:
                        next_state[node] = I

                elif current_state[node] == I:
                    if np.random.rand() < gamma:
                        next_state[node] = R

            Xt[t + 1] = next_state

        # 当前这一张网络的20个时刻全部生成完
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

        for t in range(num_steps):
            print(f"t={t}时刻的网络状态：", Xt[t])


# =========================
# 生成用于网络结构恢复的邻接矩阵 B
# =========================

structure_seed = 3000
structure_rng = np.random.default_rng(structure_seed)

p_structure = 0.3

B_upper = np.triu(
    (structure_rng.random((num_nodes, num_nodes)) < p_structure).astype(int),
    k=1
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
    "generated_data.npz",
    all_A=all_A_array,
    all_Xt=all_Xt_array,
    graph_types=graph_types_array,
    B=B
)

print()
print("========================================")
print("数据保存完成")