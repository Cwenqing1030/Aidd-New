import numpy as np
import networkx as nx
np.random.seed(2000)
num_nodes = 10
num_steps = 20
p = 0.3
seed=2000

S = 0
I = 1
R = 2

x0 = np.zeros(num_nodes, dtype=np.int64)
# t=0 时刻，10 个节点的状态全部都是 S

G = nx.erdos_renyi_graph(
    n=num_nodes,
    p=p,
    seed=seed,
    directed=False
)

A = nx.to_numpy_array(G, dtype=int)

x0[0] = I
print("t=0时刻的网络状态：", x0)
print("邻接矩阵 A：",A)

Xt = np.zeros((num_steps, num_nodes), dtype=np.int64)
Xt[0] = x0

beta = 0.7
# 表示感染概率，一个处于 S 状态的节点受到感染者影响后，有多大可能从：S→I
gamma = 0.1
# 表示恢复概率，一个处于 I 状态的节点，有多大可能从：I→R
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
    print(f"t={t + 1}时刻的网络状态：", Xt[t + 1])