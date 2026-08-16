import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# ================
# 第一个神经网络 NN1
class NN1(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN1, self).__init__()

        self.n2e = nn.Linear(input_dim, hidden_dim)
        self.e2e = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.n2e(x))
        x = F.relu(self.e2e(x))

        return x

# NN2
class NN2(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN2, self).__init__()

        self.e2n = nn.Linear(input_dim, hidden_dim)
        self.n2n = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.e2n(x))
        x = F.relu(self.n2n(x))

        return x


# NN3
class NN3(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NN3, self).__init__()

        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        x = self.output(x)

        return x

# ======
# 读取数据
data = np.load("generated_data.npz")

all_Xt = data["all_Xt"]
graph_types = data["graph_types"]
B = data["B"]

# 选择ER网络的数据
er_mask = (graph_types == "ER")
Xt_er = all_Xt[er_mask]

# 选择第一张ER网络
Xt = Xt_er[0]

# 选择目标节点 i
target_node = 0

# 取B中目标节点i对应的列
B_column = B[:, target_node]

# 当前测试时间
t = 0
# 当前时刻整个网络状态 X^t
X_t = Xt[t]

# 当前时刻目标节点状态 x_i^t
x_i_t = Xt[t, target_node]

# 下一时刻目标节点状态 x_i^(t+1)
target_next = Xt[t + 1, target_node]

# X_t变成1×100
X_t = X_t.reshape(1, -1)

# x_i_t变成1×1
x_i_t = np.array([[x_i_t]])

# 构造NN1输入
nn1_input = np.concatenate(
    (
        x_i_t,
        X_t
    ),
    axis=1
)

# 转换成PyTorch张量
nn1_input = torch.tensor(
    nn1_input,
    dtype=torch.float32
)

# =========
# 创建 NN1
# =========
# NN1输入维度
# 当前输入：[x_i^t, X^t]
# 维度为1×101
nn1_input_dim = nn1_input.shape[1]

# NN1隐藏层维度
# 这里先设置为输入维度
nn1_hidden_dim = nn1_input_dim

# NN1输出维度,暂时保持一致
nn1_output_dim = nn1_hidden_dim

nn1 = NN1(
    nn1_input_dim,
    nn1_hidden_dim,
    nn1_output_dim
)

h1 = nn1(nn1_input)
h1_neighbor = h1[:,1:]

# 提取B中目标节点i对应的列
B_column = torch.tensor(
    B[:, target_node],
    dtype=torch.float32
)

# (100,) -> (1,100),保证可以和h1_neighbor相乘
B_column = B_column.reshape(1,-1)

# 使用邻接矩阵筛选真实邻居
h1_filtered = h1_neighbor * B_column

# ===========
# 邻居信息聚合
# ===========

# 对100个邻居节点的信息求和
# 得到目标节点i收到的总邻居影响
neighbor_sum = torch.sum(
    h1_filtered,
    dim=1,
    keepdim=True
)
# =========
# 创建 NN2
# =========

# NN2输入就是邻居聚合后的结果
nn2_input_dim = neighbor_sum.shape[1]

# 隐藏维度先保持一致
nn2_hidden_dim = nn2_input_dim

# 输出维度先保持一致
nn2_output_dim = nn2_hidden_dim


nn2 = NN2(
    nn2_input_dim,
    nn2_hidden_dim,
    nn2_output_dim
)

h2 = nn2(neighbor_sum)
# =============
# 构造 NN3 输入
# =============

# h2 + 当前节点状态 x_i^t
nn3_input = torch.cat(
    (
        h2,
        torch.tensor(
            x_i_t,
            dtype=torch.float32
        )
    ),
    dim=1
)

# =========
# 创建 NN3
# =========

nn3_input_dim = nn3_input.shape[1]

# 输出一个状态值
nn3_output_dim = 1


nn3 = NN3(
    nn3_input_dim,
    nn3_output_dim
)

prediction = nn3(nn3_input)

# 定义损失函数F
criterion = nn.MSELoss()

# 定义优化器
# 训练NN1、NN2、NN3
# B不参与训练
optimizer = torch.optim.Adam(
    list(nn1.parameters())
    + list(nn2.parameters())
    + list(nn3.parameters()),
    lr=0.01
)

# ====
# 训练
# ====
num_epochs = 100

for epoch in range(1, num_epochs + 1):

    # 清空上一轮梯度
    optimizer.zero_grad()

    # =================
    # NN1
    # =================

    # 输入[x_i^t, X^t]，得到h1
    h1 = nn1(nn1_input)

    # h1第一列对应目标节点自身信息
    # 取后100列作为100个节点的信息
    h1_neighbor = h1[:, 1:]

    # 使用B[:, i]筛选真正的邻居
    h1_filtered = h1_neighbor * B_column

    neighbor_sum = torch.sum(
        h1_filtered,
        dim=1,
        keepdim=True
    )

    # =================
    # NN2
    # =================

    h2 = nn2(neighbor_sum)

    # =================
    # NN3
    # =================

    # 当前目标节点状态x_i^t
    x_i_t_tensor = torch.tensor(
        x_i_t,
        dtype=torch.float32
    )

    # NN3输入：
    # NN2输出h2 + 当前节点状态x_i^t
    nn3_input = torch.cat(
        (
            h2,
            x_i_t_tensor
        ),
        dim=1
    )

    # NN3预测下一时刻状态x_i^(t+1)
    prediction = nn3(nn3_input)

    # =================
    # 真实值
    # =================

    target_label = torch.tensor(
        [[target_next]],
        dtype=torch.float32
    )

    # =================
    # 计算损失
    # =================

    loss = criterion(
        prediction,
        target_label
    )

    # 反向传播
    loss.backward()

    # 更新NN1、NN2、NN3参数
    optimizer.step()

    # 每10轮打印一次
    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch}/{num_epochs}, "
            f"Prediction = {prediction.item():.6f}, "
            f"Target = {target_label.item():.0f}, "
            f"Loss = {loss.item():.6f}"
        )