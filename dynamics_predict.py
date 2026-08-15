import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 第一个神经网络 NN1
class NN1(nn.Module):

    def __init__(self):
        super(NN1, self).__init__()

        # 候选节点3维 + 目标节点3维 = 6维
        # NN1内部保持3维
        self.n2e = nn.Linear(6, 3)
        self.e2e = nn.Linear(3, 3)

    def forward(self, x):

        x = F.relu(self.n2e(x))
        x = F.relu(self.e2e(x))

        return x

# NN2
class NN2(nn.Module):

    def __init__(self):
        super(NN2, self).__init__()

        # 输入3维，内部和输出都保持3维
        self.e2n = nn.Linear(3, 3)
        self.n2n = nn.Linear(3, 3)

    def forward(self, x):

        x = F.relu(self.e2n(x))
        x = F.relu(self.n2n(x))

        return x

# NN3
class NN3(nn.Module):

    def __init__(self):
        super(NN3, self).__init__()

        # 邻居影响3维 + 节点自身状态3维 = 6维
        # 输出S、I、R三个状态分数
        self.output = nn.Linear(6, 3)

    def forward(self, x):

        x = self.output(x)

        return x


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

# 将S、I、R状态转换成3维
# S=0 -> [1, 0, 0]
# I=1 -> [0, 1, 0]
# R=2 -> [0, 0, 1]

Xt = np.eye(3, dtype=np.float32)[Xt]

# 构造 t -> t+1
X_current = Xt[:-1]
X_next = Xt[1:]

# 选择目标节点 i
target_node = 0

# 取B中目标节点i对应的列
B_column = B[:, target_node]

# 取目标节点i当前状态
target_series = X_current[:, target_node, :]

# 将目标节点状态复制到100个候选节点
target_expand = np.repeat(
    target_series[:, np.newaxis, :],
    repeats=100,
    axis=1
)

# 构造NN1输入
# 候选节点状态 + 目标节点状态
nn1_input = np.concatenate(
    (X_current, target_expand),
    axis=2
)

# 转换成PyTorch张量
nn1_input = torch.tensor(
    nn1_input,
    dtype=torch.float32
)

target_series_tensor = torch.tensor(
    target_series,
    dtype=torch.float32
)

B_column = torch.tensor(
    B_column,
    dtype=torch.float32
)

# 从(100,)变成(1,100,1)
B_column = B_column.view(1, 100, 1)

# 构造真实下一时刻标签
target_next = X_next[:, target_node, :]

target_label = np.argmax(
    target_next,
    axis=1
)

target_label = torch.tensor(
    target_label,
    dtype=torch.long
)

# 创建三个神经网络
nn1 = NN1()
nn2 = NN2()
nn3 = NN3()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
# 训练NN1、NN2、NN3
# B不参与训练
optimizer = torch.optim.Adam(
    list(nn1.parameters())
    + list(nn2.parameters())
    + list(nn3.parameters()),
    lr=0.01
)


# 训练
num_epochs = 100

for epoch in range(1, num_epochs + 1):

    # 清空上一轮梯度
    optimizer.zero_grad()

    # NN1
    h1 = nn1(nn1_input)

    # 使用B[:, i]筛选真正的邻居
    h1_filtered = h1 * B_column

    # 将所有邻居影响聚合
    neighbor_sum = torch.sum(
        h1_filtered,
        dim=1
    )

    # NN2
    h2 = nn2(neighbor_sum)

    # NN3输入
    # 邻居影响 + 节点自身状态
    nn3_input = torch.cat(
        (h2, target_series_tensor),
        dim=1
    )

    # NN3
    logits = nn3(nn3_input)

    # 计算损失
    loss = criterion(
        logits,
        target_label
    )

    # 反向传播
    loss.backward()

    # 更新三个NN参数
    optimizer.step()

    # 每10轮打印一次
    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch}/{num_epochs}, "
            f"Loss = {loss.item():.6f}"
        )