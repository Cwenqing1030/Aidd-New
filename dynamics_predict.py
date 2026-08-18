import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# ================
# 第一个神经网络 NN1
# ================
class NN1(nn.Module):

    def __init__(self, hidden_dim):
        super(NN1, self).__init__()

        # 1维节点状态 -> F维隐藏特征
        self.linear = nn.Linear(
            1,
            hidden_dim
        )

    def forward(self, x):

        x = F.relu(
            self.linear(x)
        )

        return x

# ================
# 第二个神经网络 NN2
# ================
class NN2(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN2, self).__init__()

        self.e2n = nn.Linear(input_dim, hidden_dim)
        self.n2n = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.e2n(x))
        x = F.relu(self.n2n(x))

        return x

# ================
# 第三个神经网络 NN3
# ================
class NN3(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NN3, self).__init__()

        self.output = nn.Linear(
            input_dim,
            output_dim
        )

    def forward(self, x):

        x = self.output(x)

        return x

# ==========
# 读取数据
# ==========
data = np.load("generated_data.npz")

all_Xt = data["all_Xt"]
graph_types = data["graph_types"]
B = data["B"]

# ===============
# 选择ER网络的数据
# ===============
er_mask = (graph_types == "ER")
Xt_er = all_Xt[er_mask]


# =====================
# 节点信息融合层
# =====================
# 输入：[x_i^t, X^t]
# x_i^t：1×1
# X^t：1×100
# 拼接后：1×101
# 融合后：1×100
fusion_layer = nn.Linear(
    101,
    100
)

# =========
# 创建 NN1
# =========
# 隐藏维度 F
F_dim = 32
nn1 = NN1(
    F_dim
)

# =========
# 创建 NN2
# =========
# NN1最终输出的特征维度为F
# B筛选以后仍然得到1×F
# 所以NN2： 1×F -> 1×F
nn2_input_dim = F_dim
nn2_hidden_dim = F_dim
nn2_output_dim = F_dim

nn2 = NN2(
    input_dim=nn2_input_dim,
    hidden_dim=nn2_hidden_dim,
    output_dim=nn2_output_dim
)

# =========
# 创建 NN3
# =========
# NN2输出： h2 = 1×F
# 当前节点状态： x_i^t = 1×1
# 拼接以后： 1×(F+1)

nn3_input_dim = F_dim + 1

# SIR三个类别：
# 0 -> S
# 1 -> I
# 2 -> R

nn3_output_dim = 3

nn3 = NN3(
    input_dim=nn3_input_dim,
    output_dim=nn3_output_dim
)

# ===========
# 定义损失函数
# ===========
criterion = nn.CrossEntropyLoss()
# ===========
# 定义优化器
# ===========
optimizer = torch.optim.Adam(
    list(fusion_layer.parameters())
    + list(nn1.parameters())
    + list(nn2.parameters())
    + list(nn3.parameters()),
    lr=0.01
)

# ====
# 训练
# ====
num_epochs = 100

# =============
# 第一层：epoch
# =============
for epoch in range(num_epochs):

    # 用来统计当前epoch的loss
    epoch_loss = 0.0

    # 当前epoch一共训练了多少个节点样本
    sample_count = 0

    # =============
    # 第二层：graph
    # =============
    # 遍历所有ER网络
    for graph_idx in range(len(Xt_er)):
        # 当前ER网络
        Xt = Xt_er[graph_idx]

        # ============
        # 第三层：time
        # ============
        # Xt共有20个时间点
        # t=0  -> t=1
        # t=1  -> t=2
        # ...
        # t=18 -> t=19
        # 这里只遍历到倒数第二个时刻

        for t in range(Xt.shape[0] - 1):

            # 当前时刻整个网络状态 X^t
            # 原始维度：(100,)
            X_t = Xt[t]

            # ============
            # 第四层：node
            # ============
            # 遍历当前网络的100个节点
            for target_node in range(Xt.shape[1]):
                # 每个节点开始前清空梯度
                optimizer.zero_grad()
                # 当前节点状态
                # 当前时刻目标节点状态 x_i^t
                x_i_t = Xt[
                    t,
                    target_node
                ]

                # 下一时刻目标节点真实状态 x_i^(t+1)
                target_next = Xt[
                    t + 1,
                    target_node
                ]

                # =========
                # 整理 X^t
                # =========
                # X_t：
                # (100,) -> (1,100)
                X_t_input = X_t.reshape(
                    1,
                    -1
                )

                # =====================
                # 整理 x_i^t
                # =====================
                # 标量 -> 1×1
                x_i_input = np.array(
                    [[x_i_t]]
                )

                # =============
                # 构造融合层输入
                # =============
                # x_i^t： 1×1
                # X^t：1×100
                # 拼接：1×101
                fusion_input = np.concatenate(
                    (
                        x_i_input,
                        X_t_input
                    ),
                    axis=1
                )

                # numpy -> torch
                fusion_input = torch.tensor(
                    fusion_input,
                    dtype=torch.float32
                )

                # =====================
                # 当前节点状态转Tensor
                # =====================
                # 维度：
                # 1×1
                x_i_tensor = torch.tensor(
                    x_i_input,
                    dtype=torch.float32
                )

                # =========
                # 信息融合
                # =========
                # 1×101 -> 1×100
                nn1_input = fusion_layer(
                    fusion_input
                )

                # ===========
                # NN1输入准备
                # ===========
                # 当前：1×100
                # 转换：100×1
                nn1_input = nn1_input.T

                # =====
                # NN1
                # =====
                # 输入： 100×1
                # 输出： 100×32
                h1 = nn1(
                    nn1_input
                )

                # ==================
                # 取当前节点对应的B列
                # ==================
                # 当前节点是 target_node
                # 取：B[:, target_node]
                # 原始维度：(100,)
                B_column = torch.tensor(
                    B[:, target_node],
                    dtype=torch.float32
                )

                # ==========
                # B列整理维度
                # ==========
                # (100,) -> (100,1)
                B_column = B_column.reshape(
                    100,
                    1
                )

                # (100,1)-> (1,100)
                B_column = B_column.T

                # =============
                # B筛选邻居信息
                # =============
                # B_column： 1×100
                # h1：100×32
                # 矩阵乘法： (1×100)(100×32)
                # 得到： 1×32
                neighbor_sum = torch.matmul(
                    B_column,
                    h1
                )

                # ======
                # NN2
                # ======

                # 输入：1×32
                # 输出：1×32

                h2 = nn2(
                    neighbor_sum
                )

                # =============
                # 构造 NN3 输入
                # =============
                # h2：1×32
                # x_i_tensor：1×1
                # 拼接：1×33
                nn3_input = torch.cat(
                    (
                        h2,
                        x_i_tensor
                    ),
                    dim=1
                )

                # =========
                # NN3预测
                # =========
                # 输入：1×33
                # 输出：1×3
                # 三个logits分别对应：
                # 0 -> S
                # 1 -> I
                # 2 -> R
                prediction = nn3(
                    nn3_input
                )

                # ============
                # 构造真实标签
                # ============
                # target_next：
                # 0 / 1 / 2
                # CrossEntropyLoss要求：
                # dtype = long
                # shape = [1]
                target_label = torch.tensor(
                    [target_next],
                    dtype=torch.long
                )

                # ========
                # 计算损失
                # ========
                loss = criterion(
                    prediction,
                    target_label
                )

                # =========
                # 反向传播
                # =========
                # 注意：backward就在node循环里面,当前node算完loss以后立即反向传播
                loss.backward()

                # =========
                # 更新参数
                # =========
                # 当前node反向传播以后,立即更新参数
                optimizer.step()

                # ==================
                # 得到最终SIR预测状态
                # ==================
                # 从三个logits中
                # 取最大值对应的位置
                # 得到： 0 / 1 / 2
                state_prediction = torch.argmax(
                    prediction,
                    dim=1
                )

                # =========
                # 记录loss
                # =========
                epoch_loss += loss.item()
                sample_count += 1

    # ==================
    # 当前epoch平均loss
    # ==================
    average_loss = (
        epoch_loss
        / sample_count
    )

    # ===========
    # 打印训练结果
    # ===========
    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Average Loss = {average_loss:.6f}"
    )