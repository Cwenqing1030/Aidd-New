import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ================
# 第一个神经网络 NN1
# ================
class NN1(nn.Module):
    def __init__(self, hidden_dim, n, d):
        super().__init__()

        self.linear1 = nn.Linear(n + 1, n)
        # 1维节点状态 -> F维隐藏特征
        self.linear = nn.Linear(d, hidden_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = x.T
        x = F.relu(self.linear(x))

        return x


# ================
# 第二个神经网络 NN2
# ================
class NN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

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
        super().__init__()

        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.output(x)
        return x


# ==========
# 读取数据
# ==========
data = np.load("data/generated_data.npz")
all_Xt = data["all_Xt"]
all_A = data["all_A"]
graph_types = data["graph_types"]


# ===============
# 选择ER网络的数据
# ===============
er_mask = graph_types == "ER"
Xt_er = all_Xt[er_mask]
A_er = all_A[er_mask]


# ===================
# 选择一个ER网络
# ===================
graph_idx = 0
# 当前ER网络状态序列
Xt = Xt_er[graph_idx]
# 当前ER网络真实邻接矩阵
A_true = A_er[graph_idx]
print("==============================")
print(f"开始恢复 ER 网络 {graph_idx}")
print("==============================")


# =========
# 创建 NN1
# =========
# 隐藏维度 F
F_dim = 32
n = 100
d = 1
nn1 = NN1(F_dim, n, d).cuda()


# ===================
# 创建可训练邻接矩阵
# ===================
theta = nn.Parameter(torch.randn(n,n).cuda())

# 不允许节点自连接
identity_mask = 1.0 - torch.eye(n,device=theta.device)


# =========
# 创建 NN2
# =========
# NN1最终输出的特征维度为F
# A_hat筛选以后仍然得到1×F
# 所以NN2：1×F -> 1×F
nn2_input_dim = F_dim
nn2_hidden_dim = F_dim
nn2_output_dim = F_dim

nn2 = NN2(
    input_dim=nn2_input_dim, hidden_dim=nn2_hidden_dim, output_dim=nn2_output_dim
).cuda()


# =========
# 创建 NN3
# =========
# NN2输出：h2 = 1×F
# 当前节点状态：x_i^t = 1×1
# 拼接以后：1×(F+1)
nn3_input_dim = F_dim + 1

# SIR三个类别：
# 0 -> S
# 1 -> I
# 2 -> R
nn3_output_dim = 3

nn3 = NN3(input_dim=nn3_input_dim, output_dim=nn3_output_dim).cuda()


# ===========
# 定义损失函数
# ===========
criterion = nn.CrossEntropyLoss()


# ===========
# 定义优化器
# ===========
optimizer = torch.optim.Adam(
    list(nn1.parameters()) + list(nn2.parameters()) + list(nn3.parameters()) + [theta],
    lr=0.01,
)


# ====
# 训练
# ====
num_epochs = 100
RESET_INTERVAL = 100

# =============
# 第一层：epoch
# =============
for epoch in range(num_epochs):

    # 用来统计当前epoch的loss
    epoch_loss = 0.0

    # 当前epoch一共训练了多少个节点样本
    sample_count = 0


    # ============
    # 第二层：time
    # ============
    # Xt共有1000个时间点
    # t=0   -> t=1
    # t=1   -> t=2
    # ...
    # t=998 -> t=999
    # 这里只遍历到倒数第二个时刻
    # 每100个时间步的人为重置转移不参与训练
    for t in range(Xt.shape[0] - 1):
        # 跳过人为重置产生的状态转移
        if (t + 1) % RESET_INTERVAL == 0:
            continue

        # 当前时刻整个网络状态 X^t
        # 原始维度：(100,)
        X_t = Xt[t]


        # ============
        # 第三层：node
        # ============
        # 遍历当前网络的100个节点
        for target_node in range(Xt.shape[1]):

            # 每个节点开始前清空梯度
            optimizer.zero_grad()

            # 当前节点状态
            # 当前时刻目标节点状态 x_i^t
            x_i_t = Xt[t,target_node]

            # 下一时刻目标节点真实状态 x_i^(t+1)
            target_next = Xt[t+1,target_node]


            # =========
            # 整理 X^t
            # =========
            # X_t：
            # (100,) -> (1,100)
            X_t_input = X_t.reshape(1,-1)


            # =====================
            # 整理 x_i^t
            # =====================
            # 标量 -> 1×1
            x_i_input = np.array([[x_i_t]])


            # =============
            # 构造融合层输入
            # =============
            # x_i^t：1×1
            # X^t：1×100
            # 拼接：1×101
            NN1_input = np.concatenate((x_i_input,X_t_input),axis=1)


            # numpy -> torch
            nn1_input = torch.tensor(NN1_input,dtype=torch.float32).cuda()


            # =====================
            # 当前节点状态转Tensor
            # =====================
            # 维度：1×1
            x_i_tensor = torch.tensor(x_i_input,dtype=torch.float32).cuda()


            # =====
            # NN1
            # =====
            # 输入：1×101
            # 输出：100×32
            h1 = nn1(nn1_input)


            # =======================
            # 生成可学习的邻接概率矩阵
            # =======================
            # theta中的任意实数
            # 经过sigmoid后映射到0~1
            A_hat = torch.sigmoid(theta)

            # 对角线强制为0
            A_hat = A_hat * identity_mask


            # ======================
            # 取目标节点对应的列
            # ======================
            # A_hat[:,target_node]
            # 维度：(100,)
            A_column = A_hat[:,target_node]

            # (100,) -> (1,100)
            A_column = A_column.reshape(1,n)


            # ======================
            # 聚合邻居信息
            # ======================
            # A_column：1×100
            # h1：100×32
            # 得到：neighbor_sum：1×32
            neighbor_sum = torch.matmul(A_column,h1)


            # ======
            # NN2
            # ======
            # 输入：1×32
            # 输出：1×32
            h2 = nn2(neighbor_sum)


            # =============
            # 构造 NN3 输入
            # =============
            # h2：1×32
            # x_i_tensor：1×1
            # 拼接：1×33
            nn3_input = torch.cat((h2,x_i_tensor),dim=1)


            # =========
            # NN3预测
            # =========
            # 输入：1×33
            # 输出：1×3
            # 三个logits分别对应：
            # 0 -> S
            # 1 -> I
            # 2 -> R
            prediction = nn3(nn3_input)


            # ============
            # 构造真实标签
            # ============
            # target_next：
            # 0 / 1 / 2
            # CrossEntropyLoss要求：
            # dtype = long
            # shape = [1]
            target_label = torch.tensor([target_next],dtype=torch.long).cuda()


            # ========
            # 计算损失
            # ========
            loss = criterion(prediction,target_label)


            # =========
            # 反向传播
            # =========
            # 当前node算完loss以后立即反向传播
            loss.backward()


            # =========
            # 更新参数
            # =========
            # 当前node反向传播以后立即更新参数
            optimizer.step()


            # ==================
            # 得到最终SIR预测状态
            # ==================
            # 从三个logits中取最大值对应的位置
            # 得到：0 / 1 / 2
            state_prediction = torch.argmax(prediction,dim=1)


            # =========
            # 记录loss
            # =========
            epoch_loss += loss.item()
            sample_count += 1


    # ==================
    # 当前epoch平均loss
    # ==================
    average_loss = epoch_loss / sample_count


    # ===========
    # 打印训练结果
    # ===========
    print(f"ER_{graph_idx} Epoch {epoch+1}/{num_epochs}, Average Loss = {average_loss:.6f}")


# =============================
# 训练结束，得到最终邻接概率矩阵
# =============================
with torch.no_grad():
    A_hat_final = torch.sigmoid(theta)

    # 去除自环
    A_hat_final = A_hat_final * identity_mask


# =========================
# 恢复0/1邻接矩阵
# =========================
threshold = 0.5
A_recovered = (A_hat_final >= threshold).float()


# ==========
# 输出结果
# ==========
print("最终邻接概率矩阵 A_hat_final：",A_hat_final)
print("恢复后的0/1邻接矩阵 A_recovered：",A_recovered)


# =========================
# 只取邻接矩阵上三角部分
# =========================
upper_indices = np.triu_indices(n,k=1)

A_true_eval = A_true_np[upper_indices]
A_hat_eval = A_hat_np[upper_indices]
A_recovered_eval = A_recovered_np[upper_indices]


# =========================
# 计算评价指标
# =========================
roc_auc = roc_auc_score(A_true_eval,A_hat_eval)
pr_auc = average_precision_score(A_true_eval,A_hat_eval)

precision = precision_score(A_true_eval,A_recovered_eval,zero_division=0)
recall = recall_score(A_true_eval,A_recovered_eval,zero_division=0)
f1 = f1_score(A_true_eval,A_recovered_eval,zero_division=0)
accuracy = accuracy_score(A_true_eval,A_recovered_eval)


# =========================
# 计算SHD
# =========================
shd = np.sum(A_true_eval != A_recovered_eval)


# =========================
# 输出评价指标
# =========================
print()
print("==============================")
print("网络结构恢复评价结果")
print("==============================")
print(f"ROC-AUC = {roc_auc:.6f}")
print(f"PR-AUC = {pr_auc:.6f}")
print(f"Precision = {precision:.6f}")
print(f"Recall = {recall:.6f}")
print(f"F1 = {f1:.6f}")
print(f"Accuracy = {accuracy:.6f}")
print(f"SHD = {shd}")