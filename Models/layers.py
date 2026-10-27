"""共享的消息编码、聚合后变换和输出层。"""

import torch.nn as nn
import torch.nn.functional as F


class NN1(nn.Module):
    # 对一个候选交互编码，输入维度由调用者指定。
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class NN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.e2n = nn.Linear(input_dim, hidden_dim)
        self.n2n = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.e2n(x))
        x = F.relu(self.n2n(x))
        return x


class NN3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.output(x)
