# 导入 PyTorch；这个文件只做张量公式计算，不依赖别的模块。
import torch


def contagion_F(x: torch.Tensor, B: torch.Tensor, beta: float, delta: float, dt: float, eps: float = 1e-12) -> torch.Tensor:
    r"""
    Eq. (16)/(17): contagion process iterative equation.

    x(t+1) = (1 - delta * dt) * x(t)
             + dt * beta * ((1 - x(t)) / x(t)) * B e_prod

    e_prod[j] = prod_k x_k(t)^{B[k,j]}

    Args:
        x: node states, [N] or [N, batch].
        B: incidence or surrogate matrix, [N, E].
    """
    # 先把 x 中过小的值截断到 eps，避免后面对 `log(x)` 时出现 log(0) 的数值错误。
    x_safe = torch.clamp(x, min=eps)
    # 如果输入是一维向量，说明当前只处理“单个时刻的一组节点状态”。
    if x_safe.dim() == 1:
        # 根据公式 e_prod[j] = prod_k x_k^{B[k,j]} 计算每条超边上的乘积项。
        # 这里用 `exp(log(x) @ B)` 是把乘积改写成“对数求和再指数”的稳定写法。
        e_prod = torch.exp(torch.matmul(torch.log(x_safe), B))
        # 把每条超边的影响再聚合回节点层面，得到每个节点接收到的总传播效应。
        Be_prod = torch.matmul(B, e_prod)
        # 返回下一时刻状态：
        # 第一项是当前状态在衰减/恢复后的保留量；
        # 第二项是超图传播带来的新增变化量。
        return (1.0 - delta * dt) * x_safe + dt * beta * ((1.0 - x_safe) / x_safe) * Be_prod
    # 如果输入是二维矩阵，说明一次并行处理多个状态列，形状通常是 [N, batch]。
    if x_safe.dim() == 2:
        # 批量版本的超边乘积项计算；这里转置 B 是为了让矩阵维度能正确对齐。
        e_prod = torch.exp(torch.matmul(B.T, torch.log(x_safe)))
        # 把批量超边效应再映射回节点层面。
        Be_prod = torch.matmul(B, e_prod)
        # 返回批量输入对应的下一时刻状态。
        return (1.0 - delta * dt) * x_safe + dt * beta * ((1.0 - x_safe) / x_safe) * Be_prod
    # 当前实现只接受一维或二维输入，别的维度没有定义明确的物理含义。
    raise ValueError("x must be a 1D tensor [N] or a 2D tensor [N, batch].")
