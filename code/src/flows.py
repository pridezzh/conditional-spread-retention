# -*- coding: utf-8 -*-
"""条件流匹配（conditional flow matching / rectified flow）最小实现。

代码来源说明（缝合记录，README 与论文 §5.1 会引用）：
  - 训练目标与采样器**按公开论文的算法描述从零实现**：
      条件流匹配  Lipman et al., ICLR 2023（arXiv:2210.02747）式 (5)-(8)
      整流流/重流  Liu et al., ICLR 2023（arXiv:2209.03003）Alg. 1-2
      小批量 OT 耦合 Tong et al., TMLR 2024（arXiv:2302.00482）§4.1
      时间位移调度 Esser et al., ICML 2024（arXiv:2403.03206）§3.1 + C20 的高噪声偏置
  - 我们没有复用任何作者的仓库源码，但复用了上述**公开发表的目标函数定义**；
    所有改造点（上下文拼接方式、时间采样、求解器、重流开关）都在 cfg 里显式记录。
"""
import numpy as np
import torch
import torch.nn as nn

from common import DEVICE


# ---------------------------------------------------------------- 模型
class SinusoidalTime(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t[:, None] * freq[None, :] * 2 * np.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class MLPCondVelocity(nn.Module):
    """v_theta(x_t, t, c)。输入为 [x, sin/cos(t), c]。"""

    def __init__(self, x_dim, c_dim, hidden=256, n_layers=4, time_dim=32):
        super().__init__()
        self.temb = SinusoidalTime(time_dim)
        dims = [x_dim + time_dim + c_dim] + [hidden] * (n_layers - 1) + [x_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, c):
        if c is None:
            inp = torch.cat([x, self.temb(t)], dim=-1)
        else:
            inp = torch.cat([x, self.temb(t), c], dim=-1)
        return self.net(inp)


class SmallConvVelocity(nn.Module):
    """图像用的小型卷积速度场（消融：架构是否改变结论）。"""

    def __init__(self, side, c_dim, hidden=64, time_dim=32):
        super().__init__()
        self.side = side
        self.temb = SinusoidalTime(time_dim)
        self.fc_in = nn.Linear(time_dim + c_dim, side * side)
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1))

    def forward(self, x, t, c):
        n = x.shape[0]
        e = self.temb(t)
        if c is None:
            cc = e
        else:
            cc = torch.cat([e, c], dim=-1)
        z = self.fc_in(cc).view(n, 1, self.side, self.side)
        inp = torch.cat([x.view(n, 1, self.side, self.side), z], dim=1)
        return self.net(inp).view(n, -1)


def build_model(x_dim, c_dim, arch="mlp", hidden=256, n_layers=4, side=None):
    if arch == "mlp":
        return MLPCondVelocity(x_dim, c_dim, hidden=hidden, n_layers=n_layers).to(DEVICE)
    if arch == "conv":
        return SmallConvVelocity(side, c_dim, hidden=hidden).to(DEVICE)
    raise ValueError(arch)


# ---------------------------------------------------------------- 时间采样
def sample_time(n, mode="uniform", alpha=1.0, mu=0.0, sigma=1.0, device=DEVICE):
    """t ∈ [0,1]；t 越接近 0 噪声越大。

    uniform     : t ~ U[0,1]
    power       : t = u^alpha，alpha>1 时向高噪声（t≈0）偏移（C20 / SD3 思路）
    logitnormal : t = sigmoid(N(mu, sigma^2))
    """
    u = torch.rand(n, device=device)
    if mode == "uniform":
        return u
    if mode == "power":
        return u ** alpha
    if mode == "logitnormal":
        z = torch.randn(n, device=device) * sigma + mu
        return torch.sigmoid(z)
    raise ValueError(mode)


# ---------------------------------------------------------------- 训练
def train_fm(model, C, X, n_steps=20000, batch=512, lr=3e-4, time_mode="uniform",
             time_alpha=1.0, time_mu=0.0, time_sigma=1.0, weight_decay=0.0,
             coupling="independent", seed=0, log_every=2000, verbose=False,
             X0_fixed=None):
    """条件流匹配训练。x0 ~ N(0, I)，与 x1 独立（或 mini-batch OT）。

    X0_fixed: 若给出，则**使用给定的源样本**（重流/2-RF 第二轮需要：
    源样本与目标是上一轮 ODE 生成的配对，不能再重新采样噪声）。
    """
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Ct = torch.as_tensor(C, dtype=torch.float32, device=DEVICE)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    X0t = None
    if X0_fixed is not None:
        X0t = torch.as_tensor(X0_fixed, dtype=torch.float32, device=DEVICE)
        assert len(X0t) == len(Xt)
    if C.shape[1] == 0:
        Ct = None
    n = X.shape[0]
    idx = np.arange(n)
    losses = []
    n_epoch = max(1, n_steps * batch // n)
    step = 0
    model.train()
    for ep in range(n_epoch):
        rng.shuffle(idx)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            x1 = Xt[b]
            if X0t is not None:
                x0 = X0t[b]
            else:
                x0 = torch.randn_like(x1)
            if coupling == "ot" and x1.shape[0] <= 512:
                x0, x1 = _minibatch_ot(x0, x1)
            t = sample_time(x1.shape[0], time_mode, time_alpha, time_mu, time_sigma)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            tgt = x1 - x0
            c = None if Ct is None else Ct[b]
            v = model(xt, t, c)
            loss = ((v - tgt) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
            if verbose and step % log_every == 0:
                print("    step %d/%d loss %.4f" % (step, n_steps, losses[-1]))
            if step >= n_steps:
                break
        if step >= n_steps:
            break
    return model, float(np.mean(losses[-200:]))


def _minibatch_ot(x0, x1):
    """小批量最优传输耦合（匈牙利算法，n≤512 时精确）。"""
    from scipy.optimize import linear_sum_assignment
    with torch.no_grad():
        a = x0.detach().cpu().numpy()
        b = x1.detach().cpu().numpy()
        cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        r, cidx = linear_sum_assignment(cost)
        x0n = x0[r]
        x1n = x1[cidx]
    return x0n.detach(), x1n.detach()


# ---------------------------------------------------------------- 采样
@torch.no_grad()
def sample_euler(model, x0, c, n_steps=1, solver="euler", clip=None):
    """从 x0 出发用 n_steps 步求解 ODE。n_steps=1 即"一步生成"。"""
    model.eval()
    x = x0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        v = model(x, t, c)
        if solver == "euler":
            x = x + dt * v
        elif solver == "heun":
            t2 = torch.full((x.shape[0],), (i + 1) * dt, device=x.device)
            x2 = x + dt * v
            v2 = model(x2, t2, c)
            x = x + dt * 0.5 * (v + v2)
        else:
            raise ValueError(solver)
        if clip is not None:
            x = torch.clamp(x, -clip, clip)
    return x


@torch.no_grad()
def reflow_pairs(model, C, n, n_steps=64, x_dim=2, seed=0):
    """重流（2-RF）：用已训练模型生成 (x0, x1) 配对，作为下一轮的耦合。"""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, x_dim)).astype(np.float32)
    xt = torch.as_tensor(x0, dtype=torch.float32, device=DEVICE)
    ct = None if C is None else torch.as_tensor(C, dtype=torch.float32, device=DEVICE)
    x1 = sample_euler(model, xt, ct, n_steps=n_steps).cpu().numpy()
    return (None if C is None else C), x0.astype(np.float32), x1.astype(np.float32)
