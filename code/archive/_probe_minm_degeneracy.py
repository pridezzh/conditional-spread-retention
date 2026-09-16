# -*- coding: utf-8 -*-
"""取证脚本：证明 min-of-M 分支（旧版）与点式 L2 **逐比特等价**。

背景
----
results/method_map.json 里 onestep_l2@1 与 onestep_minM@1 的 rho 与 rho_std
在 3 个种子上**逐比特相同**（0.00010386879583898434 / 0.00011148913286755626）。
这不可能来自两次独立训练，只可能是两条代码路径算的是同一个损失。

原因
----
旧版 min-of-M 用 `X0 = np.repeat(x0, M, axis=0)` 造 M 个候选：同一个 x0、
同一个 t=0、同一个 c。一步网络是**确定性映射**（MLP，无 dropout/BN），
所以 M 次前向给出**完全相同**的输出，min 的 argmin 恒为 0，损失恒等于
在同一个 batch 上算的点式 L2 —— 与"是否塌缩"无关。

本脚本用 200 步小规模训练，逐参数比较两种损失训练出的网络：
  * 旧版实现：应完全相同（max|Δw| = 0）；
  * 修正实现（每个候选独立抽源噪声）：应明显不同。
"""
import os
import sys


def _find_root(d):
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("root not found")
        d = p


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_method_map as M  # noqa: E402


def flat_params(net):
    return np.concatenate([p.detach().numpy().ravel() for p in net.parameters()])


def train(steps, min_m):
    M.STEPS = steps
    return M.train_onestep(lambda bs, rng: M.sample_cond(bs, rng), 0,
                           tag="probe", min_m=min_m)


STEPS = 200
print("== 同种子同批数：min_m=1 vs min_m=%d，%d 步 ==" % (M.M_MIN, STEPS))
net_l2 = train(STEPS, 1)
net_mm = train(STEPS, M.M_MIN)
a, b = flat_params(net_l2), flat_params(net_mm)
diff = float(np.abs(a - b).max())
print("  max|Δw| = %.3e   allclose = %s" % (diff, bool(np.allclose(a, b, atol=0, rtol=0))))
if diff == 0.0:
    print("  >>> 结论：旧版 min-of-M 与点式 L2 **逐比特相同**，该行不构成对 IMLE 的检验。")
else:
    print("  >>> 结论：两条代码路径已可区分（修正生效）。")
