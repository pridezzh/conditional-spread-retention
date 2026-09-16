# -*- coding: utf-8 -*-
"""实验二：条件图像（MNIST 16×16）—— 目标边际不变，只改变条件强度。

条件阶梯（同一批目标图像，条件从强到弱）：
  full  : 8×8 粗化 + 标签
  mid   : 4×4 粗化 + 标签
  half  : 左半张图的 4×4 粗化 + 标签
  weak  : 只有标签
  none  : 无条件

指标：边际切片 W2、PCA-64 空间的 PRDC（Kynkaanniemi et al. 2019）、
      逐上下文条件切片 W2。条件参考集只能由脚本实际提供给模型的条件 c 构造；
      禁止额外使用测试标签。
"""
import argparse
import os
import sys
import time
import warnings

# 必须在 import numpy/torch **之前**设置（同一坑：run_rl.py 曾因此 OMP Error #15 崩溃）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """向上找到同时含 code/ 与 paper/ 的一级；与脚本自身深度无关。"""
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d


_ROOT_ = _find_root(_HERE)
_CODE_ = os.path.join(_ROOT_, "code")
sys.path.insert(0, os.path.join(_CODE_, "src"))
sys.path.insert(0, _HERE)
from common import DEVICE, Timer, save_result, set_seed  # noqa: E402
from discriminant import compute_discriminant  # noqa: E402
from flows import build_model, sample_euler, train_fm  # noqa: E402
from metrics import conditional_reference_samples, prdc, sliced_wasserstein  # noqa: E402
from tasks import load_mnist, make_image_condition  # noqa: E402

NFES = [1, 2, 4, 8, 16, 32]
MODES = ["full", "mid", "half", "weak", "none"]


def run_cell(mode="mid", seed=0, size=16, n_train=20000, n_test=2000,
             train_steps=5000, batch=256, lr=1e-3, hidden=384, n_layers=3,
             time_mode="uniform", time_alpha=1.0, arch="mlp",
             n_ctx=200, m_per=8, verbose=False, threads=0):
    set_seed(seed)
    torch.set_num_threads(int(threads) if threads > 0
                          else min(8, os.cpu_count() or 4))
    xtr, ytr, xte, yte, data_meta = load_mnist(size=size, return_metadata=True)
    xtr = (xtr[:n_train] * 2 - 1).astype(np.float32)
    xte = (xte[:n_test] * 2 - 1).astype(np.float32)
    ytr, yte = ytr[:n_train], yte[:n_test]
    Xtr = xtr.reshape(len(xtr), -1)
    Xte = xte.reshape(len(xte), -1)
    Ctr = make_image_condition(xtr, ytr, mode)
    Cte = make_image_condition(xte, yte, mode)
    x_dim, c_dim = Xtr.shape[1], Ctr.shape[1]

    # 1) 训练前判别量
    t0 = time.time()
    disc = compute_discriminant(Ctr[:12000], Xtr[:12000], Cte[:1500], Xte[:1500],
                                estimator="knn", k=10, kmax=8, seed=seed)
    t_disc = time.time() - t0

    # 2) 训练
    model = build_model(x_dim, c_dim, arch=arch, hidden=hidden, n_layers=n_layers,
                        side=size)
    tt = Timer()
    model, loss = train_fm(model, Ctr, Xtr, n_steps=train_steps, batch=batch, lr=lr,
                           time_mode=time_mode, time_alpha=time_alpha, seed=seed)
    t_train = tt.elapsed()

    # 3) 采样与指标
    rng = np.random.default_rng(seed + 5)
    sel = rng.choice(len(Xte), n_ctx, replace=False)
    Csel = torch.as_tensor(Cte[sel], dtype=torch.float32, device=DEVICE)
    X0 = rng.normal(size=(n_ctx, m_per, x_dim)).astype(np.float32)
    X0f = torch.as_tensor(X0.reshape(-1, x_dim), dtype=torch.float32, device=DEVICE)
    Crep = Csel.repeat_interleave(m_per, dim=0)

    # 每个查询条件的真实参考集只由 Cte 决定。旧实现始终按 yte 类别取图，
    # 这在 full/mid/half 中把条件扩大成了整类分布，在 none 中更是直接泄漏标签。
    real_by_context = conditional_reference_samples(
        Cte, Xte, Cte[sel], n_ref=min(256, len(Xte) - 1), seed=seed + 101,
        exclude_indices=sel)
    out = dict(mode=mode, seed=seed, arch=arch, hidden=hidden, steps=train_steps,
               disc=disc, loss=loss, t_disc=t_disc, t_train=t_train, nfe={},
               schema_version=2, metric_protocol="condition-knn-v2-no-label-leakage",
               dataset_source=data_meta["dataset_source"], n_contexts=n_ctx,
               references_per_context=min(256, len(Xte) - 1))
    for nfe in NFES:
        gen = sample_euler(model, X0f, Crep, n_steps=nfe)
        gen = gen.detach().cpu().numpy().astype(np.float64)
        pooled = gen.reshape(-1, x_dim)
        sw_marg = sliced_wasserstein(Xte[:2000].astype(np.float64), pooled,
                                     n_proj=512, seed=seed)
        # 对全部查询上下文计算条件 SW；不再只取每 4 个中的 1 个。
        cs = []
        for i, real in enumerate(real_by_context):
            cs.append(sliced_wasserstein(real, gen[i * m_per:(i + 1) * m_per],
                                         n_proj=256, seed=seed + i))
        csw = float(np.mean(cs))
        p = prdc(Xte[:2000].astype(np.float64), pooled, k=5, seed=seed)
        out["nfe"][str(nfe)] = dict(sw_marg=sw_marg, csw=csw,
                                    precision=p["precision"], recall=p["recall"],
                                    coverage=p["coverage"])
    out["sw_1"] = out["nfe"]["1"]["sw_marg"]
    out["sw_32"] = out["nfe"]["32"]["sw_marg"]
    out["sw_gap"] = out["sw_1"] - out["sw_32"]
    out["recall_1"] = out["nfe"]["1"]["recall"]
    out["recall_32"] = out["nfe"]["32"]["recall"]
    out["recall_gap"] = out["nfe"]["32"]["recall"] - out["nfe"]["1"]["recall"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--hidden", type=int, default=384,
                    help="宽度；conv 架构下是通道数（64 足够且快得多）")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-suffix", default="", help="结果文件名后缀，避免覆盖")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    import torch as _t
    _t.set_num_threads(max(1, int(args.threads)))
    if args.quick:
        args.modes = ["mid"]
        args.seeds = [0]
        args.steps = min(args.steps, 800)
    for mode in args.modes:
        for seed in args.seeds:
            t = Timer()
            res = run_cell(mode=mode, seed=seed, train_steps=args.steps,
                           arch=args.arch, hidden=args.hidden,
                           threads=args.threads)
            tag = "mnist_%s_%s%s_s%d" % (mode, args.arch, args.out_suffix, seed)
            save_result(tag, res)
            print("%-24s D=%.3f Lam=%.2f K*=%d | sw1=%.3f sw32=%.3f rec1=%.2f "
                  "rec32=%.2f | %.1fs"
                  % (tag, res["disc"]["D_hat"], res["disc"]["Lam_hat"],
                     res["disc"]["K_star"], res["sw_1"], res["sw_32"],
                     res["recall_1"], res["recall_32"], t.elapsed()), flush=True)


if __name__ == "__main__":
    main()
