# -*- coding: utf-8 -*-
"""实验三：多目标连续控制上的离线策略提取（闭环回报）。

受控维度（对应 C19/C20 的消融轴）：
  ctx   条件强度：exact / noisy(p=0.5) / partial(象限) / none（完全不给目标）
  K     目标数（决定条件动作分布的模态数）
  H     动作块 horizon（1 / 4 / 8），对应"拉长动作 horizon 会抹掉单步增益"
  NFE   采样步数 1 / 2 / 4 / 8 / 32

同一组预抽回合（相同初始状态、相同目标、相同观测掩码）上比较不同 NFE，
因此不同 NFE 的回报差是**配对比较**，方差远小于独立评估。
"""
import argparse
import os
import sys
import time
import warnings

# 必须在 import numpy/torch **之前**设置：MKL/OMP 运行时在库加载时就完成初始化，
# 之后再设 OMP_NUM_THREADS 是无效的；缺 KMP_DUPLICATE_LIB_OK 会直接
# "OMP: Error #15 ... libiomp5md.dll already initialized" 崩溃（本项目实测踩过）。
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
from tasks import MultiGoalNav  # noqa: E402

NFES = [1, 2, 4, 8, 32]


def run_cell(ctx="none", K=6, H=1, seed=0, n_train=30000, train_steps=8000,
             batch=512, lr=1e-3, hidden=256, n_layers=4, n_ep=200,
             time_mode="uniform", time_alpha=1.0, verbose=False, threads=4,
             max_T=40):
    set_seed(seed)
    torch.set_num_threads(threads)
    env = MultiGoalNav(K=K, seed=seed)
    C, A, S, G = env.build_dataset(n=n_train, ctx=ctx, p_obs=0.5, H=H, seed=seed)
    a_dim, c_dim = A.shape[1], C.shape[1]

    # 1) 训练前判别量
    t0 = time.time()
    disc = compute_discriminant(C[:20000], A[:20000], C[-2000:], A[-2000:],
                                estimator="knn", k=10, kmax=10, seed=seed)
    t_disc = time.time() - t0

    # 2) 行为克隆：条件流匹配
    model = build_model(a_dim, c_dim, hidden=hidden, n_layers=n_layers)
    tt = Timer()
    model, loss = train_fm(model, C, A, n_steps=train_steps, batch=batch, lr=lr,
                           time_mode=time_mode, time_alpha=time_alpha, seed=seed)
    t_train = tt.elapsed()

    # 3) 配对回合上评估各 NFE（**批量闭环**：所有回合同时推进，
    #    语义与逐回合完全一致，只是把"一次一个前向"换成"一次一批前向"）
    episodes = env.sample_episodes(n_ep, seed=seed + 4242, p_obs=0.5)
    out = dict(ctx=ctx, K=K, H=H, seed=seed, disc=disc, loss=loss,
               t_disc=t_disc, t_train=t_train, n_ep=n_ep, nfe={})

    def eval_nfe(nfe, rng_seed):
        rng = np.random.default_rng(rng_seed)
        s0, gi, reveal = episodes
        n = len(s0)
        S = s0.astype(np.float64).copy()
        done = np.zeros(n, dtype=bool)
        ret = np.zeros(n)
        mind = np.linalg.norm(S - env.goals[gi], axis=1)
        t = 0
        while t < max_T and not done.all():
            C = np.concatenate([env.context_one(S[i], int(gi[i]), ctx,
                                                 bool(reveal[i])) for i in range(n)], 0)
            ct = torch.as_tensor(C, dtype=torch.float32, device=DEVICE)
            x0 = torch.as_tensor(rng.normal(size=(n, a_dim)), dtype=torch.float32,
                                 device=DEVICE)
            with torch.no_grad():
                a = sample_euler(model, x0, ct, n_steps=nfe)
            a = a.detach().cpu().numpy().astype(np.float64).reshape(n, -1, 2)
            for h in range(a.shape[1]):
                ah = a[:, h, :]
                nrm = np.linalg.norm(ah, axis=1, keepdims=True)
                ah = np.where(nrm > 0.12, ah / np.maximum(nrm, 1e-9) * 0.12, ah)
                act = ~done
                S[act] = np.clip(S[act] + ah[act], -1.2, 1.2)
                t += 1
                cur = np.linalg.norm(S - env.goals[gi], axis=1)
                mind = np.minimum(mind, cur)
                d = act & (cur < 0.12)
                ret[d] += 1.0
                done = done | d
                if t >= max_T:
                    break
        # 成功率为稀疏指标（会饱和），故同时报告**到真实目标的距离**：
        # 这是不受饱和影响、直接反映动作质量的连续指标。
        return (float(done.mean()), float(ret.mean()),
                float(mind.mean()),
                float(np.linalg.norm(S - env.goals[gi], axis=1).mean()),
                done.astype(float))

    for nfe in NFES:
        succ, ret, md, fd, arr = eval_nfe(nfe, seed + 31)
        out["nfe"][str(nfe)] = dict(success=float(succ), ret=float(ret),
                                    min_dist=float(md), final_dist=float(fd),
                                    succ_arr=arr.tolist())
    out["succ_1"] = out["nfe"]["1"]["success"]
    out["succ_32"] = out["nfe"]["32"]["success"]
    out["succ_gap"] = out["nfe"]["32"]["success"] - out["nfe"]["1"]["success"]
    out["dist_1"] = out["nfe"]["1"]["min_dist"]
    out["dist_32"] = out["nfe"]["32"]["min_dist"]
    out["dist_gap"] = out["nfe"]["1"]["min_dist"] - out["nfe"]["32"]["min_dist"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    cells = []
    for ctx in ["exact", "noisy", "partial", "none"]:
        cells.append(dict(ctx=ctx, K=6, H=1))
    for K in [2, 12]:
        cells.append(dict(ctx="none", K=K, H=1))
    for H in [4, 8]:
        cells.append(dict(ctx="none", K=6, H=H))
    if args.quick:
        cells = [dict(ctx="none", K=6, H=1)]
        args.seeds = [0]
        args.steps = min(args.steps, 3000)

    for cell in cells:
        for seed in args.seeds:
            t = Timer()
            res = run_cell(ctx=cell["ctx"], K=cell["K"], H=cell["H"], seed=seed,
                           train_steps=args.steps, threads=args.threads)
            tag = "rl_%s_K%d_H%d_s%d" % (cell["ctx"], cell["K"], cell["H"], seed)
            save_result(tag, res)
            print("%-28s D=%.3f Lam=%.2f | succ1=%.3f succ32=%.3f (gap %+.3f) | "
                  "dist1=%.3f dist32=%.3f (excess %+.3f) | %.1fs"
                  % (tag, res["disc"]["D_hat"], res["disc"]["Lam_hat"],
                     res["succ_1"], res["succ_32"], res["succ_gap"],
                     res["dist_1"], res["dist_32"], res["dist_gap"],
                     t.elapsed()), flush=True)


if __name__ == "__main__":
    main()
