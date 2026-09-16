# -*- coding: utf-8 -*-
"""实验一：二维（及 8 维）环形混合高斯 —— 真值模态结构已知。

每个单元做的事（严格顺序，保证"训练前判别量"确实在训练前算）：
  1. 生成数据 -> 2. 算判别量 D-hat / Lam-hat（只用数据） -> 3. 训练条件流匹配
  -> 4. 用**同一组初始噪声 x0** 在 NFE ∈ {1,2,4,8,16,32,128} 下采样
  -> 5. 计算条件 SW、边际 SW、模态召回/塌缩、条件方差实现比（spread ratio）
  -> 6. 全部落盘 results/toy_<tag>_s<seed>.json

用法:
  python run_toy.py --quick                 # 单单元冒烟测试
  python run_toy.py --grid                  # 完整网格
  python run_toy.py --cell 0 --seeds 0 1 2  # 指定单元
"""
import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 必须在 import numpy 之前
os.environ.setdefault("OMP_NUM_THREADS", "8")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402

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
from discriminant import compute_discriminant, fit_conditional_mean  # noqa: E402
from flows import build_model, sample_euler, train_fm  # noqa: E402
from metrics import conditional_sw, mode_stats, mode_stats_band, sliced_wasserstein  # noqa: E402
from tasks import RingMixture  # noqa: E402

NFES = [1, 2, 4, 8, 16, 32, 64]
N_CTX = 64
M_PER_CTX = 64


def make_grid():
    """设计网格：在 (D-hat, Lam-hat) 平面上铺开，而不是在参数上乱扫。"""
    cells = []
    for K in [1, 2, 4, 8, 16]:
        for eta in [0.05, 0.2, 0.5, 1.5]:
            cells.append(dict(K=K, sep=6.0, eta=eta, d=2, tag="K%d_sep6_eta%.2f" % (K, eta)))
    for sep in [2.0, 15.0]:
        for eta in [0.05, 0.5]:
            cells.append(dict(K=8, sep=sep, eta=eta, d=2,
                              tag="K8_sep%g_eta%.2f" % (sep, eta)))
    cells.append(dict(K=4, sep=6.0, eta=0.05, d=8, tag="K4_sep6_eta0.05_d8"))
    cells.append(dict(K=4, sep=6.0, eta=0.50, d=8, tag="K4_sep6_eta0.50_d8"))
    return cells


def run_cell(cell, seed, n_train=20000, train_steps=5000, batch=512, lr=1e-3,
             hidden=256, n_layers=4, k_knn=10, kmax=8, time_mode="uniform",
             time_alpha=1.0, coupling="independent", reflow=False,
             arch="mlp", solver="euler", verbose=False, threads=4):
    set_seed(seed)
    K, sep, eta, d = cell["K"], cell["sep"], cell["eta"], cell["d"]
    sigma = 0.12
    task = RingMixture(K=K, sep_ratio=sep, sigma=sigma, eta=eta, d=d,
                       seed=seed, n_train=n_train)
    (Ctr, Xtr), (Cte, Xte) = task.arrays(n_train=n_train, n_test=4000)

    # ---- 1) 训练前判别量（只用数据） --------------------------------
    t0 = time.time()
    disc = compute_discriminant(Ctr[:15000], Xtr[:15000], Cte[:1500], Xte[:1500],
                                estimator="knn", k=k_knn, kmax=kmax, seed=seed)
    t_disc = time.time() - t0
    # 闭式真值 D（只有玩具任务可算）：用来标定估计器是否有偏
    D_true = float(task.analytic_D())
    var_total_true = float(((task.centers - task.centers.mean(0)) ** 2).sum(-1).mean()
                           + d * sigma ** 2)
    if verbose:
        print("    [disc] %.1fs D_hat=%.3f (knn10=%.3f, true=%.3f) Lam=%.2f"
              % (t_disc, disc["D_hat"], disc["D_hat_knn"], D_true,
                 disc["Lam_hat"]), flush=True)

    # ---- 2) 训练条件流匹配 ------------------------------------------
    torch.set_num_threads(threads)
    model = build_model(d, d, arch=arch, hidden=hidden, n_layers=n_layers)
    tt = Timer()
    model, last_loss = train_fm(model, Ctr, Xtr, n_steps=train_steps, batch=batch,
                                lr=lr, time_mode=time_mode, time_alpha=time_alpha,
                                coupling=coupling, seed=seed, verbose=verbose)
    t_train = tt.elapsed()
    if verbose:
        print("    [train] %.1fs loss=%.4f" % (t_train, last_loss), flush=True)

    # 重流（2-RF）：先用 64 步 ODE 生成配对，再在新耦合上重训
    if reflow:
        from flows import reflow_pairs
        C_r, X0_r, X1_r = reflow_pairs(model, Ctr[:n_train], n_train,
                                       n_steps=64, x_dim=d, seed=seed)
        model = build_model(d, d, arch=arch, hidden=hidden, n_layers=n_layers)
        model, last_loss = train_fm(model, C_r, X1_r, n_steps=train_steps,
                                    batch=batch, lr=lr, time_mode=time_mode,
                                    time_alpha=time_alpha, seed=seed + 1,
                                    X0_fixed=X0_r)

    # ---- 3) 固定 x0，扫 NFE -----------------------------------------
    C_ctx = torch.as_tensor(Cte[:N_CTX], dtype=torch.float32, device=DEVICE)
    C_rep = C_ctx.repeat_interleave(M_PER_CTX, dim=0)
    rng = np.random.default_rng(seed + 999)
    X0 = rng.normal(size=(N_CTX * M_PER_CTX, d)).astype(np.float32)
    X0t = torch.as_tensor(X0, dtype=torch.float32, device=DEVICE)

    # 真值条件样本与真值条件方差
    true_samples = task.true_conditional_samples(Cte[:N_CTX], M_PER_CTX)  # (ctx,m,d)
    true_cvar = float(np.mean(true_samples.var(axis=1).sum(axis=-1)) + 1e-12)
    true_cmean = true_samples.mean(axis=1)

    # ---- 2b) 理想一步（oracle one-step）：直接输出经验条件均值 ----------
    # 独立源耦合下，最优一步速度场给出 x_hat = m(c)，即**必然输出条件均值**。
    # 这里用一个外挂的 kNN 条件均值（完全不经过神经网络训练）当作"最好情形
    # 的一步采样器"，用来把**原理性地板**与**训练不足**分开：
    #   csw_oracle  ≈ 地板本身；csw(learned,NFE=1) 与它的差 = 可归因于训练的损失。
    pred_knn = fit_conditional_mean(Ctr, Xtr, estimator="knn", k=k_knn, seed=seed)
    X_or = np.asarray(pred_knn(Cte[:N_CTX]), dtype=np.float64)      # (n_ctx,d)
    or_samples = np.repeat(X_or[:, None, :], M_PER_CTX, axis=1)
    csw_oracle, _ = conditional_sw(true_samples, or_samples, n_proj=128, seed=seed)
    # 理想一步相对"真值条件均值"的偏差（纯估计误差，不含任何原理性项）
    oracle_bias = float(np.mean(((X_or - true_cmean) ** 2).sum(-1))
                        / (disc["var_total"] + 1e-12))

    out = dict(cell=cell, seed=seed, disc=disc, last_loss=last_loss,
               t_disc=t_disc, t_train=t_train, true_cvar=true_cvar,
               var_total=disc["var_total"], D_true=D_true,
               var_total_true=var_total_true, csw_oracle=csw_oracle,
               oracle_bias=oracle_bias,
               radius=float(task.R), sigma=float(sigma), nfe={})
    # 旧口径（容差球）容差，仅用于附录里对比口径差异
    cd = np.linalg.norm(task.centers[:, None, :] - task.centers[None, :, :], axis=-1)
    np.fill_diagonal(cd, np.inf)
    mode_tol = float(cd.min() / 2.0) if K > 1 else None

    t_eval = time.time()
    for nfe in NFES:
        gen = sample_euler(model, X0t, C_rep, n_steps=nfe, solver=solver)
        gen = gen.detach().cpu().numpy().astype(np.float64).reshape(N_CTX, M_PER_CTX, d)
        csw_mean, csw_std = conditional_sw(true_samples, gen, n_proj=128, seed=seed)
        pooled = gen.reshape(-1, d)
        mb = mode_stats_band(pooled, task.centers, radius=task.R, sigma=sigma)
        ms = mode_stats(pooled, task.centers, tol=mode_tol)      # 旧口径（对照）
        sw_marg = sliced_wasserstein(Xte[:2000].astype(np.float64), pooled,
                                     n_proj=256, seed=seed)
        gen_cvar = float(np.mean(gen.var(axis=1).sum(axis=-1)))
        gen_mean = gen.mean(axis=1)
        mean_bias = float(np.mean(((gen_mean - true_cmean) ** 2).sum(-1))
                          / (disc["var_total"] + 1e-12))
        out["nfe"][str(nfe)] = dict(
            csw=csw_mean, csw_std=csw_std, sw_marg=sw_marg,
            band_frac=mb["band_frac"], recall_band=mb["recall_band"],
            drop_band=mb["drop_band"], band_min_ratio=mb["min_share_ratio"],
            band_entropy=mb["entropy_ratio"], on_band=mb["on_band"],
            off_frac=mb["off_frac"], radial_off=mb["radial_off"], tol=mb["tol"],
            recall_ball=ms["recall"], off_frac_ball=ms["off_frac"],
            off_ball=ms["off_frac"],
            gen_cvar=gen_cvar, spread_ratio=gen_cvar / true_cvar,
            mean_bias=mean_bias)
        if verbose:
            print("    [nfe=%3d] %.1fs csw=%.4f band=%.2f recB=%.2f" %
                  (nfe, time.time() - t_eval, csw_mean, mb["band_frac"],
                   mb["recall_band"]), flush=True)
        t_eval = time.time()
    sqrt_var = float(np.sqrt(disc["var_total"]))
    out["gap_csw"] = out["nfe"]["1"]["csw"] - out["nfe"]["64"]["csw"]
    out["rel_gap"] = out["gap_csw"] / max(out["nfe"]["64"]["csw"], 1e-9)
    # 归一化误差（论文定量定律 e1 = kappa * sqrt(D) 用的就是这些量）
    out["e1"] = out["nfe"]["1"]["csw"] / sqrt_var
    out["eref"] = out["nfe"]["64"]["csw"] / sqrt_var
    out["efloor"] = csw_oracle / sqrt_var
    out["band_frac_1"] = out["nfe"]["1"]["band_frac"]
    out["band_frac_64"] = out["nfe"]["64"]["band_frac"]
    out["mode_drop_1"] = out["nfe"]["1"]["drop_band"]
    out["mode_drop_64"] = out["nfe"]["64"]["drop_band"]
    out["spread_ratio_1"] = out["nfe"]["1"]["spread_ratio"]
    out["spread_ratio_64"] = out["nfe"]["64"]["spread_ratio"]
    out["nfe_ref"] = 64
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell", type=int, default=-1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="toy")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已存在结果文件的 (cell, seed)，用于进程被杀后续跑")
    args = ap.parse_args()

    cells = make_grid()
    all_cells = list(cells)
    if args.quick:
        cells = [cells[10]]
        args.seeds = [0]
        args.steps = min(args.steps, 3000)
    if args.cell >= 0:
        cells = [all_cells[args.cell]]

    for ci, cell in enumerate(cells):
        for seed in args.seeds:
            if args.resume and os.path.exists(os.path.join(
                    _ROOT_,
                    "results", "toy_%s_s%d.json" % (cell["tag"], seed))):
                print("[%02d] %-24s seed=%d SKIP (已完成)" % (ci, cell["tag"], seed),
                      flush=True)
                continue
            t = Timer()
            res = run_cell(cell, seed, train_steps=args.steps,
                           verbose=args.verbose, threads=args.threads)
            tag = "toy_%s_s%d" % (cell["tag"], seed)
            save_result(tag, res)
            print("[%02d] %-24s seed=%d D=%.3f(D_true=%.3f) Lam=%.2f exc=%.2f | "
                  "csw1=%.4f csw64=%.4f orc=%.4f | dropB1=%.2f band1=%.2f spr1=%.2f "
                  "| %.1fs"
                  % (ci, cell["tag"], seed, res["disc"]["D_hat"], res["D_true"],
                     res["disc"]["Lam_hat"], res["disc"]["Lam_excess"],
                     res["nfe"]["1"]["csw"], res["nfe"]["64"]["csw"],
                     res["csw_oracle"], res["mode_drop_1"], res["band_frac_1"],
                     res["spread_ratio_1"], t.elapsed()), flush=True)


if __name__ == "__main__":
    main()
