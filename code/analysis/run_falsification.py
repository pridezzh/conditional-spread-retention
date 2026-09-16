# -*- coding: utf-8 -*-
"""最小可证伪实验：Lambda 在"非高斯单峰"分布上的假阳性压力测试。

对应校核报告 P1 第 1 项，以及"必须预注册的否决条件"第 2 条：
    所谓多峰分离统计在重尾 / 弯曲单峰上假阳性过高。

为什么必须先做这一步
--------------------
Lambda 的单峰零假设是"与样本均值、协方差匹配的高斯"（`_gaussian_null`）。
真实数据的条件分布完全可以是**单峰但明显非高斯**的：重尾、偏态、弯曲流形、
异方差、紧支撑。若这类形状被系统性误判为多峰，Lambda 就无法再把
"结构性丢模态"与"形状奇怪的单峰"区分开，论文的第二条判别轴
（良性模糊 vs 灾难性丢模态）直接失去意义。

这个测试**不需要训练、不需要 GPU，几分钟内出结果**，而它决定整条判别量路线
是否成立。所以在投入数小时重跑 162 个实验之前，先跑它。

设计
----
构造一批**确知单峰**的分布（含多种非高斯形状）与一批**确知多峰**的分布，
对每组独立重复多次调用 `_select_K`，统计返回 K >= 2 的比例：

    单峰组 → 假阳性率 FPR，名义显著水平 alpha = 0.05
    多峰组 → 灵敏度 TPR，用来确认检验没有退化成"永远返回 K = 1"

事前写死的判据（不根据结果调整）
------------------------------
    FAIL_FPR = 0.20   任一单峰形状的 FPR 超过它 → 该形状下零假设失效
    FAIL_TPR = 0.80   多峰形状的 TPR 低于它   → 检验失去功效
    两者任一命中，即按预注册规则停止"判别量"主张。
"""
import json
import os
import sys
import time


def _find_root(d):
    """自愈式锚点：向上找同时含 code/ 与 paper/ 的那一级。"""
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project root not found")
        d = parent


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
from discriminant import _select_K, _lam_of  # noqa: E402

# ---- 实验参数（事前固定） ----
N_REP = 20
N_SAMP = 200
N_NULL = 49
KMAX = 5
ALPHA = 0.05
FAIL_FPR = 0.20
FAIL_TPR = 0.80

# 确知单峰（含高斯对照与多种非高斯形状）
UNIMODAL = ["gauss", "t_df3", "t_df5", "laplace", "uniform_ball",
            "banana", "lognorm", "exp_skew", "hetero"]
# 确知多峰（用于灵敏度，不是假阳性）
MULTIMODAL = ["two_modes", "ring8"]


def sample_shape(name, n, rng):
    """生成一种形状。除 two_modes / ring8 外，全部是单峰分布。"""
    if name == "gauss":                      # 对照：零假设的真实分布
        return rng.normal(size=(n, 2))
    if name == "t_df3":                      # 重尾，方差几乎不收敛
        return rng.standard_t(3.0, size=(n, 2)) * 0.7
    if name == "t_df5":                      # 中等重尾
        return rng.standard_t(5.0, size=(n, 2)) * 0.8
    if name == "laplace":                    # 尖峰 + 比高斯重的尾
        return rng.laplace(size=(n, 2))
    if name == "uniform_ball":               # 紧支撑，尾部比高斯轻得多
        d = rng.normal(size=(n, 2))
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        r = np.sqrt(rng.uniform(0.0, 1.0, size=(n, 1)))
        return d * r * 2.0
    if name == "banana":                     # 弯曲流形：一维弯曲嵌入二维
        x1 = rng.normal(scale=1.0, size=(n, 1))
        x2 = x1 ** 2 + rng.normal(scale=0.30, size=(n, 1))
        return np.hstack([x1, x2 - 1.0])
    if name == "lognorm":                    # 强右偏
        z = rng.normal(size=(n, 2))
        return np.exp(0.6 * z) - np.exp(0.18)
    if name == "exp_skew":                   # 指数型偏态
        return rng.exponential(size=(n, 2)) - 1.0
    if name == "hetero":                     # 异方差：展布随位置变化
        x1 = rng.normal(size=(n, 1))
        sd = 0.20 + 0.80 * np.abs(x1)
        return np.hstack([x1, rng.normal(scale=sd)])
    if name == "two_modes":                  # 真·双峰，分离充分
        half = n // 2
        a = rng.normal(-2.5, 0.25, size=(half, 2))
        b = rng.normal(2.5, 0.25, size=(n - half, 2))
        return np.r_[a, b]
    if name == "ring8":                      # 论文合成族的八模态环
        ang = rng.integers(0, 8, size=n)
        th = ang * (2.0 * np.pi / 8.0)
        cen = np.c_[np.cos(th), np.sin(th)] * 3.0
        return cen + rng.normal(scale=0.25, size=(n, 2))
    raise ValueError("unknown shape: %s" % name)


def run_one(name, rep):
    rng = np.random.default_rng(1000 * (rep + 1) + 7)
    R = sample_shape(name, N_SAMP, rng)
    t0 = time.time()
    K = _select_K(R, kmax=KMAX, seed=7919 * (rep + 1), n_null=N_NULL, alpha=ALPHA)
    lam = 0.0
    if K >= 2:
        lam, _ = _lam_of(R, K, seed=7919 * (rep + 1))
    return int(K), float(lam), float(time.time() - t0)


def main():
    report = {
        "experiment": "lambda_false_positive_stress",
        "purpose": "测试 Lambda 的高斯单峰零假设在非高斯单峰分布上的假阳性率",
        "params": dict(N_REP=N_REP, N_SAMP=N_SAMP, N_NULL=N_NULL,
                       KMAX=KMAX, ALPHA=ALPHA),
        "criteria": dict(FAIL_FPR=FAIL_FPR, FAIL_TPR=FAIL_TPR),
        "unimodal": {},
        "multimodal": {},
    }

    for group, names in (("unimodal", UNIMODAL), ("multimodal", MULTIMODAL)):
        for name in names:
            ks, lams, secs = [], [], []
            for rep in range(N_REP):
                K, lam, dt = run_one(name, rep)
                ks.append(K)
                lams.append(lam)
                secs.append(dt)
                print("  [%s] rep %2d/%d  K=%d  lambda=%.3f  (%.2fs)"
                      % (name, rep + 1, N_REP, K, lam, dt), flush=True)
            ks = np.array(ks)
            rate = float((ks >= 2).mean())
            entry = {
                "K>=2_rate": rate,
                "K_mean": float(ks.mean()),
                "lambda_mean": float(np.mean(lams)),
                "n_rep": N_REP,
                "sec_per_call": float(np.mean(secs)),
            }
            report[group][name] = entry
            print("== %-14s K>=2 比例 = %.2f  (K 均值 %.2f, lambda 均值 %.3f)"
                  % (name, rate, ks.mean(), np.mean(lams)), flush=True)

    # ---- 判据 ----
    fp_fail = {k: v["K>=2_rate"] for k, v in report["unimodal"].items()
               if v["K>=2_rate"] > FAIL_FPR}
    tp_fail = {k: v["K>=2_rate"] for k, v in report["multimodal"].items()
               if v["K>=2_rate"] < FAIL_TPR}
    report["false_positive_failures"] = fp_fail
    report["power_failures"] = tp_fail
    report["verdict"] = "PASS" if not (fp_fail or tp_fail) else "FAIL"

    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "falsification_lambda_stress.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print("预注册判据：单峰 FPR > %.2f 判失败；多峰 TPR < %.2f 判失败"
          % (FAIL_FPR, FAIL_TPR))
    print("-" * 62)
    for name in UNIMODAL:
        r = report["unimodal"][name]["K>=2_rate"]
        flag = "  <-- 假阳性过高" if name in fp_fail else ""
        print("  单峰 %-14s FPR = %.2f%s" % (name, r, flag))
    for name in MULTIMODAL:
        r = report["multimodal"][name]["K>=2_rate"]
        flag = "  <-- 功效不足" if name in tp_fail else ""
        print("  多峰 %-14s TPR = %.2f%s" % (name, r, flag))
    print("-" * 62)
    print("VERDICT: %s" % report["verdict"])
    print("报告已写入 %s" % out)
    print("=" * 62)


if __name__ == "__main__":
    main()
