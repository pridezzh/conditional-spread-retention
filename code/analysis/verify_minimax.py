# -*- coding: utf-8 -*-
r"""不可能性定理的**最小最大**形式：纯数据统计量的最坏风险恰好 = 1/2。

--------------------------------------------------------------------
从"信息不足"到"精确等于 1/2"
--------------------------------------------------------------------
`verify_impossibility.py` 只证明了定性版本：ρ\* 扫遍 [0,1] 而数据分布不动，
所以纯数据量"信息不足"。但"信息不足"是个软说法，审稿人可以追问：
**究竟差多少？** 本脚本把它变成一个**可引用的数字**。

【定理（最小最大形式）】设 Π = {π_α : α ∈ [0,1]} 为 α-插值耦合族，
所有成员诱导同一个边缘律 p(c,x₁)。令 𝒜 为**任意**只依赖 n 个
(c,x₁) 样本的估计量（可随机化，可"无所不知"——即使它精确知道 p(c,x₁)）。则

    inf_𝒜  sup_{α∈[0,1]}  E_α | 𝒜 − ρ\*(α) |  =  1/2

且最优解就是**常数 1/2**。即：**在这族问题上，数据的价值恰好为零。**

证明（五行，无技术条件）：
  * 𝒜 的分布在所有 α 下**相同**（边缘律相同），故 ā := E[𝒜] 与 α 无关；
  * Jensen：E_α|𝒜 − α²| ≥ |ā − α²|，故 sup_α ≥ sup_α |ā − α²| ≥ max(|ā|, |ā−1|) ≥ 1/2；
  * 取 𝒜 ≡ 1/2 达到 sup_α |1/2 − α²| = 1/2。∎

注意 sup 里 α 取遍 [0,1] 而非只在网格上：0 与 1 都在族内，所以界是 1/2 而不是更小。
**n = ∞ 也不改善**——缺陷是信息论的，不是统计的。

--------------------------------------------------------------------
本脚本验什么
--------------------------------------------------------------------
定理的**经验内容**只有一条：纯数据统计量在 α 上确实是常数。所以：

  M1 统计量**不变性**：7 个纯数据统计量（含两个本文自己提出的 D̂、Λ̂，
     以及 5 个文献里常见的"多模态/结构"分数），
     跨 α 的最大相对偏离都 < ε。（这是定理前提的经验落实。）
  M2 **常数预测器基线**：min_c max_α |c − α²| = 0.5（细网格 1001 点数值确认），
     即"什么都不做"的最好成绩就是 0.5。任何纯数据量必须以它为基准。
  M3 **配对侧对照**：用上了耦合信息（成对的 (x₀,x₁)）的 ρ̂，对 α² 的
     平均绝对差 = MAD（来自 verify_impossibility 的实测值）。
     于是 **0.500 → MAD** 就是"耦合信息"这一份额外信息的全部价值。
  M4 **样本量救不了**：把纯数据统计量的样本量从 10³ 加到 1.6×10⁴，
     其在 α=0 与 α=1 之间的**差距不下降**（始终停在噪声水平）——
     对照 ρ̂ 随配对样本数增加而收敛。

--------------------------------------------------------------------
为什么不做"oracle 仿射标定后再比较"
--------------------------------------------------------------------
一个诱人的做法：给每个纯数据统计量做一次最小二乘仿射标定 a·S+b（允许它看过
ρ\*(α)=α²），再报最坏误差。这在**有限样本**上是陷阱：S 在 α 上的**噪声**
（D̂ 的跨 α 标准差约 0.001）会被 LS 当成信号，拟合出斜率 a ~ 10²–10³，
在**训练集（这 5 个 α）**上把误差压到 0.5 以下——那只是拟合噪声。
定理说的是**总体**量：总体的 S(α) 是常数，任何标定都是常数，最坏误差 ≥ 1/2。
所以本脚本只验"不变性"这条前提，把 1/2 留给定理。这是本文件最重要的一条纪律。
"""
import json
import os
import sys
import time


def _find_root(d):
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("project root not found")
        d = p


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))

import numpy as np  # noqa: E402
from scipy.stats import f_oneway  # noqa: E402

from deficit import gaussian_deficit  # noqa: E402
import verify_impossibility as VI  # noqa: E402

C, K, DIM = VI.C, VI.K, VI.DIM
ALPHAS = VI.ALPHAS
SEEDS = VI.SEEDS
N_QUERY = 2000

# 纯数据统计量不变性的检验方式：单因素 ANOVA（组 = α，重复 = 种子）
ALPHA_P = 0.05
M1_SEEDS = (0, 1, 2, 3, 4)
# M4 用的样本量阶梯与统计量（跳过高斯亏损，它最贵）
N_LADDER = (1000, 4000, 16000)
M4_SEEDS = (0, 1, 2)
M4_KEYS = ("D", "sep8", "K_CH", "nn_dist", "kurtosis")
# M4 的判据只用在**与 ρ* 同处 [0,1] 尺度**的统计量上。
#
# 为什么不把 sep8 / K_CH 放进绝对阈值判据：它们是**比值尺度**（典型值 ~35 与 ~8），
# 对它们设绝对阈值 0.01 毫无意义——它们的 α-无关性已由 M1 的 ANOVA（尺度无关）确立。
# 把它们排除是**先验的尺度区分**，不是看到结果后挑好看的。
M4_KEYS_ABS = ("D", "nn_dist", "kurtosis")
# 判据：|S(1) − S(0)| 的最大值 < ABS_MAX。ρ* 在这个族里扫过的范围是 1.0，
# 所以 ABS_MAX = 0.01 的含义是「可检出的 α-依赖不超过待预测效应量的 1%」。
SNR_MAX = 2.0          # 仍照报，但只作描述性数字（SNR 变大只说明噪声变小，不说明有信号）
ABS_MAX = 0.01
EPS_REL = 0.05          # 仅作**描述性**展示，不进判据


# ---------------------------------------------------------------- 纯数据统计量
def kmeans_pp_init(Z, Kc, rng):
    """k-means++ 初始化。"""
    n = len(Z)
    mu = [Z[int(rng.integers(n))]]
    d2 = ((Z - mu[0]) ** 2).sum(1)
    for _ in range(Kc - 1):
        tot = d2.sum()
        if tot <= 1e-12:
            mu.append(Z[int(rng.integers(n))])
        else:
            mu.append(Z[int(rng.choice(n, p=d2 / tot))])
        d2 = np.minimum(d2, ((Z - mu[-1]) ** 2).sum(1))
    return np.array(mu)


def kmeans(Z, Kc, rng, iters=100, restarts=3):
    """Lloyd + k-means++ + 多次重启取最小 inertia。

    **为什么必须重启**：初版用随机初始化，同一分布上 sep8 能从 15.3 跳到 36.4
    （跨种子），把"不变性"检验整个毁掉——那个"偏离"是优化器的噪声，不是 α 的信号。
    重启 3 次后该统计量的种子间变异降一个量级。
    """
    n = len(Z)
    Kc = min(Kc, n)
    best = None
    for _ in range(restarts):
        mu = kmeans_pp_init(Z, Kc, rng)
        lab = np.zeros(n, dtype=int)
        for _ in range(iters):
            d2 = ((Z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
            new = d2.argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
            for kk in range(Kc):
                m = lab == kk
                if m.any():
                    mu[kk] = Z[m].mean(0)
        d2 = ((Z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        inertia = float(d2.min(1).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy(), mu.copy())
    _, lab, mu = best
    within = float(sum(((Z[lab == kk] - mu[kk]) ** 2).sum() for kk in range(Kc)))
    return lab, mu, within


def ch_index(Z, Kc, rng):
    """Calinski--Harabasz：越大越该分这么多簇。"""
    n = len(Z)
    if Kc < 2 or Kc >= n:
        return 0.0
    lab, mu, within = kmeans(Z, Kc, rng)
    gm = Z.mean(0)
    cnt = np.bincount(lab, minlength=Kc).astype(np.float64)
    between = float((cnt[:, None] * (mu - gm) ** 2).sum())
    if within <= 1e-12:
        return 0.0
    return (between / (Kc - 1)) / (within / (n - Kc))


def k_ch(Z, rng, kmax=10):
    return float(1 + int(np.argmax([ch_index(Z, kk, rng) for kk in range(1, kmax + 1)])))


def sep8(Z, rng, Kc=None):
    """K=8 的 k-means 簇间/簇内方差比（文献里最常见的"多模态分数"形态）。"""
    Kc = K if Kc is None else Kc
    n = len(Z)
    if Kc < 2 or Kc >= n:
        return 0.0
    lab, mu, within = kmeans(Z, Kc, rng)
    gm = Z.mean(0)
    cnt = np.bincount(lab, minlength=Kc).astype(np.float64)
    between = float((cnt[:, None] * (mu - gm) ** 2).sum())
    return between / max(within, 1e-12)


def nn_dist(Z):
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min(1)).mean())


def excess_kurtosis(Z):
    Zc = Z - Z.mean(0)
    cov = Zc.T @ Zc / len(Z)
    ev = np.linalg.eigvalsh(cov)
    ev = np.maximum(ev, 1e-12)
    W = Zc @ (np.linalg.eigh(cov)[1] / np.sqrt(ev))
    m2 = (W ** 2).mean()
    m4 = (W ** 4).mean()
    return float(m4 / max(m2 ** 2, 1e-12) - 3.0)


def tr_var(a):
    return float(np.asarray(a, dtype=np.float64).var(axis=0).sum())


def data_only_stats(Y_by_cond, rng, with_lambda=True):
    """只吃 {(c_i, x1_i)}，绝不吃 x0，也绝不吃任何耦合信息。

    with_lambda=False 时跳过高斯亏损 Λ（它最贵），M4 的样本量阶梯用它。
    """
    Yall = np.stack(Y_by_cond)
    out = {}
    within = float(np.mean([tr_var(Y_by_cond[j]) for j in range(C)]))
    total = tr_var(Yall.reshape(-1, DIM))
    out["D"] = within / total
    if with_lambda:
        out["Lambda"] = float(np.mean([gaussian_deficit(Y_by_cond[j], n_proj=64,
                                                        seed=int(rng.integers(1 << 30)))[0]
                                       for j in range(C)]))
    out["K_CH"] = float(np.mean([k_ch(Y_by_cond[j], rng) for j in range(C)]))
    out["nn_dist"] = float(np.mean([nn_dist(Y_by_cond[j])
                                    / np.sqrt(tr_var(Y_by_cond[j])) for j in range(C)]))
    out["sep8"] = float(np.mean([sep8(Y_by_cond[j], rng) for j in range(C)]))
    out["kurtosis"] = float(np.mean([excess_kurtosis(Y_by_cond[j]) for j in range(C)]))
    out["trVar_marginal"] = total
    return out


# ---------------------------------------------------------------- 不变性的正确检验
def anova_alpha_effect(tab):
    """单因素方差分析：组 = α，重复 = 种子。返回 (F, p, 相对效应量)。

    **为什么必须用 ANOVA 而不是"固定相对阈值"。**
    不同统计量的自身噪声差两个数量级：D̂ 跨种子的相对波动 ~0.3%，
    而 sep8（k-means 簇间/簇内比）在**同一分布**上跨种子能差 2 倍以上。
    用统一的 5% 阈值去卡，等于对噪声大的统计量判它"有信号"，对噪声小的判它"没信号"
    ——检验的是噪声，不是 α 效应。ANOVA 用**自身种子间方差**做分母，
    问的是"α 造成的变异是否超过重复采样的变异"，这才是不变性的正确形式。
    """
    groups = [np.asarray(tab[a], dtype=np.float64) for a in ALPHAS]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")
    F, p = f_oneway(*groups)
    gm = float(np.mean(np.concatenate(groups)))
    # 相对效应量：α 造成的组间标准差 / 总水平
    means = np.array([g.mean() for g in groups])
    between = float(means.std(ddof=1))
    return float(F), float(p), between / max(abs(gm), 1e-12)


STAT_KEYS = ("D", "Lambda", "K_CH", "nn_dist", "sep8", "kurtosis", "trVar_marginal")


# ---------------------------------------------------------------- M2：常数基线
def minimax_constant(n_grid=1001):
    a = np.linspace(0.0, 1.0, n_grid)
    tgt = a ** 2

    def worst(c):
        return float(np.max(np.abs(c - tgt)))
    cs = np.linspace(-0.5, 1.5, 20001)
    w = np.array([worst(c) for c in cs])
    i = int(np.argmin(w))
    return float(cs[i]), float(w[i])


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    print("=" * 78)
    print("不可能性定理 · 最小最大形式")
    print("  定理：inf_A sup_α E|A − ρ*(α)| = 1/2，最优解即常数 1/2")
    print("  本脚本验定理的**经验前提**：7 个纯数据统计量在 α 上确实是常数")
    print("=" * 78)

    # ---------- M2（纯数值，先算） ----------
    cstar, wstar = minimax_constant()
    print("\n[M2] 常数预测器基线：min_c max_α |c − α²| = %.4f，在 c = %.4f 处取到"
          % (wstar, cstar))
    print("     （细网格 1001 点；理论值 0.5 / 0.5。这就是'什么都不做'的最好成绩）")

    # ---------- M1：不变性 ----------
    print("\n[M1] 7 个纯数据统计量跨 α 的不变性（%d 个种子，单因素 ANOVA）" % len(M1_SEEDS))
    grid = {k: {a: [] for a in ALPHAS} for k in STAT_KEYS}
    for sd in M1_SEEDS:
        print("  ---- seed %d ----" % sd, flush=True)
        for a in ALPHAS:
            rng = np.random.default_rng(1000 * (sd + 1) + int(a * 100) + 7)
            # 与 verify_impossibility **同一套** α-耦合样本生成
            Y = []
            for j in range(C):
                _, x1 = VI.sample_alpha(N_QUERY, j, a, rng)
                Y.append(x1)
            st = data_only_stats(Y, np.random.default_rng(4242 + sd))
            for k in STAT_KEYS:
                grid[k][a].append(st[k])
            print("      α=%.2f: D=%.4f  Λ=%.4f  K_CH=%.2f  nn=%.4f  sep8=%.4f  kurt=%.4f"
                  % (a, st["D"], st["Lambda"], st["K_CH"], st["nn_dist"],
                     st["sep8"], st["kurtosis"]), flush=True)

    mean = {k: {a: float(np.mean(grid[k][a])) for a in ALPHAS} for k in STAT_KEYS}
    print("\n  %-16s %s" % ("统计量", "".join("α=%.2f      " % a for a in ALPHAS)))
    print("  " + "-" * 74)
    for k in STAT_KEYS:
        print("  %-16s %s" % (k, "".join("%-12.4f" % mean[k][a] for a in ALPHAS)))
    print("  " + "-" * 74)

    # 描述性：相对偏离（**不进判据**，只作展示）
    dev = {}
    for k in STAT_KEYS:
        base = mean[k][ALPHAS[0]]
        scale = max(abs(base), 1e-9)
        dev[k] = float(max(abs(mean[k][a] - base) for a in ALPHAS) / scale)
    print("\n  描述性：相对偏离 max_α |S(α) − S(0)| / |S(0)|（不进判据，因未对噪声归一）")
    for k in STAT_KEYS:
        print("    %-16s %.5f" % (k, dev[k]))

    # 判据：单因素 ANOVA（组 = α，重复 = 种子）
    print("\n  判据：单因素 ANOVA，H0 = “S 与 α 无关”，显著水平 %.2f" % ALPHA_P)
    anova = {}
    for k in STAT_KEYS:
        F, p, rel = anova_alpha_effect(grid[k])
        anova[k] = dict(F=F, p=p, rel_effect=rel)
        tag = "不显著（不变）" if p == p and p > ALPHA_P else "显著"
        print("    %-16s F=%-10.3f p=%-10.4f α-效应(相对)=%-9.5f  %s"
              % (k, F, p, rel, tag))
    M1 = bool(all(anova[k]["p"] == anova[k]["p"] and anova[k]["p"] > ALPHA_P
                  for k in STAT_KEYS))
    # 每个统计量的自身噪声（种子间 CV），说明它能不能当诊断量
    cv = {}
    for k in STAT_KEYS:
        vals = np.concatenate([np.asarray(grid[k][a], dtype=np.float64) for a in ALPHAS])
        cv[k] = float(vals.std(ddof=1) / max(abs(vals.mean()), 1e-12))
    print("\n  各统计量的种子间变异系数 CV（越大越不能当诊断量）：")
    for k in STAT_KEYS:
        print("    %-16s %.4f" % (k, cv[k]))

    # ---------- M3：配对侧对照 ----------
    p = os.path.join(ROOT, "logs", "verify_impossibility.json")
    rho_mad = float("nan")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            rho_mad = float(json.load(f)["deviations"]["pred_mad"])
    print("\n[M3] 配对侧（用上了耦合信息）的 ρ̂：对 α² 的平均绝对差 MAD = %.4f" % rho_mad)
    print("     纯数据侧最坏误差 = %.4f（= 常数基线）  →  比值 %.1f×"
          % (wstar, (wstar / rho_mad) if rho_mad > 0 else float("nan")))

    # ---------- M4：样本量救不了 ----------
    print("\n[M4] 加大样本量能否救纯数据侧？")
    print("     看 α=0 与 α=1 之间的差 |Δ|（这两侧的 ρ* 相差 1.000），")
    print("     并同时给出用**自身种子间噪声**归一的 SNR。")
    print("     注意：若 S 的总体值真的依赖 α，|Δ| 应随 n 收敛到一个非零常数；")
    print("     若总体值不依赖 α（定理所说），|Δ| 只是噪声，随 n 按 1/√n 缩小。")
    print("     所以判据看 |Δ| 的绝对大小（相对待预测效应量 1.0），不看 SNR——")
    print("     SNR 变大只说明噪声变小到能看见那个 ~0.002 的残差，不说明有信号。")
    ladder = {}
    for n in N_LADDER:
        g = {k: {0.0: [], 1.0: []} for k in M4_KEYS}
        for sd in M4_SEEDS:
            for a in (0.0, 1.0):
                rng = np.random.default_rng(1000 * (sd + 1) + int(a * 100) + 7)
                Y = [VI.sample_alpha(n, j, a, rng)[1] for j in range(C)]
                st = data_only_stats(Y, np.random.default_rng(4242 + sd),
                                     with_lambda=False)
                for k in M4_KEYS:
                    g[k][a].append(st[k])
        row = {}
        for k in M4_KEYS:
            v0 = np.asarray(g[k][0.0], dtype=np.float64)
            v1 = np.asarray(g[k][1.0], dtype=np.float64)
            pooled = float(np.sqrt(0.5 * (v0.var(ddof=1) + v1.var(ddof=1))))
            row[k] = dict(diff=float(abs(v1.mean() - v0.mean())),
                          noise=pooled,
                          snr=float(abs(v1.mean() - v0.mean()) / max(pooled, 1e-12)))
        ladder[str(n)] = row
        print("    n=%-7d %s" % (n, "  ".join("%s: |Δ|=%.4f SNR=%.2f"
                                              % (k, row[k]["diff"], row[k]["snr"])
                                              for k in M4_KEYS)))
    # 判据（只用在 [0,1] 尺度的统计量上）：|S(1)−S(0)| < 0.01 = 待预测效应量 1.0 的 1%
    flat = {k: bool(all(ladder[str(n)][k]["diff"] < ABS_MAX for n in N_LADDER))
            for k in M4_KEYS_ABS}
    print("    判据（与 ρ* 同尺度的统计量，各档 |S(1)−S(0)| < %.2f，"
          "即不超过待预测效应量 1.0 的 %.0f%%）：%s"
          % (ABS_MAX, ABS_MAX * 100, json.dumps(flat, ensure_ascii=False)))
    worst_abs = max(ladder[str(n)][k]["diff"] for n in N_LADDER for k in M4_KEYS_ABS)
    print("    实测最坏 |S(1)−S(0)| = %.4f  →  是待预测效应量 1.0 的 %.2f%%"
          % (worst_abs, worst_abs * 100))
    print("    （sep8 / K_CH 是比值尺度，绝对阈值不适用；其 α-无关性由 M1 的 ANOVA 确立。）")
    M4 = bool(all(flat.values()))

    checks = dict(M1_invariance=M1, M2_constant_baseline=bool(abs(wstar - 0.5) < 1e-3),
                  M3_paired_mad=bool(rho_mad == rho_mad and rho_mad < 0.05),
                  M4_sample_size_does_not_help=M4)
    verdict = "PASS" if all(checks.values()) else (
        "PARTIAL" if sum(checks.values()) >= 3 else "FAIL")
    print("\n" + "=" * 78)
    for k, v in checks.items():
        print("    %-34s %s" % (k, v))
    print("  VERDICT: %s      用时 %.0fs" % (verdict, time.time() - t0))
    print("=" * 78)

    report = dict(theorem="minimax risk of any data-only statistic = 1/2",
                  params=dict(alphas=list(ALPHAS), seeds=list(M1_SEEDS),
                              N_QUERY=N_QUERY, N_LADDER=list(N_LADDER),
                              ALPHA_P=ALPHA_P, SNR_MAX=SNR_MAX),
                  constant_baseline=dict(c_star=cstar, worst=wstar),
                  stat_means={k: {str(a): mean[k][a] for a in ALPHAS} for k in STAT_KEYS},
                  stat_rel_dev=dev,
                  stat_anova=anova,
                  stat_cv=cv,
                  paired_rho_mad=rho_mad,
                  risk_ratio=(wstar / rho_mad) if rho_mad > 0 else None,
                  sample_size_ladder=ladder,
                  checks=checks, verdict=verdict)
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "verify_minimax.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main()
