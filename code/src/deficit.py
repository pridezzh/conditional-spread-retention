# -*- coding: utf-8 -*-
"""Lambda 的新定义：条件分布到"矩匹配高斯"的切片 W2 亏损（Gaussian deficit）。

**纯 numpy 实现，不依赖 sklearn / scipy。**

--------------------------------------------------------------------
一、为什么必须改定义（而不是修旧实现的零假设）
--------------------------------------------------------------------
旧 Lambda 声称检验"多峰"，但它实际检验的是"非高斯"：零假设是与样本同均值
同协方差的高斯。而

        高斯 ⊊ 单峰

所以任何**单峰但非高斯**的条件分布都会被系统性误判成多峰。压力测试
（``code/analysis/run_falsification.py``，预注册否决线 FPR > 0.20）实测：

    lognorm 0.85   banana 0.95   uniform_ball 1.00   exp_skew 1.00

全部命中否决线。注意这里有一个**反直觉但重要**的细节：重尾反而让旧检验更
保守（t_df3 / t_df5 / laplace 的 FPR 都是 0.00）——因为重尾把"池化簇内方差"
撑大，压低了分离度。所以这不是"调一调阈值"能解决的偏差，而是**统计量问的
问题本身就是错的**。

这个病根**换零假设也治不好**：逐坐标秩高斯化只改边缘分布，改不掉 banana
的弯曲、uniform_ball 的紧支撑这类**联合结构**；而做严格的多元单峰检验要用
Hartigan dip，其临界值依赖"均匀分布是最不利单峰"这一渐近结论，有限样本下
难以校准，且多元单峰本身就不是个良定义的概念。

--------------------------------------------------------------------
二、新定义：不再问"是不是多峰"
--------------------------------------------------------------------
从业者真正关心的其实是一个**不需要零假设就能回答**的问题：

    一步映射只能输出条件均值 m(c) 这一个点。最常见的补救是
    "在 m(c) 周围加一个协方差匹配的高斯噪声"。
    问：这个补救到底管不管用？

把这个问题写成量：

    Lam(c) = E_theta [ W2^2( theta#p(·|c) , N(theta'm(c), theta'Sigma(c)theta) ) ]
             / E_theta [ theta' Sigma(c) theta ]

其中 theta 取单位球面上的随机方向，theta#p 表示 1-D 投影分布，
Sigma(c) = Cov(x|c)。分母是"平均每个方向的条件方差"（= tr Sigma(c)/d）。

性质：
  * Lam ∈ [0, ~1]，**尺度无关**；
  * Lam = 0 当且仅当 p(·|c) 在**每个** 1-D 投影上都是高斯；
  * 没有零假设、没有 p 值、没有 FPR/TPR —— 它是一个可直接读取的指数，
    因此校核报告里那条针对"多峰检验"的预注册否决线**不再适用**。

为什么用 W2 而不是 CH / 卡方型统计量：
    W2 是**输运距离**，问的是"质量要搬多远"，对尾部的敏感度是多项式的；
    而 CH 本质是簇间/簇内方差比，重尾会把它撑爆（或反过来压低）。这正是
    对付 t 分布、对数正态这些形状所需要的稳健性。

--------------------------------------------------------------------
三、有限样本偏差与一阶校正
--------------------------------------------------------------------
用经验分位数估计 W2^2 有一个 O(1/n) 的正偏：极端次序统计量会系统性地偏离
高斯分位数，且该偏差在尾部按 1/phi(z) 发散（有限 n 下被 u = 1/(2n) 截断）。
本模块用**同规模高斯样本的蒙特卡洛基线**做一阶校正：

    W2^2_corr = W2^2_raw - b(n) * s^2

其中 b(n) = E[W2^2(n 个标准正态样本, 其矩匹配高斯)]，按 n 缓存。
校正后依构造有 Lam(gauss) = 0（见 ``validate_deficit.py`` 的检验 R1）。
"""
import numpy as np

_BASELINE_CACHE = {}


# ---------------------------------------------------------------- 反正态 CDF
def _norm_ppf(u):
    """反标准正态 CDF（Acklam 有理逼近，相对误差 < 1.2e-9），保持 numpy-only。"""
    u = np.asarray(u, dtype=np.float64)
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02,
                  -2.759285104469687e+02, 1.383577518672690e+02,
                  -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02,
                  -1.556989798598866e+02, 6.680131188771972e+01,
                  -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                  -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01,
                  2.445134137142996e+00, 3.754408661907416e+00])
    out = np.empty_like(u)
    lo = u < 0.02425
    hi = u > 1.0 - 0.02425
    mid = ~(lo | hi)
    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(u[lo]))
        out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - u[hi]))
        out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(mid):
        q = u[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return out


# ---------------------------------------------------------------- 一维 W2
def w2_to_gaussian_1d(p):
    """一维样本到其矩匹配高斯的 W2^2。

    对经验测度（等质量 1/n），W2^2 = ∫_0^1 (F_n^{-1}(u) - F_g^{-1}(u))^2 du，
    用 u_j = (j+1/2)/n 离散化即得分位数的均方误差。

    返回 (w2_squared, sample_variance)。
    """
    p = np.sort(np.asarray(p, dtype=np.float64).ravel())
    n = p.size
    if n < 3:
        return 0.0, 0.0
    u = (np.arange(n) + 0.5) / n
    mu = float(p.mean())
    var = float(p.var())
    if not np.isfinite(var) or var <= 1e-300:
        return 0.0, 0.0
    gq = mu + np.sqrt(var) * _norm_ppf(u)
    return float(((p - gq) ** 2).mean()), var


def _baseline(n, seed=91711, n_rep=256):
    """单位方差高斯在同样流程下的 W2^2（有限样本正偏），按 n 缓存。

    n_rep 必须足够大：该统计量的方差主要来自**极端次序统计量**
    （u = 1/(2n) 处分母 phi(z) 很小），分布明显右偏，重复次数少了
    蒙特卡洛误差会把校正量本身带偏。实测 n_rep=24 时基线与独立重复
    的实测均值差 34%，取 256 后收敛。
    """
    key = (int(n), int(n_rep))
    if key not in _BASELINE_CACHE:
        rng = np.random.default_rng(seed)
        tot = 0.0
        for _ in range(n_rep):
            tot += w2_to_gaussian_1d(rng.normal(size=n))[0]
        _BASELINE_CACHE[key] = tot / n_rep
    return _BASELINE_CACHE[key]


# ---------------------------------------------------------------- 核心统计量
def gaussian_deficit(Z, n_proj=64, seed=0, correct_bias=True, proj_chunk=32):
    """单个分布 p 的切片 W2 亏损。

    Z : (n, d) 来自 p 的样本。

    返回 (lam, lam_uncorrected, mean_proj_var)
      lam           偏差校正后的切片 W2 亏损 ∈ [0, ~1]
      lam_uncorrect 未校正值（用于报告校正量有多大）
      mean_proj_var 随机方向上的平均方差（= tr Sigma / d 的估计）

    **方向必须分块**：一次把所有方向的投影铺成 (n, n_proj) 矩阵，在
    n = 2e5、n_proj = 256 时就是 409 MB，会直接把进程撑爆。分块后峰值
    内存降到 O(n * proj_chunk)。
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n, d = Z.shape
    if n < 8:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n_proj, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-300
    raw = np.empty(n_proj)
    var = np.empty(n_proj)
    for s in range(0, n_proj, proj_chunk):
        e = min(s + proj_chunk, n_proj)
        P = Z @ G[s:e].T                                # (n, chunk)
        for j in range(e - s):
            raw[s + j], var[s + j] = w2_to_gaussian_1d(P[:, j])
        del P
    vbar = float(var.mean())
    if not np.isfinite(vbar) or vbar <= 1e-300:
        return 0.0, 0.0, 0.0
    raw_ratio = float(raw.mean()) / vbar
    lam = raw_ratio
    if correct_bias:
        lam = raw_ratio - _baseline(n)
    return float(np.clip(lam, 0.0, None)), float(raw_ratio), vbar


def conditional_gaussian_deficit(C, X, n_anchor=48, n_nb=150, n_proj=64,
                                 seed=0, pca_dim=32, correct_bias=True,
                                 min_size=40):
    """条件版的 Lambda：在条件空间的锚点邻域内计算切片 W2 亏损。

    与旧 ``conditional_mode_separation`` 用同一套锚点-邻域协议，便于对照。

    返回 dict：
      lam          各锚点 Lambda 的均值
      lam_std      各锚点之间的标准差（刻画条件间异质性）
      lam_raw      未做偏差校正的均值
      per_anchor   逐锚点值
      n_anchor, n_nb
    """
    C = np.asarray(C, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(C)
    n_nb = int(min(max(min_size, n_nb), max(min_size, n // 4)))
    Cs = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-8)   # 条件量纲标准化
    Xr = X
    if X.shape[1] > pca_dim:                              # 高维先降维
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        Xr = Xc @ U[:, :pca_dim].T
    rng = np.random.default_rng(seed)
    anchor = rng.choice(n, size=min(n_anchor, n), replace=False)
    vals, raws = [], []
    for a in anchor:
        d2 = ((Cs[a][None, :] - Cs) ** 2).sum(-1)
        nb = np.argpartition(d2, n_nb - 1)[:n_nb]
        lam, raw, _ = gaussian_deficit(Xr[nb], n_proj=n_proj,
                                       seed=seed + int(a),
                                       correct_bias=correct_bias)
        vals.append(lam)
        raws.append(raw)
    vals = np.asarray(vals)
    return dict(lam=float(vals.mean()), lam_std=float(vals.std()),
                lam_raw=float(np.mean(raws)), per_anchor=vals,
                n_anchor=int(vals.size), n_nb=n_nb)
