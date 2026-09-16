# -*- coding: utf-8 -*-
"""训练前可算的判别量（论文 §4）。**纯 numpy 实现，不依赖 sklearn**。

（工程备注：Windows + MKL 下 sklearn 的 KMeans 会在多线程环境中卡死，
因此这里的 k-means / kNN / PCA 全部自己写，顺便把"老方法"复现得更透明。）

两个量，都**只用数据集 (c_i, x_i)，不需要训练任何生成模型**：

  D-hat  **未被条件解释的目标方差占比**（"一步生成误差地板"的估计）
         = 条件均值回归在留出集上的残差方差 / 目标总方差。
         估计器可换（kNN / RFF-岭回归 / 线性岭），默认 kNN（Cover & Hart, 1967），
         并用"最近邻配对估计器"做**无回归器**的交叉校验。

  Lam-hat **不可解释部分（残差）的模态分离度**
         对残差 r = x - m_hat(c) 做 k-means（Lloyd, 1982 / MacQueen, 1967），
         簇数由 Calinski-Harabasz 准则（1974）在 1..Kmax 中选；
         Lam = 最近的两个簇心距离 / (2 × 池化簇内标准差)；单簇时定义为 0。

两者的分工（论文命题 1 与 §4）：
  D-hat 决定一步相对多步的**误差地板有多大**；
  Lam-hat 决定这个地板表现为**丢模态（结构化、灾难性）**还是**模糊（弥散、良性）**。
"""
import numpy as np


def _as2(a):
    a = np.asarray(a, dtype=np.float64)
    return a if a.ndim > 1 else a.reshape(-1, 1)


# ------------------------------------------------------------ 基础件
def knn_predict(Ctr, Xtr, Cq, k=10, chunk=256, eps=1e-8):
    """k 近邻回归（距离加权），分块计算避免大矩阵。"""
    Ctr, Xtr, Cq = _as2(Ctr), _as2(Xtr), _as2(Cq)
    n, d = Xtr.shape
    k = min(k, max(1, n - 1))
    out = np.empty((len(Cq), d), dtype=np.float64)
    for s in range(0, len(Cq), chunk):
        q = Cq[s:s + chunk]
        d2 = ((q[:, None, :] - Ctr[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        w = 1.0 / (np.sqrt(np.take_along_axis(d2, idx, axis=1)) + eps)
        xs = Xtr[idx]                                   # (chunk, k, d)
        out[s:s + chunk] = (xs * w[:, :, None]).sum(1) / w.sum(1)[:, None]
    return out


def kmeans_lloyd(X, K, n_init=5, iters=50, seed=0):
    """Lloyd 迭代 + k-means++ 初始化。返回 (centers, labels, inertia)。"""
    X = _as2(X)
    n = len(X)
    if K >= n:
        return X.copy(), np.arange(n), 0.0
    rng = np.random.default_rng(seed)
    best = None
    for init in range(n_init):
        # k-means++
        idx0 = rng.integers(0, n)
        cent = [X[idx0]]
        for _ in range(1, K):
            d2 = np.min(((X[:, None, :] - np.array(cent)[None, :, :]) ** 2).sum(-1), axis=1)
            prob = d2 / (d2.sum() + 1e-12)
            cent.append(X[rng.choice(n, p=prob)])
        C = np.array(cent)
        for _ in range(iters):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d2.argmin(1)
            newC = C.copy()
            for j in range(K):
                m = lab == j
                if m.any():
                    newC[j] = X[m].mean(0)
                else:
                    newC[j] = X[rng.integers(0, n)]
            if np.allclose(newC, C, atol=1e-9):
                C = newC
                break
            C = newC
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        inertia = float(d2[np.arange(n), lab].sum())
        if best is None or inertia < best[2]:
            best = (C, lab, inertia)
    return best


def calinski_harabasz(X, lab):
    X = _as2(X)
    K = len(np.unique(lab))
    n = len(X)
    if K < 2 or K >= n:
        return 0.0
    gm = X.mean(0)
    cent = np.array([X[lab == j].mean(0) for j in range(K)])
    counts = np.array([(lab == j).sum() for j in range(K)], dtype=float)
    bg = (counts[:, None] * (cent - gm) ** 2).sum()
    wg = float(((X - cent[lab]) ** 2).sum())
    if wg < 1e-12:
        return 0.0
    return float((bg / (K - 1)) / (wg / (n - K)))


def pca_project(X, n_comp=16, fit_on=None):
    """返回 (低维表示, 均值, 投影矩阵)。"""
    X = _as2(X)
    base = X if fit_on is None else _as2(fit_on)
    mu = base.mean(0)
    U, S, Vt = np.linalg.svd(base - mu, full_matrices=False)
    W = Vt[:min(n_comp, Vt.shape[0])]
    return (X - mu) @ W.T, mu, W


def ridge_fit(Z, Y, alpha=1.0):
    Z = _as2(Z)
    Y = _as2(Y)
    p = Z.shape[1]
    A = Z.T @ Z + alpha * np.eye(p)
    return np.linalg.solve(A, Z.T @ Y)


def rff_features(C, n_comp=256, gamma=None, seed=0):
    C = _as2(C)
    d = C.shape[1]
    if gamma is None:
        gamma = 1.0 / max(d, 1)
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(d, n_comp))
    b = rng.uniform(0, 2 * np.pi, size=n_comp)
    return np.sqrt(2.0 / n_comp) * np.cos(C @ W + b)


# ------------------------------------------------------------ 判别量
def fit_conditional_mean(Ctr, Xtr, estimator="knn", k=10, n_rff=256,
                         alpha=1.0, seed=0):
    """拟合 E[x|c]，返回 predict(C)。默认 kNN（Cover & Hart, 1967）。"""
    Ctr, Xtr = _as2(Ctr), _as2(Xtr)
    mu_c, sd_c = Ctr.mean(0), Ctr.std(0) + 1e-8
    Ztr = (Ctr - mu_c) / sd_c

    def _prep(C):
        return (_as2(C) - mu_c) / sd_c

    if estimator == "knn":
        return lambda C: knn_predict(Ztr, Xtr, _prep(C), k=k)
    if estimator == "linear":
        Wb = ridge_fit(np.hstack([Ztr, np.ones((len(Ztr), 1))]), Xtr, alpha)

        def _f(C):
            Z = np.hstack([_prep(C), np.ones((len(_as2(C)), 1))])
            return Z @ Wb
        return _f
    if estimator == "rff":
        Ftr = rff_features(Ztr, n_comp=n_rff, seed=seed)
        Wb = ridge_fit(np.hstack([Ftr, np.ones((len(Ftr), 1))]), Xtr, alpha)

        def _f(C):
            F = rff_features(_prep(C), n_comp=n_rff, seed=seed)
            return np.hstack([F, np.ones((len(F), 1))]) @ Wb
        return _f
    if estimator == "mean":
        mu_x = Xtr.mean(0)
        return lambda C: np.tile(mu_x, (len(_as2(C)), 1))
    raise ValueError(estimator)


def d_hat_from_predictor(pred, Cte, Xte, var_total=None):
    Cte, Xte = _as2(Cte), _as2(Xte)
    if var_total is None:
        var_total = Xte.var(axis=0).sum() + 1e-12
    res = Xte - pred(Cte)
    return float((res ** 2).sum(axis=1).mean() / var_total)


def d_hat_pair(C, X, k=1, chunk=512):
    """无回归器的交叉校验：最近邻配对估计器。

    E||x_i - x_nn(i)||^2 ≈ 2 tr Var(x|c)（邻域收缩时偏差趋于 0），
    故 D_pair = mean(d^2) / (2 tr Var(x))。
    """
    C, X = _as2(C), _as2(X)
    # 条件的各维量纲可能完全不同（例如图像像素与 one-hot 标签）。
    # 最近邻搜索前必须标准化；否则数值范围最大的维度会独占距离。
    C = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-8)
    n = len(C)
    tot = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = ((C[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
        for i in range(e - s):
            d2[i, s + i] = np.inf          # 排除自身
        nb = d2.argmin(1)
        tot[s:e] = ((X[s:e] - X[nb]) ** 2).sum(-1)
    var_total = X.var(axis=0).sum() + 1e-12
    return float(np.clip(tot.mean() / (2 * var_total), 0.0, None))


def _lam_of(Rr, K, seed):
    """给定簇数 K，计算分离度 Lam = 最近簇心距 / (2 × 池化簇内标准差)。"""
    if K < 2:
        return 0.0, None
    C, lab, _ = kmeans_lloyd(Rr, K, n_init=3, iters=30, seed=seed)
    if len(np.unique(lab)) < K:
        return 0.0, None
    dmin = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            dmin = min(dmin, np.linalg.norm(C[i] - C[j]))
    # “池化簇内方差”要按样本数加权。逐簇等权平均会让很小的簇获得与大簇
    # 相同的权重，从而在人为切出小簇时系统性扭曲 Lambda。
    within = ((Rr - C[lab]) ** 2).sum(axis=1).mean()
    sbar = np.sqrt(max(within, 1e-12) / Rr.shape[1])
    return float(dmin / (2 * sbar)), (C, lab)


def _best_K_by_ch(Rr, kmax, seed, n_init=2, iters=30):
    """返回 K>=2 的最佳 CH 候选及其分数；没有有效候选时返回 (1, 0)。"""
    kmax = max(1, min(kmax, len(Rr) // 20))
    best_score, best_K = 0.0, 1
    for K in range(2, kmax + 1):
        C, lab, _ = kmeans_lloyd(Rr, K, n_init=n_init, iters=iters, seed=seed + K)
        if len(np.unique(lab)) < K:
            continue
        s = calinski_harabasz(Rr, lab)
        if s > best_score:
            best_score, best_K = s, K
    return best_K, float(best_score)


def _select_K(Rr, kmax, seed, n_null=99, alpha=0.05):
    """用“最大 CH 分数”的高斯零假设检验在 K=1 与 K>=2 之间选择。

    只在 K=2..Kmax 中取最大 CH 会必然返回多簇，旧实现所谓的 K=1 实际上
    永远不可选。本实现把“搜索过多个 K”也包含进零假设：每个高斯自助样本
    同样搜索全部 K，观测最大分数超过 (1-alpha) 分位数时才接受多峰模型。
    """
    best_K, observed = _best_K_by_ch(Rr, kmax, seed)
    if best_K == 1 or n_null < 1:
        return best_K
    null_scores = []
    for b in range(n_null):
        rng = np.random.default_rng(seed + 7919 * (b + 1))
        Zn = _gaussian_null(Rr, len(Rr), rng)
        _, score = _best_K_by_ch(
            Zn, kmax, seed + 104729 * (b + 1), n_init=1, iters=20)
        null_scores.append(score)
    threshold = float(np.quantile(null_scores, 1.0 - alpha))
    return int(best_K if observed > threshold else 1)


def _gaussian_null(Rr, n, rng):
    """与给定样本二阶矩匹配的高斯样本（单峰零假设）。"""
    mu = Rr.mean(0)
    cov = np.cov(Rr.T) + 1e-10 * np.eye(Rr.shape[1])
    if Rr.shape[0] < 2:
        return Rr
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-10)))
    return rng.normal(size=(n, Rr.shape[1])) @ L.T + mu


def conditional_mode_separation(C, X, n_anchor=48, n_nb=150, kmax=8, seed=0,
                                pca_dim=16, n_null=99):
    """**条件分布**的模态分离度（核心口径）。

    做法：在数据里取 A 个锚点上下文，每个锚点取条件空间中的 n_nb 个近邻，
    对这 n_nb 个**原始目标样本**做聚类（不需要减条件均值，因此不会引入
    回归器噪声），用 Calinski-Harabasz 选簇数，算
        Lam_a = 最近的两个簇心距离 / (2 × 池化簇内标准差)
    并对**同协方差单峰高斯零假设**做参数化自助，得到 null 基线。

    返回 (Lam_hat, Lam_null, Lam_p95, frac_multi, K_mean)
    """
    C, X = _as2(C), _as2(X)
    n = len(C)
    n_nb = min(n_nb, max(30, n // 10))
    rng = np.random.default_rng(seed)
    anchor = rng.choice(n, size=min(n_anchor, n), replace=False)
    Xr = X
    if X.shape[1] > pca_dim:
        Xr, _, _ = pca_project(X, n_comp=pca_dim)
    lams, nulls, ks, multi = [], [], [], []
    for a in anchor:
        d2 = ((C[a][None, :] - C) ** 2).sum(-1)
        nb = np.argpartition(d2, n_nb - 1)[:n_nb]
        Z = Xr[nb]
        # K=1 必须由含模型选择步骤的零假设检验得到，而不是只比较 K>=2 的 CH。
        K = _select_K(Z, kmax, seed + int(a), n_null=n_null)
        if K < 2:
            lams.append(0.0)
            nulls.append(0.0)
            ks.append(1)
            multi.append(0.0)
            continue
        lam, _ = _lam_of(Z, K, seed + int(a))
        nl = []
        for b in range(n_null):
            Zn = _gaussian_null(Z, len(Z), np.random.default_rng(seed + 7919 * (int(a) + 1) + b))
            # K 的选择显著性已由 _select_K 用“最大 CH”零假设校正；这里在已接受的
            # K 上校准 Lambda 的尺度，避免再嵌套一层极昂贵的自助检验。
            ln, _ = _lam_of(Zn, K, seed + 130363 * (int(a) + 1) + b)
            nl.append(ln)
        nl = np.array([v for v in nl if v is not None])
        lams.append(lam)
        nulls.append(float(nl.mean()) if len(nl) else 0.0)
        ks.append(K)
        multi.append(1.0)  # K>=2 已通过含多重 K 搜索的 5% 零假设检验
    return (float(np.mean(lams)), float(np.mean(nulls)),
            float(np.mean(multi)), float(np.mean(ks)),
            np.array(lams), np.array(nulls))


def residual_mode_separation(R, kmax=10, seed=0, min_size=20, pca_dim=16,
                             n_null=20):
    """残差的多模态性与分离度。

    返回 (Lam, K_star, Lam_null_mean, Lam_null_p95, multimodal):
      K_star 由 Calinski-Harabasz 准则在 1..kmax 中选择（K=1 时 Lam=0）；
      Lam_null_* 是**参数化自助零假设**（与残差同协方差的各向同性高斯）下的
      同统计量分布，用于判定"这个分离度是否超出单峰噪声能产生的范围"。
    """
    R = _as2(R)
    n, d = R.shape
    if n < 4 * min_size:
        return 0.0, 1, 0.0, 0.0, False
    Rr = R
    if d > pca_dim:                      # 高维时先投到主子空间（模态结构通常低维）
        Rr, _, _ = pca_project(R, n_comp=pca_dim)
    kmax = max(1, min(kmax, n // min_size))
    best_score, best_K = 0.0, 1
    for K in range(2, kmax + 1):
        C, lab, _ = kmeans_lloyd(Rr, K, n_init=3, iters=30, seed=seed + K)
        if len(np.unique(lab)) < K:
            continue
        s = calinski_harabasz(Rr, lab)
        if s > best_score:
            best_score, best_K = s, K
    if best_K == 1:
        return 0.0, 1, 0.0, 0.0, False
    Lam, _ = _lam_of(Rr, best_K, seed)
    # 零假设：与残差二阶矩匹配的单峰高斯
    rng = np.random.default_rng(seed + 12345)
    mu = Rr.mean(0)
    cov = np.cov(Rr.T) + 1e-10 * np.eye(Rr.shape[1])
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov)))
    nulls = []
    for b in range(n_null):
        Z = rng.normal(size=(len(Rr), Rr.shape[1])) @ L.T + mu
        lb, _ = _lam_of(Z, best_K, seed + 777 + b)
        nulls.append(lb)
    nulls = np.array(nulls)
    p95 = float(np.percentile(nulls, 95))
    return float(Lam), int(best_K), float(nulls.mean()), p95, bool(Lam > p95)


def marginal_separation(X, kmax=10, seed=0, min_size=20, pca_dim=16):
    """对照量：目标**边际**的簇间方差占比（完全不考虑条件）。"""
    X = _as2(X)
    n, d = X.shape
    Xr = X
    if d > pca_dim:
        Xr, _, _ = pca_project(X, n_comp=pca_dim)
    kmax = max(1, min(kmax, n // min_size))
    best_score, best_K, best = 0.0, 1, None
    for K in range(2, kmax + 1):
        C, lab, _ = kmeans_lloyd(Xr, K, n_init=3, iters=30, seed=seed + K)
        if len(np.unique(lab)) < K:
            continue
        s = calinski_harabasz(Xr, lab)
        if s > best_score:
            best_score, best_K, best = s, K, (C, lab)
    if best is None:
        return 0.0, 1
    C, lab = best
    within = ((Xr - C[lab]) ** 2).sum(axis=1).mean()
    # within 已经对所有坐标求和，total 也只应把逐维方差求和一次。
    # 旧代码额外乘以维度，使高维数据的分离度虚高并破坏对照实验。
    total = Xr.var(axis=0).sum() + 1e-12
    return float(np.clip(1.0 - within / total, 0.0, 1.0)), int(best_K)


def compute_discriminant(Ctr, Xtr, Cte, Xte, estimator="knn", k=10, kmax=10,
                         seed=0, n_rff=256, alpha=1.0, with_competitors=True,
                         k_list=(10, 25, 50)):
    """判别量 + 一组对照预测因子（供论文表 2 的秩相关比较）。

    估计器口径（重要）：
      D 的本质是"**最优**条件均值回归的残差方差占比"，它是 Var(x|c)/Var(x) ∈ [0,1]。
      但任何**有限样本**回归器都会把自己的估计方差 Var(m_hat) 掺进残差，于是
      D_hat ≈ D + Var(m_hat)/Var(x)。最直白的是 kNN：残差里含 ≈ 1/k_eff 的
      估计噪声，**会让 D_hat 突破理论上界 1**。实测 K=1（条件完全无信息，
      闭式 D 恰为 1.000）时 kNN(k=10) 给出 1.143 —— 这是正的 14% 偏差。

      固定 k 的 kNN、固定特征数的 RFF 和线性模型都不能笼统称为“一致估计器”。
      这里用训练集内部的验证划分选择一个候选，再在独立测试集上报告 D_hat。
      这样测试集只承担一次最终评估，避免旧实现“在同一测试集上取五个风险
      的最小值”造成的乐观选择偏差。各候选的测试风险仍保留供诊断。
    """
    Ctr, Xtr, Cte, Xte = _as2(Ctr), _as2(Xtr), _as2(Cte), _as2(Xte)
    var_total = float(Xte.var(axis=0).sum() + 1e-12)

    specs = [("knn%d" % kk, "knn", {"k": kk}) for kk in k_list]
    specs += [("linear", "linear", {"alpha": alpha}),
              ("rff", "rff", {"n_rff": n_rff, "alpha": alpha})]
    rng = np.random.default_rng(seed + 1709)
    order = rng.permutation(len(Ctr))
    n_val = min(max(200, len(Ctr) // 5), max(1, len(Ctr) // 3))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    if len(fit_idx) < 2:
        fit_idx, val_idx = order, order
    val_total = float(Xtr[val_idx].var(axis=0).sum() + 1e-12)
    selection_scores = {}
    for name, est, kw in specs:
        pred = fit_conditional_mean(Ctr[fit_idx], Xtr[fit_idx], estimator=est,
                                    seed=seed, **kw)
        selection_scores[name] = d_hat_from_predictor(
            pred, Ctr[val_idx], Xtr[val_idx], val_total)
    selected = min(selection_scores, key=selection_scores.get)

    ens = {}
    for name, est, kw in specs:
        pred = fit_conditional_mean(Ctr, Xtr, estimator=est, seed=seed, **kw)
        ens[name] = d_hat_from_predictor(pred, Cte, Xte, var_total)
    mean_pred = fit_conditional_mean(Ctr, Xtr, estimator="mean", seed=seed)
    ens["mean_null"] = d_hat_from_predictor(mean_pred, Cte, Xte, var_total)

    members = {name: value for name, value in ens.items() if name != "mean_null"}
    D_ens = float(members[selected])
    D_hat_knn = float(ens.get("knn%d" % k, ens["knn10"]))

    Lam, lam_null, frac_multi, k_mean, _, _ = conditional_mode_separation(
        Ctr, Xtr, n_anchor=48, n_nb=150, kmax=kmax, seed=seed)
    out = dict(D_hat=D_ens, D_hat_knn=D_hat_knn, D_ens=ens,
               Lam_hat=Lam, Lam_null=lam_null,
               Lam_excess=float(Lam - lam_null),
               Lam_ratio=float(Lam / max(lam_null, 1e-9)),
               K_star=k_mean, frac_multi=frac_multi,
               D_hat_pair=d_hat_pair(Cte, Xte),
               var_total=var_total, dim=int(Xte.shape[1]),
               D_selection_scores=selection_scores,
               D_hat_estimator="validation-selected:%s" % selected)
    if with_competitors:
        out["D_lin"] = ens["linear"]
        out["D_rff"] = ens["rff"]
        sep_m, km = marginal_separation(Xte, kmax=kmax, seed=seed)
        out["sep_marg"] = sep_m
        out["K_marg"] = km
    return out
