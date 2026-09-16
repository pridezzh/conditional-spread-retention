# -*- coding: utf-8 -*-
"""评测指标：全部为公开、可独立实现的度量（不依赖任何预训练网络）。

  sliced_wasserstein  切片 Wasserstein-2（Kolouri et al. 2019 的广义切片 W 距离特例）
  energy_distance     （等价于一阶能量距离，作为稳健性交叉校验）
  mode_recall/precision  真值模态已知时的模态覆盖/塌缩率（玩具任务）
  prdc                Kynkaanniemi et al., NeurIPS 2019 的改进 precision/recall（图像任务）
"""
import numpy as np


def conditional_reference_samples(C_ref, X_ref, C_query, n_ref=256, seed=0,
                                  exclude_indices=None):
    """仅依据可观测条件，为每个查询条件构造经验条件参考集。

    该函数故意不接收类别标签。图像实验旧实现按真实数字标签选参考图，即使
    ``mode='none'`` 时条件中根本没有标签，因而把不可用信息泄漏进了指标。

    连续条件用标准化后的条件空间近邻；条件完全常数时从整个参考集独立抽样，
    对应经验边际分布。``exclude_indices[i]`` 可排除查询样本自身。
    """
    C_ref = np.asarray(C_ref, dtype=np.float64)
    C_query = np.asarray(C_query, dtype=np.float64)
    X_ref = np.asarray(X_ref)
    if C_ref.ndim == 1:
        C_ref = C_ref[:, None]
    if C_query.ndim == 1:
        C_query = C_query[:, None]
    if len(C_ref) != len(X_ref):
        raise ValueError("C_ref 与 X_ref 的样本数必须一致")
    if n_ref < 1:
        raise ValueError("n_ref 必须为正整数")
    n_take = min(int(n_ref), len(C_ref) - (1 if exclude_indices is not None else 0))
    if n_take < 1:
        raise ValueError("参考集样本不足")

    scale = C_ref.std(axis=0)
    active = scale > 1e-12
    center = C_ref.mean(axis=0)
    z_ref = ((C_ref[:, active] - center[active]) / scale[active]
             if np.any(active) else None)
    rng = np.random.default_rng(seed)
    refs = []
    for i, cq in enumerate(C_query):
        banned = None if exclude_indices is None else int(exclude_indices[i])
        if not np.any(active):
            pool = np.arange(len(C_ref))
            if banned is not None:
                pool = pool[pool != banned]
            idx = rng.choice(pool, size=n_take, replace=False)
        else:
            z_q = (cq[active] - center[active]) / scale[active]
            d2 = ((z_ref - z_q) ** 2).sum(axis=1)
            if banned is not None:
                d2[banned] = np.inf
            # 随机微扰只打破完全相同条件的并列，不改变非并列距离的次序。
            d2 = d2 + rng.uniform(0.0, 1e-12, size=len(d2))
            idx = np.argpartition(d2, n_take - 1)[:n_take]
        refs.append(X_ref[idx])
    return refs


def sliced_wasserstein(A, B, n_proj=1024, seed=0, p=2):
    """样本间切片 W_2 距离（对投影方向做蒙特卡洛）。"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    d = A.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_proj, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    pa = np.sort(A @ W.T, axis=0)
    pb = np.sort(B @ W.T, axis=0)
    n = min(len(A), len(B))
    if len(A) != len(B):
        qa = np.quantile(A @ W.T, np.linspace(0, 1, n), axis=0)
        qb = np.quantile(B @ W.T, np.linspace(0, 1, n), axis=0)
        pa, pb = qa, qb
    val = np.mean(np.abs(pa - pb) ** p) ** (1.0 / p)
    return float(val)


def energy_distance(A, B, seed=0, max_n=2000):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]

    def _cross(X, Y):
        return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) + 1e-12).mean()

    return float(2 * _cross(A, B) - _cross(A, A) - _cross(B, B))


def mode_stats(samples, centers, tol=None):
    """真值模态已知时的模态统计。

    关键口径：样本只有落在**某个模态的容差球内**才算"命中该模态"；
    否则记为 off-mode。若不做这一步，塌缩到环心的一团样本会被"最近中心"
    均匀分摊给各模态，从而虚假地报告 recall=1。

    返回 dict：
      recall   : 经验质量 ≥ 名义质量一半的模态占比（模态召回率）
      drop     : 经验质量 < 名义质量 10% 的模态占比（模态塌缩率）
      off_frac : 不在任何模态容差球内的样本占比（一步塌缩的直接证据）
      min_ratio: 最差模态的经验质量 / 名义质量
      entropy_ratio : 经验模态分布熵 / 均匀熵
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    K = len(centers)
    d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    lab = d2.argmin(axis=1)
    dmin = d2.min(axis=1)
    if tol is None:
        # 默认容差 = 最近的两个模态中心距离的一半
        cd = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(cd, np.inf)
        tol = float(cd.min() / 2.0) if K > 1 else np.inf
    off = dmin > tol ** 2
    hit = ~off
    prop = np.bincount(lab[hit], minlength=K) / max(len(samples), 1)
    nominal = 1.0 / K
    recall = float(np.mean(prop >= 0.5 * nominal))
    drop = float(np.mean(prop < 0.1 * nominal))
    min_ratio = float(prop.min() / nominal)
    ent = -(prop[prop > 0] * np.log(prop[prop > 0])).sum()
    return dict(recall=recall, drop=drop, min_ratio=min_ratio,
                off_frac=float(off.mean()),
                entropy_ratio=float(ent / np.log(K)) if K > 1 else 0.0,
                prop=prop.tolist())


def mode_stats_band(samples, centers, radius, sigma, band=3.0, min_share=0.01):
    """环形混合任务的模态统计（**σ 尺度球**口径，用于替代容差球）。

    为什么要换口径（本项目实测踩到两个坑）：
      1) 旧容差取"最近模态间距的一半"，会随 K 变小。K=16、sep_ratio=6 时
         相邻模态间距仅 2R·sin(pi/16)*sigma 尺度上约 2.34σ，半间距 ≈1.17σ
         **比模态自身的径向尺度还小**，真样本被误判为 off-mode（旧口径下
         K=16 的 recall 假性掉到 0.75~0.88）。
      2) K=1 时没有"最近模态间距"，旧口径退化成"最近中心"分摊，**恰好掩盖
         一步采样塌缩**这一最重要的失败模式（旧口径下 K=1 报 recall=1.00）。

    新口径：容差取与 K 无关的 **σ 尺度球** tol = sigma*(band + sqrt(d))：
      off_frac      : 到最近模态中心的距离 > tol 的样本占比
                      —— 一步塌缩（落在环内/模态之间）会直接被记进来
      recall_band   : 球内样本按最近中心分配后，经验质量 ≥ min_share 的模态占比
      drop_band     : 1 - recall_band
      entropy_ratio : 球内经验模态分布的归一化熵
      radial_off    : 径向读数 | ||x|| - R | > tol 的占比（交叉校验，免疫角度错位）
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    K = len(centers)
    d = samples.shape[1]
    tol = float(sigma * (band + np.sqrt(d)))
    d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    lab = d2.argmin(axis=1)
    dmin = np.sqrt(d2[np.arange(len(samples)), lab])
    on = dmin <= tol
    rad = np.linalg.norm(samples, axis=1)
    radial_off = float(np.mean(np.abs(rad - radius) > tol))
    off_frac = float(1.0 - on.mean())
    out = dict(tol=tol, off_frac=off_frac, radial_off=radial_off,
               on_band=bool(on.mean() >= min_share),
               band_frac=float(on.mean()),
               recall_band=0.0, drop_band=1.0, min_share_ratio=0.0,
               entropy_ratio=0.0)
    if not out["on_band"]:
        out["drop_band"] = 1.0
        return out
    prop = np.bincount(lab[on], minlength=K) / int(on.sum())   # 球内经验质量
    recall = float(np.mean(prop >= min_share))
    ent = -(prop[prop > 0] * np.log(prop[prop > 0])).sum()
    out.update(recall_band=recall, drop_band=1.0 - recall,
               min_share_ratio=float(prop.min() / (1.0 / K)),
               entropy_ratio=float(ent / np.log(K)) if K > 1 else 0.0,
               prop_band=prop.tolist())
    return out


def conditional_sw(true_samples, gen_samples, n_proj=512, seed=0):
    """条件 SW：对每个上下文先算 SW，再对上下文取均值。

    true_samples: (n_ctx, m, d)；gen_samples: (n_ctx, m, d)
    """
    true_samples = np.asarray(true_samples, dtype=np.float64)
    gen_samples = np.asarray(gen_samples, dtype=np.float64)
    vals = [sliced_wasserstein(true_samples[i], gen_samples[i],
                               n_proj=n_proj, seed=seed + i)
            for i in range(len(true_samples))]
    return float(np.mean(vals)), float(np.std(vals))


def _pairwise_dist(X, Y, chunk=512):
    """分块计算 ||x-y||^2，避免一次性开 n×n 大矩阵。"""
    n, m = len(X), len(Y)
    out = np.empty((n, m), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = ((X[s:e, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.maximum(out, 0.0))


def prdc(real, fake, k=5, seed=0, max_n=3000):
    """Kynkaanniemi et al., NeurIPS 2019 的 precision / recall / density / coverage。

    precision: 生成样本落在真实样本 k-邻域球内的比例（生成质量）
    recall   : 真实样本被生成样本邻域球覆盖的比例（模态覆盖，丢模态 → 下降）
    density  : 每个生成样本平均被多少个真实球覆盖 / k
    coverage : 至少含一个生成样本的真实球比例
    """
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(real) > max_n:
        real = real[rng.choice(len(real), max_n, replace=False)]
    if len(fake) > max_n:
        fake = fake[rng.choice(len(fake), max_n, replace=False)]
    nr, nf = len(real), len(fake)
    kk = min(k, nr - 1, nf - 1)

    # 真实样本 / 生成样本各自的 k 邻域半径（不含自身）
    Drr = _pairwise_dist(real, real)
    Dff = _pairwise_dist(fake, fake)
    r_real = np.sort(Drr, axis=1)[:, kk]         # 第 kk 近（0 列为自身）
    r_fake = np.sort(Dff, axis=1)[:, kk]

    Drf = _pairwise_dist(real, fake)             # (nr, nf)
    # precision: 每个 fake 找最近的 real，看是否落在其球内
    j = Drf.argmin(axis=0)                       # 每个 fake 最近的 real
    d_fake = Drf[j, np.arange(nf)]
    precision = float(np.mean(d_fake <= r_real[j]))
    # recall: 每个 real 找最近的 fake
    i = Drf.argmin(axis=1)                       # 每个 real 最近的 fake
    d_real = Drf[np.arange(nr), i]
    recall = float(np.mean(d_real <= r_fake[i]))
    # density: 每个 fake 被多少个真实球覆盖 / k
    cnt = (Drf <= r_real[:, None]).sum(axis=0)
    density = float(np.mean(cnt / kk))
    # coverage: 至少覆盖一个 fake 的真实球比例
    coverage = float(np.mean((Drf <= r_real[:, None]).sum(axis=1) > 0))
    return dict(precision=precision, recall=recall, density=density,
                coverage=coverage)


def pca_features(X, n_comp=64, seed=0, fit_on=None):
    """PCA 投影（numpy SVD 实现，不依赖 sklearn）。fit_on 给定时用它拟合基。

    返回 (基样本的低维表示, X 的低维表示)。
    """
    X = np.asarray(X, dtype=np.float64)
    base = X if fit_on is None else np.asarray(fit_on, dtype=np.float64)
    mu = base.mean(0)
    U, S, Vt = np.linalg.svd(base - mu, full_matrices=False)
    W = Vt[:min(n_comp, Vt.shape[0])]
    return (base - mu) @ W.T, (X - mu) @ W.T
