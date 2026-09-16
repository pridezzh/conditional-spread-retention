# -*- coding: utf-8 -*-
"""三类受控任务：二维环形混合高斯 / 条件图像（MNIST）/ 多目标连续控制（离线 RL）。

设计原则（论文 §5.1 的来源）：
  - **目标的边际分布不变，只改变"条件 c 对目标 x 的解释程度"与"不可解释部分的几何"**。
    这样一步与多步的差距就只能归因于结构，而不是数据难度或网络容量。
  - 三个任务族分别覆盖：可计算真值模态（玩具）、高维真实数据（图像）、
    真实闭环回报（强化学习）。
"""
import gzip
import os
import struct

import numpy as np
import torch

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)


# --------------------------------------------------------------------------
# 1) 二维（可推广到 d 维）环形混合高斯：真值模态结构已知
# --------------------------------------------------------------------------
class RingMixture(object):
    """条件混合高斯：x = mu_k + sigma*xi,  c = mu_k^c + eta*zeta,  k ~ U[K].

    三个受控维度：
      K            模态数（1,2,4,8,16）
      sep_ratio    R / sigma，模态间距与模态宽度之比（拓扑轴）
      eta          上下文噪声，越小则 c 越能确定 k（条件强度轴）
      d            目标维度（默认 2，可升到 8 检验维度效应）
    """

    def __init__(self, K=8, sep_ratio=6.0, sigma=0.12, eta=0.25, d=2,
                 seed=0, n_train=40000):
        self.K = K
        self.sigma = sigma
        self.eta = eta
        self.d = d
        self.R = sep_ratio * sigma
        rng = np.random.default_rng(seed)
        self.centers = self._make_centers(rng, K, d, self.R)
        self.ctx_centers = self._make_centers(
            np.random.default_rng(seed + 1000), K, d, 1.0)
        self.rng = np.random.default_rng(seed + 7)
        self.n_train = n_train

    @staticmethod
    def _make_centers(rng, K, d, R):
        if d == 2:
            ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
            return R * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        C = rng.normal(size=(K, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        return R * C

    def sample(self, n):
        k = self.rng.integers(0, self.K, size=n)
        x = self.centers[k] + self.sigma * self.rng.normal(size=(n, self.d))
        c = self.ctx_centers[k] + self.eta * self.rng.normal(size=(n, self.d))
        return c.astype(np.float32), x.astype(np.float32), k

    def posterior_weights(self, c):
        """p(k | c) 的精确后验（上下文噪声为各向同性高斯）。"""
        c = np.atleast_2d(c)
        d2 = ((c[:, None, :] - self.ctx_centers[None, :, :]) ** 2).sum(-1)
        logp = -d2 / (2.0 * max(self.eta, 1e-6) ** 2)
        logp -= logp.max(axis=1, keepdims=True)
        w = np.exp(logp)
        w /= w.sum(axis=1, keepdims=True)
        return w  # (n, K)

    def true_conditional_samples(self, c, m):
        """从真实条件分布 p(x|c) 采样 m 个样本 / 每个上下文。"""
        c = np.atleast_2d(c)
        w = self.posterior_weights(c)                       # (n,K)
        n = c.shape[0]
        out = np.empty((n, m, self.d), dtype=np.float32)
        for i in range(n):
            k = self.rng.choice(self.K, size=m, p=w[i])
            out[i] = self.centers[k] + self.sigma * self.rng.normal(size=(m, self.d))
        return out

    def analytic_D(self, n=40000, seed=0):
        """总体判别量 D 的**闭式值**（仅玩具任务可用，用来校准估计器）。

        Var(x|c) = sum_k w_k(c) [ ||mu_k - m(c)||^2 + d*sigma^2 ]（混合分布的协方差迹）
        D        = E_c[tr Var(x|c)] / tr Var(x)
        """
        rng = np.random.default_rng(seed + 555)
        k = rng.integers(0, self.K, size=n)
        c = self.ctx_centers[k] + self.eta * rng.normal(size=(n, self.d))
        w = self.posterior_weights(c)                       # (n,K)
        m = w @ self.centers                                # (n,d)
        d2 = ((self.centers[None, :, :] - m[:, None, :]) ** 2).sum(-1)   # (n,K)
        var_cond = (w * (d2 + self.d * self.sigma ** 2)).sum(1).mean()
        total = ((self.centers - self.centers.mean(0)) ** 2).sum(-1).mean() \
            + self.d * self.sigma ** 2
        return float(var_cond / (total + 1e-12))

    def arrays(self, n_train=None, n_test=4000):
        n = n_train or self.n_train
        ctr, xtr, _ = self.sample(n)
        cte, xte, _ = self.sample(n_test)
        return (ctr, xtr), (cte, xte)


# --------------------------------------------------------------------------
# 2) MNIST 条件图像：目标边际不变，只变条件强度
# --------------------------------------------------------------------------
def _download(url, dst):
    import subprocess
    for attempt in range(5):
        p = subprocess.run(["curl", "-sSL", "-m", "240", "-A", "Mozilla/5.0",
                            "-o", dst, url],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return True
        print("   重试下载", url, str(p.stderr[:100]))
    return False


def _load_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 2051:
        h, w = struct.unpack(">II", raw[8:16])
        return np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, h, w).astype(np.float32) / 255.0
    return np.frombuffer(raw[8:], dtype=np.uint8).astype(np.int64)


def load_mnist(size=16, max_train=40000, max_test=8000, seed=0,
               return_metadata=False):
    """加载 MNIST；可选返回数据来源，避免静默回退后仍把结果标成 MNIST。"""
    cache = os.path.join(DATA, "mnist_%d.npz" % size)
    if os.path.exists(cache):
        z = np.load(cache)
        source = str(z["source"].item()) if "source" in z.files else (
            "mnist" if len(z["xtr"]) > 1200 else "sklearn_digits_legacy_cache")
        values = (z["xtr"], z["ytr"], z["xte"], z["yte"])
        return values + ({"dataset_source": source, "cache": cache},) if return_metadata else values

    base = "https://ossci-datasets.s3.amazonaws.com/mnist"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    ok = True
    for fn in files:
        dst = os.path.join(DATA, fn)
        if not os.path.exists(dst):
            ok = ok and _download(base + "/" + fn, dst)
    xtr = ytr = xte = yte = None
    if ok:
        try:
            xtr = _load_idx(os.path.join(DATA, files[0]))
            ytr = _load_idx(os.path.join(DATA, files[1]))
            xte = _load_idx(os.path.join(DATA, files[2]))
            yte = _load_idx(os.path.join(DATA, files[3]))
        except Exception as ex:
            print("   MNIST 解析失败:", ex)
            xtr = None
    if xtr is None:
        from sklearn.datasets import load_digits
        print("   [回退] 使用 sklearn 自带 8x8 digits 数据集")
        d = load_digits()
        x = d.images.astype(np.float32) / 16.0
        y = d.target.astype(np.int64)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        xtr, ytr = x[perm[:1200]], y[perm[:1200]]
        xte, yte = x[perm[1200:1600]], y[perm[1200:1600]]
        xtr = np.stack([_resize(im, size) for im in xtr])
        xte = np.stack([_resize(im, size) for im in xte])
        source = "sklearn_digits"
    else:
        xtr = np.stack([_resize(im, size) for im in xtr[:max_train]])
        xte = np.stack([_resize(im, size) for im in xte[:max_test]])
        ytr, yte = ytr[:max_train], yte[:max_test]
        source = "mnist"
    np.savez_compressed(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte,
                        source=np.array(source))
    values = (xtr, ytr, xte, yte)
    return values + ({"dataset_source": source, "cache": cache},) if return_metadata else values


def _resize(im, size):
    """最近邻-块平均缩放（不依赖 cv2/torchvision）。"""
    h, w = im.shape
    if h == size:
        return im
    bh, bw = h // size, w // size
    return im[:size * bh, :size * bw].reshape(size, bh, size, bw).mean(axis=(1, 3))


def make_image_condition(x, y, mode):
    """把图像 x 压成条件 c。mode 控制条件强度（越弱 -> 判别量越大）。"""
    n, h, w = x.shape
    if mode == "full":        # 8x8 粗化 + 标签：强条件
        coarse = _resize_batch(x, 8).reshape(n, -1)
        onehot = _onehot(y, 10)
        return np.concatenate([coarse, onehot], axis=1).astype(np.float32)
    if mode == "mid":         # 4x4 粗化 + 标签
        coarse = _resize_batch(x, 4).reshape(n, -1)
        return np.concatenate([coarse, _onehot(y, 10)], axis=1).astype(np.float32)
    if mode == "weak":        # 只有标签
        return _onehot(y, 10).astype(np.float32)
    if mode == "half":        # 只看到左半张图 + 标签
        half = x[:, :, : w // 2]
        coarse = _resize_batch(half, 4).reshape(n, -1)
        return np.concatenate([coarse, _onehot(y, 10)], axis=1).astype(np.float32)
    if mode == "none":        # 无条件
        return np.zeros((n, 1), dtype=np.float32)
    raise ValueError(mode)


def _resize_batch(x, size):
    n, h, w = x.shape
    bh, bw = h // size, w // size
    return x[:, :size * bh, :size * bw].reshape(n, size, bh, size, bw).mean(axis=(2, 4))


def _onehot(y, k):
    n = len(y)
    o = np.zeros((n, k), dtype=np.float32)
    o[np.arange(n), y] = 1.0
    return o


# --------------------------------------------------------------------------
# 3) 多目标连续控制：离线数据集 + 可闭环评估的环境
# --------------------------------------------------------------------------
class MultiGoalNav(object):
    """2D 连续导航；K 个目标均匀分布在半径 0.8 的圆上。

    - 状态 s ∈ [-1,1]^2，动作 a ∈ R^2（速度，||a|| ≤ 0.12），步长上限 T=40；
    - 到达真实目标 0.12 以内即成功并终止，稀疏奖励 +1，否则 0；
    - 上下文 c = (s, 目标观测)；目标观测的强度是**条件强度轴**：
        'exact'  : 目标 one-hot（完全确定）
        'noisy'  : one-hot 以概率 1-p 被换成均匀分布（p 可调）
        'partial': 只给目标所在象限（4 类）
        'none'   : 不给目标信息（条件最弱 -> 条件动作分布 K 峰）
    - 动作块 horizon H：一次生成 H 步动作（H=1/4/8），对应 C19/C20 的"拉长 horizon"消融。
    """

    def __init__(self, K=6, T=40, radius=0.8, seed=0):
        ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
        self.goals = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
        self.K = K
        self.T = T
        self.rng = np.random.default_rng(seed)

    def expert_action(self, s, g):
        v = g - s
        n = np.linalg.norm(v) + 1e-8
        a = v / n * 0.12
        return a

    def build_dataset(self, n=40000, ctx="none", p_obs=0.5, H=1, seed=0,
                      sigma_a=0.02):
        rng = np.random.default_rng(seed)
        s = rng.uniform(-1, 1, size=(n, 2))
        g_idx = rng.integers(0, self.K, size=n)
        g = self.goals[g_idx]
        a = np.stack([self.expert_action(s[i], g[i]) for i in range(n)])
        a = a + sigma_a * rng.normal(size=a.shape)
        # 动作块：把未来 H 步的专家动作拼起来（用直线外推近似）
        if H > 1:
            blocks = [a]
            s_cur = s.copy()
            for h in range(1, H):
                s_cur = np.clip(s_cur + blocks[-1], -1, 1)
                blocks.append(np.stack([self.expert_action(s_cur[i], g[i])
                                        for i in range(n)]) +
                              sigma_a * rng.normal(size=(n, 2)))
            a = np.concatenate(blocks, axis=1)
        c = self._context(s, g_idx, ctx, p_obs, rng)
        return c.astype(np.float32), a.astype(np.float32), s.astype(np.float32), g_idx

    def _context(self, s, g_idx, ctx, p_obs, rng):
        n = len(s)
        if ctx == "exact":
            return np.concatenate([s, self._obs(g_idx, 1.0, rng)], axis=1)
        if ctx == "noisy":
            return np.concatenate([s, self._obs(g_idx, p_obs, rng)], axis=1)
        if ctx == "partial":
            ang = np.arctan2(self.goals[g_idx][:, 1], self.goals[g_idx][:, 0])
            quad = np.floor((ang + np.pi) / (np.pi / 2)).astype(np.int64)
            quad = np.clip(quad, 0, 3)
            onehot = np.zeros((n, 4), dtype=np.float32)
            onehot[np.arange(n), quad] = 1.0
            return np.concatenate([s, onehot], axis=1)
        if ctx == "none":
            return s.copy()
        raise ValueError(ctx)

    def _obs(self, g_idx, p, rng):
        n = len(g_idx)
        onehot = np.zeros((n, self.K), dtype=np.float32)
        onehot[np.arange(n), g_idx] = 1.0
        if p >= 1.0:
            return onehot
        keep = rng.random(n) < p
        unif = np.full((n, self.K), 1.0 / self.K, dtype=np.float32)
        onehot[~keep] = unif[~keep]
        return onehot

    def sample_episodes(self, n, seed, p_obs=0.5):
        """预先抽好 n 个回合的 (初始状态, 真实目标, 是否揭示目标)，
        使不同 NFE 的策略在**完全相同的回合**上比较（配对比较，降方差）。"""
        rng = np.random.default_rng(seed)
        s0 = rng.uniform(-1, 1, size=(n, 2))
        gi = rng.integers(0, self.K, size=n)
        reveal = rng.random(n) < p_obs
        return s0, gi, reveal

    def context_one(self, s, gi, ctx, reveal=True):
        """构造单条上下文（评估时用，不引入新的随机性）。"""
        s = np.asarray(s, dtype=np.float32).reshape(1, 2)
        g_idx = np.asarray([gi], dtype=np.int64)
        if ctx in ("exact", "noisy"):
            onehot = np.zeros((1, self.K), dtype=np.float32)
            if reveal:
                onehot[0, gi] = 1.0
            else:
                onehot[0, :] = 1.0 / self.K
            return np.concatenate([s, onehot], axis=1).astype(np.float32)
        if ctx == "partial":
            ang = np.arctan2(self.goals[gi][1], self.goals[gi][0])
            quad = int(np.clip(np.floor((ang + np.pi) / (np.pi / 2)), 0, 3))
            onehot = np.zeros((1, 4), dtype=np.float32)
            onehot[0, quad] = 1.0
            return np.concatenate([s, onehot], axis=1).astype(np.float32)
        if ctx == "none":
            return s.astype(np.float32)
        raise ValueError(ctx)

    def run_episodes(self, policy_fn, episodes, ctx, H=1, max_T=40):
        """在预先抽好的回合上闭环评估；动作块按 open-loop 执行 H 步。"""
        s0, gi, reveal = episodes
        succ, ret = [], []
        for i in range(len(s0)):
            s = s0[i].copy()
            g = self.goals[int(gi[i])]
            done, R, t = False, 0.0, 0
            while not done and t < max_T:
                c = self.context_one(s, int(gi[i]), ctx, bool(reveal[i]))
                a = np.asarray(policy_fn(c), dtype=np.float32).reshape(-1)
                n_blk = len(a) // 2
                for h in range(n_blk):
                    ah = a[2 * h:2 * h + 2]
                    na = np.linalg.norm(ah)
                    if na > 0.12:
                        ah = ah / na * 0.12
                    s = np.clip(s + ah, -1.2, 1.2)
                    t += 1
                    if np.linalg.norm(s - g) < 0.12:
                        R += 1.0
                        done = True
                        break
                    if t >= max_T:
                        break
            succ.append(1.0 if done else 0.0)
            ret.append(R)
        return float(np.mean(succ)), float(np.mean(ret)), np.array(succ)

    def evaluate(self, policy_fn, n_episodes=200, seed=0, H=1):
        """闭环评估：policy_fn(s, g_idx) -> 动作（或动作块）。"""
        rng = np.random.default_rng(seed)
        succ, ret = [], []
        for _ in range(n_episodes):
            s = rng.uniform(-1, 1, size=2)
            gi = int(rng.integers(0, self.K))
            g = self.goals[gi]
            done = False
            R = 0.0
            t = 0
            while not done and t < self.T:
                a = policy_fn(s, gi)
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                a = a[:2]
                na = np.linalg.norm(a)
                if na > 0.12:
                    a = a / na * 0.12
                s = np.clip(s + a, -1.2, 1.2)
                R += 1.0 if np.linalg.norm(s - g) < 0.12 else 0.0
                if np.linalg.norm(s - g) < 0.12:
                    done = True
                t += 1
            succ.append(1.0 if done else 0.0)
            ret.append(R)
        return float(np.mean(succ)), float(np.mean(ret))
