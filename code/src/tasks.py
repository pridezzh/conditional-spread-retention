# -*- coding: utf-8 -*-
"""Three families of controlled tasks: 2-D ring mixture of Gaussians / conditional
images (MNIST) / multi-objective continuous control (offline RL).

Design principles (source of paper §5.1):
  - **The marginal distribution of the target is held fixed; only "how much the
    condition c explains the target x" and "the geometry of the unexplained part"
    are varied.** This way the gap between one step and many steps can only be
    attributed to structure, not data difficulty or network capacity.
  - The three task families respectively cover: computable ground-truth modes (toy),
    high-dimensional real data (images), and real closed-loop returns (reinforcement
    learning).
"""
import gzip
import os
import struct

import numpy as np
import torch

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)


# --------------------------------------------------------------------------
# 1) 2-D (generalizable to d-D) ring mixture of Gaussians: ground-truth mode
#    structure is known
# --------------------------------------------------------------------------
class RingMixture(object):
    """Conditional mixture of Gaussians: x = mu_k + sigma*xi,  c = mu_k^c + eta*zeta,
    k ~ U[K].

    Three controlled dimensions:
      K             number of modes (1,2,4,8,16)
      sep_ratio    R / sigma, ratio of mode spacing to mode width (topology axis)
      eta          context noise; smaller means c determines k more (condition-strength axis)
      d             target dimension (default 2, can rise to 8 to test dimension effects)
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
        """Exact posterior p(k | c) (context noise is isotropic Gaussian)."""
        c = np.atleast_2d(c)
        d2 = ((c[:, None, :] - self.ctx_centers[None, :, :]) ** 2).sum(-1)
        logp = -d2 / (2.0 * max(self.eta, 1e-6) ** 2)
        logp -= logp.max(axis=1, keepdims=True)
        w = np.exp(logp)
        w /= w.sum(axis=1, keepdims=True)
        return w  # (n, K)

    def true_conditional_samples(self, c, m):
        """Sample m samples from the true conditional distribution p(x|c) per context."""
        c = np.atleast_2d(c)
        w = self.posterior_weights(c)                       # (n,K)
        n = c.shape[0]
        out = np.empty((n, m, self.d), dtype=np.float32)
        for i in range(n):
            k = self.rng.choice(self.K, size=m, p=w[i])
            out[i] = self.centers[k] + self.sigma * self.rng.normal(size=(m, self.d))
        return out

    def analytic_D(self, n=40000, seed=0):
        """**Closed-form** value of the population discriminant D (toy task only; used
        to calibrate the estimator).

        Var(x|c) = sum_k w_k(c) [ ||mu_k - m(c)||^2 + d*sigma^2 ] (trace of the mixture
        covariance)
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
# 2) MNIST conditional images: target marginal fixed, only condition strength varies
# --------------------------------------------------------------------------
def _download(url, dst):
    import subprocess
    for attempt in range(5):
        p = subprocess.run(["curl", "-sSL", "-m", "240", "-A", "Mozilla/5.0",
                            "-o", dst, url],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return True
        print("   retrying download", url, str(p.stderr[:100]))
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
    """Load MNIST; optionally return the data source to avoid silently falling back yet
    still labeling the result as MNIST."""
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
            print("   MNIST parsing failed:", ex)
            xtr = None
    if xtr is None:
        from sklearn.datasets import load_digits
        print("   [fallback] using sklearn's built-in 8x8 digits dataset")
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
    """Nearest-neighbor block-average resize (no cv2/torchvision dependency)."""
    h, w = im.shape
    if h == size:
        return im
    bh, bw = h // size, w // size
    return im[:size * bh, :size * bw].reshape(size, bh, size, bw).mean(axis=(1, 3))


def make_image_condition(x, y, mode):
    """Compress image x into condition c. mode controls condition strength (weaker ->
    larger discriminant)."""
    n, h, w = x.shape
    if mode == "full":        # 8x8 coarsening + label: strong condition
        coarse = _resize_batch(x, 8).reshape(n, -1)
        onehot = _onehot(y, 10)
        return np.concatenate([coarse, onehot], axis=1).astype(np.float32)
    if mode == "mid":         # 4x4 coarsening + label
        coarse = _resize_batch(x, 4).reshape(n, -1)
        return np.concatenate([coarse, _onehot(y, 10)], axis=1).astype(np.float32)
    if mode == "weak":        # label only
        return _onehot(y, 10).astype(np.float32)
    if mode == "half":        # only the left half of the image + label
        half = x[:, :, : w // 2]
        coarse = _resize_batch(half, 4).reshape(n, -1)
        return np.concatenate([coarse, _onehot(y, 10)], axis=1).astype(np.float32)
    if mode == "none":        # unconditional
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
# 3) Multi-objective continuous control: offline dataset + closed-loop evaluation env
# --------------------------------------------------------------------------
class MultiGoalNav(object):
    """2-D continuous navigation; K goals uniformly placed on a circle of radius 0.8.

    - State s ∈ [-1,1]^2, action a ∈ R^2 (velocity, ||a|| ≤ 0.12), max steps T=40;
    - Reaching within 0.12 of the true goal succeeds and terminates, sparse reward +1,
      otherwise 0;
    - Context c = (s, goal observation); the strength of the goal observation is the
      **condition-strength axis**:
        'exact'  : goal one-hot (fully determined)
        'noisy'  : one-hot replaced by a uniform distribution with probability 1-p (p tunable)
        'partial': only the goal's quadrant is given (4 classes)
        'none'   : no goal information (weakest condition -> multi-modal conditional action dist.)
    - Action block horizon H: generate H steps of actions at once (H=1/4/8), corresponding
      to the "lengthened horizon" ablation of C19/C20.
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
        # Action block: concatenate the expert actions of the next H steps (using a
        # straight-line extrapolation approximation)
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
        """Pre-draw n episodes' (initial state, true goal, whether goal is revealed), so
        that policies with different NFEs are compared on **exactly the same episodes**
        (paired comparison, reduced variance)."""
        rng = np.random.default_rng(seed)
        s0 = rng.uniform(-1, 1, size=(n, 2))
        gi = rng.integers(0, self.K, size=n)
        reveal = rng.random(n) < p_obs
        return s0, gi, reveal

    def context_one(self, s, gi, ctx, reveal=True):
        """Construct a single context (used at evaluation, introduces no new randomness)."""
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
        """Closed-loop evaluation on pre-drawn episodes; action blocks are executed
        open-loop for H steps."""
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
        """Closed-loop evaluation: policy_fn(s, g_idx) -> action (or action block)."""
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
