# -*- coding: utf-8 -*-
"""MNIST 16x16 endpoint regression: the minimal non-synthetic verification of the
mean-dependence criterion on real data.

--------------------------------------------------------------------
Design (2026-09-17, P0 minimal implementation of "at least one public, non-synthetic,
no-hardware-baseline" experiment)
--------------------------------------------------------------------
Source x0 ~ N(0, I_256); three couplings share the same source distribution, differing
only in the pairing rule:

  A  independent      y = x~, x~ independent of x0
      E[y|x0]=mu (data mean map) -> L2 optimal map collapses (reproduces the conclusion
      of shou2026nfm)

  B  dependent but mean-independent
      y = mu + a(x0)·(x~ - mu), a(x0) = 1 + 0.8·tanh(x0[1])
      E[y|x0] = mu + a(x0)(E[x~]-mu) = mu: mean-independent;
      but Var(y|x0) = a(x0)^2 Var(x~) varies strongly with x0: statistical dependence
      (significant dCor). -> "dependence" alone does not rescue collapse; only the
      mean-dependence criterion can predict that it will still collapse.

  C  mini-batch OT    per batch, optimal transport assignment between (x0 batch, image batch)
      E[y|x0] ≈ deterministic map -> no collapse.

  D  sorted quantile pairing (greedy mean-dependence maximization construction, added 2026-09-17)
      per batch, sort x0 by x0[:,1] ascending, sort images by their projection on the
      data's first principal axis v1 ascending, and pair rank to rank (O(m log m), no
      assignment solver).
      E[y|x0] ≈ mu + q(x0[:,1])·v1: the mean depends deterministically on x0 through a
      scalar statistic -> the criterion predicts no collapse; but rho(f*(D)) ≈ lambda1/trVar(y)
      (the PC1 explained-variance fraction), so fidelity is lower than full-space OT.
      A/B (collapse) < D (partial release) < C (full release) form a ladder of
      mean-dependence strength: the criterion not only diagnoses, but also points to the fix.

Pre-registered criteria (frozen after seed-0 calibration; thresholds in the JSON's
protocol.thresholds)
  M1  dCor(B) permutation p < 0.01 and dCor(A) permutation p >= 0.05
      (B: all seeds p<0.01; A's 0.05-level false positives scale with the seed count:
      5 seeds allow <=1, 10 seeds allow <=2 -- multiplicity control; the per-seed AND
      would have an ~40% false-rejection probability under 10 seeds)
  M2  rho_hat_corr(A) < 0.10 and rho_hat_corr(B) < 0.10   (data-side diagnostics both predict collapse)
  M3  rho(f_hat)(A) < 0.10 and rho(f_hat)(B) < 0.10       (after training both truly collapse)
  M4  rho(f_hat)(C) > 0.25                        (OT un-collapses after training)
      Note: rho_hat_corr(C) does not enter the criteria -- at d=256, n=1000 the kNN
      estimator returns ≈0 for all couplings (curse of dimensionality: the kNN neighbours'
      OT target is unrelated to the query point), and the estimator's high-dimensional
      failure region is honestly recorded as a paper limitation; on real data the
      criterion's value is verified through the trained map.
  M5  cSW(C) < cSW(A) and cSW(C) < cSW(B)      (fidelity)
  M6  mu_err(A) < 0.5 and mu_err(B) < 0.5      (output mean map lands near mu rather than 0)
  M7  dcor(D) > dcor(A) (per seed)              (the constructed pairing produces detectable dependence)
  M8  rho_tr(D) > rho_tr(A) and > rho_tr(B)     (per-seed directional: no collapse)
  M9  cSW(D) < cSW(A) and < cSW(B)              (per-seed directional: fidelity improves)
  (M7-M9 are directional criteria with no constant threshold; D's permutation p-value is
  only reported as a diagnostic -- the single-coordinate -> single-direction dependence
  effect size is small and the n=1000 permutation test is underpowered, so it is not used
  as a criterion.)

The criteria only state direction and thresholds, not multiples (following the discipline
of run_loss_ladder: a multiple depends on optimization depth and cannot be pre-registered).
rho_hat's 1/k bias is corrected with the paper's own finite-neighbor law:
rho_hat_corr = (rho_hat_raw - 1/k)/(1 - 1/k).

Implementation notes (pitfalls hit during 2026-09-17 calibration):
- dCor uses the U-statistic version: after double-centering, **zero out the diagonal**.
  After centering the distance matrix, A_ii ≈ -2×row-mean (~-40 at 256 dims); including
  the diagonal, dc2 is dominated by the diagonal product (contribution ≈ diag²/n), so
  under high dimensions even independent data gives dcor≈0.7.
- All pairwise distances use the Gram trick (‖u‖²+‖v‖²−2u·v); never construct an
  n×m×d broadcast array.
- The coupling pool uses the full 20000 training images + hidden 128 + weight decay to
  suppress network memorization (2k pool + 384 width gave a spuriously high rho(f_hat) of
  0.63; 20k pool + 384 width still 0.78).
- The C configuration's rho_hat needs a fixed OT pairing pool (per-batch OT during training
  has no fixed pool available).
"""
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")


def _find_root(d):
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "results")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("project root not found")
        d = p


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from provenance import (attach_provenance, protocol_fingerprint,  # noqa: E402
                        require_merge_compatible)

PROTOCOL_FILES = ["code/experiments/run_mnist_collapse.py", "code/src/provenance.py"]

SEEDS = tuple(range(10))
DIM = 256                      # 16x16
K_KNN = 32                     # data-side kNN estimator's neighbour count (same order of magnitude as paper Sec. 6)
N_POOL = 20000                 # coupling pool = full MNIST training set (suppress memorization)
N_OTPOOL = 1000                # C-config fixed OT pool for rho_hat diagnostics
N_DIAG = 2000                  # dCor / rho_hat query sample count
N_EVAL = 4000                  # post-training evaluation sample count
TRAIN_STEPS = 4000
BATCH = 256
HIDDEN = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_PROJ = 256
THRESHOLDS = {"M1_p_B": 0.01, "M1_p_A": 0.05, "M2_rho_hat": 0.10,
              "M3_rho_train": 0.10, "M4_rho_train_C": 0.25,
              "M6_mu_err": 0.5}
torch.set_num_threads(8)


# ---------------------------------------------------------------- basics
def _sqdist(A, B):
    """Pairwise squared Euclidean distance (Gram trick); never constructs an A×B×d broadcast array."""
    aa = (A ** 2).sum(1)[:, None]
    bb = (B ** 2).sum(1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def load_mnist16():
    d = np.load(os.path.join(ROOT, "code", "data", "mnist_16.npz"))
    x = d["xtr"].reshape(len(d["xtr"]), -1).astype(np.float64)
    return x  # (20000, 256), [0,1]


# ---------------------------------------------------------------- couplings
def a_scale(x0):
    """Dependence amplitude for the B pairing: depends only on x0's 1st coordinate; mean unchanged, variance strongly varied."""
    return 1.0 + 0.8 * np.tanh(x0[:, 1:2])


def make_pairs_B(x0, xtilde, mu):
    a = a_scale(x0)
    return mu[None, :] + a * (xtilde - mu[None, :])


def ot_pairs(x0, y):
    """In-batch optimal transport assignment; returns y reordered to x0's row order."""
    r, cc = linear_sum_assignment(_sqdist(x0, y))
    out = np.empty_like(y)
    out[r] = y[cc]
    return out


def pc1_axis(data):
    """First principal-component direction of the image pool (data-driven, no coordinate picking; 256x256 eigen-decomposition cost is negligible)."""
    c = data - data.mean(0)
    cov = (c.T @ c) / len(c)
    _w, v = np.linalg.eigh(cov)
    return v[:, -1]


def sorted_pairs(x0, y, v1):
    """D pairing: within batch, sort x0 by x0[:,1] ascending, sort y by v1 projection
    ascending, and pair rank to rank.

    Makes E[y|x0] ≈ a deterministic function of x0 (greedy construction maximizing
    mean dependence); uses the same coordinate x0[:,1] as B to guarantee comparability.
    """
    ox = np.argsort(x0[:, 1])
    oy = np.argsort(y @ v1)
    out = np.empty_like(y)
    out[ox] = y[oy]
    return out


# ---------------------------------------------------------------- diagnostics
def _center(d):
    """Double centering: d_ij - row_mean_i - col_mean_j + grand_mean (each term added once)."""
    return d - d.mean(0, keepdims=True) - d.mean(1, keepdims=True) + d.mean()


def dcor(x0, y):
    """Distance correlation (U-statistic version: diagonal zeroed after centering, removing the high-dimensional diagonal bias)."""
    a = _center(np.sqrt(_sqdist(x0, x0)))
    b = _center(np.sqrt(_sqdist(y, y)))
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(b, 0.0)
    dc2 = (a * b).mean()
    va = (a * a).mean()
    vb = (b * b).mean()
    if va <= 0 or vb <= 0:
        return 0.0
    return float(np.sqrt(max(dc2, 0.0) / np.sqrt(va * vb)))


def dcor_perm_p(x0, y, rng, n_perm=200):
    """Permutation-test p-value for dcor: permute the rows of y, recompute dc2, and see
    where the observed value falls in the null distribution.

    Why not a fixed threshold: even under independence the dcor estimator retains an
    O(1/sqrt(n)) positive bias and fluctuation (measured dcor=0.058 for an independent
    config at seed 3), so a fixed threshold would overfit the calibration; the p-value
    calibrates the null distribution separately for each seed, making it a statistically
    valid criterion.
    """
    a = _center(np.sqrt(_sqdist(x0, x0)))
    b = _center(np.sqrt(_sqdist(y, y)))
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(b, 0.0)
    obs = float((a * b).mean())
    n = len(y)
    cnt = 0
    for _ in range(n_perm):
        pm = rng.permutation(n)
        if (a * b[pm][:, pm]).mean() >= obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def knn_rho(x0_pool, y_pool, x0_query, y_ref, k=K_KNN):
    """Data-side mean-dependence ratio: kNN regression estimates E[y|x0]; returns (raw, corrected)."""
    preds = np.empty((len(x0_query), y_pool.shape[1]))
    pool_sq = (x0_pool ** 2).sum(1)
    for i in range(0, len(x0_query), 256):
        q = x0_query[i:i + 256]
        d2 = (q ** 2).sum(1)[:, None] + pool_sq[None, :] - 2.0 * (q @ x0_pool.T)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        preds[i:i + 256] = y_pool[idx].mean(axis=1)
    num = np.trace(np.cov(preds, rowvar=False))
    den = np.trace(np.cov(y_ref, rowvar=False))
    raw = float(num / den)
    corrected = float((raw - 1.0 / k) / (1.0 - 1.0 / k))
    return raw, corrected


def sliced_w2(A, B, rng, n_proj=N_PROJ):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    d = A.shape[1]
    th = rng.normal(size=(d, n_proj))
    th /= np.linalg.norm(th, axis=0, keepdims=True)
    pa = np.sort(A @ th, axis=0)
    pb = np.sort(B @ th, axis=0)
    return float(np.mean((pa - pb) ** 2) ** 0.5)


def rho_of_map(outputs, y_ref):
    """Realized conditional spread ratio of the trained map (context-free case): trVar(f_hat)/trVar(y)."""
    num = np.trace(np.cov(outputs, rowvar=False))
    den = np.trace(np.cov(y_ref, rowvar=False))
    return float(num / den)


# ---------------------------------------------------------------- network
def build_net():
    return nn.Sequential(
        nn.Linear(DIM, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, DIM),
    )


def train_net(y_pool_t, seed, x0_pool, data, pair="none", v1=None,
              steps=TRAIN_STEPS, stop_at_baseline=None):
    """pair="none": take batches from the fixed pairing pool y_pool_t indexed by x0;
    "ot": per-step in-batch optimal transport assignment; "sorted": per-step in-batch D sorted pairing.

    stop_at_baseline: A/B's population optimum is the conditional mean mu, whose training
    loss is exactly the mu baseline; stop once the loss EMA drops to the baseline, to
    prevent continued training from sliding into the "memorize the training pairs"
    memorization region (2026-09-17 calibration: without early stopping, loss_ratio drops
    to 0.55-0.61 and rho_tr spuriously rises to 0.44-0.51). C's optimum is far below the
    baseline, so pass None to run the full step count.
    Returns (net, final_loss, steps_done).
    """
    torch.manual_seed(seed)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(7000 + seed)
    x0_t = torch.from_numpy(x0_pool).float()
    loss_v = float("nan")
    ema = None
    steps_done = steps
    for step in range(steps):
        idx = rng.integers(0, len(x0_pool), size=BATCH)
        xb = x0_t[idx]
        if pair == "ot":
            j = rng.integers(0, len(data), size=BATCH)
            yb = torch.from_numpy(ot_pairs(x0_pool[idx], data[j])).float()
        elif pair == "sorted":
            j = rng.integers(0, len(data), size=BATCH)
            yb = torch.from_numpy(sorted_pairs(x0_pool[idx], data[j], v1)).float()
        else:
            yb = y_pool_t[idx]
        loss = ((net(xb) - yb) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_v = float(loss.detach())
        if stop_at_baseline is not None:
            ema = loss_v if ema is None else 0.98 * ema + 0.02 * loss_v
            if step >= 200 and ema <= stop_at_baseline:
                steps_done = step + 1
                break
    return net, loss_v, steps_done


# ---------------------------------------------------------------- main flow
def run_seed(seed, data, mu, smoke=False):
    rng = np.random.default_rng(1000 + seed)
    steps = 800 if smoke else TRAIN_STEPS
    n_pool = 2000 if smoke else N_POOL
    n_diag = 500 if smoke else N_DIAG
    n_eval = 1000 if smoke else N_EVAL
    n_ot = min(N_OTPOOL, n_pool)

    x0_pool = rng.normal(size=(n_pool, DIM))
    xt_pool = data[rng.integers(0, len(data), size=n_pool)]
    x0_diag = rng.normal(size=(n_diag, DIM))
    x0_eval = rng.normal(size=(n_eval, DIM))
    xt_eval = data[rng.integers(0, len(data), size=n_eval)]
    xt_diag = data[rng.integers(0, len(data), size=n_diag)]
    v1 = pc1_axis(data[:n_pool])

    out = {}
    nd = min(1000, n_diag)  # in smoke mode when n_diag<1000, shrink the diagnostic sample accordingly
    for cfg in ("A_independent", "B_dependent_meandep0", "C_ot",
                "D_sorted_meanmax"):
        t0 = time.time()
        # ---- the coupling's training pairs (C does per-batch OT at training time, but diagnostics need a fixed pool)
        if cfg == "A_independent":
            y_pool = xt_pool.copy()
            y_pool_t = torch.from_numpy(y_pool).float()
        elif cfg == "B_dependent_meandep0":
            y_pool = make_pairs_B(x0_pool, xt_pool, mu)
            y_pool_t = torch.from_numpy(y_pool).float()
        elif cfg == "D_sorted_meanmax":
            y_pool = sorted_pairs(x0_pool[:n_ot],
                                  data[rng.integers(0, len(data), size=n_ot)],
                                  v1)
            y_pool_t = None
        else:
            y_pool = ot_pairs(x0_pool[:n_ot],
                              data[rng.integers(0, len(data), size=n_ot)])
            y_pool_t = None

        # ---- data-side diagnostics (before training)
        if cfg == "C_ot":
            diag_pairs = ot_pairs(x0_diag[:nd],
                                  data[rng.integers(0, len(data), size=nd)])
        elif cfg == "A_independent":
            diag_pairs = xt_diag[:nd].copy()
        elif cfg == "D_sorted_meanmax":
            diag_pairs = sorted_pairs(x0_diag[:nd],
                                      data[rng.integers(0, len(data), size=nd)],
                                      v1)
        else:
            diag_pairs = make_pairs_B(x0_diag[:nd],
                                      data[rng.integers(0, len(data), size=nd)], mu)
        dcor_val = dcor(x0_diag[:nd], diag_pairs)
        dcor_p = dcor_perm_p(x0_diag[:nd], diag_pairs,
                             np.random.default_rng(5000 + seed))
        if cfg == "B_dependent_meandep0":
            y_ref_diag = make_pairs_B(x0_diag, xt_diag, mu)
        else:
            y_ref_diag = xt_diag
        rho_raw, rho_corr = knn_rho(x0_pool[:len(y_pool)], y_pool,
                                    x0_diag, y_ref_diag)

        # ---- training (A/B early-stop at the mu baseline = population optimum; C runs full steps)
        if cfg in ("C_ot", "D_sorted_meanmax"):
            stop_bl = None
        else:
            stop_bl = float(((y_pool - y_pool.mean(0)) ** 2).sum(1).mean())
        pair = ("ot" if cfg == "C_ot" else
                "sorted" if cfg == "D_sorted_meanmax" else "none")
        net, final_loss, steps_done = train_net(
            y_pool_t, seed, x0_pool, data, pair=pair, v1=v1,
            steps=steps, stop_at_baseline=stop_bl)

        # ---- post-training evaluation
        with torch.no_grad():
            outs = net(torch.from_numpy(x0_eval).float()).numpy()
        if cfg == "B_dependent_meandep0":
            y_ref_eval = make_pairs_B(x0_eval, xt_eval, mu)
        else:
            y_ref_eval = xt_eval
        rho_tr = rho_of_map(outs, y_ref_eval)
        csw = sliced_w2(outs, y_ref_eval, np.random.default_rng(9000 + seed))
        mu_err = float(np.linalg.norm(outs.mean(0) - y_ref_eval.mean(0))
                       / max(np.linalg.norm(y_ref_eval.mean(0)), 1e-12))
        baseline = float(((y_ref_eval - y_ref_eval.mean(0)) ** 2).sum(1).mean())
        loss_ratio = float(final_loss / max(baseline, 1e-12))

        out[cfg] = {
            "dcor": dcor_val,
            "dcor_p": dcor_p,
            "rho_hat_raw": rho_raw,
            "rho_hat_corrected": rho_corr,
            "rho_trained": rho_tr,
            "csw1": csw,
            "mu_err": mu_err,
            "loss_ratio": loss_ratio,
            "steps_done": steps_done,
            "mean_output": [float(v) for v in outs.mean(0)],
            "seconds": round(time.time() - t0, 1),
        }
        print("  %s dcor=%.4f p=%.4f rho_hat=%.4f/%.4f rho_tr=%.4f csw=%.4f mu_err=%.4f lr=%.3f (%.0fs)"
              % (cfg, dcor_val, dcor_p, rho_raw, rho_corr, rho_tr, csw,
                 mu_err, loss_ratio, time.time() - t0), flush=True)
    return out


def evaluate_checks(per_seed):
    """Pre-registered criteria: judged per seed, one-by-one (any seed failing makes it False)."""
    th = THRESHOLDS
    ps = list(per_seed.values())
    checks = {
        # A side allows 1 false positive under multiple comparisons (each of 5 seeds at
        # the 0.05 level; the probability all are >=0.05 at once is only 0.95^5≈0.77, so
        # the per-seed AND would overfit the calibration)
        "M1_B_is_dependent": (
            all(r["B_dependent_meandep0"]["dcor_p"] < th["M1_p_B"] for r in ps)
            and sum(1 for r in ps
                    if r["A_independent"]["dcor_p"] < th["M1_p_A"])
            <= max(1, len(ps) // 5)),
        "M2_diag_predicts_collapse_A_B": all(
            r["A_independent"]["rho_hat_corrected"] < th["M2_rho_hat"] and
            r["B_dependent_meandep0"]["rho_hat_corrected"] < th["M2_rho_hat"]
            for r in ps),
        "M3_trained_collapse_A_B": all(
            r["A_independent"]["rho_trained"] < th["M3_rho_train"] and
            r["B_dependent_meandep0"]["rho_trained"] < th["M3_rho_train"]
            for r in ps),
        "M4_ot_escapes": all(
            r["C_ot"]["rho_trained"] > th["M4_rho_train_C"] for r in ps),
        "M5_csw_ot_best": all(
            r["C_ot"]["csw1"] < r["A_independent"]["csw1"] and
            r["C_ot"]["csw1"] < r["B_dependent_meandep0"]["csw1"] for r in ps),
        "M6_mean_output_at_mu": all(
            r["A_independent"]["mu_err"] < th["M6_mu_err"] and
            r["B_dependent_meandep0"]["mu_err"] < th["M6_mu_err"] for r in ps),
        "M7_sorted_dependent": all(
            r["D_sorted_meanmax"]["dcor"] > r["A_independent"]["dcor"]
            for r in ps),
        "M8_sorted_noncollapse": all(
            r["D_sorted_meanmax"]["rho_trained"] > r["A_independent"]["rho_trained"]
            and r["D_sorted_meanmax"]["rho_trained"]
            > r["B_dependent_meandep0"]["rho_trained"] for r in ps),
        "M9_sorted_csw_better": all(
            r["D_sorted_meanmax"]["csw1"] < r["A_independent"]["csw1"]
            and r["D_sorted_meanmax"]["csw1"] < r["B_dependent_meandep0"]["csw1"]
            for r in ps),
    }
    return checks


def main(smoke=False, seeds=SEEDS, merge=False):
    t0 = time.time()
    data = load_mnist16()
    mu = data.mean(0)
    name = "mnist_collapse_smoke.json" if smoke else "mnist_collapse.json"
    out = os.path.join(ROOT, "results", name)
    protocol_id, _ = protocol_fingerprint(ROOT, PROTOCOL_FILES)
    per_seed = {}
    if merge and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            existing = json.load(f)
        require_merge_compatible(existing, protocol_id, out)
        per_seed = dict(existing.get("per_seed", {}))
        print("merge: loading existing seeds %s" % sorted(per_seed, key=int), flush=True)
    for sd in seeds:
        if str(sd) in per_seed:
            print("skip seed %d (result already in report)" % sd, flush=True)
            continue
        print("== seed %d ==" % sd, flush=True)
        per_seed[str(sd)] = run_seed(sd, data, mu, smoke=smoke)

    def agg(cfg, key):
        vals = [per_seed[s][cfg][key] for s in sorted(per_seed)]
        return {"mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    summary = {cfg: {k: agg(cfg, k) for k in
                     ("dcor", "rho_hat_raw", "rho_hat_corrected",
                      "rho_trained", "csw1", "mu_err", "loss_ratio", "steps_done", "dcor_p")}
               for cfg in ("A_independent", "B_dependent_meandep0", "C_ot",
                           "D_sorted_meanmax")}
    checks = evaluate_checks(per_seed)
    verdict = {k: ("PASS" if v else "FAIL") for k, v in checks.items()}

    report = {
        "protocol": {
            "dataset": "MNIST 16x16 (code/data/mnist_16.npz, 20000 train)",
            "source": "x0 ~ N(0, I_256)",
            "couplings": {
                "A_independent": "y = x~ (independent)",
                "B_dependent_meandep0": "y = mu + (1+0.8*tanh(x0[1]))*(x~ - mu)",
                "C_ot": "mini-batch optimal-transport assignment",
                "D_sorted_meanmax": ("per-batch sorted quantile pairing: sort "
                                     "x0 by x0[:,1], sort images by projection "
                                     "on the data first principal axis, match "
                                     "ranks (greedy mean-dependence "
                                     "maximization)"),
            },
            "seeds": sorted(int(k) for k in per_seed),
            "train_steps": 800 if smoke else TRAIN_STEPS,
            "batch": BATCH, "hidden": HIDDEN, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "k_knn": K_KNN,
            "early_stop": ("A/B: EMA train loss <= mu-baseline (population "
                           "optimum); C/D: fixed steps"),
            "limitation_rho_hat_highd": (
                "kNN data-side rho_hat is uninformative at d=256, n=1000 "
                "(all three couplings give corrected rho_hat ~ 0): the "
                "curse of dimensionality makes kNN neighbours' OT targets "
                "uninformative for the query. The real-data experiment "
                "therefore validates the coupling-level criterion through "
                "trained maps, not through the high-d estimator."),
            "thresholds": THRESHOLDS,
            "thresholds_frozen_after_calibration_seed0": True,
            "m1_a_allow_fp": "max(1, n_seeds // 5) at level 0.05 (multiplicity control)", 
        },
        "summary": summary, "checks": checks, "verdict": verdict,
        "per_seed": per_seed,
        "wall_seconds": round(time.time() - t0, 1),
    }
    attach_provenance(report, ROOT, "code/experiments/run_mnist_collapse.py",
                      PROTOCOL_FILES, merged=merge)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report written to %s" % out, flush=True)
    print("verdict: %s" % verdict, flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    _argv = [a for a in sys.argv[1:] if a not in ("--smoke", "--merge")]
    _seeds = SEEDS
    if _argv:
        _seeds = tuple(int(x) for x in " ".join(_argv).replace(",", " ").split())
    sys.exit(main(smoke="--smoke" in sys.argv, seeds=_seeds,
                  merge="--merge" in sys.argv))
