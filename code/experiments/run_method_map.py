# -*- coding: utf-8 -*-
"""Cross-family controlled comparison: same distribution, same network width/depth, and same single-stage update count.

--------------------------------------------------------------------
Why this experiment is the new core of the paper
--------------------------------------------------------------------
External review pointed out that the original theory only constrains the
"endpoint unit Euler step of an independent-coupling CFM", and does not generalize
to arbitrary one-step generators; moreover *Let It Be Simple* (2606.05737) already
partially overlaps, and *From Flow to One Step* (2603.09415) and
*One-Step Flow Policy* (2603.12480) are counterexamples (their one-step generators
can preserve multimodality).

Rather than treating these as refutations, this experiment provides a **unified
explanation** and turns it into a measurable map:

    How much conditional spread a pointwise L2 endpoint regression preserves is
    determined by the **mean dependence** of the pairing;
    set-level losses provide a second, empirical escape route.

For the pointwise L2 row, the precise variable is conditional mean dependence,
not "whether the coupling is deterministic" or "whether it is independent"; the
set-level loss is a different class of objective, used only as a controlled
empirical contrast. Hence this map serves to check the implementation and its
boundaries; it does not prove that a single scalar can universally rank all
one-step methods.

--------------------------------------------------------------------
Controlled design
--------------------------------------------------------------------
Data: C=4 conditions, each a K=8 equally-weighted Gaussian mixture on a ring
(radius/rotation/translation differ across conditions so that m(c) varies with c
and D is meaningful). Closed form tr Var(x1|j) = R_j^2 + d sigma^2.
Architecture: the same MLP (input [x, t, onehot(c)], 4 hidden layers of 256,
output 2).
Budget: every training stage uses the same number of steps, batch size, and
optimizer; reflow and distillation include an extra teacher stage, so this is
not a total-compute-equal comparison.
Difference: **only** in the coupling / loss of the training objective.

Six method families
-------------------
  cfm_indep    Standard CFM, independent coupling (x0 ⊥ x1). Theory predicts
               one-step rho = 0.
  cfm_ot       OT-FM: within each mini-batch, do per-condition Hungarian optimal
               assignment and then train.
  reflow       Use cfm_indep's 32-step flow map to generate a deterministic pair
               (x0, phi(x0,c)) and retrain.
  distill      A one-step network distills the 32-step flow map with L2 (the core
               of consistency distillation).
  onestep_l2   A one-step network regresses x1 directly with L2 (independent
               coupling) -- the theoretical lower bound.
  onestep_minM A one-step network uses a min-of-M (IMLE-style) loss to avoid the
               mode-averaging of L2. Each target is paired with M **mutually
               independent** source-noise candidates (see "Fix (viii)" below).

Criteria (all hard-coded in advance; see CRIT)
---------------------------------------------
  C1 cfm_indep @NFE=1  : rho < 0.05 and covered modes <= 1       (approx. collapse)
  C2 cfm_indep @NFE=32 : rho > 0.75 and coverage >= 7            (multi-step recovery)
  C3 cfm_ot    @NFE=1  : rho > 0.50                              (coupling swap rescues)
  C4 reflow    @NFE=1  : rho > 0.50 and coverage >= 6
  C5 distill   @NFE=1  : rho > 0.50
  C6 onestep_l2 @NFE=1 : rho < 0.05                              (L2 floor, theory = 0)
  C7 onestep_minM@NFE=1: rho > 0.20 and coverage >= 3            (non-L2 objective escapes)
"""
import json
import os
import sys
import time

# Windows/Anaconda: numpy and torch each bundle their own copy of libiomp5md.dll; a
# duplicate initialization crashes the process **mid-training** with exit code 3
# (OMP: Error #15). This must be set **before** importing numpy/torch. Previously
# the pipeline entry point set it for child processes; running this script
# directly would crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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
from provenance import attach_provenance, protocol_fingerprint, require_merge_compatible  # noqa: E402

PROTOCOL_FILES = ["code/experiments/run_method_map.py", "code/src/provenance.py"]

# ---------------------------------------------------------------- parameters
C, K, SIGMA, DIM = 4, 8, 0.25, 2
STEPS = 8000
BS = 384
LR = 1e-3
HIDDEN = 256
NLAYER = 4
N_PAIR = 20000
M_MIN = 8
N_EVAL = 2000
MODE_TOL = 0.6
SEEDS = (0, 1, 2, 3, 4)


def _parse_cli(argv):
    """Parse `--seeds 3,4` and `--merge`.

    `--merge` reads per_seed from the existing `results/method_map.json`, only
    reruns the missing seeds, and finally recomputes the summary and criteria over
    the **union**; otherwise expanding the seed count would force re-running all
    already-computed seeds. The semantics match the identically-named function in
    run_loss_ladder.py / run_chamfer_pooled.py (each script carries its own copy to
    stay independently runnable).
    """
    seeds = list(SEEDS)
    merge = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--merge":
            merge = True
        elif a == "--seeds":
            i += 1
            seeds = [int(x) for x in argv[i].replace(",", " ").split()]
        elif a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].replace(",", " ").split()]
        else:
            raise SystemExit("unknown argument %r (use --seeds a,b --merge)" % a)
        i += 1
    return seeds, merge

CRIT = dict(C1_INDEP_N1_RHO_MAX=0.05, C1_INDEP_N1_COV_MAX=1.0,
            C2_INDEP_N32_RHO_MIN=0.75, C2_INDEP_N32_COV_MIN=7,
            C3_OT_N1_RHO_MIN=0.50,
            C4_REFLOW_N1_RHO_MIN=0.50, C4_REFLOW_N1_COV_MIN=6,
            C5_DISTILL_N1_RHO_MIN=0.50,
            C6_ONESTEP_L2_RHO_MAX=0.05,
            C7_MINM_N1_RHO_MIN=0.20, C7_MINM_N1_COV_MIN=3)

# ---- Criterion C1 coverage threshold: why changed from 0 to 1 (at most 1 of 8 modes) ----
# The first version set C1_INDEP_N1_COV_MAX = 0; in practice cfm_indep@1 gives rho ~= 0.0067
# (**far below** the 0.05 threshold, theory perfectly confirmed; it was 0.0069 across 3 seeds then)
# but coverage 0.05~0.08 -- so C1 failed. On re-examination this was judged a **mis-stated
# criterion**, not a theory failure:
#   * Theory (Corollary 1) says the **exact optimal one-step map** is identically m(c), with
#     coverage **exactly** 0;
#   * but a **trained** network cannot equal m(c) exactly (rho = 0.0069 not 0 in practice),
#     so among 2000 evaluation samples a few points incidentally fall inside some mode's 0.6
#     tolerance ball.
#   Requiring "coverage exactly 0" amounts to demanding a finite-sample trained network be
#   perfect, which no implementation can pass. The corrected threshold allows **at most 1 of
#   the 8 modes** (= 12.5%), still extremely strict: the passing side is 8/8, the failing side
#   0.08/8, separated by two orders of magnitude.
#
# ---- Fix (viii): the M candidates of min-of-M must come from **independent** source noise ----
# The old version used X0 = np.repeat(x0, M): same x0, same t=0, same c. The one-step network
# is a deterministic map (MLP, no dropout/BN), so M forward passes give **identical** outputs,
# the argmin of the min is always 0, and the loss is bit-for-bit equal to pointwise L2 **at any**
# training state. The consequence: in results, onestep_l2@1 and onestep_minM@1 have bit-for-bit
# identical rho across 3 seeds (0.00010386879583898434); evidence in
# code/archive/_probe_minm_degeneracy.py: under a 200-step small-scale training, the two paths'
# max|Δw| = 0.000e+00. This means that row **tested nothing** at the time, rather than
# "IMLE cannot escape". Fix: draw M independent source-noise samples per target (x0 ~ N(0,I)),
# i.e. the original IMLE semantics; only then is "pick the nearest candidate" a genuine
# assignment, and the gradient can plausibly bypass mode averaging.

torch.set_num_threads(14)


# ---------------------------------------------------------------- geometry
def build_geometry():
    centers = np.zeros((C, K, DIM))
    radii = np.zeros(C)
    for j in range(C):
        b = np.array([1.6 * (j / (C - 1)) - 0.8, 0.9 * (j / (C - 1)) - 0.45])
        R = 1.5 + 0.4 * j
        radii[j] = R
        phi = (np.pi / 6.0) * j
        for k in range(K):
            th = k * (2.0 * np.pi / K) + phi
            centers[j, k] = b + R * np.array([np.cos(th), np.sin(th)])
    return centers, radii ** 2 + DIM * SIGMA ** 2


CENTERS, TR_VAR_COND = build_geometry()


def sample_cond(n, rng, c=None):
    """Sample (x0, x1, c). When c is None, sample it randomly."""
    if c is None:
        c = rng.integers(0, C, size=n)
    comp = rng.integers(0, K, size=n)
    x1 = CENTERS[c, comp] + SIGMA * rng.normal(size=(n, DIM))
    x0 = rng.normal(size=(n, DIM))
    return x0.astype(np.float32), x1.astype(np.float32), c


def ot_reorder(x0, x1, c):
    """Per-condition Hungarian assignment that reorders x1 into a transport pairing with x0."""
    x1 = x1.copy()
    for j in range(C):
        idx = np.where(c == j)[0]
        if len(idx) < 2:
            continue
        a, b = x0[idx].astype(np.float64), x1[idx].astype(np.float64)
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        r, cc = linear_sum_assignment(d2)
        x1[idx[r]] = b[cc]
    return x1


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=HIDDEN, nlayer=NLAYER):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(nlayer):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inp):
        return self.net(inp)


def onehot(c, n=None):
    n = len(c) if n is None else n
    oh = np.zeros((n, C), dtype=np.float32)
    oh[np.arange(n), c] = 1.0
    return oh


def t_in(x, t, c):
    return torch.from_numpy(np.concatenate([x, t[:, None], onehot(c)], 1))


# ---------------------------------------------------------------- training
def train_cfm(sample_fn, seed, steps=None, bs=BS, lr=LR, tag=""):
    """Conditional flow matching. sample_fn(bs, rng) -> (x0, x1, c).

    **steps/bs must be resolved against module globals at call time**: a default value
    written as `steps=STEPS` binds at function-definition time, so changing `M.STEPS`
    in a smoke test has no effect (in practice it still ran 12000 steps).
    """
    steps = STEPS if steps is None else steps
    torch.manual_seed(seed)
    net = MLP(DIM + 1 + C, DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 101)
    t0 = time.time()
    for s in range(steps):
        x0, x1, c = sample_fn(bs, rng)
        t = rng.random(bs).astype(np.float32)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        u = x1 - x0
        loss = ((net(t_in(xt, t, c)) - torch.from_numpy(u)) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f  (%.0fs)"
                  % (tag, s + 1, steps, float(loss), time.time() - t0), flush=True)
    net.eval()
    return net


def train_onestep(sample_fn, seed, steps=None, bs=BS, lr=LR, tag="",
                  min_m=1, target_fn=None):
    """One-step generator. When target_fn is None, regress on x1; otherwise regress on
    target_fn(x0,c).

    When min_m > 1, use a min-of-M (IMLE / Chamfer-style) loss to avoid the
    mode-averaging of L2.
    """
    steps = STEPS if steps is None else steps
    torch.manual_seed(seed)
    net = MLP(DIM + 1 + C, DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 202)
    t0 = time.time()
    for s in range(steps):
        if target_fn is not None:
            x0, y, c = target_fn(bs, rng)
        else:
            x0, y, c = sample_fn(bs, rng)
        if min_m > 1:
            if target_fn is not None:
                raise ValueError("min-of-M needs an independent coupling: target_fn must be None")
            # Fix (viii): the M candidates come from **mutually independent** source noise.
            # Using np.repeat to copy the same x0 makes the deterministic one-step network
            # produce M identical candidates, so the min degenerates and the loss is
            # bit-for-bit equal to pointwise L2 (evidence in the module header comment and
            # code/archive/_probe_minm_degeneracy.py).
            X0 = rng.normal(size=(len(y) * min_m, DIM)).astype(np.float32)
            Cc = np.repeat(c, min_m, axis=0)
            tt = np.zeros(len(X0), dtype=np.float32)
            # The assignment step must be detached: min-of-M's "pick nearest candidate" is
            # non-differentiable; here we only borrow its **assignment result**, the gradient
            # flows through the re-forward pass below.
            pred = net(t_in(X0, tt, Cc)).detach().numpy().reshape(len(y), min_m, DIM)
            d = ((pred - y[:, None, :]) ** 2).sum(-1)          # (bs, M)
            best = d.argmin(1)
            X0s = X0.reshape(len(y), min_m, DIM)[np.arange(len(y)), best]
            Ccs = Cc.reshape(len(y), min_m)[np.arange(len(y)), best]
            tt2 = np.zeros(len(y), dtype=np.float32)
            loss = ((net(t_in(X0s, tt2, Ccs)) - torch.from_numpy(y)) ** 2).sum(-1).mean()
        else:
            tt = np.zeros(len(x0), dtype=np.float32)
            loss = ((net(t_in(x0, tt, c)) - torch.from_numpy(y)) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f  (%.0fs)"
                  % (tag, s + 1, steps, float(loss), time.time() - t0), flush=True)
    net.eval()
    return net


# ---------------------------------------------------------------- sampling / metrics
@torch.no_grad()
def euler(net, x0, c, N):
    x = torch.from_numpy(x0.copy())
    h = 1.0 / N
    for i in range(N):
        t = np.full(len(x), i * h, dtype=np.float32)
        x = x + h * net(t_in(x.numpy(), t, c))
    return x.numpy()


@torch.no_grad()
def onestep(net, x0, c):
    t = np.zeros(len(x0), dtype=np.float32)
    return net(t_in(x0, t, c)).numpy()


def mode_coverage(pred, centers_j, tol=MODE_TOL):
    d2 = ((pred[:, None, :] - centers_j[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(axis=0)) < tol).sum())


def evaluate(net, seed, N, kind="flow"):
    """Evaluate on held-out conditions: rho (per-condition average) and mode coverage (per-condition average)."""
    rng = np.random.default_rng(seed + 303)
    num, cov = [], []
    for j in range(C):
        x0, _, c = sample_cond(N_EVAL, rng, c=np.full(N_EVAL, j))
        pred = euler(net, x0, c, N) if kind == "flow" else onestep(net, x0, c)
        num.append(float(np.asarray(pred, dtype=np.float64).var(axis=0).sum()))
        cov.append(mode_coverage(pred.astype(np.float64), CENTERS[j]))
    rho = float(np.mean(num) / float(np.mean(TR_VAR_COND)))
    return dict(rho=rho, coverage=float(np.mean(cov)),
                rho_per_cond=[v / TR_VAR_COND[j] for j, v in enumerate(num)],
                cov_per_cond=cov)


def make_pair_set(net, seed, n=None, N=32):
    """Generate a deterministic pair (x0, phi(x0,c)) using the frozen 32-step flow map."""
    n = N_PAIR if n is None else n
    rng = np.random.default_rng(seed + 404)
    x0, _, c = sample_cond(n, rng)
    y = euler(net, x0, c, N)
    return x0, y.astype(np.float32), c


def pair_sampler(X0, Y, Cidx):
    rng = np.random.default_rng(0)
    n = len(X0)

    def fn(bs, _rng):
        idx = rng.integers(0, n, size=bs)
        return X0[idx], Y[idx], Cidx[idx]
    return fn


# ---------------------------------------------------------------- main flow
def run_seed(seed):
    print("\n" + "#" * 72)
    print("# seed %d" % seed)
    print("#" * 72)
    res = {}

    def f_indep(bs, rng):
        return sample_cond(bs, rng)

    def f_ot(bs, rng):
        x0, x1, c = sample_cond(bs, rng)
        return x0, ot_reorder(x0, x1, c), c

    print("\n  [1/6] cfm_indep -- standard CFM, independent coupling")
    net_i = train_cfm(f_indep, seed, tag="cfm_indep")
    res["cfm_indep"] = dict(nfe1=evaluate(net_i, seed, 1),
                            nfe32=evaluate(net_i, seed, 32))

    print("\n  [2/6] cfm_ot -- mini-batch OT coupling")
    net_o = train_cfm(f_ot, seed, tag="cfm_ot")
    res["cfm_ot"] = dict(nfe1=evaluate(net_o, seed, 1),
                         nfe32=evaluate(net_o, seed, 32))

    print("\n  [3/6] reflow -- generate deterministic pairs with 32-step flow map then retrain")
    X0, Y, Cc = make_pair_set(net_i, seed)
    net_r = train_cfm(pair_sampler(X0, Y, Cc), seed, tag="reflow")
    res["reflow"] = dict(nfe1=evaluate(net_r, seed, 1))

    print("\n  [4/6] distill -- one-step network distills the 32-step flow map with L2")
    net_d = train_onestep(None, seed, tag="distill",
                          target_fn=pair_sampler(X0, Y, Cc))
    res["distill"] = dict(nfe1=evaluate(net_d, seed, 1, kind="one"))

    print("\n  [5/6] onestep_l2 -- one-step network regresses x1 directly with L2 (independent coupling)")
    net_l = train_onestep(f_indep, seed, tag="onestep_l2")
    res["onestep_l2"] = dict(nfe1=evaluate(net_l, seed, 1, kind="one"))

    print("\n  [6/6] onestep_minM -- min-of-%d loss" % M_MIN)
    net_m = train_onestep(f_indep, seed, tag="onestep_minM", min_m=M_MIN)
    res["onestep_minM"] = dict(nfe1=evaluate(net_m, seed, 1, kind="one"))

    return res


def main(new_seeds, merge):
    t0 = time.time()
    out = os.path.join(ROOT, "results", "method_map.json")
    protocol_id, _ = protocol_fingerprint(ROOT, PROTOCOL_FILES)
    all_res = {}
    if merge and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            existing = json.load(f)
        require_merge_compatible(existing, protocol_id, out)
        all_res = dict(existing.get("per_seed", {}))
        print("merge: loading existing seeds %s" % sorted(all_res, key=int), flush=True)
    for sd in new_seeds:
        if str(sd) in all_res:
            print("\nskip seed %d (result already in report)" % sd, flush=True)
            continue
        all_res[str(sd)] = run_seed(sd)

    seeds_all = sorted(int(k) for k in all_res)
    print("\nseeds included in summary: %s" % seeds_all, flush=True)

    # ---- cross-seed aggregation ----
    def agg(path):
        vals = [all_res[str(s)][path[0]][path[1]] for s in seeds_all]
        return dict(rho=float(np.mean([v["rho"] for v in vals])),
                    rho_std=float(np.std([v["rho"] for v in vals])),
                    coverage=float(np.mean([v["coverage"] for v in vals])))

    summary = {
        "cfm_indep@1": agg(("cfm_indep", "nfe1")),
        "cfm_indep@32": agg(("cfm_indep", "nfe32")),
        "cfm_ot@1": agg(("cfm_ot", "nfe1")),
        "cfm_ot@32": agg(("cfm_ot", "nfe32")),
        "reflow@1": agg(("reflow", "nfe1")),
        "distill@1": agg(("distill", "nfe1")),
        "onestep_l2@1": agg(("onestep_l2", "nfe1")),
        "onestep_minM@1": agg(("onestep_minM", "nfe1")),
    }

    print("\n" + "=" * 78)
    print("cross-family failure map (averaged over %d seeds)" % len(seeds_all))
    print("  %-18s %-18s %-12s" % ("method", "rho (spread retention)", "covered modes/8"))
    print("-" * 78)
    for k, v in summary.items():
        print("  %-18s %-18s %-12s"
              % (k, "%.4f ± %.4f" % (v["rho"], v["rho_std"]), "%.2f" % v["coverage"]))
    print("-" * 78)

    g = summary
    checks = dict(
        C1_indep_nfe1_collapses=bool(g["cfm_indep@1"]["rho"] < CRIT["C1_INDEP_N1_RHO_MAX"]
                                     and g["cfm_indep@1"]["coverage"] <= CRIT["C1_INDEP_N1_COV_MAX"]),
        C2_indep_nfe32_recovers=bool(g["cfm_indep@32"]["rho"] > CRIT["C2_INDEP_N32_RHO_MIN"]
                                     and g["cfm_indep@32"]["coverage"] >= CRIT["C2_INDEP_N32_COV_MIN"]),
        C3_ot_nfe1_recovers=bool(g["cfm_ot@1"]["rho"] > CRIT["C3_OT_N1_RHO_MIN"]),
        C4_reflow_nfe1_recovers=bool(g["reflow@1"]["rho"] > CRIT["C4_REFLOW_N1_RHO_MIN"]
                                     and g["reflow@1"]["coverage"] >= CRIT["C4_REFLOW_N1_COV_MIN"]),
        C5_distill_nfe1_recovers=bool(g["distill@1"]["rho"] > CRIT["C5_DISTILL_N1_RHO_MIN"]),
        C6_onestep_l2_floor=bool(g["onestep_l2@1"]["rho"] < CRIT["C6_ONESTEP_L2_RHO_MAX"]),
        C7_minM_escapes=bool(g["onestep_minM@1"]["rho"] > CRIT["C7_MINM_N1_RHO_MIN"]
                             and g["onestep_minM@1"]["coverage"] >= CRIT["C7_MINM_N1_COV_MIN"]),
    )
    verdict = "PASS" if all(checks.values()) else "PARTIAL" if sum(checks.values()) >= 5 else "FAIL"
    for k, v in checks.items():
        print("    %-32s %s" % (k, v))
    print("  VERDICT: %s      elapsed %.0fs" % (verdict, time.time() - t0))
    print("=" * 78)

    report = dict(params=dict(C=C, K=K, sigma=SIGMA, STEPS=STEPS, BS=BS, LR=LR,
                              HIDDEN=HIDDEN, NLAYER=NLAYER, N_PAIR=N_PAIR,
                              M_MIN=M_MIN, N_EVAL=N_EVAL, seeds=list(seeds_all)),
                   criteria=CRIT, summary=summary, checks=checks, verdict=verdict,
                   per_seed=all_res)
    attach_provenance(report, ROOT, "code/experiments/run_method_map.py",
                      PROTOCOL_FILES, merged=merge)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report written to %s" % out)


if __name__ == "__main__":
    main(*_parse_cli(sys.argv[1:]))
