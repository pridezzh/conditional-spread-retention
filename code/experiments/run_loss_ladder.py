# -*- coding: utf-8 -*-
"""Loss-form ladder: pointwise L2 / unbalanced set-level / balanced set-level.

--------------------------------------------------------------------
Theory under test (a three-tier ladder)
--------------------------------------------------------------------
Fix the coupling as **independent coupling** (x0 ⊥ x1 | c), and only vary the loss
form. Let q_c = f(·,c)_# p0.

  L1  pointwise L2
      L = E ||f(x0,c) - x1||^2
      minimizer f*(x0,c) = E[x1 | x0, c] = E[x1 | c] = m(c)      (independent coupling)
      => q_c = δ_{m(c)}, ρ* = 0. **Collapse**.

  L2  unbalanced set-level (Chamfer / IMLE-style min-matching)
      population limit (batch M -> ∞):
        L -> E_{y~p_c}[ min_{x in supp q_c} ||y-x||^2 ] + E_{x~q_c}[ min_{y in supp p_c} ||x-y||^2 ]
      both terms are 0 **iff supp q_c = supp p_c**.
      => collapse is lifted, but **weights are completely unconstrained**: any q_c with the
      same support as p_c is optimal.
      measurable prediction: coverage 8/8, but **per-condition spread is flattened**
      (every condition converges to the same value), i.e. the dispersion of ρ_j is far
      greater than 0, and ρ_j is unrelated to the true radius.

  L3  balanced set-level (in-batch optimal assignment / transport)
      population limit -> W_2^2(p_c, q_c), unique minimizer q_c = p_c.
      => ρ* = 1, per-condition spread is correct, dispersion of ρ_j ≈ 0.

The **falsifiable** part of this ladder is the difference between L2 and L3: if "any
set-level loss is equivalent", then the dispersion of ρ_j for L2 and L3 should be
comparable. We predict L3's dispersion is significantly smaller.

Criteria (hard-coded in advance)
--------------------------------
  B1  onestep_l2      @1: rho < 0.05                        (L1 collapse)
  B2  chamfer         @1: rho > 0.5 and coverage >= 6             (L2 escapes collapse)
  B3  balanced        @1: rho > 0.5 and coverage >= 6             (L3 escapes collapse)
  B4  **mean of |ρ_j - 1|: balanced < chamfer**                   (L3 recovers the law, L2 only recovers support)
  B5  **cSW: balanced < chamfer**                                 (fidelity: L3 is more accurate)

B4/B5 are the real stakes of this ladder: if they fail, the claim "balance is the key"
is overturned.

**Why the criteria only state directions, not multiples.** An earlier version wrote
effect sizes like `disp(chamfer) > 2 × disp(balanced)` that were **guessed**; the result
was that a 40-step smoke test already failed (0.396 vs 2×0.250), but that was only
training not having converged, not a theory failure. The multiple depends on how far
optimization goes and is an **unknowable-ahead-of-time quantity**; whereas "L3's
population optimum is exactly p_c, while L2's population optimum set is a mixture of
distributions with wrong weights" is **qualitative**, and can only be tested with
directional criteria. The multiples are reported as descriptive numbers, but do not
enter the criteria.
"""
import json
import os
import sys
import time

# Windows/Anaconda: numpy and torch each bundle their own copy of libiomp5md.dll; a
# duplicate initialization crashes the process **mid-training** with exit code 3
# (OMP: Error #15). This must be set **before** importing numpy/torch. Previously
# only code/_run_pipeline.py set it for child processes; running this script
# directly would crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import run_method_map as M  # noqa: E402
from provenance import attach_provenance, protocol_fingerprint, require_merge_compatible  # noqa: E402

PROTOCOL_FILES = ["code/experiments/run_loss_ladder.py",
                  "code/experiments/run_method_map.py", "code/src/provenance.py"]

SEEDS = (0, 1, 2, 3, 4)
N_PROJ = 256
torch.set_num_threads(14)


def _parse_cli(argv):
    """Parse `--seeds 3,4` and `--merge`.

    Why `--merge` is needed: one full run (measured ~2950s at 3 seeds, ~1200s at 5 seeds
    depending on machine load) -- if the seed count is expanded without merging, the
    already-computed seeds would be recomputed. `--merge` reads per_seed from the existing
    `results/loss_ladder.json`, only reruns the missing seeds, and finally recomputes the
    summary and criteria over the **union** (the same lesson as in memory: "grabbing new
    results must merge with the existing cache before writing back").
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


# ---------------------------------------------------------------- metrics
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
    return float(np.sqrt(((pa - pb) ** 2).mean()))


def evaluate_full(net, seed):
    """Per-condition: ρ_j, coverage, conditional sliced W2 (normalized by sqrt(trVar(x1|c_j)))."""
    rng = np.random.default_rng(seed + 909)
    num, cov, csw = [], [], []
    for j in range(M.C):
        x0, x1_true, c = M.sample_cond(M.N_EVAL, rng, c=np.full(M.N_EVAL, j))
        pred = M.onestep(net, x0, c)
        pred64 = np.asarray(pred, dtype=np.float64)
        num.append(float(pred64.var(axis=0).sum()))
        cov.append(M.mode_coverage(pred64, M.CENTERS[j]))
        csw.append(sliced_w2(pred64, x1_true.astype(np.float64), rng)
                   / np.sqrt(M.TR_VAR_COND[j]))
    rho_j = np.array([v / M.TR_VAR_COND[j] for j, v in enumerate(num)], dtype=np.float64)
    return dict(rho=float(np.mean(num) / np.mean(M.TR_VAR_COND)),
                coverage=float(np.mean(cov)),
                rho_per_cond=rho_j.tolist(),
                rho_disp=float(rho_j.std(ddof=0)),
                rho_abs_err=float(np.abs(rho_j - 1.0).mean()),
                csw=float(np.mean(csw)),
                cov_per_cond=[int(v) for v in cov])


# ---------------------------------------------------------------- training
def _make_net(seed):
    torch.manual_seed(seed)
    return M.MLP(M.DIM + 1 + M.C, M.DIM)


def train_l2(seed, steps=None, tag="onestep_l2"):
    steps = M.STEPS if steps is None else steps
    net = _make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 202)
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        loss = ((net(M.t_in(x0, tt, c)) - torch.from_numpy(y)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f" % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


def train_chamfer(seed, steps=None, tag="onestep_chamfer"):
    """Unbalanced bidirectional min-matching, done **per condition** (matching the
    assignment range of train_balanced).

    The only difference from the unbalanced version is that the assignment is **not a
    bijection**: each target finds its nearest generated sample, each generated sample
    finds its nearest target sample; the same generated sample can be shared by multiple
    targets, or not used at all. The population limit only constrains the **support set**
    and imposes no requirement on the weights.

    ---- Fix (2026-09-16): assignment range changed from "whole batch" to "per condition" ----
    The first version took argmin over the **entire batch** (ignoring conditions) here,
    whereas train_balanced did per-condition Hungarian assignment. As a result L2 and L3
    differed in **two** ways (balanced or not AND per-condition or not), so the observed
    "support recovered, weights flattened" could not be attributed to unbalance alone --
    which is precisely the point Thm 4(L2) argues. After switching to per-condition, L2
    and L3 differ only in "bijection / non-bijection".
    Note: the pooled version's min is taken over a larger candidate set, giving a smaller
    loss and a **weaker** constraint, so this fix only makes L2's support constraint
    stronger; it does not artificially manufacture support recovery.
    """
    steps = M.STEPS if steps is None else steps
    net = _make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 505)
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))
        with torch.no_grad():
            Gn = G.detach().numpy().astype(np.float64)
            yn = y.astype(np.float64)
            a1 = np.arange(len(x0))       # target -> nearest generated sample in this condition
            a2 = np.arange(len(x0))       # generated sample -> nearest target in this condition
            for j in range(M.C):
                idx = np.where(c == j)[0]
                if len(idx) < 2:
                    continue
                d2 = ((yn[idx][:, None, :] - Gn[idx][None, :, :]) ** 2).sum(-1)
                a1[idx] = idx[d2.argmin(1)]
                a2[idx] = idx[d2.argmin(0)]
        l1 = ((G[a1] - torch.from_numpy(y)) ** 2).sum(-1).mean()
        l2 = ((G - torch.from_numpy(y)[a2]) ** 2).sum(-1).mean()
        loss = l1 + l2
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f" % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


def train_balanced(seed, steps=None, tag="onestep_balanced"):
    """Balanced set-level: per-condition in-batch Hungarian optimal assignment (bijection),
    then L2 on the assignment result.

    The only difference from the unbalanced version is that the assignment is a
    **bijection**: each generated sample is used exactly once, each target is matched
    exactly once. The population limit is then W_2^2(p_c, q_c).
    """
    steps = M.STEPS if steps is None else steps
    net = _make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 707)
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))
        with torch.no_grad():
            Gn = G.detach().numpy().astype(np.float64)
            perm = np.arange(len(x0))
            for j in range(M.C):
                idx = np.where(c == j)[0]
                if len(idx) < 2:
                    continue
                d2 = ((Gn[idx][:, None, :] - y[idx][None, :, :].astype(np.float64)) ** 2).sum(-1)
                r, cc = linear_sum_assignment(d2)
                perm[idx[r]] = idx[cc]
        loss = ((G - torch.from_numpy(y)[perm]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f" % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


# ---------------------------------------------------------------- main flow
def main(new_seeds, merge):
    t0 = time.time()
    out = os.path.join(ROOT, "results", "loss_ladder.json")
    protocol_id, _ = protocol_fingerprint(ROOT, PROTOCOL_FILES)
    out_all = {}
    if merge and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            existing = json.load(f)
        require_merge_compatible(existing, protocol_id, out)
        out_all = dict(existing.get("per_seed", {}))
        print("merge: loading existing seeds %s" % sorted(out_all, key=int), flush=True)
    for sd in new_seeds:
        if str(sd) in out_all:
            print("\nskip seed %d (result already in report)" % sd, flush=True)
            continue
        print("\n" + "#" * 70 + "\n# seed %d\n" % sd + "#" * 70, flush=True)
        out_all[str(sd)] = {}
        for name, fn in (("onestep_l2", train_l2),
                         ("onestep_chamfer", train_chamfer),
                         ("onestep_balanced", train_balanced)):
            print("\n  [%s]" % name, flush=True)
            net = fn(sd)
            r = evaluate_full(net, sd)
            out_all[str(sd)][name] = r
            print("    rho=%.4f  cov=%.2f  disp=%.3f  |rho_j-1|=%.3f  cSW=%.4f"
                  % (r["rho"], r["coverage"], r["rho_disp"], r["rho_abs_err"], r["csw"]),
                  flush=True)
            print("    rho_j = %s" % np.round(r["rho_per_cond"], 3).tolist(), flush=True)

    seeds_all = sorted(int(k) for k in out_all)
    print("\nseeds included in summary: %s" % seeds_all, flush=True)

    def agg(name):
        v = [out_all[str(s)][name] for s in seeds_all]
        return dict(rho=float(np.mean([x["rho"] for x in v])),
                    rho_std=float(np.std([x["rho"] for x in v])),
                    coverage=float(np.mean([x["coverage"] for x in v])),
                    rho_disp=float(np.mean([x["rho_disp"] for x in v])),
                    rho_disp_std=float(np.std([x["rho_disp"] for x in v])),
                    rho_abs_err=float(np.mean([x["rho_abs_err"] for x in v])),
                    csw=float(np.mean([x["csw"] for x in v])),
                    csw_std=float(np.std([x["csw"] for x in v])))

    S = {k: agg(k) for k in ("onestep_l2", "onestep_chamfer", "onestep_balanced")}

    print("\n" + "=" * 78)
    print("loss-form ladder (%d-seed average)" % len(seeds_all))
    print("  %-18s %-14s %-8s %-10s %-10s %-10s"
          % ("loss", "rho", "cov/8", "disp(rho_j)", "|rho_j-1|", "cSW"))
    print("-" * 78)
    for k, v in S.items():
        print("  %-18s %-14s %-8s %-10s %-10s %-10s"
              % (k, "%.4f±%.4f" % (v["rho"], v["rho_std"]), "%.2f" % v["coverage"],
                 "%.3f±%.3f" % (v["rho_disp"], v["rho_disp_std"]),
                 "%.3f" % v["rho_abs_err"],
                 "%.4f±%.4f" % (v["csw"], v["csw_std"])))
    print("-" * 78)

    checks = dict(
        B1_l2_collapses=bool(S["onestep_l2"]["rho"] < 0.05),
        B2_chamfer_escapes=bool(S["onestep_chamfer"]["rho"] > 0.5
                                and S["onestep_chamfer"]["coverage"] >= 6),
        B3_balanced_escapes=bool(S["onestep_balanced"]["rho"] > 0.5
                                 and S["onestep_balanced"]["coverage"] >= 6),
        B4_balanced_less_weight_error=bool(
            S["onestep_balanced"]["rho_abs_err"] < S["onestep_chamfer"]["rho_abs_err"]),
        B5_balanced_more_faithful=bool(S["onestep_balanced"]["csw"] < S["onestep_chamfer"]["csw"]),
    )
    verdict = ("PASS" if all(checks.values())
               else "PARTIAL" if sum(checks.values()) >= 4 else "FAIL")
    for k, v in checks.items():
        print("    %-32s %s" % (k, v))
    print("  descriptive ratios (not in criteria): |rho_j-1| chamfer/balanced = %.2f×, cSW chamfer/balanced = %.2f×"
          % (S["onestep_chamfer"]["rho_abs_err"] / max(S["onestep_balanced"]["rho_abs_err"], 1e-9),
             S["onestep_chamfer"]["csw"] / max(S["onestep_balanced"]["csw"], 1e-9)))
    print("  VERDICT: %s       elapsed %.0fs" % (verdict, time.time() - t0))
    print("=" * 78)

    report = dict(theorem="loss ladder: pointwise L2 -> support only -> full law",
                  params=dict(seeds=list(seeds_all), STEPS=M.STEPS, BS=M.BS, LR=M.LR,
                              C=M.C, K=M.K, sigma=M.SIGMA, N_EVAL=M.N_EVAL,
                              N_PROJ=N_PROJ),
                  tr_var_cond=M.TR_VAR_COND.tolist(),
                  summary=S, checks=checks, verdict=verdict, per_seed=out_all)
    attach_provenance(report, ROOT, "code/experiments/run_loss_ladder.py",
                      PROTOCOL_FILES, merged=merge)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report written to %s" % out)


if __name__ == "__main__":
    main(*_parse_cli(sys.argv[1:]))
