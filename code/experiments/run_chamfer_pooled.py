# -*- coding: utf-8 -*-
"""Controlled experiment: **pooled** (cross-condition) unbalanced Chamfer -- i.e. the
version that is "actually written in practice".

Why a separate script
---------------------
`run_loss_ladder.py`'s `train_chamfer` has been corrected to **per-condition** matching,
strictly aligned with the statement of Thm 4(L2) (p_c, q_c per condition). But Chamfer
/ IMLE are in practice usually computed over the **whole batch**, and pooling introduces
an extra failure channel:

    the generated samples of condition c' can serve as the "nearest point" for the
    targets of condition c  ==>  the model can **share samples across conditions** while
    keeping the loss small  ==>  each condition's generated set becomes a subset of the
    full set  ==>  the conditional spread is flattened.

This is exactly the observable of "support recovered, weights free". So the paper needs
**two rows**:
  (a) pooled version (the practical write-up): spread is flattened;
  (b) per-condition version (the Thm 4 setup): spread is essentially recovered.
Their difference isolates "pooling" as a single variable.

This script only runs (a) and writes `results/chamfer_pooled.json`; the other three rows
(L1 / per-condition L2 / balanced L3) are written to `results/loss_ladder.json` by
`run_loss_ladder.py`. The aggregation convention is fully identical to the `agg()` in
`run_loss_ladder.py` (rho is the seed mean; disp/abs_err/csw are seed means).

Note: this script also serves as a **reproducibility self-check** -- if
`results/loss_ladder_pooled_20260916.json` (the pre-fix raw output) is present, the
script compares the new and old numbers field by field and prints the differences.
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
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_method_map as M  # noqa: E402
import run_loss_ladder as L  # noqa: E402
from provenance import attach_provenance, protocol_fingerprint, require_merge_compatible  # noqa: E402

PROTOCOL_FILES = ["code/experiments/run_chamfer_pooled.py",
                  "code/experiments/run_loss_ladder.py",
                  "code/experiments/run_method_map.py", "code/src/provenance.py"]

SEEDS = L.SEEDS
torch.set_num_threads(14)


def train_chamfer_pooled(seed, steps=None, tag="chamfer_pooled"):
    """Unbalanced bidirectional min-matching, taking argmin over the **whole batch**
    (the pre-fix implementation).

    It is kept for the controlled comparison described above; it is **not** the Thm 4(L2)
    setup.
    """
    steps = M.STEPS if steps is None else steps
    net = L._make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 505)          # same seed as run_loss_ladder
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))
        with torch.no_grad():
            Gn = G.detach().numpy().astype(np.float64)
            yn = y.astype(np.float64)
            d2 = ((yn[:, None, :] - Gn[None, :, :]) ** 2).sum(-1)
            a1 = d2.argmin(1)
            a2 = d2.argmin(0)
        l1 = ((G[a1] - torch.from_numpy(y)) ** 2).sum(-1).mean()
        l2 = ((G - torch.from_numpy(y)[a2]) ** 2).sum(-1).mean()
        loss = l1 + l2
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f"
                  % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


def main(new_seeds, merge):
    """`--seeds a,b --merge` semantics are the same as run_loss_ladder.py (reuses its parser).

    Purpose: in Table 1, L2a(pooled) must be compared against L2b/L3 with the **same seed
    count**, otherwise the ± intervals are not comparable.
    """
    t0 = time.time()
    out = os.path.join(ROOT, "results", "chamfer_pooled.json")
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
        net = train_chamfer_pooled(sd)
        r = L.evaluate_full(net, sd)
        out_all[str(sd)] = r
        print("    rho=%.4f  cov=%.2f  disp=%.3f  |rho_j-1|=%.3f  cSW=%.4f"
              % (r["rho"], r["coverage"], r["rho_disp"], r["rho_abs_err"], r["csw"]),
              flush=True)
        print("    rho_j = %s" % np.round(r["rho_per_cond"], 3).tolist(), flush=True)

    seeds_all = sorted(int(k) for k in out_all)
    print("\nseeds included in summary: %s" % seeds_all, flush=True)

    def _summ(seed_list):
        v = [out_all[str(s)]["onestep_chamfer"] if "onestep_chamfer" in out_all[str(s)]
             else out_all[str(s)] for s in seed_list]
        return dict(rho=float(np.mean([x["rho"] for x in v])),
                    rho_std=float(np.std([x["rho"] for x in v])),
                    coverage=float(np.mean([x["coverage"] for x in v])),
                    rho_disp=float(np.mean([x["rho_disp"] for x in v])),
                    rho_disp_std=float(np.std([x["rho_disp"] for x in v])),
                    rho_abs_err=float(np.mean([x["rho_abs_err"] for x in v])),
                    csw=float(np.mean([x["csw"] for x in v])),
                    csw_std=float(np.std([x["csw"] for x in v])))

    summary = _summ(seeds_all)

    print("\n" + "=" * 78)
    print("pooled unbalanced Chamfer (%d-seed average)" % len(seeds_all))
    print("  rho=%.4f±%.4f  cov=%.2f  disp=%.3f  |rho_j-1|=%.3f  cSW=%.4f±%.4f"
          % (summary["rho"], summary["rho_std"], summary["coverage"],
             summary["rho_disp"], summary["rho_abs_err"],
             summary["csw"], summary["csw_std"]))
    print("  elapsed %.0fs" % (time.time() - t0))
    print("=" * 78)

    report = dict(variant="onestep_chamfer_pooled",
                  note="argmin taken over the whole batch (the practically used form); "
                       "run_loss_ladder.py's onestep_chamfer matches within condition",
                  params=dict(seeds=list(seeds_all), STEPS=M.STEPS, BS=M.BS, LR=M.LR,
                              C=M.C, K=M.K, sigma=M.SIGMA, N_EVAL=M.N_EVAL,
                              N_PROJ=L.N_PROJ),
                  tr_var_cond=M.TR_VAR_COND.tolist(),
                  summary=summary,
                  per_seed=out_all)
    attach_provenance(report, ROOT, "code/experiments/run_chamfer_pooled.py",
                      PROTOCOL_FILES, merged=merge)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report written to %s" % out)

    # ---------------- reproducibility self-check: field-by-field compare with pre-fix raw output ----------------
    # Note: the self-check must be done on **common seeds**. The raw output only has seeds
    # 0/1/2; comparing against a 5-seed summary directly would mix in the irrelevant factor
    # of "two extra seeds computed", falsely reporting a reproduction failure.
    old = os.path.join(ROOT, "results", "loss_ladder_pooled_20260916.json")
    if os.path.exists(old):
        with open(old, encoding="utf-8") as f:
            orep = json.load(f)
        o = orep["summary"]["onestep_chamfer"]
        o_seeds = [int(s) for s in orep.get("params", {}).get("seeds", seeds_all)]
        common = [s for s in seeds_all if s in set(o_seeds)]
        print("\nreproducibility self-check (vs pre-fix raw output, limited to common seeds %s):" % common)
        worst = 0.0
        for k, val in _summ(common).items():
            if k in o:
                d = abs(float(val) - float(o[k]))
                worst = max(worst, d)
                print("  %-14s new=%.10f  old=%.10f  |diff|=%.3e" % (k, val, o[k], d))
        print("  max field difference = %.3e" % worst)


if __name__ == "__main__":
    main(*L._parse_cli(sys.argv[1:]))
