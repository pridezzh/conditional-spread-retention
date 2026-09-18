# -*- coding: utf-8 -*-
"""Add a 7th method family: a one-step generator with **bidirectional Chamfer (set-level)** loss.

Why this one must be added
--------------------------
`onestep_minM` (min-of-M / IMLE) in `run_method_map.py` differs mechanistically from
the Chamfer in this script, and the two cannot substitute for each other:

    min-of-M: each **target** picks its nearest candidate only among **its own** M
              candidates; the M branches are each sampled independently (original
              IMLE semantics).
    bidirectional Chamfer: the first term picks, for each target, the nearest generated
              sample over the **whole batch**, and the second term goes the other way.
              With 384 samples and 8 modes, the "nearest generated sample" most often
              comes from the correct mode, different targets get assigned to different
              generated samples, and the network is forced to **spread out**.

Note (fix 2026-09-16): the earlier min-of-M implementation repeated the same source
noise M times; under a deterministic network the M candidates are **identical**, and
the loss degenerates bit-for-bit into pointwise L2. Hence the earlier observation that
"minM and onestep_l2 have exactly the same loss (4.5537 vs 4.5537)" was a product of a
**code defect**, not a property of IMLE. After the fix, min-of-M's M candidates are
each independently sampled and constitute a genuine set-level objective; its results
should be taken from `results/method_map.json` after re-running `run_method_map.py`.

This script trains the **whole-batch pooled** version of Chamfer (the first term's
argmin covers the entire batch), which is exactly the form used in external review
*From Flow to One Step* (2603.09415) and in most implementations. Its difference from
the "per-condition" Chamfer is isolated as a separate variable in paper Sec. 5.4.

This script trains only this one method family; its protocol is **fully identical** to
`run_method_map.py` (same geometry family, same architecture, same bs/steps/lr, same
evaluation), so its results can be merged directly into the failure map.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_method_map as M  # noqa: E402
from provenance import (attach_provenance, protocol_fingerprint,  # noqa: E402
                        require_merge_compatible)

PROTOCOL_FILES = ["code/experiments/run_chamfer_supplement.py",
                  "code/experiments/run_method_map.py", "code/src/provenance.py"]

SEEDS = (0, 1, 2, 3, 4)


def _parse_cli(argv):
    """`--seeds a,b --merge`; semantics match the identically-named function in run_method_map.py / run_loss_ladder.py."""
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


torch.set_num_threads(14)


def train_chamfer(sample_fn, seed, steps=None, bs=None, lr=M.LR, tag="chamfer"):
    steps = M.STEPS if steps is None else steps
    bs = M.BS if bs is None else bs
    torch.manual_seed(seed)
    net = M.MLP(M.DIM + 1 + M.C, M.DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 505)
    t0 = time.time()
    for s in range(steps):
        x0, y, c = sample_fn(bs, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))                       # (bs, d), with grad
        with torch.no_grad():
            Gn = G.detach().numpy()
            d2 = ((y[:, None, :] - Gn[None, :, :]) ** 2).sum(-1)   # (bs, bs)
            a1 = d2.argmin(1)                            # target i -> nearest generated sample
            a2 = d2.argmin(0)                            # generated sample j -> nearest target
        l1 = ((G[a1] - torch.from_numpy(y)) ** 2).sum(-1).mean()
        l2 = ((G - torch.from_numpy(y)[a2]) ** 2).sum(-1).mean()
        loss = l1 + l2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f  (l1 %.4f / l2 %.4f)  (%.0fs)"
                  % (tag, s + 1, steps, float(loss), float(l1), float(l2),
                     time.time() - t0), flush=True)
    net.eval()
    return net


def main(new_seeds, merge):
    t0 = time.time()
    out = os.path.join(ROOT, "results", "method_map_chamfer.json")
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
        print("\n# seed %d" % sd, flush=True)
        net = train_chamfer(lambda bs, rng: M.sample_cond(bs, rng), sd,
                            tag="onestep_chamfer")
        res = M.evaluate(net, sd, 1, kind="one")
        out_all[str(sd)] = res
        print("  chamfer@1  rho=%.4f  cov=%.2f  rho_per_cond=%s  cov_per_cond=%s"
              % (res["rho"], res["coverage"],
                 np.round(res["rho_per_cond"], 4).tolist(), res["cov_per_cond"]),
              flush=True)

    seeds_all = sorted(int(k) for k in out_all)
    print("\nseeds included in summary: %s" % seeds_all, flush=True)

    rhos = [out_all[str(s)]["rho"] for s in seeds_all]
    covs = [out_all[str(s)]["coverage"] for s in seeds_all]
    summary = dict(rho=float(np.mean(rhos)), rho_std=float(np.std(rhos)),
                   coverage=float(np.mean(covs)),
                   rho_per_seed=[float(r) for r in rhos])
    ok = summary["rho"] > M.CRIT["C7_MINM_N1_RHO_MIN"] and \
        summary["coverage"] >= M.CRIT["C7_MINM_N1_COV_MIN"]
    print("\n" + "=" * 70)
    print("onestep_chamfer@1  rho = %.4f ± %.4f   coverage = %.2f"
          % (summary["rho"], summary["rho_std"], summary["coverage"]))
    print("criterion (reusing C7): rho > %.2f and coverage >= %d   -> %s"
          % (M.CRIT["C7_MINM_N1_RHO_MIN"], M.CRIT["C7_MINM_N1_COV_MIN"],
             "PASS" if ok else "FAIL"))
    print("elapsed %.0fs" % (time.time() - t0))
    print("=" * 70)

    report = dict(method="onestep_chamfer",
                  params=dict(STEPS=M.STEPS, BS=M.BS, LR=M.LR, seeds=list(seeds_all)),
                  criteria={k: M.CRIT[k] for k in
                            ("C7_MINM_N1_RHO_MIN", "C7_MINM_N1_COV_MIN")},
                  summary=summary, per_seed=out_all,
                  passes_C7=bool(ok))
    attach_provenance(report, ROOT, "code/experiments/run_chamfer_supplement.py",
                      PROTOCOL_FILES, merged=merge)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report written to %s" % out)


if __name__ == "__main__":
    main(*_parse_cli(sys.argv[1:]))
