# -*- coding: utf-8 -*-
"""Compute the "margin / seed noise" ratio for each pre-registered criterion and
output a Markdown table.

Why this script
----------------
The substance of the "only 3 (or 5) seeds" criticism is: **whether the criterion's
margin is drowned out by seed noise**. The quantitative question is: how many
seed standard deviations sit between the observed value and the threshold? The
more of them, the less likely adding seeds would overturn the conclusion.
Per discipline (D8) no mental arithmetic is done; everything is computed by script
and reads the named fields of results/*.json directly.

Usage: python margin_vs_noise.py [out.md]
"""
import json
import os
import sys

ROOT = None
d = os.path.dirname(os.path.abspath(__file__))
while d != os.path.dirname(d):
    if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
        ROOT = d
        break
    d = os.path.dirname(d)
if ROOT is None:
    raise SystemExit("project root not found")


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ratios(margin, sigma):
    if sigma is None or sigma <= 0:
        return "inf (inter-seed variance is 0)"
    return "%.0f×" % (abs(margin) / sigma)


def main():
    rows = []
    ll = load(os.path.join("results", "loss_ladder.json"))
    if ll:
        n = len(ll["params"]["seeds"])
        s = ll["summary"]
        C = {"B1_l2_collapses": 0.05, "B2_chamfer_escapes": 0.5,
             "B3_balanced_escapes": 0.5, "B5_balanced_more_faithful": None}
        rows.append(("B1 L1 collapse", "rho<0.05", "%.5f" % s["onestep_l2"]["rho"],
                     "%.5f" % s["onestep_l2"]["rho_std"], n,
                     ratios(0.05 - s["onestep_l2"]["rho"], s["onestep_l2"]["rho_std"])))
        rows.append(("B2 pooled/per-condition Chamfer escape", "rho>0.5 and coverage>=6",
                     "%.4f" % s["onestep_chamfer"]["rho"], "%.5f" % s["onestep_chamfer"]["rho_std"], n,
                     ratios(s["onestep_chamfer"]["rho"] - 0.5, s["onestep_chamfer"]["rho_std"])))
        rows.append(("B3 balanced assignment escape", "rho>0.5 and coverage>=6",
                     "%.4f" % s["onestep_balanced"]["rho"], "%.5f" % s["onestep_balanced"]["rho_std"], n,
                     ratios(s["onestep_balanced"]["rho"] - 0.5, s["onestep_balanced"]["rho_std"])))
        # B4 negative verdict: look at "per-seed separation" rather than threshold margin
        per = [ll["per_seed"][k]["onestep_chamfer"]["rho_abs_err"] for k in ll["per_seed"]]
        bal = [ll["per_seed"][k]["onestep_balanced"]["rho_abs_err"] for k in ll["per_seed"]]
        sep = "per-seed fully separated: Chamfer max %.4f < balanced min %.4f" % (max(per), min(bal))
        rows.append(("B4 balanced reduces weight error", "Chamfer<balanced", "%.4f vs %.4f"
                     % (s["onestep_chamfer"]["rho_abs_err"], s["onestep_balanced"]["rho_abs_err"]),
                     "—", n, "**verdict negative**; " + sep))
        d_csw = s["onestep_balanced"]["csw"] - s["onestep_chamfer"]["csw"]
        comb = (s["onestep_chamfer"]["csw_std"] ** 2 + s["onestep_balanced"]["csw_std"] ** 2) ** 0.5
        rows.append(("B5 balanced more faithful (cSW)", "balanced<Chamfer",
                     "%.4f vs %.4f" % (s["onestep_balanced"]["csw"], s["onestep_chamfer"]["csw"]),
                     "combined sigma=%.4f (%.1f sigma separation)" % (comb, abs(d_csw) / comb), n,
                     "%s (criterion takes direction)" % ("holds" if d_csw < 0 else "does not hold")))
    mm = load(os.path.join("results", "method_map.json"))
    if mm:
        n = len(mm["params"]["seeds"])
        g = mm["summary"]
        rows.append(("C1 independent coupling @1 collapse", "rho<0.05", "%.5f" % g["cfm_indep@1"]["rho"],
                     "%.5f" % g["cfm_indep@1"]["rho_std"], n,
                     ratios(0.05 - g["cfm_indep@1"]["rho"], g["cfm_indep@1"]["rho_std"])))
        rows.append(("C2 same network @32 recovers", "rho>0.75", "%.4f" % g["cfm_indep@32"]["rho"],
                     "%.5f" % g["cfm_indep@32"]["rho_std"], n,
                     ratios(g["cfm_indep@32"]["rho"] - 0.75, g["cfm_indep@32"]["rho_std"])))
        rows.append(("C3 OT one-step recovery", "rho>0.50", "%.4f" % g["cfm_ot@1"]["rho"],
                     "%.5f" % g["cfm_ot@1"]["rho_std"], n,
                     ratios(g["cfm_ot@1"]["rho"] - 0.50, g["cfm_ot@1"]["rho_std"])))
        rows.append(("C4 reflow one-step recovery", "rho>0.50", "%.4f" % g["reflow@1"]["rho"],
                     "%.5f" % g["reflow@1"]["rho_std"], n,
                     ratios(g["reflow@1"]["rho"] - 0.50, g["reflow@1"]["rho_std"])))
        rows.append(("C5 distillation one-step recovery", "rho>0.50", "%.4f" % g["distill@1"]["rho"],
                     "%.5f" % g["distill@1"]["rho_std"], n,
                     ratios(g["distill@1"]["rho"] - 0.50, g["distill@1"]["rho_std"])))
        rows.append(("C6 pointwise L2 floor", "rho<0.05", "%.5f" % g["onestep_l2@1"]["rho"],
                     "%.5f" % g["onestep_l2@1"]["rho_std"], n,
                     ratios(0.05 - g["onestep_l2@1"]["rho"], g["onestep_l2@1"]["rho_std"])))
        rows.append(("C7 min-of-M escape", "rho>0.20", "%.4f" % g["onestep_minM@1"]["rho"],
                     "%.5f" % g["onestep_minM@1"]["rho_std"], n,
                     ratios(g["onestep_minM@1"]["rho"] - 0.20, g["onestep_minM@1"]["rho_std"])))

    lines = ["| criterion | threshold (direction) | observed (5-seed mean) | seed sigma | margin/sigma |",
             "|---|---|---|---|---|"]
    for name, thr, obs, sd, n, r in rows:
        lines.append("| %s | %s | %s | %s | %s |" % (name, thr, obs, sd, r))
    out = "\n".join(lines)
    print(out)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print("\nwrote %s" % sys.argv[1])


if __name__ == "__main__":
    main()
