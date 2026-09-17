# -*- coding: utf-8 -*-
"""Generate paper/numbers_theory.tex: all numeric macros needed by the new theory
section.

Conventions (consistent with make_tables.py)
--------------------------------------------
* The paper body is **not allowed** to contain hand-written experimental numbers;
  everything goes through macros;
* **LaTeX control sequences may only contain letters** -- `\\RhoOne` is valid,
  `\\rho1` is a fatal error. So all digits in macro names are spelled out as
  English words.
* When a source file is missing, **skip it and record a warning** rather than
  failing hard (to allow running experiments in batches).
* Every macro carries a `% source: ...` comment, so any number in the body can be
  traced back to its JSON field.
"""
import json
import os
import re
import sys


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
_NAME_RE = re.compile(r"^[A-Za-z]+$")


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Bag:
    """Collect (macro name, value, note). Values are uniformly formatted as
    strings with at most 4 significant figures."""

    def __init__(self):
        self.items = []
        self.missing = []

    def add(self, name, value, note):
        if not _NAME_RE.match(name):
            raise ValueError("illegal macro name (letters only): %s" % name)
        if value is None:
            self.missing.append(name)
            return
        if isinstance(value, float):
            if value == 0:
                s = "0"
            elif abs(value) < 1e-4:
                s = "%.2e" % value
            elif abs(value) < 1:
                s = ("%.4f" % value).rstrip("0").rstrip(".")
            elif abs(value) < 1000:
                s = ("%.4f" % value).rstrip("0").rstrip(".")
            else:
                s = "%.1f" % value
        elif isinstance(value, (int,)):
            s = str(value)
        else:
            s = str(value)
        self.items.append((name, s, note))

    def fmt(self, v, nd=4):
        if v is None:
            return None
        return float(v)


def sigfigs(v, n=4):
    """Output with 4 significant figures.

    Use `%.4g` rather than computing decimal places by hand: the hand-computed
    version would compute a negative significant-figure count for numbers like
    0.00299, output "0", and erase the information (observed in practice). `%.4g`
    automatically switches to scientific notation for both 1e-16 and 1e5.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    v = float(v)
    if v == 0:
        return "0"
    if v != v:  # NaN
        return None
    s = ("%." + str(n) + "g") % v
    # "1e-05" is invalid in LaTeX; always write it as 1\times10^{-5}.
    #
    # It must be wrapped in \ensuremath: `%.4g` switches to scientific notation
    # when abs(v) < 1e-4, and `\times` is illegal in **text mode**. This is not a
    # hypothetical problem -- after re-running with 5 seeds, rho changed from
    # 1.039e-4 (output "0.0001039", text-safe) to 9.716e-5 (output
    # "9.716\times 10^{-5}"); Table 1's cells are in text mode, so the table end
    # reported "Missing $ inserted" / "Extra }, or forgotten $". \ensuremath makes
    # the macro enter math automatically in text mode and act as a no-op in math
    # mode, so both usages are safe.
    if "e" in s:
        mant, expo = s.split("e")
        expo = int(expo)
        return "\\ensuremath{%s\\times 10^{%d}}" % (mant, expo)
    return s


def main():
    B = Bag()

    # ------------------------------------------------ impossibility: alpha sweep
    imp = load(os.path.join("logs", "verify_impossibility.json"))
    if imp:
        cur = imp["curve"]
        al = cur["alpha"]
        rho = cur["rho_hat"]
        names = ["Zero", "Quarter", "Half", "ThreeQ", "One"]
        for i, a in enumerate(al):
            B.add("ImpRho" + names[i] if i < len(names) else "ImpRho%d" % i,
                  sigfigs(rho[i]), "verify_impossibility: curve.rho_hat[%d] (alpha=%.2f)" % (i, a))
        B.add("ImpAlphaSweep", sigfigs(rho[-1] - rho[0]), "rho_hat(1) - rho_hat(0)")
        B.add("ImpDDev", sigfigs(imp["deviations"]["max_d_dev"]), "deviations.max_d_dev")
        B.add("ImpLamDev", sigfigs(imp["deviations"]["max_lam_dev"]), "deviations.max_lam_dev")
        B.add("ImpPredMad", sigfigs(imp["deviations"]["pred_mad"]), "deviations.pred_mad")
        B.add("ImpKnnK", imp["params"]["KNN_K"][0], "params.KNN_K[0]")
        B.add("ImpSeeds", len(imp["params"]["seeds"]), "params.seeds")
        B.add("ImpAlphas", len(imp["params"]["alphas"]), "params.alphas")
        B.add("ImpC", imp["params"]["C"], "params.C")
        B.add("ImpK", imp["params"]["K"], "params.K")
        B.add("ImpVerdict", imp["verdict"], "verdict")
    else:
        B.missing.append("verify_impossibility.json")

    # ------------------------------------------------ minimax bounds
    mm = load(os.path.join("logs", "verify_minimax.json"))
    if mm:
        cb = mm.get("constant_baseline", {})
        B.add("MinimaxConst", sigfigs(cb.get("worst")), "verify_minimax: constant_baseline.worst")
        B.add("MinimaxConstAt", sigfigs(cb.get("c_star")), "constant_baseline.c_star")
        dev = mm.get("stat_rel_dev", {})
        anv = mm.get("stat_anova", {})
        cvs = mm.get("stat_cv", {})
        m = {"D": "D", "Lambda": "Lam", "K_CH": "KCH", "nn_dist": "NNDist",
             "sep8": "Sep", "kurtosis": "Kurt", "trVar_marginal": "TrVar"}
        worst = 0.0
        for k, short in m.items():
            if k in dev:
                B.add("MinimaxDev" + short, sigfigs(dev[k]), "stat_rel_dev.%s" % k)
                worst = max(worst, dev[k])
            if k in anv:
                B.add("MinimaxP" + short, sigfigs(anv[k].get("p"), 3), "stat_anova.%s.p" % k)
                B.add("MinimaxF" + short, sigfigs(anv[k].get("F"), 3), "stat_anova.%s.F" % k)
            if k in cvs:
                B.add("MinimaxCV" + short, sigfigs(cvs[k], 3), "stat_cv.%s" % k)
        B.add("MinimaxDevMax", sigfigs(worst), "max over stat_rel_dev")
        B.add("MinimaxPAlpha", sigfigs(mm.get("params", {}).get("ALPHA_P"), 2),
              "params.ALPHA_P (ANOVA level)")
        B.add("MinimaxPairedMad", sigfigs(mm.get("paired_rho_mad")), "paired_rho_mad")
        B.add("MinimaxRiskRatio", sigfigs(mm.get("risk_ratio")), "risk_ratio")
        B.add("MinimaxNStats", len(dev), "number of data-only statistics tested")
        lad = mm.get("sample_size_ladder", {})
        if lad:
            keys = sorted(lad.keys(), key=lambda s: int(s))
            for k in ("D", "sep8", "K_CH", "nn_dist", "kurtosis"):
                if k not in lad[keys[0]]:
                    continue
                # ladder[n][k] is compatible with both shapes: the new version is
                # {"diff","noise","snr"}, the old version is a bare float (in which
                # case there is no snr, so skip).
                first, last = lad[keys[0]][k], lad[keys[-1]][k]
                if not (isinstance(first, dict) and isinstance(last, dict)):
                    continue
                B.add("LadderSnr" + m[k] + "First", sigfigs(first.get("snr"), 3),
                      "sample_size_ladder[%s].%s.snr" % (keys[0], k))
                B.add("LadderSnr" + m[k] + "Last", sigfigs(last.get("snr"), 3),
                      "sample_size_ladder[%s].%s.snr" % (keys[-1], k))
            B.add("LadderNFirst", keys[0], "smallest n")
            B.add("LadderNLast", keys[-1], "largest n")
        # The maximum |delta| across sample sizes for the statistics on the same
        # scale as rho* (the three used by the M4 criterion)
        ABS_KEYS = ("D", "nn_dist", "kurtosis")
        if lad:
            worst = 0.0
            for _nk, row in lad.items():
                for k in ABS_KEYS:
                    if k in row and isinstance(row[k], dict):
                        worst = max(worst, row[k]["diff"])
            B.add("LadderWorstAbsDiff", sigfigs(worst, 3),
                  "max over n and over {%s} of |S(1)-S(0)|" % ",".join(ABS_KEYS))
        B.add("LadderAbsMax", sigfigs(mm.get("params", {}).get("ABS_MAX", 0.01), 2),
              "params.ABS_MAX")
        B.add("MinimaxVerdict", mm.get("verdict"), "verdict")
    else:
        B.missing.append("verify_minimax.json")

    # ------------------------------------------------ conditional control (coupling only)
    ct = load(os.path.join("logs", "verify_conditional_theory.json"))
    if ct:
        def g(d, *ks):
            cur = d
            for k in ks:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur
        B.add("CondDIndep", sigfigs(g(ct, "D", "independent")), "verify_conditional_theory: D.independent")
        B.add("CondDTrans", sigfigs(g(ct, "D", "transport")), "D.transport")
        B.add("CondDDiff", sigfigs(g(ct, "D", "abs_diff")), "D.abs_diff")
        bk = (ct.get("best_contrast") or {}).get("k")
        row = None
        for e in (ct.get("knn_sweep") or []):
            if bk is None or e.get("k") == bk:
                row = e
        if row:
            B.add("CondRhoInd", sigfigs(row.get("rho_ind")), "knn_sweep[k=%s].rho_ind" % bk)
            B.add("CondRhoDet", sigfigs(row.get("rho_det")), "knn_sweep[k=%s].rho_det" % bk)
        B.add("CondContrast", sigfigs(g(ct, "best_contrast", "ratio")), "best_contrast.ratio")
        B.add("CondKnnK", bk, "best_contrast.k")
        # 1/k law: k * rho_hat_ind should be ~ 1
        for e in (ct.get("knn_sweep") or []):
            kk = e.get("k")
            word = {1: "One", 2: "Two", 5: "Five", 10: "Ten", 20: "Twenty",
                    30: "Thirty", 50: "Fifty", 100: "Hundred"}.get(kk)
            if word:
                B.add("CondKTimes" + word, sigfigs(e.get("k_times_rho_ind")),
                      "knn_sweep[k=%s].k_times_rho_ind" % kk)
                B.add("CondRhoInd" + word, sigfigs(e.get("rho_ind")),
                      "knn_sweep[k=%s].rho_ind" % kk)
        B.add("CondVerdict", ct.get("verdict"), "verdict")
    else:
        B.missing.append("verify_conditional_theory.json")

    # ------------------------------------------------ multistep (exact velocity field)
    ms = load(os.path.join("logs", "verify_multistep_theory.json"))
    if ms:
        B.add("MSEndpointErr", sigfigs(ms.get("endpoint_identity_max_err")),
              "verify_multistep_theory: endpoint_identity_max_err")
        B.add("MSVarOne", sigfigs(ms.get("one_step_spread")), "one_step_spread")
        word = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen",
                32: "ThirtyTwo", 64: "SixtyFour", 128: "OneTwentyEight",
                256: "TwoFiftySix"}
        for e in (ms.get("sweep") or []):
            w = word.get(e.get("N"))
            if w:
                B.add("MSRho" + w, sigfigs(e.get("rho")), "sweep[N=%s].rho" % e["N"])
                B.add("MSCov" + w, e.get("modes_covered"), "sweep[N=%s].modes_covered" % e["N"])
        B.add("MSVerdict", ms.get("verdict"), "verdict")
    else:
        B.missing.append("verify_multistep_theory.json")

    # ------------------------------------------------ method-family map
    mmap = load(os.path.join("results", "method_map.json"))
    if mmap:
        S = mmap["summary"]
        key = {"cfm_indep@1": "IndepOne", "cfm_indep@32": "IndepMany", "cfm_ot@1": "OtOne",
               "reflow@1": "ReflowOne", "distill@1": "DistillOne",
               "onestep_l2@1": "LtwoOne", "onestep_minM@1": "MinMOne"}
        for k, short in key.items():
            if k in S:
                B.add("MM" + short + "Rho", sigfigs(S[k]["rho"]), "method_map: summary.%s.rho" % k)
                B.add("MM" + short + "Std", sigfigs(S[k].get("rho_std")), "summary.%s.rho_std" % k)
                B.add("MM" + short + "Cov", sigfigs(S[k]["coverage"]), "summary.%s.coverage" % k)
        B.add("MMVerdict", mmap.get("verdict"), "verdict")
    else:
        B.missing.append("method_map.json")

    ch = load(os.path.join("results", "method_map_chamfer.json"))
    if ch:
        s = ch["summary"]
        B.add("MMChamferRho", sigfigs(s["rho"]), "method_map_chamfer: summary.rho")
        B.add("MMChamferStd", sigfigs(s["rho_std"]), "summary.rho_std")
        B.add("MMChamferCov", sigfigs(s["coverage"]), "summary.coverage")
    else:
        B.missing.append("method_map_chamfer.json")

    # ------------------------------------------------ loss ladder
    ll = load(os.path.join("results", "loss_ladder.json"))
    if ll:
        S = ll["summary"]
        key = {"onestep_l2": "Ltwo", "onestep_chamfer": "Chamfer", "onestep_balanced": "Bal"}
        for k, short in key.items():
            if k in S:
                v = S[k]
                B.add("LL" + short + "Rho", sigfigs(v["rho"]), "loss_ladder: summary.%s.rho" % k)
                B.add("LL" + short + "Std", sigfigs(v["rho_std"]), "summary.%s.rho_std" % k)
                B.add("LL" + short + "Cov", sigfigs(v["coverage"]), "summary.%s.coverage" % k)
                B.add("LL" + short + "Disp", sigfigs(v["rho_disp"]), "summary.%s.rho_disp" % k)
                B.add("LL" + short + "AbsErr", sigfigs(v["rho_abs_err"]), "summary.%s.rho_abs_err" % k)
                B.add("LL" + short + "Csw", sigfigs(v["csw"]), "summary.%s.csw" % k)
        if "onestep_chamfer" in S and "onestep_balanced" in S:
            a, b = S["onestep_chamfer"], S["onestep_balanced"]
            B.add("LLRatioAbsErr", sigfigs(a["rho_abs_err"] / max(b["rho_abs_err"], 1e-9)),
                  "chamfer/balanced mean|rho_j-1|")
            B.add("LLRatioCsw", sigfigs(a["csw"] / max(b["csw"], 1e-9)), "chamfer/balanced cSW")
            B.add("LLRatioDisp", sigfigs(a["rho_disp"] / max(b["rho_disp"], 1e-9)),
                  "chamfer/balanced std_j(rho_j)")
        # The signature that unbalanced loss "only recovers the support": the
        # generated per-condition spread tends toward a constant.
        # Metric = the coefficient of variation (CV) across conditions of the
        # generated trVar_j (:= rho_j * trVar_j^true), compared against the CV of
        # the true trVar_j. The closer the two, the less the model discriminates
        # between conditions.
        tv = ll.get("tr_var_cond")
        if tv and ll.get("per_seed"):
            import statistics as _st
            for meth, short in (("onestep_chamfer", "Chamfer"),
                                ("onestep_balanced", "Bal")):
                cvs = []
                for sd, d in ll["per_seed"].items():
                    r = d.get(meth, {}).get("rho_per_cond")
                    if not r:
                        continue
                    gen = [r[j] * tv[j] for j in range(len(r))]
                    mu = sum(gen) / len(gen)
                    sdv = _st.pstdev(gen)
                    cvs.append(sdv / mu if mu else 0.0)
                if cvs:
                    B.add("LL" + short + "GenCV", sigfigs(sum(cvs) / len(cvs), 3),
                          "%s: CV over j of generated trVar_j" % meth)
            tv_arr = [float(x) for x in tv]
            mu = sum(tv_arr) / len(tv_arr)
            B.add("LLTrueCV", sigfigs(_st.pstdev(tv_arr) / mu, 3),
                  "CV over j of true trVar(x1|c_j)")
        B.add("LLVerdict", ll.get("verdict"), "verdict")
    else:
        B.missing.append("loss_ladder.json")

    # ------------------------------------------------ pooled comparison (Chamfer, argmin across conditions)
    # The paper needs two Chamfer rows: the pooled version (the practical
    # implementation) and the per-condition version (the setup of Thm 4).
    # Their difference isolates the single variable "pooling"; this is the true
    # source of the "support recovered, weights free" signature.
    cp = load(os.path.join("results", "chamfer_pooled.json"))
    if cp:
        import statistics as _st
        v = cp["summary"]
        for suf, fld in (("Rho", "rho"), ("Std", "rho_std"), ("Cov", "coverage"),
                         ("Disp", "rho_disp"), ("AbsErr", "rho_abs_err"),
                         ("Csw", "csw")):
            B.add("LLChamferPooled" + suf, sigfigs(v[fld]),
                  "chamfer_pooled: summary.%s" % fld)
        tvp = cp.get("tr_var_cond")
        if tvp and cp.get("per_seed"):
            cvs = []
            for sd, d in cp["per_seed"].items():
                r = d.get("rho_per_cond")
                if not r:
                    continue
                gen = [r[j] * tvp[j] for j in range(len(r))]
                mu = sum(gen) / len(gen)
                sdv = _st.pstdev(gen)
                cvs.append(sdv / mu if mu else 0.0)
            if cvs:
                B.add("LLChamferPooledGenCV", sigfigs(sum(cvs) / len(cvs), 3),
                      "chamfer_pooled: CV over j of generated trVar_j")
    else:
        B.missing.append("chamfer_pooled.json")

    # ------------------------------------------------ MNIST real-data confirmation
    mn = load(os.path.join("results", "mnist_collapse.json"))
    if mn:
        ms = mn["summary"]
        key = {"A_independent": "A", "B_dependent_meandep0": "B", "C_ot": "C",
               "D_sorted_meanmax": "D"}
        for cfg, short in key.items():
            if cfg not in ms:
                continue
            g = ms[cfg]
            B.add("Mnist" + short + "Dcor", sigfigs(g["dcor"]["mean"]),
                  "mnist_collapse: summary.%s.dcor.mean" % cfg)
            B.add("Mnist" + short + "RhoTr", sigfigs(g["rho_trained"]["mean"]),
                  "mnist_collapse: summary.%s.rho_trained.mean" % cfg)
            B.add("Mnist" + short + "Csw", sigfigs(g["csw1"]["mean"]),
                  "mnist_collapse: summary.%s.csw1.mean" % cfg)
        B.add("MnistAMuErr", sigfigs(ms["A_independent"]["mu_err"]["mean"]),
              "mnist_collapse: summary.A_independent.mu_err.mean")
        B.add("MnistBMuErr", sigfigs(ms["B_dependent_meandep0"]["mu_err"]["mean"]),
              "mnist_collapse: summary.B_dependent_meandep0.mu_err.mean")
        B.add("MnistNSeeds", len(mn.get("per_seed", {})),
              "mnist_collapse: len(per_seed)")
        # Cross-seed dispersion: the largest standard deviation among the reported
        # quantities (rho_trained/csw1/dcor) and which one it belongs to
        best = None
        for cfg in ("A_independent", "B_dependent_meandep0", "C_ot",
                    "D_sorted_meanmax"):
            if cfg not in ms:
                continue
            for k in ("rho_trained", "csw1", "dcor"):
                sd = ms[cfg].get(k, {}).get("std")
                if isinstance(sd, (int, float)) and (best is None or sd > best[0]):
                    best = (sd, "%s.%s" % (cfg, k))
        if best:
            B.add("MnistMaxStd", sigfigs(best[0]),
                  "mnist_collapse: max summary std over rho_trained/csw1/dcor")
            # LaTeX text mode forbids bare _ and bare .: convert "A_independent.dcor"
            # into a typesettable form
            B.add("MnistMaxStdOf", best[1].replace("_", "-").replace(".", ", "),
                  "mnist_collapse: which quantity has the max std")
        # Training time cost (mean per-seed seconds): D sorted pairing vs C Hungarian assignment
        for cfg, short in (("C_ot", "C"), ("D_sorted_meanmax", "D")):
            if not mn.get("per_seed"):
                continue
            secs = [d[cfg]["seconds"] for d in mn["per_seed"].values()
                    if cfg in d and isinstance(d[cfg].get("seconds"), (int, float))]
            if secs:
                B.add("Mnist" + short + "Seconds", sigfigs(sum(secs) / len(secs), 3),
                      "mnist_collapse: mean per-seed seconds of %s" % cfg)
    else:
        B.missing.append("mnist_collapse.json")

    # ------------------------------------------------ Lambda negative results
    vd = load(os.path.join("logs", "validate_deficit.json"))
    if vd:
        pop = vd.get("R6_population") or {}
        for k, short in (("gauss", "Gauss"), ("ring8", "RingEight"), ("banana", "Banana"),
                         ("two_modes", "TwoModes"), ("uniform_ball", "Ball"),
                         ("laplace", "Laplace"), ("lognorm", "LogNorm"),
                         ("exp_skew", "ExpSkew"), ("t_df3", "StudentT"),
                         ("t_df5", "StudentTFive"), ("hetero", "Hetero")):
            v = pop.get(k)
            if isinstance(v, dict):
                v = v.get("value") or v.get("lam") or v.get("mean")
            B.add("Lam" + short, sigfigs(v), "validate_deficit: R6_population.%s" % k)
        ceil = vd.get("ceiling")
        B.add("LamCeiling", sigfigs(ceil), "validate_deficit: ceiling")
        # [IMPORTANT] the ceiling field is itself the analytic expression
        # 1-2*sqrt(2/pi)+1, so |ceiling - theory| is identically 0; reporting it
        # says nothing (a past bug: the paper wrote "reproduced to 0", which reads
        # like an exact reproduction but is actually just tautology).
        # What is actually informative is the difference between the **estimator**
        # and the analytic upper bound, R1_ceiling[*].abs_err.
        r1 = vd.get("R1_ceiling") or []
        errs = [e.get("abs_err") for e in r1 if isinstance(e.get("abs_err"), (int, float))]
        if errs:
            B.add("LamCeilingEstErr", sigfigs(max(errs)),
                  "max over R1_ceiling[*].abs_err (estimator vs analytic ceiling)")
            n_min = min(e.get("n") for e in r1 if e.get("n") is not None)
            n_max = max(e.get("n") for e in r1 if e.get("n") is not None)
            B.add("LamCeilingNMin", n_min, "R1 smallest sample size")
            B.add("LamCeilingNMax", n_max, "R1 largest sample size")
        # Cross-sample-size consistency (R2): the difference for the same shape at
        # n=2000 vs n=200000
        r2 = vd.get("R2_consistency") or []
        d2 = [e.get("abs_diff") for e in r2 if isinstance(e.get("abs_diff"), (int, float))]
        if d2:
            B.add("LamConsistencyMaxDiff", sigfigs(max(d2)),
                  "max over R2_consistency[*].abs_diff")
        # Rank correlation: R4 = convergence validity vs exact OT (this is the one
        # cited in the paper); R5 = construct validity vs "Gaussian-patch erroneous
        # quality". The two are different things; do not mix them.
        rc4 = vd.get("R4_rank_corr")
        rc5 = vd.get("R5_rank_corr")
        if isinstance(rc4, (int, float)):
            B.add("LamRankCorrOT", sigfigs(rc4, 3), "R4_rank_corr (vs exact OT)")
        if isinstance(rc5, (int, float)):
            B.add("LamRankCorrConstruct", sigfigs(rc5, 3), "R5_rank_corr (vs constructed)")
        mad4 = vd.get("R4_mean_abs_diff")
        if isinstance(mad4, (int, float)):
            B.add("LamMeanAbsDiffOT", sigfigs(mad4), "R4_mean_abs_diff (vs exact OT)")
        B.add("LamModeDetector", str(bool(vd.get("usable_as_mode_detector"))).lower(),
              "validate_deficit: usable_as_mode_detector")
    else:
        B.missing.append("validate_deficit.json")

    # ------------------------------------------------ write file
    lines = ["% auto-generated, do not edit by hand. Read from logs/ and results/ by code/analysis/make_theory_macros.py.",
             "% the source of every macro is recorded in the end-of-line comment."]
    for name, val, note in B.items:
        lines.append("\\newcommand{\\%s}{%s}  %% %s" % (name, val, note))
    out = os.path.join(ROOT, "paper", "numbers_theory.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s  (%d macros)" % (out, len(B.items)))
    if B.missing:
        print("MISSING (skipped): %s" % ", ".join(B.missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
