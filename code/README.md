# code/ Directory Guide

This directory contains all the code for the paper "When Does One-Step Endpoint
Regression Preserve Conditional Spread? Coupling, Mean Dependence, and Data-Side
Non-Identifiability" (ICLR 2027 submission).

> **Note.** This guide and the in-repository source comments describe the code
> directory for the paper; `REPRODUCE.md` at the repository root is the
> end-to-end reproduction quickstart. Directory names mirror the paper's
> structure: `experiments/` (Sec. 5), `analysis/` (verification, macros, figures,
> audit), `src/` (shared library), `tests/` (integrity guards), `archive/`
> (frozen legacy, not part of the pipeline).

**Artifacts are not stored in this directory**: raw experiment results, figures,
and run logs are written to the same-named directories at the project root
(`../results`, `../figures`, `../logs`).

---

## 1. Current structure (after the third-round rewrite, 2026-09-16)

```
code/
├── README.md               ← this file
├── requirements.txt / requirements-lock.txt
├── _run_pipeline.py        sole entry point: rebuild macros, figures, and the paper from existing results (no training run)
│
├── experiments/            ① run experiments → ../results/*.json
│   ├── run_method_map.py       cross-method-family failure map (8 method families × NFE∈{1,32} × 5 seeds)
│   │                           also imported by the two scripts below; do not move it
│   ├── run_chamfer_supplement.py  Table 1: 7th-family bidirectional Chamfer (**whole-batch pooled** assignment) × 5 seeds
│   ├── run_loss_ladder.py      loss ladder L1 / L2b (per-condition Chamfer) / L3 (per-condition balanced) × 5 seeds
│   ├── run_chamfer_pooled.py   pooled comparison of the loss ladder, L2a (whole-batch argmin) × 5 seeds
│   └── run_mnist_collapse.py   MNIST 16×16 real-data endpoint regression (four couplings A, B, C, D; M1–M9 × 10 seeds)
│                               all five scripts support `--seeds a,b --merge` (resume run; do not recompute already-computed seeds)
│
├── analysis/              ② verification, aggregation, and checks
│   ├── verify_multistep_theory.py     multi-step Euler (closed-form exact velocity field) → logs/*.json
│   ├── verify_conditional_theory.py   conditional control: same D, different couplings
│   ├── verify_coupling_theory.py      earliest version of the coupling comparison (kept as corroborating evidence)
│   ├── verify_impossibility.py        **impossibility theorem, constructive version** (alpha family, ρ*(α)=α²)
│   ├── verify_minimax.py              **impossibility theorem, minimax version** (7 data-side statistics)
│   ├── validate_deficit.py            estimator and discriminative power of the sliced-W₂ Gaussian deficit Λ (incl. the R6 negative result)
│   ├── run_falsification.py           pre-registered refutation experiment for Λ (false-positive rate)
│   ├── margin_vs_noise.py             computes the "margin / seed-noise" ratio for each criterion (evidence-strength audit §2.4)
│   ├── make_theory_macros.py          logs/ + results/ → the manuscript's ../paper/numbers_theory.tex
│   ├── make_fig_impossibility.py      → ../figures/fig_impossibility.pdf (paper Figure 1)
│   ├── make_fig_ladder.py             → ../figures/fig_loss_ladder.pdf (paper Figure 2)
│   ├── check_macros.py                pre-build: missing macros / illegal macro names / dead macros (--strict for pipeline use)
│   └── build_paper.py                 four-pass build + error summary (page count / overfull / undefined references)
│                                      (manuscript-tree only: needs ../paper/, not shipped in this repo)
│
├── src/                   ③ library
│   ├── deficit.py          **in use**: sliced-W₂ Gaussian deficit Λ (pure numpy, includes Acklam's inverse-normal CDF)
│   ├── common.py           \
│   ├── tasks.py             |  task/flow/metric library from the earlier paper draft; the current pipeline uses
│   ├── flows.py             |  only deficit.py, while the other four are kept for reference and are used by the
│   ├── metrics.py           |  legacy experiment scripts under code/archive/stale_protocol/
│   └── discriminant.py     /
│
├── tests/                 ④ regression tests
│   └── test_audit_guards.py    guard tests during the audit (estimator bias, conditional-metric scope)
│
├── data/                  data cache directory (the current pipeline **does not use** it: all experiments are synthetic families, no external downloads)
└── archive/               ⑤ archival, **not part of the pipeline**
    ├── stale_protocol/        experiments under the old protocol (run_toy/rl/mnist/ablations) — results voided,
    │                          kept so the data-generation code can be reused when later ported to the new protocol
    ├── _reorg_code_tree.py    directory-organization script (one-off, already run)
    ├── _probe_mnist.py        per-bracket comparison of marginal vs conditional statistics (historical forensics)
    ├── _probe_minm_degeneracy.py  forensics on the old min-of-M implementation being bit-identical to pointwise L2
    ├── apply_macro_alias.py   macro names with digits → pure-letter macro names (incl. --check dry run)
    ├── compress_main{,2,3}.py scripts that moved the body down to 9 pages (old version)
    └── rewrite_{abl,claims,mnist_conv}.py  body-rewriting scripts, already applied
```

Scripts moved to the project-root `_archive_20260916/` (old pipeline; their outputs
are no longer referenced by the body): `make_tables.py`, `make_figures.py`,
`make_vis.py`, `audit_format.py`, `_final_check.py`, `_verify_estimator_bias.py`,
`_verify_kappa.py`.

---

## 2. How to run

```bash
PY=python                     # requires numpy / scipy / matplotlib / torch (cpu)

# ① one-shot rebuild (no training): make_theory_macros → two figures → check_macros → build_paper
$PY -u code/_run_pipeline.py          # log goes to logs/pipeline.log; at the end look for "PIPELINE OK"

# ② run experiments step by step (each writes results/*.json)
$PY -u code/experiments/run_method_map.py          # ~75 min (CPU, 14 threads; incl. the revised min-of-8)
$PY -u code/experiments/run_chamfer_supplement.py  # ~5 min (Table 1 pooled-Chamfer row)
$PY -u code/experiments/run_loss_ladder.py         # ~20 min (Table 2 + Figure 2: L1/L2b/L3)
$PY -u code/experiments/run_chamfer_pooled.py      # ~7 min (Table 2 + Figure 2: L2a pooled comparison)

# ③ run verification step by step (each writes logs/*.json)
$PY -u code/analysis/verify_multistep_theory.py
$PY -u code/analysis/verify_conditional_theory.py
$PY -u code/analysis/verify_impossibility.py
$PY -u code/analysis/verify_minimax.py             # ~10 min
$PY -u code/analysis/validate_deficit.py

# ④ regression tests
$PY -m unittest discover -s code/tests -v
```

The environment variable `PIPELINE_PY` can override the interpreter used by
`_run_pipeline.py`.

---

## 3. Three conventions (read before changing anything)

**1) Path anchors must be self-healing; do not hard-code the directory depth.**
All scripts uniformly walk up to **the level that contains both `code/` and `paper/`**:

```python
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_root(d):
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d

_ROOT_ = _find_root(_HERE)
_CODE_ = os.path.join(_ROOT_, "code")
```

**New scripts should copy this block verbatim.** A hard-coded `dirname` count
breaks as soon as the directory layout moves.

**2) Do not place artifact directories under `code/`.**
`configs/ figures/ logs/ results/ tables/` once existed under `code/` and were all
empty; the real artifacts have always lived at the project root — such empty
directories only send people looking in the wrong place. All new output is written
to the same-named directory at the project root.

**3) The paper body must not contain hand-written numbers.**
All experimental numbers in the body come from `../paper/numbers_theory.tex`,
generated by `analysis/make_theory_macros.py` from `../logs/*.json` and
`../results/*.json`; **every macro carries a `% source: file.field` comment at the
end of its line**. To change a number, change the experiment first, then rerun the
pipeline; do not edit `numbers_theory.tex` directly (it gets overwritten).
Two further hard LaTeX constraints: **macro names may contain only letters**
(`\RhoAlOne` is valid, `\rho1` is a fatal error); values containing scientific
notation like `\times 10^{-16}` **must be inside math mode**, otherwise LaTeX
reports `Missing $ inserted`.

> **2026-09-16 fix**: `sigfigs` uses `%.4g`, which auto-switches values with
> `abs(v) < 1e-4` into scientific notation — after re-running with 5 seeds the loss
> ladder's $\rho$ changed from `1.039e-4` to `9.716e-5`, crossing the threshold, so
> Table 1's **text-mode** cells failed to compile immediately. Now
> `make_theory_macros.py` emits such values as `\ensuremath{...}`, which is safe in
> both text and math mode; numeric columns in tables are still **recommended to be
> wrapped entirely** in math delimiters for cleaner typesetting.

> `check_macros.py` lists macros that are "defined but not referenced in the body"
> as `[DEAD]` (currently 74 of 183). This is an **informational item, not a
> failure**: they record intermediate statistics that the paper does not reference
> (the ANOVA $F/p/\eta^2$ of each statistic, the per-bracket signal-to-noise ratios
> of the sample-size ladder, the $\Lambda$ ten-distribution values, etc.) and serve
> as evidence of traceability. Only `[ILLEGAL]` and `[MISSING]` make `--strict`
> return non-zero.

---

## 4. Result-schema versions

**The current paper references only four result files**; all others are
old-protocol artifacts, already archived:

| File | Corresponding conclusion |
|---|---|
| `../results/method_map.json` | Table 1: cross-method-family failure map |
| `../results/method_map_chamfer.json` | Table 1, last row: bidirectional Chamfer (whole-batch pooled) |
| `../results/loss_ladder.json` | Table 2 + Figure 2: L1 / L2b / L3 |
| `../results/chamfer_pooled.json` | Table 2 + Figure 2: L2a pooled comparison |

The JSON files under `../logs/` are the verification scripts' reports
(`verify_*.json`, `validate_deficit.json`, `falsification_lambda_stress.json`).

**Two red lines of the schema** (do not regress):

1. **One-step penalty is always decided by the conditional statistic cSW
   (conditional sliced $W_2$), never by marginal sliced $W_2$** — the latter
   conflates the "one-step floor" with "mode collapse" and would report a penalty
   for underfit fields too.
2. **Λ must not be called a multimodality detector**. It has been demoted to a
   descriptive index: in practice it ranks the unimodal banana above the 8-mode
   ring (`validate_deficit.py`'s R6 is a **deliberately-negative** negative
   result).
3. **The M candidates of min-of-M must come from mutually independent source
   noise**. A one-step network is a deterministic map, so if the same `x0` is
   copied with `np.repeat`, the M candidates are identical under any training
   state, `argmin` is always 0, and the loss is **bit-identical to pointwise L2**
   (forensics: `archive/_probe_minm_degeneracy.py` measures `max|Δw| = 0` under
   small-scale training). The old implementation hit this pitfall;
   `run_method_map.py`'s `onestep_minM` draws noise independently with
   `rng.normal(size=(bs*M, DIM))`.
4. **The "scope" of the two `argmin` calls in unbalanced Chamfer must match the
   balanced version.** The paper's Thm 4(L2) explicitly requires `argmin` to be
   taken **per condition**; the first L2 draft took it **whole-batch**, differing
   from L3's per-condition Hungarian by two variables (whether balanced AND whether
   split by condition). After the fix, `run_loss_ladder.py::train_chamfer` takes
   `argmin` per condition, while the pooled version is reproduced separately by
   `run_chamfer_pooled.py`. The empirical conclusion is therefore rewritten:
   **what flattens the conditional spread is "pooling", not "unbalance"** — L2b
   (per-condition unbalanced) essentially returns to the diagonal, while L2a
   (pooled) is what produces the "support recovered, weights free" signature.

Old MNIST results must contain `schema_version=2` and
`metric_protocol=condition-knn-v2-no-label-leakage`; the old version used true
class labels to construct the "conditional reference distribution" and is voided.
The aggregation scripts raise an error on encountering such legacy JSON, to prevent
old numbers from re-entering the tables or the paper.
