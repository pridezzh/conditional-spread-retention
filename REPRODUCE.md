# Reproduction guide

> **Current evidence status:** the shipped five-seed training JSON files predate the
> provenance schema and combine cached seeds `0,1,2` with rerun seeds `3,4`. Level 1
> checks deterministic assembly and numerical consistency only. A submission-grade
> reproduction requires Level 2 without `--merge`, followed by the strict audit.

Two levels: **verify the shipped results** (minutes), and **re-run everything from
scratch** (~2 h CPU).

---

## Level 1 — verify the shipped results

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s code/tests    # 10 regression guards
python code/analysis/verify_minimax.py       # each verify_*.py asserts internally
```

`_run_pipeline.py` runs five steps in order and stops at the first failure. Steps 4 and 5
(`check_macros.py`, `build_paper.py`) read the manuscript, so the pipeline cannot reach
`PIPELINE OK` outside the authoring tree; run the stages you need individually instead.

## Scope of this package

This repository is the **reproduction payload for the experiments and results**: the
training runs, the verification suite, the released result files, and the figures they
produce. The **manuscript and the ICLR 2027 style files are submitted separately** via
OpenReview and are deliberately not part of this repository. Consequently the final
stages of the one-shot pipeline — which emit the manuscript's `numbers_theory.tex` and
compile the PDF — are out of scope here; the stages that produce `results/`, `logs/` and
`figures/` are fully covered.

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `code/analysis/make_theory_macros.py` | `results/*.json` + `logs/*.json` → `paper/numbers_theory.tex` |
| 2 | `code/analysis/make_fig_impossibility.py` | → `figures/fig_impossibility.{pdf,png}` (Figure 1) |
| 3 | `code/analysis/make_fig_ladder.py` | → `figures/fig_loss_ladder.{pdf,png}` (Figure 2) |
| 4 | `code/analysis/check_macros.py --strict` | every referenced macro exists; no macro name contains a digit |
| 5 | `code/analysis/build_paper.py --clean` | `pdflatex → bibtex → pdflatex ×2`, then reports errors |

Success criteria (all must hold):

```
hard errors    : 0
undefined refs : none
undefined cites: none
overfull boxes : 0
verdict        : OK
PIPELINE OK
```

The build script treats a compile as successful only if **no** `pdflatex` pass exits
non-zero. Note that `-file-line-error` makes TeX report errors as `./main.tex:587:
<message>` rather than starting with `!`; the checker matches the `file:line:` form,
which is why it catches errors such as `Missing $ inserted` that a `!`-only grep misses.

---

## Level 2 — re-run the experiments and verifications

```bash
# experiments -> results/*.json
python code/experiments/run_method_map.py           # ~75 min
python code/experiments/run_chamfer_supplement.py   # ~5 min  (Table 1, pooled Chamfer row)
python code/experiments/run_loss_ladder.py          # ~20 min (Table 2 + Fig 2: L1 / L2b / L3)
python code/experiments/run_chamfer_pooled.py       # ~7 min  (Table 2 + Fig 2: L2a, pooled)

# verifications -> logs/*.json
python code/analysis/verify_multistep_theory.py
python code/analysis/verify_conditional_theory.py
python code/analysis/verify_coupling_theory.py
python code/analysis/verify_impossibility.py
python code/analysis/verify_minimax.py
python code/analysis/validate_deficit.py
python code/analysis/run_falsification.py

python code/analysis/make_fig_impossibility.py      # figures -> figures/*.pdf (no manuscript needed)
python code/analysis/make_fig_ladder.py

# manuscript-tree only, hence not runnable here: make_theory_macros.py, check_macros.py,
# build_paper.py, and the page-limit check inside audit_submission.py
```

Level 2 is satisfied when every `verify_*.py` / `validate_deficit.py` / `run_falsification.py`
exits zero (they assert their own analytic predictions) and the regenerated
`results/*.json` agree with the shipped values to floating-point tolerance.

Run the experiment commands **without `--merge`** so every seed is regenerated from one
revision. New outputs include a schema version, generator, timestamp and SHA-256 protocol
fingerprint. Small floating-point differences across BLAS builds are expected.

---

## Claim → artefact map

Each row names the paper construct, the script that produces the evidence, and the JSON
field the macro reads.

| Paper claim | Script | JSON evidence |
|---|---|---|
| `rho_N → 1` under multi-step Euler, exact velocity field | `analysis/verify_multistep_theory.py` | `logs/verify_multistep_theory.json`: `sweep[*].rho`, `endpoint_identity_max_err` |
| Same `D`, different coupling (32× swing in one-step `rho`) | `analysis/verify_conditional_theory.py` | `logs/verify_conditional_theory.json`: `D.independent`, `D.transport`, `knn_sweep[*]`, `best_contrast` |
| `1/k` bias law of the `k`NN estimator | `analysis/verify_conditional_theory.py` | same file, `knn_sweep[*].k_times_rho_ind` |
| Alpha family with fixed data, `rho*(alpha) = alpha^2` | `analysis/verify_impossibility.py` | `logs/verify_impossibility.json`: `curve.rho_hat`, `deviations.pred_mad` |
| Minimax value `1/2`; seven flat data-side statistics | `analysis/verify_minimax.py` | `logs/verify_minimax.json`: `constant_baseline`, `stat_anova`, `sample_size_ladder`, `risk_ratio` |
| Method map (Table 1) | `experiments/run_method_map.py` | `results/method_map.json`: `summary` |
| Pooled bidirectional Chamfer row (Table 1) | `experiments/run_chamfer_supplement.py` | `results/method_map_chamfer.json` |
| Loss ladder (Table 2, Figure 2): L1 / L2b / L3 | `experiments/run_loss_ladder.py` | `results/loss_ladder.json`: `summary`, `tr_var_cond`, `per_seed` |
| L2a pooled row (Table 2, Figure 2) | `experiments/run_chamfer_pooled.py` | `results/chamfer_pooled.json`: `summary`, `tr_var_cond`, `per_seed` |
| Negative result: `Lambda` sorted ring8 below banana | `analysis/validate_deficit.py` | `logs/validate_deficit.json`: `R6_population`, `R1_ceiling`, `R4_rank_corr` |
| Pre-registered false-positive stress test | `analysis/run_falsification.py` | `logs/falsification_lambda_stress.json` |
| MNIST real-data confirmation (Sec. 5.6): couplings (i)-(iv), checks M1-M9 | `experiments/run_mnist_collapse.py` | `results/mnist_collapse.json`: `summary`, `checks`, `per_seed` |

`results/mnist_collapse.json` is rebuilt from scratch when its protocol fingerprint
changes (`code/src/provenance.py`); merging seeds across revisions is rejected, so
all ten seeds always come from one frozen code revision.

---

## Environment

| Item | Value |
|---|---|
| Python | 3.13 |
| Packages | `numpy>=1.26`, `scipy>=1.11`, `torch>=2.2` (CPU), `matplotlib>=3.8` |
| TeX | any TeX Live with `pdflatex` + `bibtex`; edit `TEXLIVE` in `code/analysis/build_paper.py` if yours is elsewhere |
| Hardware | CPU only; no GPU required or used |
| Network | only for the one-time MNIST download in `code/src/tasks.py` (falls back to scikit-learn digits offline). |

`code/data/mnist_16.npz` is the preprocessed 16x16 MNIST cache used by
`experiments/run_mnist_collapse.py`; it ships with the repository so the real-data
experiment reproduces offline.

---

## Regenerating the bibliography

The bibliography regeneration scripts live in the manuscript tree and are not
shipped here.

`fetch_refs.py` reads `<meta name="citation_*">` from `arxiv.org/abs/<id>` pages. The
arXiv API and OpenAlex were both rate-limited from the machine this was developed on,
which is why the page-scraping route is used; the script retries with backoff and merges
into the existing cache so a throttled run does not lose entries.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OMP: Error #15 ... libiomp5md.dll already initialized` | set `KMP_DUPLICATE_LIB_OK=TRUE` |
| `File 'iclr2027_conference.sty' not found` | manuscript-tree only: compile through `build_paper.py`, which sets `TEXINPUTS=./iclr2027;` and `BSTINPUTS=./iclr2027;` relative to `paper/` |
| TeX cannot open files although the paths exist | `TEXINPUTS` must be **relative and use forward slashes**; an absolute path containing non-ASCII characters breaks `\openin` |
| `Missing $ inserted` | a macro expanded to scientific notation (e.g. `4.41\times 10^{-5}`) used outside math mode; wrap it as `$\Macro$` |
| `UnicodeDecodeError` on a log file | PowerShell's `*>` redirect writes UTF-16; re-encode or read with the right codec |
