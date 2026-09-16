# Reproduction guide

Two levels: **rebuild the PDF from shipped results** (seconds), and **re-run everything
from scratch** (~2 h CPU).

---

## Level 1 — rebuild the paper from the shipped results

```bash
python -m pip install -r requirements.txt
python code/_run_pipeline.py
```

`_run_pipeline.py` runs five steps in order and stops at the first failure:

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

python code/_run_pipeline.py                        # rebuild macros, figures, PDF
```

All seeds are fixed and passed explicitly, so runs are reproducible on the same
platform. Small floating-point differences across BLAS builds are expected.

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

---

## Environment

| Item | Value |
|---|---|
| Python | 3.13 |
| Packages | `numpy>=1.26`, `scipy>=1.11`, `torch>=2.2` (CPU), `matplotlib>=3.8` |
| TeX | any TeX Live with `pdflatex` + `bibtex`; edit `TEXLIVE` in `code/analysis/build_paper.py` if yours is elsewhere |
| Hardware | CPU only; no GPU required or used |
| Network | only for `paper/fetch_refs.py` (arXiv metadata). No dataset download: every experiment is on synthetic families with a closed-form ground truth. |

`code/data/` exists only for the archived MNIST protocol under `code/archive/`; it is
git-ignored, and the reproduction path above never touches it.

---

## Regenerating the bibliography

```bash
python paper/fetch_refs.py     # arXiv -> paper/refs_raw.json  (needs network; retries + cache merge)
python paper/make_bib.py       # refs_raw.json -> paper/references.bib
```

`fetch_refs.py` reads `<meta name="citation_*">` from `arxiv.org/abs/<id>` pages. The
arXiv API and OpenAlex were both rate-limited from the machine this was developed on,
which is why the page-scraping route is used; the script retries with backoff and merges
into the existing cache so a throttled run does not lose entries.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OMP: Error #15 ... libiomp5md.dll already initialized` | set `KMP_DUPLICATE_LIB_OK=TRUE` |
| `File 'iclr2027_conference.sty' not found` | compile through `build_paper.py`; it sets `TEXINPUTS=./iclr2027;` and `BSTINPUTS=./iclr2027;` relative to `paper/` |
| TeX cannot open files although the paths exist | `TEXINPUTS` must be **relative and use forward slashes**; an absolute path containing non-ASCII characters breaks `\openin` |
| `Missing $ inserted` | a macro expanded to scientific notation (e.g. `4.41\times 10^{-5}`) used outside math mode; wrap it as `$\Macro$` |
| `UnicodeDecodeError` on a log file | PowerShell's `*>` redirect writes UTF-16; re-encode or read with the right codec |
