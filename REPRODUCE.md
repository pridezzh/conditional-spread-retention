# Reproduction guide

This repository is the **reproduction payload** for the experiments and results: the
training runs, the verification suite, the released per-run result files, and the
figures they produce. The manuscript is submitted separately and is not part of this
repository.

Two levels: **verify the shipped results** (minutes), and **re-run everything from
scratch** (~2 h CPU).

---

## Level 1 — verify the shipped results

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s code/tests    # 10 regression guards
python code/analysis/verify_minimax.py       # each verify_*.py asserts its prediction
```

Every `verify_*.py`, `validate_deficit.py` and `run_falsification.py` re-derives an
analytic prediction of the theory from scratch and exits non-zero if the prediction is
not met, so Level 1 needs no external reference values.

---

## Level 2 — re-run the experiments and verifications

```bash
# experiments -> results/*.json
python code/experiments/run_method_map.py           # ~75 min  (Table 1)
python code/experiments/run_loss_ladder.py          # ~20 min  (Table 2 + Fig 2: L1 / L2b / L3)
python code/experiments/run_chamfer_pooled.py       # ~7 min   (Table 2 + Fig 2: L2a, pooled)
python code/experiments/run_chamfer_supplement.py   # ~5 min   (Table 1, pooled Chamfer row)
python code/experiments/run_mnist_collapse.py       # real-image endpoint study (Sec. 5.6)

# verifications -> logs/*.json
python code/analysis/verify_multistep_theory.py
python code/analysis/verify_conditional_theory.py
python code/analysis/verify_coupling_theory.py
python code/analysis/verify_impossibility.py
python code/analysis/verify_minimax.py
python code/analysis/validate_deficit.py
python code/analysis/run_falsification.py

# figures -> figures/*.pdf
python code/analysis/make_fig_impossibility.py
python code/analysis/make_fig_ladder.py
```

Run the experiment commands **without `--merge`** so every seed is regenerated from one
revision. New outputs include a schema version, generator, timestamp and SHA-256
protocol fingerprint. Small floating-point differences across BLAS builds are expected.

Level 2 is satisfied when every verification script exits zero (they assert their own
analytic predictions) and the regenerated `results/*.json` agree with the shipped values
to floating-point tolerance.

---

## Claim → artefact map

Each row names the paper construct, the script that produces the evidence, and the JSON
field the result is read from.

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
| Python | 3.10+ (tested on 3.13) |
| Packages | `numpy>=1.26`, `scipy>=1.11`, `torch>=2.2` (CPU), `matplotlib>=3.8` |
| Hardware | CPU only; no GPU required or used |
| Network | only for the one-time MNIST download in `code/src/tasks.py` (falls back to scikit-learn digits offline) |

`code/data/mnist_16.npz` is the preprocessed 16x16 MNIST cache used by
`experiments/run_mnist_collapse.py`; it ships with the repository so the real-data
experiment reproduces offline.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OMP: Error #15 ... libiomp5md.dll already initialized` | set `KMP_DUPLICATE_LIB_OK=TRUE` |
| `UnicodeDecodeError` on a log file | PowerShell's `*>` redirect writes UTF-16; re-encode or read with the right codec |
| A script reports `project root not found` | it resolves the root as the first ancestor holding both `code/` and `results/`; run it from inside this checkout |
