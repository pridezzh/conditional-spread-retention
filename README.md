# When Does One-Step Endpoint Regression Preserve Conditional Spread?

Code, data, and results for the accompanying study of one-step generative-policy
failure boundaries under **conditional mean dependence**.

The paper studies when a single-step endpoint regression `f*(x0, c) = E[x1 | x0, c]`
preserves the conditional spread of the target. It proves an impossibility theorem
(a one-parameter target family with identical `(c, x1)` marginals yet `rho*(alpha)`
sweeping all of `[0,1]`), gives a `k`NN estimator with an analytic `1/k` bias law, and
confirms the predicted collapse/escape ladder on synthetic families plus a real-image
MNIST 16x16 endpoint study.

**This repository is the reproduction payload**: the training runs, the verification
suite, the released per-run results, and the figures they produce. The manuscript is
submitted separately and is not part of this repository.

---

## Requirements

| Item | Version |
|---|---|
| Python | 3.10+ (tested on 3.13) |
| Packages | `numpy>=1.26`, `scipy>=1.11`, `torch>=2.2` (CPU build), `matplotlib>=3.8` |
| Hardware | CPU only; no GPU required or used |

```bash
pip install -r requirements.txt
```

> **Windows:** if you hit `OMP: Error #15`, set `KMP_DUPLICATE_LIB_OK=TRUE` before
> importing numpy/torch.

---

## Quick start

```bash
python -m unittest discover -s code/tests        # 10 regression guards
python code/analysis/verify_minimax.py           # each verify_*.py asserts its prediction
python code/analysis/make_fig_ladder.py          # regenerates figures/fig_loss_ladder.pdf
```

Every `verify_*.py` checks an analytic prediction of the theory and exits non-zero if
the prediction fails, so the suite is self-checking.

---

## Full reproduction from scratch

See **[REPRODUCE.md](REPRODUCE.md)** for the step-by-step order (~2 hours CPU), the
claim-to-artefact map, and the expected outputs.

```bash
# experiments -> results/*.json
python code/experiments/run_method_map.py
python code/experiments/run_loss_ladder.py
python code/experiments/run_chamfer_pooled.py
python code/experiments/run_chamfer_supplement.py
python code/experiments/run_mnist_collapse.py

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

---

## Repository structure

```
.
├── README.md                this file
├── LICENSE                  MIT (code and released results)
├── CITATION.cff             citation metadata
├── REPRODUCE.md             full reproduction guide
├── requirements.txt         Python dependencies
│
├── code/
│   ├── experiments/         training runs      -> results/*.json
│   ├── analysis/            verification suite and figure generators -> logs/*.json, figures/
│   ├── src/                 shared library (discriminant, metrics, provenance, ...)
│   ├── tests/               regression guards
│   └── data/                mnist_16.npz (preprocessed MNIST cache; ships with the repo)
│
├── results/                 experiment output (5 JSON files; what the paper cites)
├── logs/                    verification output (7 JSON files)
└── figures/                 fig_impossibility.pdf, fig_loss_ladder.pdf
```

Figure PNGs and text run logs (`logs/*.log`) are not shipped; they are regenerated
locally.

---

## How the results are traced

No released number is hand-copied. Each `results/*.json` and `logs/*.json` is written by
the script named in its `provenance.generator` field, and carries:

- `result_schema_version` — the artefact schema,
- `provenance.protocol_id` and `provenance.source_sha256` — a SHA-256 fingerprint of the
  protocol files that produced it,
- for training runs, a `per_seed` block so every aggregate can be recomputed from the
  individual seeds.

Because the fingerprint covers the generating scripts, merging seeds produced by a
different code revision is rejected: all reported seeds come from one code revision.

To change a number, re-run the corresponding experiment or verification.

---

## Citation

If you use this code, please cite using the metadata in [`CITATION.cff`](CITATION.cff).

---

## License

- **Code and released results:** MIT — see [`LICENSE`](LICENSE).
- **Figures:** reuse with attribution.
