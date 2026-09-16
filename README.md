# Whether One Step Is Enough Is Not Determined by the Data

**Coupling, Loss Form, and an Impossibility Theorem**

Code, raw results, and the compiled manuscript for an ICLR 2027 submission
(double-blind; author information withheld).

---

## The claim in one paragraph

Whether a one-step generative policy collapses is **not** a property of the data
distribution, and not a property of the architecture or the sampler. It is a property
of the **training target** — the coupling between the source noise `x_0` and the target
`x_1`, together with the **form of the loss** used to fit it. Independent coupling plus
a pointwise `L^2` loss collapses; changing **either one** escapes. We make this sharp
with three results:

1. **The L²-optimal one-step map is the conditional mean** `f*(x0,c) = E[x1|x0,c]`, so
   the *realised conditional-spread ratio*

   ```
   rho(f) = E_c[ tr Var_{x0}( f(x0,c) | c ) ] / E_c[ tr Var(x1 | c) ]  in [0,1]
   ```

   is a functional of the coupling. `rho = 0` under an independent coupling,
   `rho = 1` under a deterministic one.

2. **An impossibility theorem.** There is a one-parameter family of training targets
   whose induced `(c, x1)` distribution is *identical sample by sample*, yet for which
   the optimal `rho*(alpha) = alpha^2` sweeps all of `[0,1]`. Consequently
   `inf_A sup_alpha E|A - rho*(alpha)| = 1/2` over **all** statistics of the data —
   randomised, and even given the population law exactly — attained by the constant
   `1/2`. The value does not depend on sample size: `n = infinity` does not help.

3. **The positive counterpart, and a ladder.** `rho` is estimable from the paired
   samples a training minibatch already contains, before any one-step model exists
   (analytic `1/k` bias law: `rho_hat(k) ~ rho* + (1-rho*)/k`). The second escape
   channel has a structure that turns on the **scope** of the matching: a pointwise
   `L^2` loss collapses; a collection loss that matches each target against the whole
   batch recovers the **support** but flattens the conditional weights; matching
   **within each condition** restores the spread, and a *balanced* within-condition
   assignment comes closest on the conditional sliced `W_2`. The variable that does the
   flattening is the scope of the `arg min`, not the absence of a bijection — a
   pre-registered prediction that failed, which we report as such.

We also report a negative result about our own earlier proposal: a shape statistic we
had advanced as a modality detector sorts a single-mode banana **above** an eight-mode
ring, and is demoted to a descriptive index.

---

## Repository layout

```
.
├── README.md                this file
├── LICENSE                  MIT (code); manuscript text shared with attribution
├── CITATION.cff
├── requirements.txt         numpy / scipy / torch(cpu) / matplotlib
├── REPRODUCE.md             step-by-step reproduction, with expected outputs
│
├── paper/
│   ├── main.tex             manuscript source (ICLR 2027 style, single column, <=9pp main text)
│   ├── numbers_theory.tex   GENERATED: every number in the paper is a macro here
│   ├── references.bib       bibliography
│   ├── iclr2027/            official style files
│   ├── main.pdf             compiled manuscript
│   └── fetch_refs.py        regenerates refs_raw.json from arXiv metadata
│       make_bib.py          refs_raw.json -> references.bib
│
├── code/
│   ├── _run_pipeline.py     single entry point: macros -> figures -> checks -> PDF
│   ├── experiments/         training runs  -> results/*.json
│   ├── analysis/            verifications, macro/figure generation, compile checks
│   ├── src/                 shared library (deficit.py, discriminant.py, metrics.py, ...)
│   ├── tests/               regression guards
│   └── archive/             superseded protocols and one-off forensic scripts (not on the repro path)
│
├── results/                 raw experiment output (5 files; these are what the paper cites)
├── logs/                    verification reports (verify_*.json, validate_deficit.json, ...)
└── figures/                 fig_impossibility.{pdf,png}, fig_loss_ladder.{pdf,png}
```

---

## Quick start: rebuild the PDF from the shipped results

This does **not** train anything. It regenerates the number macros, both figures, runs
the macro/format checks, and compiles the paper.

```bash
python -m pip install -r requirements.txt

python code/_run_pipeline.py
```

Expected tail of the output (`logs/pipeline.log`):

```
  pdflatex #1  rc=0
  bibtex       rc=0
  pdflatex #2  rc=0
  pdflatex #3  rc=0

hard errors    : 0 (last pass)
undefined refs : none
undefined cites: none
overfull boxes : 0 (last pass)
total pages    : 14  (incl. references and appendix; main text ends on p.9)
verdict        : OK
PIPELINE OK
```

Requires a TeX distribution (`pdflatex`, `bibtex`) on `PATH`. The path to the TeX
binaries can be edited at the top of `code/analysis/build_paper.py`.

> **Windows note.** If you see `OMP: Error #15 ... libiomp5md.dll already initialized`,
> the process aborts with exit code 3 — and note the abort happens **mid-training**, not
> at startup. `code/_run_pipeline.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` for its child
> processes, and each `code/experiments/*.py` script sets it for itself before importing
> numpy/torch, so running those scripts directly is safe. If you import them from your
> own entry point, set the variable before importing numpy or torch.

---

## Full reproduction from scratch

Every number in the paper comes from these runs. Outputs land in `results/` and `logs/`.

```bash
# ---- experiments (CPU; ~75 + 5 + 20 + 7 minutes) ----
python code/experiments/run_method_map.py           # Table 1
python code/experiments/run_chamfer_supplement.py   # Table 1, pooled-Chamfer row
python code/experiments/run_chamfer_pooled.py       # Table 2, L2a row (pooled scope isolated)
python code/experiments/run_loss_ladder.py          # Table 2 (L1/L2b/L3) + Figure 2

# ---- verifications ----
python code/analysis/verify_multistep_theory.py     # Figure 1(a) context, rho_N sweep
python code/analysis/verify_conditional_theory.py   # same-D / different-coupling control
python code/analysis/verify_coupling_theory.py      # earliest coupling control (supporting)
python code/analysis/verify_impossibility.py        # the alpha family
python code/analysis/verify_minimax.py              # the seven flat data-side statistics
python code/analysis/validate_deficit.py            # the Lambda negative result
python code/analysis/run_falsification.py           # pre-registered false-positive stress test

# ---- rebuild and compile ----
python code/_run_pipeline.py

# ---- regression guards ----
python -m unittest discover -s code/tests -v
```

Roughly 2 hours of CPU wall-clock in total on a 20-core workstation.

---

## Before you submit

Two things are deliberately left to the authors, because they are decisions rather than
artefacts:

1. **Anonymity.** The manuscript is double-blind (`Anonymous Author(s)`, no
   `\iclrfinalcopy`). If you attach an anonymised repository, add its link to the
   reproducibility statement in `paper/main.tex`; the repository as shipped contains no
   author-identifying information.
2. **The AI use statement.** The text in `paper/main.tex` records that generative AI
   was used for hypothesis development, literature search, mathematical critique, code
   implementation and audit, and drafting. Confirm that this matches your institution's
   policy and your actual usage before submitting.

Known open items, all stated in the paper itself and none of them affecting the
theorems: the impossibility holds for an explicit family and bounds the worst case
rather than the typical case; there is no D4RL / OGBench / LIBERO / VLA validation; and
balanced assignment costs a Hungarian solve per batch per condition, whose scaling is
not studied here. Also note the scope limit the paper states plainly: `rho` measures
conditional spread, not task reward.

---

## Results at a glance

| What | Measured |
|---|---|
| Multi-step Euler escapes collapse (exact velocity field, no training error) | `rho_N`: 0.0000 → 0.5907 → 0.8573 → 0.9233 → 0.9612 → 0.9815 → 0.9920 → 0.9974 → **1.0001** for `N = 1..256`; endpoint identity error `8.9e-16` |
| **Same data, different coupling** (the hardest control) | `D = 0.9116` in *both* arms (identical by construction); one-step `rho_hat` 0.0272 vs 0.8776 → **32.2×**; mode coverage 0/8 vs 8/8 |
| Impossibility, constructive form | `rho_hat` 0.0501 → 0.1054 → 0.2854 → 0.5747 → 0.9766 while `D` moves by `0.0030` and `Lambda` by `0.0003`; MAD against `alpha^2 + (1-alpha)^2/k` is `0.0141` |
| Impossibility, minimax form | constant-predictor baseline `0.5000`; **all seven** data-side statistics pass a one-way ANOVA over `alpha` (`p >= 0.22`); paired estimator MAD `0.0141` → ratio **35.6×**; `|Delta| < 0.01` for every `[0,1]`-scaled statistic up to `n = 16000` |
| Method map (8 families × NFE ∈ {1,32} × **5 seeds**) | independent coupling at NFE=1: `rho = 0.0067`; OT coupling `0.8949`; reflow `0.9631`; distillation `0.9654`; the same field at NFE=32 recovers `0.9646` |
| min-of-M / IMLE at one step (corrected) | `rho = 0.8165 ± 0.0212`, mode coverage **8/8** — it *does* escape, once its `M` candidates come from independent source noise. An earlier "it does not escape" was an implementation artefact (repeating one draw makes the objective literally pointwise `L^2`; the two code paths agreed to 17 digits, `0.00010386879583898434`) |
| Loss ladder (same independent coupling, only the loss changes; **5 seeds**) | L1 pointwise `L^2`: `rho = 0.0001` (collapse). L2a unbalanced Chamfer **pooled over the batch**: aggregate `rho = 0.9474` — *indistinguishable* from L2b's `0.9720`, yet the conditional spread is nearly flat across conditions (CV `0.12` vs true `0.398`), spread error `\|rho_j-1\| = 0.433`, dispersion of `rho_j` `0.508`, conditional sliced `W_2 = 0.397`. L2b same objective matched **within each condition**: `rho = 0.9720`, spread error `0.027`, dispersion `0.019`, cSW `0.129`. L3 balanced within-condition: `rho = 0.8991`, spread error `0.097`, cSW `0.079`. Pre-registered verdict **PARTIAL** — the check "balancedness reduces weight error" is FALSE (L2b beats L3 at `0.027` vs `0.097`, separated in 5/5 seeds); what flattens the spread is the **scope** (pooling), not the missing bijection. The aggregate `rho` cannot see this: only the conditional statistics can |
| Negative result | Gaussian deficit sorts the 8-mode ring (`0.0900`) **below** the single-mode banana (`0.2257`) |

---

## How numbers are traced

The paper contains **no hand-copied statistics**. Every value is a LaTeX macro in
`paper/numbers_theory.tex`, generated by `code/analysis/make_theory_macros.py`, which
reads `results/*.json` and `logs/*.json`. Each macro line ends with a comment naming
the JSON file and field it came from, e.g.

```latex
\newcommand{\ImpRhoZero}{0.05006}  % verify_impossibility: curve.rho_hat[0] (alpha=0.00)
```

To change a number: re-run the experiment or verification, then re-run
`code/_run_pipeline.py`. Never edit `numbers_theory.tex` by hand — it is overwritten.

`code/analysis/check_macros.py --strict` asserts that every macro referenced in
`main.tex` exists and that no macro name contains a digit (LaTeX control sequences may
contain letters only).

---

## Compute

Everything is **CPU-only** (a 20-core workstation, PyTorch CPU build). No GPU was used
at any point. All synthetic families have closed-form or exactly controlled ground
truth, and **no external dataset is used** — the paper's scope is deliberately
CPU-reproducible with zero downloads.

---

## Scope and known limitations

Stated in the paper, repeated here so the code is not over-read:

- The impossibility is proved for an **explicit family**. The family is not contrived
  (`alpha = 0` and `alpha = 1` are the two couplings actually in use), but a
  practitioner's training target is not literally an `alpha`-mixture, and the theorem
  bounds the **worst case**, not the typical case.
- Measurements are on controlled CPU-scale **synthetic** families. We have **not**
  validated on D4RL, OGBench, LIBERO, or a VLA.
- `rho` measures conditional spread, not task reward: a policy can have the right
  spread and still be wrong.
- Balanced assignment costs a Hungarian solve per batch per condition; its scaling is
  not studied here.
- `rho` is estimated by `k`NN regression with an analytic `1/k` bias. All claims are
  about the population quantity `rho*`; the estimator's bias is reported rather than
  corrected away.
- `code/src/` contains a few modules (`common.py`, `tasks.py`, `flows.py`) that are
  used only by `code/archive/`. They are kept for completeness; the reproduction path
  above does not need them.

## License

Code: MIT (see `LICENSE`). Manuscript text and figures: reuse with attribution.
