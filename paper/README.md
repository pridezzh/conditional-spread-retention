# This directory is a placeholder

The manuscript is **not** distributed in this repository. Its source, its bibliography and
the ICLR 2027 style files are submitted separately (double-blind review), so this
directory intentionally holds no `.tex`, no `.bib` and no style files.

**Why it exists anyway.** Every script under `code/` locates the project root by walking
upwards to the first ancestor that contains a `code/` directory *together with* a `paper/`
directory. Keeping this directory makes that anchor resolve to the repository root here
exactly as it does in the authoring tree, so every command in `REPRODUCE.md` that does not
consume the manuscript resolves its paths identically. Without it, the walk reaches the
filesystem root and those scripts stop with `RuntimeError: project root not found`.

**What this means for reproduction.** The stages that consume the manuscript — emitting
`numbers_theory.tex`, `check_macros.py`, `build_paper.py`, and the page-limit check inside
`audit_submission.py` — cannot run here. Everything that produces `results/`, `logs/` and
`figures/` runs normally; see the scope section of `REPRODUCE.md`.

Do not copy the manuscript into this directory: it would break the double-blind
submission.
