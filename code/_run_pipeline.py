# -*- coding: utf-8 -*-
"""Rebuild numeric macros, figures, and the paper from existing results/ and logs/.

Order (from 2026-09-16, the new pipeline after the third rewrite):
    make_theory_macros -> make_fig_impossibility -> make_fig_ladder
    -> check_macros -> build_paper

**This script does NOT train models or rerun experiments**, so it cannot by itself
be called "full reproduction"; it only assembles the already-produced results/*.json
and logs/*.json into the manuscript. To run experiments, use the scripts under
code/experiments/; to run verification, use the verify_*/validate_* scripts under
code/analysis/.

Why this differs from the old version: the old pipeline was
`make_tables -> make_figures -> make_vis -> check_macros -> build_paper`,
whose output `paper/numbers.tex` (162 macros) and `tables/`, `figures/fig1..fig6`
are **no longer referenced by the body text** after the third rewrite (measured
reference count = 0). Keeping the old pipeline around would blur "which is the real
pipeline", so the steps are reordered by the new dependencies. The old scripts
remain in `code/analysis/` and `code/archive/` as history.

Uses the current interpreter (sys.executable) by default; can be overridden with
the PIPELINE_PY environment variable. Logs are written to logs/pipeline.log at the
project root.
"""
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """Walk up to the top level that contains both code/ and paper/; independent of the script's own depth."""
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d


ROOT = _find_root(_HERE)
LOGDIR = os.path.join(ROOT, "logs")
os.makedirs(LOGDIR, exist_ok=True)

PY = os.environ.get("PIPELINE_PY", sys.executable)

# (step name, [script path relative to code/, extra args...])
STEPS = [
    ("make_theory_macros", ["analysis/make_theory_macros.py"]),
    ("make_fig_impossibility", ["analysis/make_fig_impossibility.py"]),
    ("make_fig_ladder", ["analysis/make_fig_ladder.py"]),
    ("check_macros", ["analysis/check_macros.py", "--strict"]),
    ("build_paper", ["analysis/build_paper.py", "--clean"]),
]

logpath = os.path.join(LOGDIR, "pipeline.log")
with open(logpath, "w", encoding="utf-8") as lf:
    def w(s):
        print(s)
        lf.write(s + "\n")
        lf.flush()

    overall_ok = True
    for name, args in STEPS:
        w("\n" + "=" * 70)
        w("STEP %s  (%s)" % (name, " ".join(args)))
        w("=" * 70)
        t0 = time.time()
        env = dict(os.environ)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONIOENCODING"] = "utf-8"
        script = os.path.join(_HERE, args[0])
        r = subprocess.run([PY, "-u", script] + args[1:],
                           capture_output=True, cwd=_HERE, env=env)
        out = r.stdout.decode("utf-8", errors="replace")
        err = r.stderr.decode("utf-8", errors="replace")
        w(out.rstrip())
        if err.strip():
            w("--- stderr ---")
            w(err.rstrip())
        dt = time.time() - t0
        w("-> rc=%d  (%.1fs)" % (r.returncode, dt))
        if r.returncode != 0:
            overall_ok = False
            w("!! STEP FAILED: %s" % name)
            break

    w("\n" + "#" * 70)
    w("PIPELINE %s" % ("OK" if overall_ok else "FAILED"))
    w("#" * 70)

sys.exit(0 if overall_ok else 1)
