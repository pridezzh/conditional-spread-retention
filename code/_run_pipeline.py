# -*- coding: utf-8 -*-
"""从已有 results/ 与 logs/ 重建数值宏、图与论文。

顺序（2026-09-16 起，第三轮重写后的新流水线）：
    make_theory_macros -> make_fig_impossibility -> make_fig_ladder
    -> check_macros -> build_paper

**本脚本不会训练模型或重跑实验**，因而不能单独称为"完整复现"；
它只把已经跑出来的 results/*.json 与 logs/*.json 装配成稿件。
跑实验请用 code/experiments/ 下的脚本，跑验证请用 code/analysis/ 下
verify_*/validate_* 脚本。

为什么和旧版不同：旧流水线是
`make_tables -> make_figures -> make_vis -> check_macros -> build_paper`，
产出的 `paper/numbers.tex`（162 个宏）与 `tables/`、`figures/fig1..fig6`
在第三轮重写后**已不被正文引用**（实测引用数 = 0）。保留旧流水线会让
"哪个才是真流水线"变模糊，故按新依赖重排。旧脚本仍留在 `code/analysis/`
与 `code/archive/` 作为历史。

默认用当前解释器（sys.executable）；可用环境变量 PIPELINE_PY 覆盖。
日志写到项目根 logs/pipeline.log。
"""
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """向上找到同时含 code/ 与 paper/ 的一级；与脚本自身深度无关。"""
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d


ROOT = _find_root(_HERE)
LOGDIR = os.path.join(ROOT, "logs")
os.makedirs(LOGDIR, exist_ok=True)

PY = os.environ.get("PIPELINE_PY", sys.executable)

# (步骤名, [相对 code/ 的脚本路径, 额外参数...])
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
