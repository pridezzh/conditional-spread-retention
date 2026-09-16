# -*- coding: utf-8 -*-
"""补齐消融实验里从未跑过的部分。

现状（查 logs/abl.log + results/ 确认）：
  - PART D 的 time_power3 只跑了 seed=0，time_logit 一次都没跑；
  - PART E（reflow）、F（solver）、G（coupling）完全没跑，
    但论文正文写了对它们的结论 —— 这是"纸面主张"，必须补数据。
本脚本顺序补齐，日志写 logs/abl2.log。
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """向上找到同时含 code/ 与 paper/ 的一级；与脚本自身深度无关。"""
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d


_ROOT_ = _find_root(_HERE)
_CODE_ = os.path.join(_ROOT_, "code")
ROOT = _ROOT_
LOG = os.path.join(ROOT, "logs", "abl2.log")
PY = sys.executable or r"E:\anaconda\python.exe"

JOBS = [
    ("D", [1], 6),        # time_uniform/power2/power3/logit 全量 seed=1
    ("E", [0, 1], 6),     # noreflow / reflow2RF
    ("F", [0, 1], 6),     # euler / heun
    ("G", [0, 1], 6),     # cpl_independent / cpl_ot
]

with open(LOG, "a", encoding="utf-8") as lg:
    lg.write("\n===== run_missing_abl start =====\n")
    for part, seeds, th in JOBS:
        cmd = [PY, "-u", os.path.join(_HERE, "run_ablations.py"),
               "--part", part, "--threads", str(th),
               "--seeds", *[str(s) for s in seeds]]
        lg.write("\n>>> %s\n" % " ".join(cmd))
        lg.flush()
        r = subprocess.run(cmd, cwd=_HERE, stdout=lg, stderr=subprocess.STDOUT)
        lg.write("<<< rc=%d\n" % r.returncode)
        lg.flush()
    lg.write("===== done =====\n")
print("done ->", LOG)
