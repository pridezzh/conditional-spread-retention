# -*- coding: utf-8 -*-
"""把 MNIST 阶梯每一档的两个统计量都列出来，判断图与正文/题注是否自洽。

两个量必须分清（早前踩过混淆的坑）：
  marginal  : sw@1 / sw@32          —— 边缘 sliced W2 之比（图 5 面板 b 画的就是这个）
  conditional: csw@1 / csw@32       —— 条件 sliced W2 之比（正文引用的 1.000 是这个）
"""
import os, json, glob
from collections import defaultdict

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
RES = os.path.join(_ROOT_, "results")

rows = []
for f in sorted(glob.glob(os.path.join(RES, "mnist_*.json"))):
    r = json.load(open(f, encoding="utf-8"))
    r["_file"] = os.path.basename(f)
    rows.append(r)

gm = defaultdict(list)
for r in rows:
    if "mode" in r:
        gm[(r["mode"], r.get("arch", "mlp"))].append(r)

MODES = ["full", "mid", "half", "weak", "none"]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


print("%-6s %-5s %3s | %8s %8s %8s | %8s %8s %8s | %8s %8s" % (
    "mode", "arch", "n",
    "sw_1", "sw_32", "marg", "csw_1", "csw_32", "cond", "D_hat", "recall1"))
print("-" * 104)
for m in MODES:
    for a in ("mlp", "conv"):
        rs = gm.get((m, a))
        if not rs:
            continue
        sw1 = [r["nfe"]["1"]["sw_marg"] for r in rs]
        sw32 = [r["nfe"]["32"]["sw_marg"] for r in rs]
        csw1 = [r["nfe"]["1"]["csw"] for r in rs]
        csw32 = [r["nfe"]["32"]["csw"] for r in rs]
        D = [r["disc"]["D_hat"] for r in rs]
        rec = [r["nfe"]["1"].get("recall", float("nan")) for r in rs]
        print("%-6s %-5s %3d | %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f | %8.3f %8.3f" % (
            m, a, len(rs), mean(sw1), mean(sw32), mean(sw1) / mean(sw32),
            mean(csw1), mean(csw32), mean(csw1) / mean(csw32),
            mean(D), mean(rec)))

print()
print("== 按架构汇总 ===")
for a in ("mlp", "conv"):
    Ds, marg, cond = [], [], []
    for m in MODES:
        rs = gm.get((m, a))
        if not rs:
            continue
        s1 = mean([r["nfe"]["1"]["sw_marg"] for r in rs])
        s32 = mean([r["nfe"]["32"]["sw_marg"] for r in rs])
        c1 = mean([r["nfe"]["1"]["csw"] for r in rs])
        c32 = mean([r["nfe"]["32"]["csw"] for r in rs])
        Ds.append(mean([r["disc"]["D_hat"] for r in rs]))
        marg.append(s1 / s32)
        cond.append(c1 / c32)
    print("%-5s  D̂ range [%.3f, %.3f]  marginal ratio [%.3f, %.3f]  conditional ratio [%.3f, %.3f]"
          % (a, min(Ds), max(Ds), min(marg), max(marg), min(cond), max(cond)))
