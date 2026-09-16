# -*- coding: utf-8 -*-
"""一次性目录整理脚本：把平铺的 code/ 分成 experiments / analysis / archive 三层。

背景与做法
----------
原来 code/ 下有 26 个 .py 平铺，另有 5 个**完全为空**的遗留目录
（configs/figures/logs/results/tables —— 真正的产物在项目根，不在 code/ 下）。

每个脚本原先都靠 `_HERE = dirname(abspath(__file__))` + `dirname(_HERE)` 找项目根，
这意味着它们**必须**正好待在 code/ 一层。移动后深度 +1，锚点会指错。

因此本脚本做两件事：
  1) 按用途把脚本移进子目录；
  2) 把每个被移动脚本的路径锚点换成**自愈式**写法——向上找到同时含
     `code/` 与 `paper/` 的那一层作为项目根，从此与脚本自身深度无关。

用法：python _reorg_code_tree.py [--check]（**一次性**，已执行完毕，仅留档）
      --check 只报告将要做的改动，不落盘。
"""
import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d


ROOT = _find_root(_HERE)
CODE = os.path.join(ROOT, "code")

MOVES = {
    "experiments": [          # 跑实验：results/*.json
        "run_toy.py", "run_mnist.py", "run_rl.py",
        "run_ablations.py", "run_missing_abl.py",
    ],
    "analysis": [             # results -> tables / figures / numbers.tex / PDF
        "make_tables.py", "make_figures.py", "make_vis.py",
        "check_macros.py", "build_paper.py",
        "audit_format.py", "_final_check.py",
    ],
    "archive": [              # 一次性改写脚本与取证脚本，已完成使命
        "apply_macro_alias.py", "compress_main.py", "compress_main2.py",
        "compress_main3.py", "rewrite_abl.py", "rewrite_claims.py",
        "rewrite_mnist_conv.py", "_probe_mnist.py",
    ],
}

# 要删除的空遗留目录（真正的产物在项目根：results/ tables/ figures/ logs/）
DROP_DIRS = ["configs", "figures", "logs", "results", "tables", "__pycache__"]

# 项目根锚点：自愈式。放在每个被移动脚本的 import os 之后。
ANCHOR = '''_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """向上找到同时含 code/ 与 paper/ 的一级；与脚本自身深度无关。"""
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d


_ROOT_ = _find_root(_HERE)
_CODE_ = os.path.join(_ROOT_, "code")
'''

# 旧写法 -> 新写法。顺序敏感：长串先替换。
RULES = [
    # --- 项目根 ---
    ("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", "_ROOT_"),
    ("os.path.dirname(_HERE)", "_ROOT_"),
    ("os.path.dirname(CODE)", "_ROOT_"),
    ("os.path.dirname(HERE)", "_ROOT_"),
    # --- 库路径：src 永远在 code/src ---
    ('os.path.join(_HERE, "src")', 'os.path.join(_CODE_, "src")'),
    # --- 旧锚点定义式（整行）会被 ANCHOR 覆盖，这里兜底 ---
    ("_HERE = os.path.dirname(os.path.abspath(__file__))", "_HERE = os.path.dirname(os.path.abspath(__file__))"),
]


def patch(text):
    """把锚点定义行换成 ANCHOR 块，并把其余旧路径写法换成新的。"""
    lines = text.splitlines(True)
    out, i = [], 0
    anchor_done = False
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        # 吃掉旧的锚点定义行（含紧随其后的 sys.path.insert(...,"src") 之前的位置）
        if (s.startswith("_HERE = os.path.dirname(os.path.abspath(__file__))")
                or s.startswith("HERE = os.path.dirname(os.path.abspath(__file__))")
                or s.startswith("CODE = os.path.dirname(os.path.abspath(__file__))")):
            if not anchor_done:
                out.append(ANCHOR)
                anchor_done = True
                # apply_macro_alias.py 里紧跟一行 sys.path.insert(0, _HERE)，保留
            i += 1
            continue
        if (s.startswith("ROOT = os.path.dirname(_HERE)")
                or s.startswith("_ROOT = os.path.dirname(_HERE)")
                or s.startswith("ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))")
                or s.startswith("RES = os.path.join(os.path.dirname(CODE), \"results\")")):
            if s.startswith("RES ="):
                out.append('RES = os.path.join(_ROOT_, "results")\n')
            elif s.startswith("_ROOT"):
                out.append("_ROOT = _ROOT_\n")
            else:
                out.append("ROOT = _ROOT_\n")
            i += 1
            continue
        # 逐行套用剩余规则
        for a, b in RULES:
            if a != b:
                ln = ln.replace(a, b)
        out.append(ln)
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告，不落盘")
    args = ap.parse_args()

    plan = []
    for sub, files in MOVES.items():
        for f in files:
            src = os.path.join(CODE, f)
            if not os.path.exists(src):
                plan.append(("MISS", src, ""))
                continue
            plan.append(("MOVE", src, os.path.join(CODE, sub, f)))
    for d in DROP_DIRS:
        p = os.path.join(CODE, d)
        if os.path.isdir(p):
            n = sum(len(x) for _, _, x in os.walk(p))
            plan.append(("DROP" if n == 0 else "SKIP", p, "files=%d" % n))

    for act, a, b in plan:
        print("%-5s %s%s" % (act, os.path.relpath(a, ROOT), ("  ->  " + os.path.relpath(b, ROOT)) if b else ""))

    if args.check:
        print("\n[--check] 未做任何改动。")
        return 0

    # 1) 建目录
    for sub in MOVES:
        os.makedirs(os.path.join(CODE, sub), exist_ok=True)

    # 2) 移动 + 打补丁
    n_patched = 0
    for sub, files in MOVES.items():
        for f in files:
            src = os.path.join(CODE, f)
            dst = os.path.join(CODE, sub, f)
            if not os.path.exists(src):
                print("  !! 缺失，跳过：%s" % f)
                continue
            txt = open(src, encoding="utf-8", errors="replace").read()
            new = patch(txt)
            shutil.move(src, dst)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(new)
            n_patched += 1
            print("  moved+patched: %-28s -> code/%s/" % (f, sub))

    # 3) 删空目录
    for d in DROP_DIRS:
        p = os.path.join(CODE, d)
        if not os.path.isdir(p):
            continue
        if d == "__pycache__":          # 可再生的编译缓存，直接删
            shutil.rmtree(p)
            print("  removed cache: code/%s" % d)
            continue
        n = sum(len(x) for _, _, x in os.walk(p))
        if n == 0:
            shutil.rmtree(p)
            print("  removed empty dir: code/%s" % d)
        else:
            print("  !! 非空，保留：code/%s (files=%d)" % (d, n))

    print("\n完成：移动并打补丁 %d 个脚本。" % n_patched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
