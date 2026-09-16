# -*- coding: utf-8 -*-
"""一次性迁移：把正文里对旧（含数字）宏名的引用改成新的纯字母名。

背景：LaTeX 的控制序列只能由字母组成，\\newcommand{\\e1Mean} 会让
"1" 脱离控制序列，pdflatex 直接报 "Missing \\begin{document}" 整篇失败。
make_tables.py 的 MACRO_ALIAS 已经把出口名换掉了，这个脚本负责把
**手写正文**里的旧引用一起搬过去，避免两边名字对不上。

只动 paper/*.tex 与 paper/ 下手工维护的 tex；tables/ 是生成物，重跑即可。

用法:
  python apply_macro_alias.py --check     # 只看会改什么，不写盘
  python apply_macro_alias.py             # 实际改写
"""
import argparse
import io
import os
import re
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
sys.path.insert(0, _HERE)
from make_tables import MACRO_ALIAS  # noqa: E402

PAPER = os.path.join(_ROOT_, "paper")

# 长的先替换：kappaD2R2 必须在 kappaD2 之前，否则会切成 kappaDimTwoR2
_PAIRS = sorted(MACRO_ALIAS.items(), key=lambda kv: -len(kv[0]))

TARGETS = ["main.tex", "macros.tex", "include.tex"]


def migrate(text):
    hits = []
    for old, new in _PAIRS:
        pat = re.compile(r"\\" + re.escape(old) + r"(?![A-Za-z0-9])")
        n = len(pat.findall(text))
        if n:
            hits.append((old, new, n))
            text = pat.sub("\\\\" + new, text)
    return text, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    total = 0
    for fn in TARGETS:
        p = os.path.join(PAPER, fn)
        if not os.path.isfile(p):
            continue
        with io.open(p, "r", encoding="utf-8") as f:
            s = f.read()
        new, hits = migrate(s)
        if not hits:
            print("%-12s no change" % fn)
            continue
        for old, nw, n in hits:
            print("%-12s \\%s -> \\%s  x%d" % (fn, old, nw, n))
            total += n
        if not args.check:
            with io.open(p, "w", encoding="utf-8") as f:
                f.write(new)
    print("total replacements: %d%s" % (total, " (dry run)" if args.check else ""))


if __name__ == "__main__":
    main()
