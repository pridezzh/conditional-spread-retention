# -*- coding: utf-8 -*-
"""两处"最"字过头的地方，按 Tab.~\\ref{tab:predictors} 的实际数字收回来。

实测（tables/tab_predictors.csv）：
  vs e_1  : ours 0.88 | raw kNN 0.89 --- 原始 kNN 其实略高（CI 重叠）
  vs off@1: null-corrected Λ 0.81 | raw Λ 0.83 --- 原始 Λ 略高（CI 重叠）
所以"$\cD$ 是最好的单变量预测子""原始 Λ 排名更差"这两句都只能对
**其中一部分目标**成立，必须写明是对哪个目标，否则是过度宣称。

用法: python rewrite_claims.py [--apply]
"""
import argparse
import io
import os
import re

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
MAIN = os.path.join(_ROOT_, "paper", "main.tex")


def _re_from(lit):
    return r"\s+".join(re.escape(p) for p in re.split(r"\s+", lit.strip()))


def rep(text, old, new, label):
    rx = _re_from(old)
    ms = list(re.finditer(rx, text))
    if not ms:
        raise SystemExit("[%s] 未匹配到" % label)
    if len(ms) > 1:
        raise SystemExit("[%s] 多处匹配 %d" % (label, len(ms)))
    m = ms[0]
    return text[: m.start()] + new + text[m.end():]


OLD = r"""
$\cD$ is the best single predictor of $e_1$ ($\rho=\rhoDGap$) and the null-corrected $\cL$
the best predictor of structural failure ($\rho=\rhoLDrop$); their product reaches AUC
$\aucDL$ for the label ``off@1$>\offThresh$'' (\nFail{} of \nCell{} cells), against
$\aucLin$ for linear $R^2$, $\aucD$ for $\cD$ alone and $\aucLam$ for $\cL$ alone. Marginal
cluster separation, which ignores the conditioning entirely, is near-useless --- the
conditioning, not the shape of the marginal, is what matters --- and the raw $\hat\Lambda$
ranks worse than the null-corrected $\cL$, so the parametric null does real work rather than
being bookkeeping.
"""

NEW = r"""
$\cD$ is the strongest predictor of $e_1$ among the natural competitors
($\rho=\rhoDGap$; its own uncalibrated variant ties within noise, $0.89$ vs.\ $0.88$) and
the null-corrected $\cL$ the strongest predictor of structural failure ($\rho=\rhoLDrop$);
their product reaches AUC $\aucDL$ for the label ``off@1$>\offThresh$'' (\nFail{} of
\nCell{} cells), against $\aucLin$ for linear $R^2$, $\aucD$ for $\cD$ alone and $\aucLam$
for $\cL$ alone. Marginal cluster separation, which ignores the conditioning entirely, is
near-useless --- the conditioning, not the shape of the marginal, is what matters. The null
correction helps against $e_1$ ($0.25\!\to\!0.36$); against off@1 raw and corrected are
within noise ($0.83$ vs.\ $0.81$), so we claim it for the former, not the latter.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    t = io.open(MAIN, encoding="utf-8").read()
    n0 = len(t)
    t = rep(t, OLD, NEW, "predictors")
    print("字符 %d -> %d (%+d)" % (n0, len(t), len(t) - n0))
    if args.apply:
        io.open(MAIN, "w", encoding="utf-8").write(t)
        print(">>> 已写入")
    else:
        print("(干跑)")


if __name__ == "__main__":
    main()
