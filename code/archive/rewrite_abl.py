# -*- coding: utf-8 -*-
"""按补齐后的消融数据改写 §5.6 与附录 app:abl。

原稿的两处断言与实测相反（这部分此前从未真正跑过，是"纸面主张"）：
  * "Heun 在一档没什么用"        -> 实测 Heun@1 反而更差且 27.8% 样本落到支撑外
  * "小批量 OT 耦合不改变结论"    -> 实测 OT 把 1 步落差抹平（relgap -0.016）但把
                                    两步都搞差了，且 1 步过散布（spread 2.76）
  * "预算不影响 spread ratio"     -> 实测预算 1k->10k 让 spread1 从 0.196 掉到 0.025
                                    （拟合越好、塌缩越彻底，正是 Prop.1 要求的）

真正被数据验证的结论是：**只有动"耦合"（reflow / OT）才能把一步落差去掉**，
动容量、深度、预算、调度都动不了；这与 §3 的 coupling remark 完全一致。

正文只留 6 行小结，细节进附录（省页）。
用法: python rewrite_abl.py [--apply]
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


OLD_MAIN = r"""
We ablate every design choice on the fixed cell ($K=8$, $\text{sep}=6$, $\eta=0.5$);
numbers are in Tab.~\ref{tab:ablest} and the discussion is in Appendix~\ref{app:abl}.
Architecture, depth and a $4\times$ training budget change absolute quality but neither the
ordering nor the spread ratio, so the floor is not a capacity artefact; the high-noise
schedule of \citep{esser2024sd3,letitbesimple2026} improves one-step quality but does
\emph{not} lift the spread ratio above the floor; reflow raises the spread ratio, as the
coupling remark predicts, at a harder $t=0$ regression; Heun and mini-batch OT coupling
change little.
"""

NEW_MAIN = r"""
We ablate every design choice on the fixed cell ($K=8$, $\text{sep}=6$, $\eta=0.5$); full
numbers are in Tab.~\ref{tab:ablest} and the discussion in Appendix~\ref{app:abl}. The
pattern is sharper than we expected and it is the strongest structural evidence in the
paper: changing capacity, depth, training budget or time schedule leaves the one-step
\emph{ordering} intact and the one-step spread ratio far below $1$, whereas the two
changes that alter the \emph{coupling} --- reflow and mini-batch OT --- remove the
one-step gap entirely. That is exactly what the coupling remark predicts, and it is the
only intervention in the whole grid that does it.
"""

OLD_APP = r"""
\section{Ablation detail}
\label{app:abl}
Removed from Sec.~\ref{sec:abl} to respect the nine-page limit; numbers are in
Tab.~\ref{tab:ablest}.

(i) \emph{Architecture and budget.} Doubling or halving the MLP width, changing the depth,
or changing the training budget by $4\times$ alters the absolute quality but neither the
ordering nor the spread ratio --- the floor is not a capacity artefact.
(ii) \emph{Time-shift schedule.} The high-noise schedule of
\citep{esser2024sd3,letitbesimple2026} improves one-step quality but does \emph{not} lift
the spread ratio above the floor, consistent with Prop.~\ref{prop:floor}.
(iii) \emph{Reflow.} As the coupling remark predicts, reflow raises the one-step spread
ratio (it genuinely recovers spread) at the cost of a harder $t=0$ regression, and the two
effects nearly cancel in cSW.
(iv) \emph{Solver and coupling.} Heun buys little at $N=1$; mini-batch OT coupling leaves
the conclusions unchanged.
(v) \emph{Estimator.} The choice is calibrated against ground truth in Tab.~\ref{tab:calib}
and Fig.~\ref{fig:calib}, and swept over $k\in[5,100]$ in Tab.~\ref{tab:est}.
"""

NEW_APP = r"""
\section{Ablation detail}
\label{app:abl}
Removed from Sec.~\ref{sec:abl} to respect the nine-page limit; all numbers are in
Tab.~\ref{tab:ablest}, which reports the mean over two seeds for every configuration
(cell, architecture, budget and data are otherwise identical).

\textbf{What does not move the one-step gap.}
(i) \emph{Capacity and depth.} Halving or doubling the MLP width ($h64$--$h512$) or
changing the depth ($L3$, $L6$) leaves both the ordering (cSW@1 $>$ cSW@64 throughout) and
the one-step spread ratio in the same range ($0.051$--$0.095$): the floor is not a capacity
artefact.
(ii) \emph{Training budget.} A $4\times$ budget sweep ($1k$--$10k$ steps) keeps the
ordering but drives the one-step spread ratio \emph{down}, from $0.196$ to $0.025$. This is
a confirmation rather than a caveat: a better-fitted field realises \emph{less} conditional
variance at one step, which is precisely what Prop.~\ref{prop:floor} requires.
(iii) \emph{Time-shift schedule.} The result is schedule-dependent and we did not
anticipate it. A logit-normal (high-noise) schedule improves one-step cSW ($0.223$ against
$0.247$ for uniform) and lifts the one-step spread ratio to $0.175$; power-shaped schedules
of the SD3 family \citep{esser2024sd3,letitbesimple2026} do the opposite --- cSW@1 rises to
$0.258$ and the spread ratio falls to $0.03$. Neither crosses the floor.

\textbf{What does move it.} Only changing the coupling.
(iv) \emph{Reflow.} As the coupling remark predicts, reflow makes the one-step map depend
on $x_0$: the one-step spread ratio reaches $1.003$ (versus $0.051$ for the independent
coupling) and the one-step gap disappears (relative gap $0.009$ versus $1.602$), at
essentially no cost in absolute quality.
(v) \emph{Mini-batch OT coupling} closes the gap too ($-0.016$) but is not a free lunch: it
degrades \emph{both} samplers (cSW@1 $0.499$ versus $0.247$, cSW@64 $0.507$ versus $0.096$)
and over-disperses the one-step output (spread ratio $2.76$). Removing the floor is
therefore not the same as becoming accurate.
(vi) \emph{Solver.} Heun does not help at one step: at the same step count the one-step
cSW is worse ($0.290$ versus $0.247$) and $27.8\%$ of samples land off-manifold, against
$1.5\%$ for Euler. At this resolution the second-order correction overshoots, so the
one-step failure mode is not exhausted by the floor.
(vii) \emph{Estimator.} The discriminant's own choices are calibrated against the
closed-form value in Tab.~\ref{tab:calib} and Fig.~\ref{fig:calib}, and swept over
$k\in[5,100]$ in Tab.~\ref{tab:est}.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    t = io.open(MAIN, encoding="utf-8").read()
    n0 = len(t)
    t = rep(t, OLD_MAIN, NEW_MAIN, "abl_main")
    t = rep(t, OLD_APP, NEW_APP, "abl_app")
    print("字符 %d -> %d" % (n0, len(t)))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
    labels = set(re.findall(r"\\label\{([^}]+)\}", t))
    print("悬空引用:", sorted(refs - labels) or "none")
    if args.apply:
        io.open(MAIN, "w", encoding="utf-8").write(t)
        print(">>> 已写入")
    else:
        print("(干跑)")


if __name__ == "__main__":
    main()
