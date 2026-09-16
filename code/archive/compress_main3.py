# -*- coding: utf-8 -*-
"""第三轮瘦身：正文 9.4 页 -> <=9 页。

动作：
  1) fig:rl 移到附录（正文只留数字叙述 + 指向附录的表/图）；
  2) Ablations 正文压成 3 行小结，细节搬到附录 app:abl；
  3) "How to use this" 与 Limitations (1)(2) 收紧。

用法: python compress_main3.py [--apply]
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
ROOT = _ROOT_
MAIN = os.path.join(ROOT, "paper", "main.tex")

FLOAT_RE = re.compile(
    r"\\begin\{(?P<env>table\*?|figure\*?)\}(?P<body>.*?)\\label\{(?P<label>[^}]+)\}"
    r"(?P<tail>.*?)\\end\{(?P=env)\}\s*", re.S)


def cut_float(text, label):
    for m in FLOAT_RE.finditer(text):
        if m.group("label") == label:
            return text[: m.start()] + text[m.end():], m.group(0).strip() + "\n"
    raise SystemExit("找不到 float: %s" % label)


def _re_from(literal):
    parts = re.split(r"\s+", literal.strip())
    return r"\s+".join(re.escape(p) for p in parts)


def rep(text, old, new, label, span=False):
    rx = _re_from(old)
    ms = list(re.finditer(rx, text))
    if not ms:
        raise SystemExit("[%s] 未匹配到" % label)
    if len(ms) > 1 and not span:
        raise SystemExit("[%s] 多处匹配(%d)" % (label, len(ms)))
    m = ms[0]
    return text[: m.start()] + new + text[m.end():]


OLD_ABL = r"""
We ablate every design choice, on the fixed cell ($K=8$, $\text{sep}=6$, $\eta=0.5$) unless
stated otherwise; all numbers are in Tab.~\ref{tab:ablest}. (i) \emph{Architecture and
budget}: doubling or halving the MLP width, changing the depth, or changing the training
budget by $4\times$ alters the absolute quality but neither the ordering nor the spread
ratio --- the floor is not a capacity artefact. (ii) \emph{Time-shift schedule}: the
high-noise schedule of \citep{esser2024sd3,letitbesimple2026} improves one-step quality but
does \emph{not} lift the spread ratio above the floor, consistent with
Prop.~\ref{prop:floor}. (iii) \emph{Reflow}: as the coupling remark predicts, reflow raises
the one-step spread ratio (it genuinely recovers spread) at the cost of a harder $t=0$
regression, and the two effects nearly cancel in cSW. (iv) \emph{Solver and coupling}: Heun
buys little at $N=1$; mini-batch OT coupling leaves the conclusions unchanged. The estimator
choice is calibrated against ground truth in Tab.~\ref{tab:calib} and Fig.~\ref{fig:calib},
and swept over $k\in[5,100]$ in Tab.~\ref{tab:est}.
"""
NEW_ABL = r"""
We ablate every design choice on the fixed cell ($K=8$, $\text{sep}=6$, $\eta=0.5$);
numbers are in Tab.~\ref{tab:ablest} and the discussion is in Appendix~\ref{app:abl}.
Architecture, depth and a $4\times$ training budget change absolute quality but neither the
ordering nor the spread ratio, so the floor is not a capacity artefact; the high-noise
schedule of \citep{esser2024sd3,letitbesimple2026} improves one-step quality but does
\emph{not} lift the spread ratio above the floor; reflow raises the spread ratio, as the
coupling remark predicts, at a harder $t=0$ regression; Heun and mini-batch OT coupling
change little.
"""

PLACEHOLDER_ABL_APP = r"""
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

OLD_HOWTO = r"""
\paragraph{How to use this.} Before committing to a one-step policy, compute $\cD$
and $\cL$ on the training set (a few seconds at $n=2\times10^4$ with the pure-numpy
implementation released with this paper), calibrate the null by parametric bootstrap, and
consult the rule (\ref{eq:rule}). If $\cD$ is large and $\cL$ exceeds the null, expect
\emph{structural} failure --- the one-step sample lands between modes, where the data does
not live --- and either increase the sampling budget or change the coupling
(reflow / a non-Gaussian source); if $\cD$ is large but $\cL$ is at the null, expect
blurring only, which may be acceptable for low-precision control but not for
contact-rich manipulation.
"""
NEW_HOWTO = r"""
\paragraph{How to use this.} Before committing to a one-step policy, compute $\cD$ and
$\cL$ on the training set (seconds at $n=2\times10^4$ with the released pure-numpy
implementation), calibrate the null by parametric bootstrap, and consult Eq.~(\ref{eq:rule}):
large $\cD$ with $\cL$ above the null means \emph{structural} failure --- raise the sampling
budget or change the coupling --- whereas large $\cD$ with $\cL$ at the null means blurring
only, acceptable for low-precision control but not for contact-rich manipulation.
"""

OLD_LIM1 = r"""
(1) \emph{One transfer attempt failed, and we know why.} On the MNIST ladder
$\cD$ moves from \mnistDmin{} to \mnistDmax{} while the one-step penalty stays flat
(\mnistCswOne{} versus \mnistCswRef{} in conditional sliced $W_2$); the training loss
\mnistLoss{} shows the field is not fitted, so the ladder cannot test the law
(Sec.~\ref{sec:images}). A $256$-dimensional target conditioned on $\le74$ dimensions
is too large for a CPU-only budget. This is a limitation of our evidence, not a
refutation --- the proposition says nothing about an unconverged field --- but it does
mean the image family does not count as support for the diagnostic.
"""
NEW_LIM1 = r"""
(1) \emph{One transfer attempt failed, and we know why.} On the MNIST ladder $\cD$ moves
from \mnistDmin{} to \mnistDmax{} while the one-step penalty stays flat (\mnistCswOne{}
versus \mnistCswRef{}); the training loss \mnistLoss{} shows the field is not fitted, so
the ladder cannot test the law (Sec.~\ref{sec:images}). This is a limitation of our
evidence, not a refutation --- the proposition says nothing about an unconverged field ---
but the image family does not count as support for the diagnostic.
"""

OLD_LIM2 = r"""
(2) \emph{The structural floor is only part of the observed penalty.}
$\cD$ bounds the one-step error from below; the observed gap also contains an
optimisation term, which the control grid exhibits directly: one step is worse in
\rlNAllWorse{} of \rlNCells{} settings, yet the gap \emph{decreases} with $\cD$
($\rho=\rhoDRl$), and a residual gap of \rlExactGap{} persists at \texttt{exact}
conditioning where $\cD=\rlExactD$ (Sec.~\ref{sec:rl}). A complete account needs a
capacity/budget term, which we do not have; this is the clearest open problem raised by
this paper.
"""
NEW_LIM2 = r"""
(2) \emph{The structural floor is only part of the observed penalty.} $\cD$ bounds the
one-step error from below; the observed gap also contains an optimisation term: one step is
worse in \rlNAllWorse{} of \rlNCells{} settings, yet the gap \emph{decreases} with $\cD$
($\rho=\rhoDRl$) and a residual gap of \rlExactGap{} persists at \texttt{exact}
conditioning where $\cD=\rlExactD$ (Sec.~\ref{sec:rl}). A complete account needs a
capacity/budget term; this is the clearest open problem raised by this paper.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t = io.open(MAIN, "r", encoding="utf-8").read()
    n0 = len(t)

    t, f_rl = cut_float(t, "fig:rl")
    t = rep(t, OLD_ABL, NEW_ABL, "abl")
    t = rep(t, OLD_HOWTO, NEW_HOWTO, "howto")
    t = rep(t, OLD_LIM1, NEW_LIM1, "lim1")
    t = rep(t, OLD_LIM2, NEW_LIM2, "lim2")

    k = t.rfind(r"\end{document}")
    t = t[:k] + PLACEHOLDER_ABL_APP + "\n" + f_rl + "\n" + t[k:]

    print("字符 %d -> %d (减 %d)" % (n0, len(t), n0 - len(t)))
    for lab in ("fig:rl", "tab:ablest", "fig:calib", "tab:calib", "tab:est"):
        print("  %-12s x%d" % (lab, t.count(r"\label{%s}" % lab)))
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
