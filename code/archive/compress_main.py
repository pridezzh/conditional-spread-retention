# -*- coding: utf-8 -*-
"""把正文压到 ICLR 9 页以内（参考文献与附录不计页数）。

策略（都是可逆的文本搬移，不改任何数值）：
  1) 把四个大 float 搬到附录 "Additional figures and tables"：
       tab:toy（整张 26 格网格表，7KB）、tab:calib、tab:mnist、fig:mnist
     正文只保留 fig1(示意)、fig2(主结果)、fig:rl(负迁移图)。
  2) 把 MNIST / 离线控制两节的正文重写得紧凑（负结果不再和 Limitations 重复叙述）。
  3) Limitations (1)(2) 去重。

用法:
  python compress_main.py            # 干跑，只报告会改动什么
  python compress_main.py --apply     # 真正写入
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


# ----------------------------------------------------------------------------
# 1) 从正文里摘掉某个 float（按 \label 定位到外层 table/figure 环境）
# ----------------------------------------------------------------------------
FLOAT_RE = re.compile(
    r"\\begin\{(?P<env>table\*?|figure\*?)\}(?P<body>.*?)\\label\{(?P<label>[^}]+)\}"
    r"(?P<tail>.*?)\\end\{(?P=env)\}\s*",
    re.S,
)


def cut_float(text, label):
    """返回 (新文本, 被摘下的 float 源码)。找不到就抛错。"""
    for m in FLOAT_RE.finditer(text):
        if m.group("label") == label:
            return text[: m.start()] + text[m.end():], m.group(0).strip() + "\n"
    raise SystemExit("找不到 float: %s" % label)


# ----------------------------------------------------------------------------
# 2) 重写后的正文（数值宏一个不动，只压叙述）
# ----------------------------------------------------------------------------
NEW_MNIST = r"""\subsection{Images: the statistic moves, the penalty does not appear}
\label{sec:images}

We test transfer on MNIST $16\times16$ with a conditioning ladder that holds the target
marginal fixed: \texttt{full} ($8\times8$ coarse image $+$ label), \texttt{mid}
($4\times4+$ label), \texttt{half} (left half $+$ label), \texttt{weak} (label only),
\texttt{none} (unconditional) --- the same intervention as ``weaken the observation''
in the VLA literature, made cheap. $\cD$ rises monotonically along the ladder, from
\mnistDmin{} to \mnistDmax: the statistic measures what it is meant to measure. The
one-step penalty, however, does not appear. Conditional sliced $W_2$ is essentially
identical at $1$ and $32$ steps on every rung (\mnistCswOne{} versus \mnistCswRef{},
ratio \mnistCswRatio; Fig.~\ref{fig:mnist}), whereas the same quantity falls by a
factor of $3.3$ between $1$ and $8$ steps on the synthetic task. The cause is visible
in the training loss, \mnistLoss{}: at this CPU-only budget the field is not fitted, so
neither sampler is near the floor and the ladder cannot test the law.
Tab.~\ref{tab:mnist} therefore \textbf{neither confirms nor refutes} it, and we say so
rather than presenting the ladder as support.

"""

NEW_RL = r"""\subsection{Offline control: the qualitative ordering holds, the quantitative one does not}
\label{sec:rl}

We build a multi-goal continuous-control task ($K$ goals on a circle), collect a
behaviour-cloning dataset from a noisy expert, and train a conditional flow-matching
policy with context $c=(s,\text{goal observation})$; we vary the goal observation
(\texttt{exact}/\texttt{noisy}/\texttt{partial}/\texttt{none}) and the action chunk
$H$, reproducing the ``lengthen the horizon'' ablation of
\citep{li2026ofp,letitbesimple2026}. Closed-loop success is measured on $200$ identical
pre-sampled episodes per seed. Tab.~\ref{tab:rl} and Fig.~\ref{fig:rl} give the result,
which splits into a prediction that holds and one that does not.

\emph{The qualitative prediction holds.} One step is worse than $32$ steps in
\rlNAllWorse{} of \rlNCells{} settings. Because the episodes are paired we can test
this directly rather than by averaging cells: over \rlPairedN{} (cell, seed) runs the
mean paired improvement is \rlPairedGap{} with bootstrap $95\%$ CI
$[\rlPairedGapLo,\rlPairedGapHi]$, positive in \rlPairedPos{} of them. Nothing about the
theory is contradicted: the floor is real, and iterative sampling does climb above it
in closed loop.

\emph{The quantitative ordering does not hold, and we can say why.} The size of the gap
\emph{decreases} in $\cD$ across conditioning types ($\rho=\rhoDRl$, CI
$[\rhoDRlLo,\rhoDRlHi]$): it is largest for the informative but multimodal
\texttt{partial} and \texttt{exact} contexts and nearly vanishes when the context is
uninformative, where both samplers are close to useless (\rlSuccOneNone{} versus
\rlSuccRefNone; gap \rlGapNone{} at \texttt{none} against \rlGapInformed{} for the
informed contexts). This is mechanical rather than noise: $\cD$ measures the floor,
whereas what we observe is the floor \emph{minus} whatever extra steps can recover
--- and extra steps recover nothing from a context that carries no information. A
second, separate discrepancy appears at \texttt{exact} conditioning, where a gap of
\rlExactGap{} persists although $\cD=\rlExactD$: no floor argument can explain a gap at
a nearly deterministic conditional, so this component is genuine optimisation error.

We therefore treat the control result as the paper's main negative result. In
closed-loop control one-step degradation has a substantial optimisation component that
$\cD$ does not model. The discriminant bounds the part of the penalty that is
structural; it does not bound the part that comes from an under-fit field.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figures/fig6_rl.pdf}
\caption{Closed-loop success versus NFE (a) and one-step success loss versus $\cD$ (b).
The gap shrinks as $\cD$ grows --- the quantitative ordering of the synthetic family
does not transfer.}
\label{fig:rl}
\end{figure}

"""

NEW_LIM_1 = r"""(1) \emph{One transfer attempt failed, and we know why.} On the MNIST ladder
$\cD$ moves from \mnistDmin{} to \mnistDmax{} while the one-step penalty stays flat
(\mnistCswOne{} versus \mnistCswRef{} in conditional sliced $W_2$); the training loss
\mnistLoss{} shows the field is not fitted, so the ladder cannot test the law
(Sec.~\ref{sec:images}). A $256$-dimensional target conditioned on $\le74$ dimensions
is too large for a CPU-only budget. This is a limitation of our evidence, not a
refutation --- the proposition says nothing about an unconverged field --- but it does
mean the image family does not count as support for the diagnostic."""

NEW_LIM_2 = r"""(2) \emph{The structural floor is only part of the observed penalty.}
$\cD$ bounds the one-step error from below; the observed gap also contains an
optimisation term, which the control grid exhibits directly: one step is worse in
\rlNAllWorse{} of \rlNCells{} settings, yet the gap \emph{decreases} with $\cD$
($\rho=\rhoDRl$), and a residual gap of \rlExactGap{} persists at \texttt{exact}
conditioning where $\cD=\rlExactD$ (Sec.~\ref{sec:rl}). A complete account needs a
capacity/budget term, which we do not have; this is the clearest open problem raised by
this paper."""


def replace_span(text, start_marker, end_marker, new):
    """把 [start_marker, end_marker) 之间（含 start_marker）整段替换为 new。"""
    i = text.find(start_marker)
    if i < 0:
        raise SystemExit("找不到起点: %r" % start_marker[:40])
    j = text.find(end_marker, i)
    if j < 0:
        raise SystemExit("找不到终点: %r" % end_marker[:40])
    return text[:i] + new + text[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    with io.open(MAIN, "r", encoding="utf-8") as f:
        t = f.read()
    orig = t

    # --- A. 摘掉四个 float ---
    t, f_calib = cut_float(t, "tab:calib")
    t, f_toy = cut_float(t, "tab:toy")
    t, f_mnist_t = cut_float(t, "tab:mnist")
    t, f_mnist_f = cut_float(t, "fig:mnist")
    # tab:rl 落在下面 RL 那一节的替换区间内，会随正文一起被换掉，
    # 所以先单独摘出来送附录，否则正文里的 \ref{tab:rl} 就悬空了。
    t, f_rl_t = cut_float(t, "tab:rl")

    # --- B. 重写两节正文；MNIST 段同时把 tab:mnist 从正文带走 ---
    t = replace_span(t, r"\subsection{Images:", r"\subsection{Offline control:", NEW_MNIST)
    t = replace_span(t, r"\subsection{Offline control:",
                     r"\subsection{Is it just a proxy for difficulty?", NEW_RL)

    # --- C. Limitations (1)(2) 去重（两段相邻，必须一次替换，否则第二段已被吞掉）---
    t = replace_span(t, r"(1) \emph{One transfer attempt failed",
                     r"(3) \emph{Scope of the coupling.}",
                     NEW_LIM_1 + "\n" + NEW_LIM_2 + "\n")

    # --- D. toy 节补一句指向被搬走的整表 ---
    t = t.replace(
        r"for \nCellsToy{} cells in total.",
        r"for \nCellsToy{} cells in total (full grid: Tab.~\ref{tab:toy}).",
        1,
    )

    # --- E. 把搬走的 float 追加到附录 ---
    add = "\n".join([f_toy, f_calib, f_mnist_t, f_rl_t, f_mnist_f])
    marker = r"\end{document}"
    k = t.rfind(marker)
    if k < 0:
        raise SystemExit("找不到 \\end{document}")
    t = t[:k] + add + "\n" + t[k:]

    # --- 一致性检查 ---
    checks = []
    for lab in ("tab:toy", "tab:calib", "tab:mnist", "fig:mnist"):
        checks.append((lab, t.count(r"\label{%s}" % lab)))
    for lab in ("fig:gap", "fig:rl", "fig:nfe", "fig:calib"):
        checks.append((lab, t.count(r"\label{%s}" % lab)))

    print("原长度 %d -> 新长度 %d  (减少 %d 字符)" % (len(orig), len(t), len(orig) - len(t)))
    print("float label 出现次数（应各为 1）:")
    for lab, c in checks:
        flag = "OK " if c == 1 else "!! "
        print("  %s%-12s %d" % (flag, lab, c))

    # 引用完整性：正文里 \ref 的 label 必须存在
    refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
    labels = set(re.findall(r"\\label\{([^}]+)\}", t))
    missing = sorted(refs - labels)
    print("悬空引用:", missing or "none")

    if args.apply:
        with io.open(MAIN, "w", encoding="utf-8") as f:
            f.write(t)
        print("\n>>> 已写入 main.tex")
    else:
        print("\n(干跑；加 --apply 才写入)")


if __name__ == "__main__":
    main()
