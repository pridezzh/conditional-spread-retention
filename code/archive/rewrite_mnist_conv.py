# -*- coding: utf-8 -*-
"""把"MNIST 负结果"改写成"拟合不足造成、换卷积场后惩罚出现"。

依据（results/ 里的真实数据，非推测）：
  同一预算 5000 步、同一条 5 档阶梯：
    MLP : 平均训练损失 113.6，sw@1/sw@32 基本持平（惩罚不出现）
    卷积: 平均训练损失 ~30，sw@1/sw@32 从强条件档的小差距升到 weak 档约 3.6x，
          且 weak 档 mode recall 从 ~0.6 掉到 0（一步把模态抹平）

同时为了不超 9 页，顺手收紧三处（Intro scope、Metrics(ii)、Limitations(1)）。

用法: python rewrite_mnist_conv.py [--apply]
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


def _re_from(literal):
    parts = re.split(r"\s+", literal.strip())
    return r"\s+".join(re.escape(p) for p in parts)


def rep(text, old, new, label):
    rx = _re_from(old)
    ms = list(re.finditer(rx, text))
    if not ms:
        raise SystemExit("[%s] 未匹配到" % label)
    if len(ms) > 1:
        raise SystemExit("[%s] 多处匹配 %d" % (label, len(ms)))
    m = ms[0]
    return text[: m.start()] + new + text[m.end():]


def span(text, start, end, new, label):
    i = text.find(start)
    if i < 0:
        raise SystemExit("[%s] 找不到起点" % label)
    j = text.find(end, i)
    if j < 0:
        raise SystemExit("[%s] 找不到终点" % label)
    return text[:i] + new + text[j:]


# ------------------------------------------------------------------ 1. 摘要
OLD_ABS = r"""
We also report
the limits of the diagnostic on two transfer families: on an image conditioning ladder
$\cD$ moves across its full range while the predicted penalty fails to appear, because
the velocity field is under-fit at a CPU-only budget; and in closed-loop control one
step is worse in \rlNAllWorse{} of \rlNCells{} settings, but the size of the gap
\emph{decreases} with $\cD$, since an uninformative context cripples the many-step
sampler too.
"""
NEW_ABS = r"""
On two transfer
families the picture is mixed and we report both halves. On an image conditioning ladder
the predicted penalty is \emph{absent} for the under-fit field we can train on CPU ---
$\cD$ moves across its full range while the penalty does not --- but it \emph{reappears}
once a convolutional field is fitted at the same budget, ordered by $\cD$; the diagnostic
therefore needs a fitted field before it binds. In closed-loop control one step is worse
in \rlNAllWorse{} of \rlNCells{} settings, but the size of the gap \emph{decreases} with
$\cD$, since an uninformative context cripples the many-step sampler too.
"""

# ------------------------------------------------------------- 2. 贡献 C3 末句
OLD_C3 = r"""
On the two transfer families the picture is
  deliberately less flattering: the image ladder moves $\cD$ across its whole range
  without the one-step penalty appearing (an under-fit field), and in closed-loop control
  one step is worse in \rlNAllWorse{} of \rlNCells{} settings but the \emph{size} of the
  gap decreases with $\cD$, since an uninformative context degrades the many-step sampler
  as well (Secs.~\ref{sec:images}, \ref{sec:rl}). So $\cD$ bounds the structural part of
  the penalty, not the part caused by an under-fit field.
"""
NEW_C3 = r"""
On the transfer families the picture is
  mixed and we report both halves: on images the penalty is absent for an under-fit MLP
  and returns once a convolutional field is fitted at the same budget (Sec.~\ref{sec:images});
  in closed-loop control one step is worse in \rlNAllWorse{} of \rlNCells{} settings but
  the \emph{size} of the gap decreases with $\cD$, since an uninformative context degrades
  the many-step sampler as well (Sec.~\ref{sec:rl}). So $\cD$ bounds the structural part
  of the penalty, not the part caused by an under-fit field.
"""

# -------------------------------------------------------- 3. MNIST 小节全文重写
NEW_MNIST = r"""\subsection{Images: the penalty appears once the field is fitted}
\label{sec:images}

We test transfer on MNIST $16\times16$ with a conditioning ladder that holds the target
marginal fixed: \texttt{full} ($8\times8$ coarse image $+$ label), \texttt{mid}
($4\times4+$ label), \texttt{half} (left half $+$ label), \texttt{weak} (label only),
\texttt{none} (unconditional) --- the ``weaken the observation'' intervention of the VLA
literature, made cheap. $\cD$ rises monotonically along the ladder, from \mnistDmin{} to
\mnistDmax: the statistic measures what it is meant to measure.

Whether the one-step penalty follows depends on whether the field is fitted at all. With
the MLP we can afford on CPU (mean training loss \mnistLoss{}) the conditional sliced $W_2$
is essentially identical at $1$ and $32$ steps on every rung (\mnistCswOne{} versus
\mnistCswRef{}, ratio \mnistCswRatio), so Tab.~\ref{tab:mnist} \textbf{neither confirms nor
refutes} the law. A small convolutional field at the \emph{same} budget fits the target
(mean loss \mnistLossConv{}) and the penalty reappears, ordered by $\cD$
(Fig.~\ref{fig:mnist}b, Tab.~\ref{tab:mnistconv}): the $1$-vs-$32$ gap in marginal sliced
$W_2$ is small on the strongly conditioned rung and reaches \mnistSwOneConv{} versus
\mnistSwRefConv{} at the \texttt{weak} rung, where mode recall collapses from
\mnistRecRefConv{} to \mnistRecOneConv{} --- one step averages the modes away. The plateau
was an under-fitting artefact, and we report both field settings rather than only the
flattering one.

"""

# --------------------------------------------- 4. Limitations (1) 收紧（已解决）
OLD_LIM1 = r"""
(1) \emph{One transfer attempt failed, and we know why.} On the MNIST ladder $\cD$ moves
from \mnistDmin{} to \mnistDmax{} while the one-step penalty stays flat (\mnistCswOne{}
versus \mnistCswRef{}); the training loss \mnistLoss{} shows the field is not fitted, so
the ladder cannot test the law (Sec.~\ref{sec:images}). This is a limitation of our
evidence, not a refutation --- the proposition says nothing about an unconverged field ---
but the image family does not count as support for the diagnostic.
"""
NEW_LIM1 = r"""
(1) \emph{The diagnostic only binds once the field is fitted.} On the MNIST ladder the
one-step penalty is invisible for our under-fit MLP and visible for a convolutional field
at the same budget (Sec.~\ref{sec:images}); a user who computes $\cD$ on a task whose
velocity field they will never fit should not expect the penalty to show up.
"""

# ------------------------------------------------- 5. Intro scope 段（省 1 行）
OLD_SCOPE = r"""
\paragraph{Scope and assumptions.}
Every experiment runs on \textbf{CPU only}, uses no proprietary data and no pretrained
backbone, and compares \emph{sampling budgets of a single trained field} rather than
different methods, so architecture and optimisation are held fixed;
Appendix~\ref{app:scope} states the rest.
"""
NEW_SCOPE = r"""
\paragraph{Scope.} Everything here is \textbf{CPU only}, with no proprietary data and no
pretrained backbone; Appendix~\ref{app:scope} gives the full statement.
"""

# ------------------------------------------- 6. Metrics (ii) 收紧（省 ~3 行）
OLD_METRIC_II = r"""
(ii) \emph{off-manifold mass} (off@$k$): the fraction of samples further than
$\sigma(3+\sqrt d)$ from \emph{every} ground-truth mode centre. This replaces the
mode-recall criterion we used in an earlier version, for a reason we consider a genuine
correction rather than a cosmetic one: a tolerance derived from the \emph{spacing}
between modes shrinks as $K$ grows and is smaller than the modes themselves at $K=16$,
and at $K=1$ it is undefined --- ``nearest centre'' assignment then reports recall $1$
even for a sampler that has collapsed to a single point. A tolerance tied to the mode
\emph{width} $\sigma$ is scale-free and catches both failures.
"""
NEW_METRIC_II = r"""
(ii) \emph{off-manifold mass} (off@$k$): the fraction of samples further than
$\sigma(3+\sqrt d)$ from \emph{every} ground-truth mode centre. It replaces the
mode-recall criterion we used earlier, for a reason we consider a genuine correction: a
tolerance derived from the \emph{spacing} between modes shrinks as $K$ grows, falls below
the mode width at $K=16$, and is undefined at $K=1$ --- where ``nearest centre''
assignment reports recall $1$ even for a sampler collapsed to a single point. A tolerance
tied to the mode \emph{width} $\sigma$ is scale-free and catches both failures.
"""

# ------------------------------------- 7. 附录 fig:mnist 题注 + 追加 conv 表
OLD_FIGMNIST_CAP = r"""
\caption{Conditioning ladder. (a) $\cD$ rises monotonically as the conditioning is
weakened --- the statistic tracks the structure it targets. (b) The one-step penalty
does not follow it: the gap between $1$ and $32$ steps stays flat across the whole
ladder, because the field is not converged at this budget. The pair (a)--(b) is the
negative transfer result of Sec.~\ref{sec:limitations}.}
"""
NEW_FIGMNIST_CAP = r"""
\caption{Conditioning ladder, one statistic, two fields. (a) $\cD$ rises monotonically as
the conditioning is weakened --- the statistic tracks the structure it targets, and it is
computed from data alone, so it is identical for both architectures. (b) The ratio of
$1$-step to $32$-step marginal sliced $W_2$: flat (no penalty) for the under-fit MLP,
rising with $\cD$ for a convolutional field at the same budget. (c) The reason is the
training loss, $3$--$6\times$ higher for the MLP.}
"""

OLD_EXTRA = r"""
\begin{table}[t]
\centering
\caption{Ablations (fixed cell $K=8$, $\mathrm{sep}=6$, $\eta=0.5$).}
\label{tab:ablest}
"""
NEW_EXTRA = r"""
\begin{table}[t]
\centering
\caption{Architecture control for the MNIST ladder: same target, same budget, same five
rungs, convolutional field instead of the MLP.}
\label{tab:mnistconv}
\vskip 0.05in
\input{../tables/tab_mnist_conv_body.tex}
\end{table}

\begin{table}[t]
\centering
\caption{Ablations (fixed cell $K=8$, $\mathrm{sep}=6$, $\eta=0.5$).}
\label{tab:ablest}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t = io.open(MAIN, "r", encoding="utf-8").read()
    n0 = len(t)

    t = rep(t, OLD_ABS, NEW_ABS, "abstract")
    t = rep(t, OLD_C3, NEW_C3, "c3")
    t = span(t, r"\subsection{Images:", r"\subsection{Offline control:", NEW_MNIST, "mnist")
    t = rep(t, OLD_LIM1, NEW_LIM1, "lim1")
    t = rep(t, OLD_SCOPE, NEW_SCOPE, "scope")
    t = rep(t, OLD_METRIC_II, NEW_METRIC_II, "metric_ii")
    t = rep(t, OLD_FIGMNIST_CAP, NEW_FIGMNIST_CAP, "figmnist_cap")
    t = rep(t, OLD_EXTRA, NEW_EXTRA, "extra_tab")

    refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
    labels = set(re.findall(r"\\label\{([^}]+)\}", t))
    print("字符 %d -> %d" % (n0, len(t)))
    print("悬空引用:", sorted(refs - labels) or "none")
    for k in ("tab:mnistconv", "fig:mnist", "tab:mnist", "sec:images"):
        print("  %-15s label x%d  ref x%d" % (k, t.count(r"\label{%s}" % k),
                                              t.count(r"\ref{%s}" % k)))
    if args.apply:
        io.open(MAIN, "w", encoding="utf-8").write(t)
        print(">>> 已写入")
    else:
        print("(干跑)")


if __name__ == "__main__":
    main()
