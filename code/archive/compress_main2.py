# -*- coding: utf-8 -*-
"""第二轮瘦身：把正文从 ~9.5 页压到 <=9 页。

只做两类动作，都不动任何数值：
  A. 收紧三段叙述（Intro 的 scope/empirical 段、Related work 的 Few-step 段）；
  B. 把 Limitations 的 (3)-(6) 与 Future work 搬到附录，正文只留 (1)(2)。

匹配用"空白弹性"正则（把原文里的空白串换成 \\s+），这样不受换行位置影响。
用法: python compress_main2.py [--apply]
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
    """把一段字面文本转成"空白弹性"正则。"""
    parts = re.split(r"\s+", literal.strip())
    return r"\s+".join(re.escape(p) for p in parts)


def rep(text, old, new, label):
    rx = _re_from(old)
    m = re.search(rx, text)
    if not m:
        raise SystemExit("[%s] 未匹配到原文" % label)
    if len(re.findall(rx, text)) > 1:
        raise SystemExit("[%s] 匹配到多处，锚点不够唯一" % label)
    return text[: m.start()] + new + text[m.end():]


# ---------------------------------------------------------------- A. 收紧叙述
OLD_SCOPE = r"""
\paragraph{Scope and assumptions.}
The field, tasks, datasets, compute budget, baselines, evaluation protocol and target
venue are stated in full in Appendix~\ref{app:scope}; the short version is that every
experiment in this paper runs on \textbf{CPU only}, uses no proprietary data and no
pretrained backbone, and compares \emph{sampling budgets of a single trained field}
rather than different methods, so that architecture and optimisation are held fixed.
"""
NEW_SCOPE = r"""
\paragraph{Scope and assumptions.}
Every experiment runs on \textbf{CPU only}, uses no proprietary data and no pretrained
backbone, and compares \emph{sampling budgets of a single trained field} rather than
different methods, so architecture and optimisation are held fixed;
Appendix~\ref{app:scope} states the rest.
"""

OLD_EMPIRICAL = r"""
The empirical picture these papers report is, however, internally inconsistent in an
instructive way. One-step policies are reported to match or beat $100$-step
diffusion policies on $56$ manipulation tasks \citep{li2026ofp} and to reach $92.8$
on D4RL MuJoCo \citep{nguyen2026bfq}, while the same families of models are reported
to degrade when the conditioning is weakened or the action horizon is lengthened
\citep{li2026ofp,letitbesimple2026}. The most explicit diagnosis so far frames
one-step success as a property of the \emph{condition--target structure} of the task
\citep{letitbesimple2026}: a VLA with a strong observation and a compact action chunk
is closer to image-to-text than to text-to-image. That paper's own stated open
problem is to \emph{quantify} when the effect holds.
"""
NEW_EMPIRICAL = r"""
The empirical picture these papers report is internally inconsistent in an instructive
way: one-step policies are reported to match or beat $100$-step diffusion policies on
$56$ manipulation tasks \citep{li2026ofp} and to reach $92.8$ on D4RL MuJoCo
\citep{nguyen2026bfq}, yet the same families degrade when the conditioning is weakened
or the action horizon lengthened \citep{li2026ofp,letitbesimple2026}. The most explicit
diagnosis frames one-step success as a property of the \emph{condition--target
structure} \citep{letitbesimple2026} --- a VLA with a strong observation and a compact
action chunk is closer to image-to-text than to text-to-image --- and that paper's own
stated open problem is to \emph{quantify} when the effect holds.
"""

OLD_FEWSTEP_TAIL = r"""
Our contribution is orthogonal and complementary: these works change the
\emph{method}; we characterise the \emph{target-side} quantity that bounds what any
one-step method can do under the standard coupling, and we give a way to measure it
before choosing among them.
"""
NEW_FEWSTEP_TAIL = r"""
Our contribution is complementary: these works change the \emph{method}; we characterise
the \emph{target-side} quantity bounding what any one-step method can do under the
standard coupling, and show how to measure it before choosing among them.
"""

OLD_C3_TAIL = r"""
On the two transfer families the picture is
  deliberately less flattering and we say so: the image ladder moves $\cD$ across its
  whole range without the one-step penalty appearing, because the field is under-fit at
  our CPU budget, and in closed-loop control one step is worse in \rlNAllWorse{} of
  \rlNCells{} settings but the \emph{size} of the gap decreases with $\cD$, since an
  uninformative context degrades the many-step sampler as well
  (Secs.~\ref{sec:images}, \ref{sec:rl}, \ref{sec:limitations}). We state the resulting
  scope precisely: $\cD$ bounds the structural part of the penalty, not the part caused
  by an under-fit field.
"""
NEW_C3_TAIL = r"""
On the two transfer families the picture is
  deliberately less flattering: the image ladder moves $\cD$ across its whole range
  without the one-step penalty appearing (an under-fit field), and in closed-loop control
  one step is worse in \rlNAllWorse{} of \rlNCells{} settings but the \emph{size} of the
  gap decreases with $\cD$, since an uninformative context degrades the many-step sampler
  as well (Secs.~\ref{sec:images}, \ref{sec:rl}). So $\cD$ bounds the structural part of
  the penalty, not the part caused by an under-fit field.
"""

# ------------------------------------------------- B. 长版 Limitations 搬附录
LIM_START = r"(3) \emph{Scope of the coupling.}"
LIM_END = r"\section{Conclusion}"

POINTER = r"""
Broader limitations and future work are in Appendix~\ref{app:limitations}.

"""

APPENDIX_SEC = r"""
\section{Broader limitations and future work}
\label{app:limitations}
Placed here to respect the nine-page limit; the two limitations that bear directly on
our claims ($1$--$2$) are stated in the main text.

\paragraph{Additional limitations.}
(3) \emph{Scope of the coupling.} Prop.~\ref{prop:floor} assumes the independent source
coupling used by essentially all practical flow-matching implementations; under a
deterministic coupling the floor disappears (reflow), and our $\cL$ still applies but
$\cD$ becomes only an upper bound on what is recoverable.
(4) \emph{High-dimensional conditioning.} $k$-NN regression and neighbourhood clustering
degrade as the conditioning dimension grows; our image experiments use
$\le74$-dimensional conditioning, and we have not tested pixel-level or
language-embedding conditioning, where a learned embedding would be needed first.
(5) \emph{Scale.} All experiments are CPU-scale (toys, MNIST, a 2-D control task). We
have \emph{not} validated on D4RL \citep{fu2020d4rl}, OGBench \citep{park2025ogbench},
LIBERO \citep{liu2023libero} or a real VLA; transfer to those settings is the obvious
next step and is unproven here.
(6) \emph{Distribution shift.} In closed-loop control the policy induces its own state
distribution; our discriminant is computed on the offline distribution only.

\paragraph{Future work.} Extending the floor to a \emph{joint} bound that includes
discretisation error at finite $N>1$; an online version that re-estimates $\cD$ during
training (the ``when is a high-noise schedule most useful'' question left open by
\citep{letitbesimple2026}); and a large-scale validation on OGBench/LIBERO with a VLA
action head, where the decision ``one step or many'' is currently made by trial and
error.

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t = io.open(MAIN, "r", encoding="utf-8").read()
    n0 = len(t)

    t = rep(t, OLD_SCOPE, NEW_SCOPE, "scope")
    t = rep(t, OLD_EMPIRICAL, NEW_EMPIRICAL, "empirical")
    t = rep(t, OLD_FEWSTEP_TAIL, NEW_FEWSTEP_TAIL, "fewstep")
    t = rep(t, OLD_C3_TAIL, NEW_C3_TAIL, "c3tail")

    # 把 (3)-(6)+Future work 摘出来
    i = t.find(LIM_START)
    j = t.find(LIM_END, i)
    if i < 0 or j < 0:
        raise SystemExit("找不到 Limitations 区间")
    moved = t[i:j]
    for must in ("(4) \\emph{High-dimensional", "(5) \\emph{Scale.", "(6) \\emph{Distribution",
                 "Future work"):
        if must not in moved:
            raise SystemExit("搬移区间缺内容: %s" % must)
    t = t[:i] + POINTER + t[j:]

    # 附录：追加到 \end{document} 之前
    k = t.rfind(r"\end{document}")
    t = t[:k] + APPENDIX_SEC + t[k:]

    # 检查：正文里不应再出现 (3)-(6) 的字样（附录里有，所以只查正文段）
    body = t[: t.find(r"\appendix")]
    leak = [s for s in ("(3) \\emph{Scope", "(4) \\emph{High", "(5) \\emph{Scale",
                        "(6) \\emph{Distribution") if s in body]
    print("正文字符 %d -> %d (减 %d)" % (n0, len(t), n0 - len(t)))
    print("正文残留 (3)-(6):", leak or "none")

    refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
    labels = set(re.findall(r"\\label\{([^}]+)\}", t))
    print("悬空引用:", sorted(refs - labels) or "none")

    if args.apply:
        io.open(MAIN, "w", encoding="utf-8").write(t)
        print(">>> 已写入")
    else:
        print("(干跑；加 --apply 才写入)")


if __name__ == "__main__":
    main()
