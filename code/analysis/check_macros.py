# -*- coding: utf-8 -*-
"""编译前置检查：正文引用的宏是否都被数字宏文件定义了。

数字宏文件（见 NUM_FILES）：第三轮重写后正文只引用 numbers_theory.tex，
由 code/analysis/make_theory_macros.py 从 logs/ 与 results/ 生成；
numbers.tex 是旧版遗留（已不再被 main.tex \\input）。

动机：main.tex 里的数字全部通过 \\newcommand 宏引用。
只要有一个宏没定义（例如某个统计量在当前数据下是 NaN，被生成脚本
跳过不写），pdflatex 会报 "Undefined control sequence" 并停在很远的位置，
排查成本很高。这个脚本把问题提前到编译之前、并直接给出缺失清单。

同时反向检查：定义了但正文从未引用的宏（死宏），
避免以为"数字都进了论文"其实一个都没用上。

用法:
  python check_macros.py            # 只报告
  python check_macros.py --strict   # 有缺失则退出码 1（供 CI / 流水线用）
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
_ROOT = _ROOT_
PAPER = os.path.join(_ROOT, "paper")
TABDIR = os.path.join(_ROOT, "tables")

# 论文数字宏的来源文件。第三轮重写后正文只引用 numbers_theory.tex；
# numbers.tex 是旧版遗留（保留在 paper/ 里但已不被 \input）。
NUM_FILES = ("numbers_theory.tex", "numbers.tex")

# 这些是 LaTeX / 宏包自带的，出现即视为已定义
BUILTIN = {
    "text", "mathrm", "mathbf", "mathcal", "mathbb", "hat", "bar", "tilde",
    "sqrt", "frac", "sum", "prod", "int", "log", "exp", "min", "max", "arg",
    "top", "mid", "bottom", "alpha", "beta", "gamma", "delta", "epsilon",
    "eta", "theta", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
    "tau", "phi", "chi", "psi", "omega", "Lambda", "Sigma", "Phi", "Psi",
    "Omega", "Theta", "Delta", "Gamma", "times", "cdot", "pm", "leq", "geq",
    "neq", "approx", "sim", "propto", "in", "subset", "forall", "exists",
    "nabla", "partial", "infty", "to", "rightarrow", "Rightarrow", "left",
    "right", "big", "Big", "bigg", "Bigg", "begin", "end", "label", "ref",
    "cite", "citep", "citet", "includegraphics", "caption", "centering",
    "item", "itemize", "enumerate", "textbf", "textit", "emph", "vskip",
    "noindent", "paragraph", "section", "subsection", "subsubsection",
    "title", "author", "date", "maketitle", "input", "bibliography",
    "bibliographystyle", "hline", "toprule", "midrule", "bottomrule",
    "multicolumn", "small", "sc", "footnote", "url", "hypersetup", "documentclass",
    "usepackage", "newcommand", "renewcommand", "def", "operatorname", "textwidth",
    "linewidth", "columnwidth", "hspace", "vspace", "quad", "qquad", "hfill",
    "Vert", "vert", "langle", "rangle", "colon", "star", "dagger", "circ",
    "bullet", "dagger", "dots", "ldots", "dotsb", "space", "rlap", "llap",
    "smash", "raisebox", "parbox", "resizebox", "scalebox", "scriptsize",
    "tiny", "large", "Large", "LARGE", "huge", "Huge", "normalsize", "bf",
    "it", "rm", "tt", "sf", "sl",     "displaystyle", "textstyle", "Bigl", "Bigr", "bigl", "bigr", "biggl",
    "biggr", "Biggl", "Biggr", "Longleftrightarrow", "longleftrightarrow",
    "Longrightarrow", "longleftarrow", "Leftrightarrow", "Leftarrow",
    "eqref", "appendix", "mapsto", "longmapsto", "perp", "square",
    "textcolor", "texttt", "textrm", "textsf", "textnormal", "underbrace",
    "overbrace", "zeta", "iota", "upsilon", "varepsilon", "varphi", "varrho",
    "varsigma", "vartheta", "notag", "nonumber", "ensuremath", "ifthenelse",
    "newcolumntype", "cline", "tiny", "scriptsize", "footnotesize",
    "normalsize", "small", "large", "Large", "LARGE", "huge", "Huge",
    "relax", "let", "edef", "gdef", "xdef", "ifx", "else", "fi", "csname",
    "endcsname", "expandafter", "noexpand", "ignorespaces", "unskip", "protect",
    "arraystretch", "tabcolsep", "abovecaptionskip", "belowcaptionskip",
    "par", "hbox", "vbox", "halign", "valign", "cr", "noalign", "omit",
    "ast", "dot", "ddot", "ge", "le", "leq", "geq", "ll", "gg", "circ",
    "prime", "tilde", "widehat", "widetilde", "overline", "underline",
    "stackrel", "overset", "underset", "binom", "choose", "pmod", "bmod",
    "mathrm", "mathsf", "mathtt", "mathnormal", "boldsymbol", "bm",
    "LaTeX", "TeX", "BibTeX", "XeTeX", "LuaTeX",
    # 定理环境：main.tex 导言区用 \theoremstyle{plain}\newtheorem{prop}{...}，
    # 这两个是 LaTeX 自带命令，不加进来会被误报成 "referenced but NOT defined"。
    "newtheorem", "theoremstyle", "theorem", "proof", "qedhere", "MakeUppercase",
    "fbox", "setlength", "texorpdfstring",
    # ---- 第三轮重写论文时新用到的（漏掉会误报 MISSING）----
    "tfrac", "succeq", "wedge", "bigm", "bigl", "bigr", "notag", "nonumber",
    "textsuperscript", "textsubscript", "left", "right", "displaystyle",
    "mathrm", "operatorname", "middle", "varnothing", "geqslant", "leqslant",
    "triangleq", "bm", "mathds", "textquotedblleft", "textquotedblright",
    # 2026-09-16：跑 --strict 时被误报的三个标准数学算符（LaTeX 内核自带）
    "equiv", "inf", "sup", "lim", "limsup", "liminf", "det", "dim", "ker",
    "deg", "gcd", "lcm", "sin", "cos", "tan", "sinh", "cosh", "tanh", "cot",
    "sec", "csc", "arcsin", "arccos", "arctan", "log", "ln", "lg", "Re", "Im",
    "lfloor", "rfloor", "lceil", "rceil", "vert", "Vert", "|", "{", "}",
}


def read(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def strip_comments(s):
    out = []
    for line in s.splitlines():
        # 去掉未被转义的 % 之后的内容
        buf, i = [], 0
        while i < len(line):
            c = line[i]
            if c == "\\" and i + 1 < len(line):
                buf.append(line[i:i + 2])
                i += 2
                continue
            if c == "%":
                break
            buf.append(c)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def defined_macros():
    """收集所有"已定义"的控制序列。

    只看 numbers.tex 是不够的：main.tex 自己的导言区定义了 \\cD \\cL \\tr
    \\Var \\Cov \\Eop \\Ncal \\Real \\todo，ICLR 模板的 math_commands.tex 又定义了
    一大批。漏掉它们会把正常引用误报成缺失。
    """
    sources = [os.path.join(PAPER, "main.tex"),
               os.path.join(PAPER, "iclr2027", "math_commands.tex")]
    sources += [os.path.join(PAPER, f) for f in NUM_FILES]
    defs, nums = set(), set()
    for p in sources:
        if not os.path.isfile(p):
            continue
        s = strip_comments(read(p))
        got = set()
        for pat in (r"\\newcommand\*?\{\\([A-Za-z]+)\}",
                    r"\\renewcommand\*?\{\\([A-Za-z]+)\}",
                    r"\\providecommand\*?\{\\([A-Za-z]+)\}",
                    r"\\DeclareRobustCommand\*?\{\\([A-Za-z]+)\}",
                    r"\\DeclareMathOperator\*?\{\\([A-Za-z]+)\}",
                    r"\\def\\([A-Za-z]+)"):
            got |= set(re.findall(pat, s))
        defs |= got
        if os.path.basename(p) in NUM_FILES:
            nums |= got
    return defs, nums


def referenced_macros():
    texts = [(os.path.join(PAPER, "main.tex"), read(os.path.join(PAPER, "main.tex")))]
    if os.path.isdir(TABDIR):
        for fn in sorted(os.listdir(TABDIR)):
            if fn.endswith(".tex"):
                p = os.path.join(TABDIR, fn)
                texts.append((p, read(p)))
    for extra in ("macros.tex",):
        p = os.path.join(PAPER, extra)
        if os.path.isfile(p):
            texts.append((p, read(p)))

    refs = {}
    for path, raw in texts:
        s = strip_comments(raw)
        # 只取连续字母：\approx0.3 里的控制序列是 \approx，不是 \approx0
        for m in re.finditer(r"\\([A-Za-z]+)", s):
            name = m.group(1)
            if name in BUILTIN:
                continue
            refs.setdefault(name, set()).add(os.path.relpath(path, _ROOT))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    defs, nums = defined_macros()
    refs = referenced_macros()
    # 只认 numbers.tex 里的定义能说明"这个数字被用上了"，
    # 样式文件里那几百个内部宏不算，否则死宏清单全是噪声。
    used_by_paper = {k for k in refs if k in nums}
    missing = sorted(k for k in refs if k not in defs)
    dead = sorted(nums - used_by_paper)
    # 含数字的宏名在 LaTeX 里是非法的（控制序列只能由字母组成）
    illegal = sorted(k for k in nums if not k.isalpha())

    print("defined in %s : %d" % ("+".join(NUM_FILES), len(nums)))
    print("defined anywhere       : %d" % len(defs))
    print("referenced (non-builtin): %d" % len(refs))
    print("matched                : %d" % len(used_by_paper))

    if illegal:
        print("\n[ILLEGAL] 宏名含数字 -> \\newcommand 会致命报错，整篇编译失败：")
        for k in illegal:
            print("   \\%s" % k)
    else:
        print("\n[ILLEGAL] none -- 宏名全部是纯字母")

    if missing:
        print("\n[MISSING] referenced but NOT defined -> pdflatex 会报错：")
        for k in missing:
            print("   \\%s   <- %s" % (k, ", ".join(sorted(refs[k]))))
    else:
        print("\n[MISSING] none -- 正文引用的宏全部有定义")

    if dead:
        print("\n[DEAD] 数字宏文件里定义但正文从未引用 (%d)：" % len(dead))
        line = "   "
        for k in dead:
            if len(line) + len(k) > 96:
                print(line)
                line = "   "
            line += k + "  "
        if line.strip():
            print(line)

    if (missing or illegal) and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
