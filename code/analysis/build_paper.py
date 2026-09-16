# -*- coding: utf-8 -*-
"""一键编译论文：pdflatex -> bibtex -> pdflatex -> pdflatex。

为什么要单独写一个脚本，而不是直接敲命令：
  1. ICLR 样式文件放在 paper/iclr2027/ 子目录里，必须给 pdflatex 设
     TEXINPUTS，否则报 "File `iclr2027_conference.sty' not found"。
  2. 本项目在 Windows/Git-Bash 下跑，shell 里 head/tail/dirname 可能缺失，
     所以所有输出都落盘再解析，不依赖管道。
  3. 编译完自动汇总：有无未定义引用 / 未定义控制序列 / Overfull 页数，
     以及正文字数，避免"编译通过"被当成"没问题"。

用法:
  python build_paper.py            # 编译并汇总
  python build_paper.py --clean     # 先删 aux 再编译
"""
import argparse
import io
import os
import re
import subprocess
import sys

TEXLIVE = r"E:\texlive\bin\windows"
PDFLATEX = os.path.join(TEXLIVE, "pdflatex.exe")
BIBTEX = os.path.join(TEXLIVE, "bibtex.exe")

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
PAPER = os.path.join(ROOT, "paper")

AUX_EXT = [".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".pdf", ".fls",
           ".fdb_latexmk", ".synctex.gz"]


def env_with_sty():
    env = dict(os.environ)
    # 两个坑，都实测过：
    # 1) kpathsea 把 TEXINPUTS 里的 \ 当转义字符 -> 必须用正斜杠；
    # 2) 本项目路径含中文，把**绝对路径**写进 TEXINPUTS 后 pdflatex 找不到文件
    #    （kpsewhich 能列出但 \openin 失败）。用相对路径即可绕开。
    # 末尾的 ';' 很关键：没有它就会丢掉系统默认搜索路径，连 article.cls 都找不到。
    env["TEXINPUTS"] = "./iclr2027;"
    # bibtex 查 .bst 用 BSTINPUTS（不是 TEXINPUTS），少了会报
    # "I couldn't open style file iclr2027_conference.bst"
    env["BSTINPUTS"] = "./iclr2027;"
    env["BIBINPUTS"] = ".;"
    env["max_print_line"] = "1000"
    return env


RCS = []


def run(cmd, label, env):
    r = subprocess.run(cmd, capture_output=True, cwd=PAPER, env=env)
    out = r.stdout.decode("utf-8", errors="replace")
    err = r.stderr.decode("utf-8", errors="replace")
    print("  %-12s rc=%d" % (label, r.returncode))
    RCS.append((label, r.returncode))
    return out + "\n" + err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean:
        # 容错清理：Windows 下 main.pdf 常被预览器/阅读器占用（WinError 5 或
        # 回收站被安全删除层拦截），此时删不掉**不该**中断编译——pdflatex 本来
        # 就会覆盖同名文件。原来直接 os.remove 会让整个 build 在第一步就挂掉。
        locked = []
        for ext in AUX_EXT:
            p = os.path.join(PAPER, "main" + ext)
            if not os.path.exists(p):
                continue
            try:
                os.remove(p)
            except OSError as e:
                locked.append("main" + ext)
                print("  [warn] 无法删除 main%s（被占用？）：%s" % (ext, e.__class__.__name__))
        if locked:
            print("  [warn] 继续编译，交由 pdflatex 覆盖：%s" % ", ".join(locked))

    env = env_with_sty()
    logs = []
    logs.append(run([PDFLATEX, "-interaction=nonstopmode", "-file-line-error",
                     "main.tex"], "pdflatex #1", env))
    logs.append(run([BIBTEX, "main"], "bibtex", env))
    for i in (2, 3):
        logs.append(run([PDFLATEX, "-interaction=nonstopmode", "-file-line-error",
                         "main.tex"], "pdflatex #%d" % i, env))

    blob = "\n".join(logs)
    with io.open(os.path.join(PAPER, "_build.log"), "w", encoding="utf-8") as f:
        f.write(blob)

    # 关键：未定义引用/引文只在**最后一遍**才有意义。
    # 第一遍 aux 还是空的，所有 \ref \cite 都会报 undefined，
    # 拿四遍拼接的日志去统计会得到一份全是假阳性的清单。
    final_log = ""
    lp = os.path.join(PAPER, "main.log")
    if os.path.isfile(lp):
        with io.open(lp, "r", encoding="utf-8", errors="replace") as f:
            final_log = f.read()

    print("\n---------------- build summary ----------------")
    pdf = os.path.join(PAPER, "main.pdf")
    if os.path.exists(pdf):
        print("PDF            : main.pdf (%.0f KB)" % (os.path.getsize(pdf) / 1024.0))
    else:
        print("PDF            : ** NOT PRODUCED **")

    last_pass = logs[-1]
    # 三类错误都要抓：
    #  1) 普通 TeX 错误以 "!" 开头；
    #  2) 因为我们传了 -file-line-error，LaTeX Error 会写成
    #     "./main.tex:247: LaTeX Error: Environment prop undefined."
    #     这种行**不以 "!" 开头**，只看 "!" 会把它们整批漏掉（真实踩过：
    #     论文里 prop/proof 环境根本没定义，却一直报 verdict OK）。
    #  3) **任何** `路径.tex:行号:` 前缀的行都是错误——`-file-line-error` 只给
    #     错误加这个前缀；警告走的是 "LaTeX Warning: ... on input line N." 格式。
    #     真实踩过：`./main.tex:587: Missing $ inserted.`（宏在文本模式下展开出
    #     `\times 10^{-5}`）不含 "LaTeX Error" 字样，被 (2) 整条漏掉，于是
    #     pdflatex 返回 1 而脚本仍报 "hard errors: 0 / verdict: OK"。
    hard = []
    for l in last_pass.splitlines():
        s = l.strip()
        if s.startswith("!"):
            hard.append(s)
        elif re.search(r"\.tex:\d+:\s*(LaTeX|Package|Class)\s+Error", s):
            hard.append(s)
        elif re.match(r"^\.?/?[^:\s]*\.tex:\d+:", s):
            hard.append(s)
        elif "Emergency stop" in s or "Fatal error occurred" in s:
            hard.append(s)
    hard = sorted(set(hard))
    print("hard errors    : %d (last pass)" % len(hard))
    for l in hard[:15]:
        print("    %s" % l[:170])

    # pdflatex 自己的退出码也要报出来：它非零而 hard==0 说明检测式漏了模式。
    bad_rc = [(lb, c) for lb, c in RCS if c != 0 and "pdflatex" in lb]
    print("pdflatex rc    : %s" % (" ".join("%s=%d" % t for t in RCS) or "?"))

    undef_cs = sorted(set(re.findall(r"Undefined control sequence.*?\\([A-Za-z]+)",
                                     final_log, re.S)))
    undef_ref = sorted(set(re.findall(r"LaTeX Warning: Reference `([^']+)'",
                                      final_log)))
    undef_cit = sorted(set(re.findall(r"LaTeX Warning: Citation `([^']+)'",
                                      final_log)))
    print("undefined ctrl : %s" % (undef_cs or "none"))
    print("undefined refs : %s" % (", ".join(undef_ref) if undef_ref else "none"))
    print("undefined cites: %s" % (", ".join(undef_cit) if undef_cit else "none"))
    print("overfull boxes : %d (last pass)" % len(re.findall(r"Overfull \\hbox", last_pass)))
    print("underfull boxes: %d (last pass)" % len(re.findall(r"Underfull \\hbox", last_pass)))

    # 页数与字数（从最后一版 pdf 的 log 里抓）
    m = re.search(r"Output written on main\.pdf \((\d+) pages?,.*?(\d+) bytes", final_log)
    if m:
        npages = int(m.group(1))
        print("total pages    : %d  (含参考文献与附录)" % npages)

    ok = (not hard) and (not bad_rc) and os.path.exists(pdf) \
        and not undef_ref and not undef_cit and not undef_cs
    print("verdict        : %s" % ("OK" if ok else "FAIL"))
    if not ok:
        if bad_rc:
            print("    pdflatex 非零退出但未被识别为错误：%s" % bad_rc)
        sys.exit(1)


if __name__ == "__main__":
    main()
