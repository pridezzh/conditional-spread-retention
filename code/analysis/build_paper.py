# -*- coding: utf-8 -*-
"""One-shot paper build: pdflatex -> bibtex -> pdflatex -> pdflatex.

Why a dedicated script instead of typing commands directly:
  1. The ICLR style files live in paper/iclr2027/, so pdflatex must be given
     TEXINPUTS, otherwise it errors "File `iclr2027_conference.sty' not found".
  2. This project runs under Windows/Git-Bash, where shell utilities such as
     head/tail/dirname may be missing, so all output is written to disk and then
     parsed -- no reliance on pipes.
  3. After building, it summarizes: undefined references / undefined control
     sequences / Overfull page count, plus main-text word count, to avoid
     treating "compiles" as "no problems".

Usage:
  python build_paper.py            # build and summarize
  python build_paper.py --clean     # delete aux files first, then build
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys

# Resolve TeX Live binaries from PATH so the build is portable across machines.
PDFLATEX = shutil.which("pdflatex") or "pdflatex"
BIBTEX = shutil.which("bibtex") or "bibtex"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """Walk up to the top level that contains both code/ and paper/; independent of the script's own depth."""
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
    # Two pitfalls, both verified empirically:
    # 1) kpathsea treats \ in TEXINPUTS as an escape character -> must use
    #    forward slashes;
    # 2) this project's path contains non-ASCII characters; writing the
    #    **absolute path** into TEXINPUTS makes pdflatex unable to find files
    #    (kpsewhich can list them but \openin fails). Use a relative path to
    #    avoid this.
    # The trailing ';' is critical: without it the system default search path is
    # dropped and even article.cls cannot be found.
    env["TEXINPUTS"] = "./iclr2027;"
    # bibtex looks up .bst via BSTINPUTS (not TEXINPUTS); omitting it raises
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
        # Tolerant cleanup: under Windows main.pdf is often locked by a previewer
        # /reader (WinError 5 or intercepted by the safe-delete layer), so it
        # cannot be deleted -- but that **must not** abort the build, since
        # pdflatex will overwrite the same-named file anyway. The original direct
        # os.remove would crash the entire build at the very first step.
        locked = []
        for ext in AUX_EXT:
            p = os.path.join(PAPER, "main" + ext)
            if not os.path.exists(p):
                continue
            try:
                os.remove(p)
            except OSError as e:
                locked.append("main" + ext)
                print("  [warn] cannot delete main%s (locked?): %s" % (ext, e.__class__.__name__))
        if locked:
            print("  [warn] continuing build; letting pdflatex overwrite: %s" % ", ".join(locked))

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

    # Critical: undefined references/citations are only meaningful on the
    # **last pass**. On the first pass the aux file is still empty, so every
    # \ref \cite reports undefined; tallying across the four concatenated logs
    # yields a list that is all false positives.
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
    # Three kinds of errors must all be caught:
    #  1) ordinary TeX errors start with "!";
    #  2) because we pass -file-line-error, LaTeX Error is written as
    #     "./main.tex:247: LaTeX Error: Environment prop undefined."
    #     Such lines do **not** start with "!", so looking only for "!" would
    #     miss them entirely (hit this in practice: the prop/proof environments
    #     were never defined in the paper, yet the verdict kept reporting OK).
    #  3) **any** line with a `path.tex:line:` prefix is an error -- `-file-line-error`
    #     adds this prefix only to errors; warnings use the "LaTeX Warning: ...
    #     on input line N." format. Hit in practice: `./main.tex:587: Missing $
    #     inserted.` (a macro expanded `\times 10^{-5}` in text mode) contains no
    #     "LaTeX Error" text, was missed entirely by (2), so pdflatex returned 1
    #     while the script still reported "hard errors: 0 / verdict: OK".
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

    # pdflatex's own exit code must also be reported: a non-zero code while
    # hard==0 means the detection pattern missed something.
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

    # page count and word count (from the last pdf build's log)
    m = re.search(r"Output written on main\.pdf \((\d+) pages?,.*?(\d+) bytes", final_log)
    if m:
        npages = int(m.group(1))
        print("total pages    : %d  (including references and appendix)" % npages)

    ok = (not hard) and (not bad_rc) and os.path.exists(pdf) \
        and not undef_ref and not undef_cit and not undef_cs
    print("verdict        : %s" % ("OK" if ok else "FAIL"))
    if not ok:
        if bad_rc:
            print("    pdflatex non-zero exit not recognized as an error: %s" % bad_rc)
        sys.exit(1)


if __name__ == "__main__":
    main()
