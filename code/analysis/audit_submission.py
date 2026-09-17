# -*- coding: utf-8 -*-
"""Pre-submission independent audit: paper, results, figures, and code fingerprint.

This script does not train any model, nor does it backfill legacy results to look
"traceable". When provenance metadata is missing it explicitly issues a BLOCK, to
avoid mistaking internal numeric consistency for full reproducibility under the
same code version.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.join(ROOT, "paper")
RESULTS = os.path.join(ROOT, "results")
# Resolve TeX Live helpers from PATH so the script is portable across machines.
PDFINFO = shutil.which("pdfinfo") or "pdfinfo"
PDFTOTEXT = shutil.which("pdftotext") or "pdftotext"
# ICLR 2027 allows a strict 9-page main text; the AI-use, ethics and reproducibility
# statement sections do not count toward that limit (see the conference template).
MAIN_TEXT_PAGE_LIMIT = 9


def load_json(name: str) -> dict:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def pdf_page_count(path: str) -> int:
    out = subprocess.check_output([PDFINFO, path], text=True, encoding="utf-8",
                                  errors="replace")
    match = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo did not return a page count")
    return int(match.group(1))


def first_page_containing(path: str, needle: str, pages: int,
                          up_to: int | None = None) -> int | None:
    # Small-caps titles are extracted by pdftotext as "R EFERENCES" (tracking
    # becomes spaces), and long titles may be broken across lines by typesetting
    # -- strip all whitespace from both sides before comparing to avoid false
    # page classification. up_to bounds the search so that a phrase recurring in
    # the appendix cannot be mistaken for the main-text location of a heading.
    key = re.sub(r"\s+", "", needle).casefold()
    for page in range(1, (up_to if up_to else pages) + 1):
        out = subprocess.check_output(
            [PDFTOTEXT, "-f", str(page), "-l", str(page), path, "-"],
            text=True, encoding="utf-8", errors="replace")
        if key in re.sub(r"\s+", "", out).casefold():
            return page
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy-results", action="store_true",
                        help="typesetting check only; does not upgrade legacy results into submission evidence")
    args = parser.parse_args()
    passed: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    tex_path = os.path.join(PAPER, "main.tex")
    tex = open(tex_path, encoding="utf-8").read()
    log_path = os.path.join(PAPER, "main.log")
    build_log = open(log_path, encoding="utf-8", errors="replace").read() \
        if os.path.isfile(log_path) else ""

    forbidden = {
        "legacy kNN correction formula": r"\alpha\^2\+\(1-\alpha\)\^2/k",
        "false 'collapse iff independent' equivalence": r"collapse\s*\\Longleftrightarrow\s*independent",
        "false all-five-seeds claim": r"every cell uses (?:at least|>=) ?5 seeds",
        "zero-cost estimation claim": r"costs nothing extra",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, tex, re.IGNORECASE):
            blockers.append("body still contains %s" % label)
    if not blockers:
        passed.append("critical-error expression scan")

    hard_patterns = [r"^!", r"Undefined control sequence",
                     r"LaTeX Warning: (?:Reference|Citation) `[^']+' .* undefined",
                     r"Overfull \\hbox"]
    hard_hits = [p for p in hard_patterns if re.search(p, build_log, re.MULTILINE)]
    if hard_hits:
        blockers.append("LaTeX log still has hard errors, undefined references, or overfull")
    elif build_log:
        passed.append("LaTeX log has no hard errors, undefined references, or overfull")
    else:
        warnings.append("main.log has not been generated yet")

    pdf_path = os.path.join(PAPER, "main.pdf")
    if os.path.isfile(pdf_path):
        pages = pdf_page_count(pdf_path)
        refs_page = first_page_containing(pdf_path, "References", pages)
        if refs_page is None:
            blockers.append("References not located in PDF")
        else:
            passed.append("PDF has %d pages; References first seen on page %d"
                          % (pages, refs_page))

        # The limit binds on the numbered main text, not on where References begin.
        # The AI-use, ethics and reproducibility statements sit between the two and
        # are exempt from the limit, so a build may legitimately start References on
        # page 10 -- an earlier version of this audit accepted exactly that while
        # section 8 was still typeset on page 10. Anchor on the page where the last
        # numbered section heading appears, searching only up to the bibliography so
        # that a phrase recurring in the appendix cannot shadow the real location.
        # Only the numbered sections before the bibliography are main text; the
        # appendix also uses \section, and must not be mistaken for the end.
        main_tex = tex.split("\\bibliography{")[0]
        numbered = [ln[len("\\section{"):-1] for ln in main_tex.splitlines()
                    if ln.startswith("\\section{") and ln.endswith("}")]
        last_title = numbered[-1] if numbered else None
        search_hi = refs_page if refs_page else pages
        end_page = (first_page_containing(pdf_path, last_title, pages, up_to=search_hi)
                    if last_title else None)
        if last_title is None:
            blockers.append("no numbered \\section found in main.tex")
        elif end_page is None:
            blockers.append("last main-text section not located in PDF: %r" % last_title)
        elif end_page > MAIN_TEXT_PAGE_LIMIT:
            blockers.append(
                "main text is not within the %d-page limit: last numbered section %r "
                "is typeset on page %d" % (MAIN_TEXT_PAGE_LIMIT, last_title, end_page))
        else:
            passed.append("main text ends by page %d (last numbered section %r on page %d)"
                          % (MAIN_TEXT_PAGE_LIMIT, last_title, end_page))
        if end_page is not None and refs_page is not None and refs_page < end_page:
            blockers.append("References start on page %d, before the main text ends on page %d"
                            % (refs_page, end_page))
    else:
        blockers.append("paper/main.pdf does not exist")

    for marker in ("AI use statement", "Reproducibility statement"):
        if marker not in tex:
            blockers.append("missing %s" % marker)
    if all(m in tex for m in ("AI use statement", "Reproducibility statement")):
        passed.append("AI use and reproducibility statements present")

    expected_seed_counts = {
        "method_map.json": 5,
        "loss_ladder.json": 5,
        "chamfer_pooled.json": 5,
        "mnist_collapse.json": 10,
        "method_map_chamfer.json": 5,
    }
    for name, expected in expected_seed_counts.items():
        report = load_json(name)
        count = len(report.get("per_seed", {}))
        if count != expected:
            blockers.append("%s has %d per-seed records, expected %d" % (name, count, expected))
        provenance = report.get("provenance", {})
        if not report.get("result_schema_version") or not provenance.get("protocol_id"):
            message = "%s missing schema/code fingerprint; cannot prove all seeds come from the same version" % name
            (warnings if args.allow_legacy_results else blockers).append(message)

    ladder = load_json("loss_ladder.json")
    for method, stored in ladder["summary"].items():
        values = [row[method]["rho"] for row in ladder["per_seed"].values()]
        if not np.isclose(np.mean(values), stored["rho"], rtol=0, atol=1e-12):
            blockers.append("loss_ladder.json: %s summary inconsistent with per-seed values" % method)
    passed.append("loss_ladder summary reproducible from per-seed values")

    for name in ("fig_impossibility.pdf", "fig_loss_ladder.pdf"):
        path = os.path.join(ROOT, "figures", name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            blockers.append("missing figure file %s" % name)
    if not any("figure file" in item for item in blockers):
        passed.append("both main-text figures exist and are non-empty")

    print("\n".join("PASS  " + item for item in passed))
    print("\n".join("WARN  " + item for item in warnings))
    print("\n".join("BLOCK " + item for item in blockers))
    print("VERDICT: %s" % ("BLOCKED" if blockers else "PASS_WITH_WARNINGS" if warnings else "PASS"))
    return 2 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
