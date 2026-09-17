# -*- coding: utf-8 -*-
"""投稿前独立校核：论文、结果、图和代码指纹。

本脚本不训练模型，也不把旧结果补写成“可追溯”。缺少来源元数据时会明确
给出 BLOCK，避免把数值内部一致误当作同一代码版本下的完整复现。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.join(ROOT, "paper")
RESULTS = os.path.join(ROOT, "results")
PDFINFO = r"E:\texlive\bin\windows\pdfinfo.exe"
PDFTOTEXT = r"E:\texlive\bin\windows\pdftotext.exe"


def load_json(name: str) -> dict:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def pdf_page_count(path: str) -> int:
    out = subprocess.check_output([PDFINFO, path], text=True, encoding="utf-8",
                                  errors="replace")
    match = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo 未返回页数")
    return int(match.group(1))


def first_page_containing(path: str, needle: str, pages: int) -> int | None:
    # 小型大写标题被 pdftotext 抽成 "R EFERENCES"（字距成空格），且长标题可能
    # 被排版换行拆开——比较前把双方所有空白去掉，避免整页误判。
    key = re.sub(r"\s+", "", needle).casefold()
    for page in range(1, pages + 1):
        out = subprocess.check_output(
            [PDFTOTEXT, "-f", str(page), "-l", str(page), path, "-"],
            text=True, encoding="utf-8", errors="replace")
        if key in re.sub(r"\s+", "", out).casefold():
            return page
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy-results", action="store_true",
                        help="仅供排版检查；不会把旧结果升级成投稿证据")
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
        "旧 kNN 修正式": r"\alpha\^2\+\(1-\alpha\)\^2/k",
        "错误耦合充要条件": r"collapse\s*\\Longleftrightarrow\s*independent",
        "虚假全五种子声明": r"every cell uses (?:at least|>=) ?5 seeds",
        "零成本估计声明": r"costs nothing extra",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, tex, re.IGNORECASE):
            blockers.append("正文仍含%s" % label)
    if not blockers:
        passed.append("关键错误表述扫描")

    hard_patterns = [r"^!", r"Undefined control sequence",
                     r"LaTeX Warning: (?:Reference|Citation) `[^']+' .* undefined",
                     r"Overfull \\hbox"]
    hard_hits = [p for p in hard_patterns if re.search(p, build_log, re.MULTILINE)]
    if hard_hits:
        blockers.append("LaTeX 日志仍有硬错误、未定义引用或 overfull")
    elif build_log:
        passed.append("LaTeX 日志无硬错误、未定义引用和 overfull")
    else:
        warnings.append("尚未生成 main.log")

    pdf_path = os.path.join(PAPER, "main.pdf")
    if os.path.isfile(pdf_path):
        pages = pdf_page_count(pdf_path)
        refs_page = first_page_containing(pdf_path, "References", pages)
        if refs_page is None:
            blockers.append("PDF 中未定位到 References")
        elif refs_page > 10:
            blockers.append("参考文献从第 %d 页开始，正文超过 9 页" % refs_page)
        else:
            passed.append("PDF 共 %d 页；参考文献首见第 %d 页" % (pages, refs_page))
    else:
        blockers.append("paper/main.pdf 不存在")

    for marker in ("AI use statement", "Reproducibility statement"):
        if marker not in tex:
            blockers.append("缺少 %s" % marker)
    if all(m in tex for m in ("AI use statement", "Reproducibility statement")):
        passed.append("AI 使用与可复现性声明齐全")

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
            blockers.append("%s 的逐种子记录为 %d，期望 %d" % (name, count, expected))
        provenance = report.get("provenance", {})
        if not report.get("result_schema_version") or not provenance.get("protocol_id"):
            message = "%s 缺少 schema/code fingerprint；不能证明全部种子来自同一版本" % name
            (warnings if args.allow_legacy_results else blockers).append(message)

    ladder = load_json("loss_ladder.json")
    for method, stored in ladder["summary"].items():
        values = [row[method]["rho"] for row in ladder["per_seed"].values()]
        if not np.isclose(np.mean(values), stored["rho"], rtol=0, atol=1e-12):
            blockers.append("loss_ladder.json 中 %s 汇总与逐种子值不一致" % method)
    passed.append("loss_ladder 汇总可由逐种子值重算")

    for name in ("fig_impossibility.pdf", "fig_loss_ladder.pdf"):
        path = os.path.join(ROOT, "figures", name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            blockers.append("缺少图文件 %s" % name)
    if not any("图文件" in item for item in blockers):
        passed.append("两张正文图存在且非空")

    print("\n".join("PASS  " + item for item in passed))
    print("\n".join("WARN  " + item for item in warnings))
    print("\n".join("BLOCK " + item for item in blockers))
    print("VERDICT: %s" % ("BLOCKED" if blockers else "PASS_WITH_WARNINGS" if warnings else "PASS"))
    return 2 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
