# -*- coding: utf-8 -*-
"""抓取拟引用文献的 arXiv 元数据（带重试 + 与已有缓存合并）。

用途：为 paper/references.bib 提供**真实可核实**的条目。
产出：paper/refs_raw.json  （每条含 id / title / authors / date / subject / abstract）

复用 99_工作台/fetch_arxiv.py 的经验：
  - arXiv 官方 API 与 OpenAlex 在本机会 429，只能抓 https://arxiv.org/abs/<id> 页面；
  - 代理环境会偶发 curl(7)，必须重试（5 次 + 线性退避）；
  - 新结果必须与已有缓存合并后再写回，否则一次限流会整体覆盖已有条目。
"""
import io
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "refs_raw.json")
CURL = "curl"

# (arXiv 编号, bibkey, 备注/预期标题关键词)
TARGETS = [
    ("2006.11239", "ddpm", "Denoising Diffusion Probabilistic Models"),
    ("2010.02502", "ddim", "Denoising Diffusion Implicit Models"),
    ("2011.13456", "scoresde", "Score-Based Generative Modeling through SDEs"),
    ("2206.00364", "edm", "Elucidating the Design Space of Diffusion-Based Generative Models"),
    ("2210.02747", "flowmatching", "Flow Matching for Generative Modeling"),
    ("2209.03003", "rectifiedflow", "Flow Straight and Fast"),
    ("2302.00482", "cfm", "Conditional Flow Matching Simulation-Free Dynamic Optimal Transport"),
    ("2303.08797", "interpolants", "Stochastic Interpolants"),
    ("2303.01469", "consistencymodels", "Consistency Models"),
    ("2407.02398", "consistencyfm", "Consistency Flow Matching"),
    ("2410.12557", "shortcut", "One Step Diffusion via Shortcut Models"),
    ("2505.13447", "meanflows", "Mean Flows for One-step Generative Modeling"),
    ("2403.03206", "sd3", "Scaling Rectified Flow Transformers"),
    ("2412.06264", "fmguide", "Flow Matching Guide and Code"),
    ("2202.00512", "progressivedistill", "Progressive Distillation for Fast Sampling"),
    ("2309.06380", "instaflow", "InstaFlow"),
    ("2311.18828", "dmd", "One-step Diffusion with Distribution Matching Distillation"),
    ("2303.04137", "diffusionpolicy", "Diffusion Policy"),
    ("2208.06193", "diffusionql", "Diffusion Policies as an Expressive Policy Class"),
    ("2502.02538", "fql", "Flow Q-Learning"),
    ("2006.04779", "cql", "Conservative Q-Learning"),
    ("2110.06169", "iql", "Offline Reinforcement Learning with Implicit Q-Learning"),
    ("2108.13264", "precipice", "Deep RL at the Edge of the Statistical Precipice"),
    ("1904.06991", "prdc", "Improved Precision and Recall Metric"),
    ("1706.08500", "fid", "GANs Trained by a Two Time-Scale Update Rule"),
    ("1902.00434", "gsw", "Generalized Sliced Wasserstein Distances"),
    ("2606.05737", "c20", "condition-target structure VLA one-step"),
    ("2603.12480", "c19", "one-step flow policy self-distillation"),
    ("2606.10613", "c18", "bootstrapped flow Q one-step"),
    ("2506.11172", "c17", "sequence-level data-policy coverage"),
    ("2004.07219", "d4rl", "D4RL datasets"),
    ("2410.20015", "ogbench", "OGBench"),
    ("2306.03310", "libero", "LIBERO"),
    ("2406.09246", "openvla", "OpenVLA"),
    ("2305.18459", "consistencypolicy", "Consistency Policy"),
    ("2405.05986", "cp2", "consistency policy visuomotor"),
    ("2502.01490", "ladd", "LADD"),
    ("2410.20092", "ogbench", "OGBench offline goal-conditioned RL"),
    ("2606.22752", "onestepfm", "One-Step Flow Matching physical fields"),
    ("2602.13810", "mvp", "MVP instantaneous velocity one-step policy"),
    ("2603.05296", "lps", "LPS latent one-step flow policy"),
    ("2602.01606", "flame", "FLAME one-step flow matching MaxEnt RL"),
    ("2605.01663", "fan", "FAN flow anchoring noise-conditioned Q"),
]


def sh(args, timeout=90):
    p = subprocess.run([CURL, "-sS", "-m", str(timeout), "-A",
                        "Mozilla/5.0 (research)"] + args,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:200])
    return p.stdout.decode("utf-8", "ignore")


def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def get_meta(aid):
    last = None
    for attempt in range(5):
        try:
            html = sh(["https://arxiv.org/abs/" + aid], timeout=60)
            break
        except Exception as ex:
            last = ex
            time.sleep(3 + 3 * attempt)
    else:
        raise last

    def m(name):
        r = re.search(r'<meta name="citation_%s" content="([^"]+)"' % name, html)
        return r.group(1) if r else ""

    ab = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', html, re.S)
    abtxt = ""
    if ab:
        abtxt = clean(re.sub(r"<[^>]+>", " ", ab.group(1)))
        abtxt = re.sub(r"^Abstract:\s*", "", abtxt)
    subj = re.search(r'<span class="primary-subject">([^<]+)</span>', html)
    # 是否有已发表/会议注释
    comments = re.findall(r'<td class="tablecell comments[^"]*">(.*?)</td>', html, re.S)
    jref = re.findall(r'<td class="tablecell jref">(.*?)</td>', html, re.S)
    return dict(
        id=aid,
        title=clean(m("title")),
        authors=re.findall(r'<meta name="citation_author" content="([^"]+)"', html),
        date=clean(m("date")),
        summary=abtxt,
        subject=clean(subj.group(1)) if subj else "",
        comments=clean(re.sub(r"<[^>]+>", " ", comments[0])) if comments else "",
        jref=clean(re.sub(r"<[^>]+>", " ", jref[0])) if jref else "",
    )


def main():
    old = {}
    if os.path.exists(OUT):
        try:
            with io.open(OUT, encoding="utf-8") as f:
                for e in json.load(f):
                    old[e["id"]] = e
        except Exception:
            old = {}
    out = []
    for aid, key, expect in TARGETS:
        if aid in old and old[aid].get("title"):
            print("[cache]", aid, old[aid]["title"][:70])
            out.append(old[aid])
            continue
        try:
            meta = get_meta(aid)
        except Exception as ex:
            print("!! FAIL", aid, key, str(ex)[:80])
            continue
        meta["key"] = key
        meta["expect"] = expect
        out.append(meta)
        print("=" * 70)
        print(key, aid, "|", meta["date"], "|", meta["subject"])
        print("  TITLE:", meta["title"])
        print("  AUTH :", ", ".join(meta["authors"][:3]),
              ("等" if len(meta["authors"]) > 3 else ""))
        if meta["comments"]:
            print("  CMT  :", meta["comments"][:160])
        if meta["jref"]:
            print("  JREF :", meta["jref"][:160])
        time.sleep(1.0)
    by_id = {}
    for e in out:
        by_id[e["id"]] = e
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump([by_id[k] for k in sorted(by_id)], f, ensure_ascii=False, indent=1)
    print("\n共 %d 条，写入 %s" % (len(by_id), OUT))


if __name__ == "__main__":
    main()
