# -*- coding: utf-8 -*-
"""算出每条预注册判据的「余量 / 种子噪声」比，输出 Markdown 表格。

为什么要这个脚本
----------------
"只有 3（或 5）个种子"这个批评的实质是：**判据的余量是否被种子噪声淹没**。
定量的问法是：观测值与阈值之间隔了多少个种子的标准差？隔得越多，再加种子就越不可能翻案。
按纪律（D8）不做心算，一律用脚本算，且直接读 results/*.json 的具名字段。

用法： python margin_vs_noise.py [out.md]
"""
import json
import os
import sys

ROOT = None
d = os.path.dirname(os.path.abspath(__file__))
while d != os.path.dirname(d):
    if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
        ROOT = d
        break
    d = os.path.dirname(d)
if ROOT is None:
    raise SystemExit("project root not found")


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ratios(margin, sigma):
    if sigma is None or sigma <= 0:
        return "inf（种子间方差为 0）"
    return "%.0f×" % (abs(margin) / sigma)


def main():
    rows = []
    ll = load(os.path.join("results", "loss_ladder.json"))
    if ll:
        n = len(ll["params"]["seeds"])
        s = ll["summary"]
        C = {"B1_l2_collapses": 0.05, "B2_chamfer_escapes": 0.5,
             "B3_balanced_escapes": 0.5, "B5_balanced_more_faithful": None}
        rows.append(("B1 L1 塌缩", "ρ<0.05", "%.5f" % s["onestep_l2"]["rho"],
                     "%.5f" % s["onestep_l2"]["rho_std"], n,
                     ratios(0.05 - s["onestep_l2"]["rho"], s["onestep_l2"]["rho_std"])))
        rows.append(("B2 池化/逐条件 Chamfer 逃逸", "ρ>0.5 且覆盖≥6",
                     "%.4f" % s["onestep_chamfer"]["rho"], "%.5f" % s["onestep_chamfer"]["rho_std"], n,
                     ratios(s["onestep_chamfer"]["rho"] - 0.5, s["onestep_chamfer"]["rho_std"])))
        rows.append(("B3 平衡指派逃逸", "ρ>0.5 且覆盖≥6",
                     "%.4f" % s["onestep_balanced"]["rho"], "%.5f" % s["onestep_balanced"]["rho_std"], n,
                     ratios(s["onestep_balanced"]["rho"] - 0.5, s["onestep_balanced"]["rho_std"])))
        # B4 判负：看"逐种子分离"而不是阈值余量
        per = [ll["per_seed"][k]["onestep_chamfer"]["rho_abs_err"] for k in ll["per_seed"]]
        bal = [ll["per_seed"][k]["onestep_balanced"]["rho_abs_err"] for k in ll["per_seed"]]
        sep = "逐种子完全分离：Chamfer 最大 %.4f < balanced 最小 %.4f" % (max(per), min(bal))
        rows.append(("B4 平衡降权重误差", "Chamfer<balanced", "%.4f vs %.4f"
                     % (s["onestep_chamfer"]["rho_abs_err"], s["onestep_balanced"]["rho_abs_err"]),
                     "—", n, "**判负**；" + sep))
        d_csw = s["onestep_balanced"]["csw"] - s["onestep_chamfer"]["csw"]
        comb = (s["onestep_chamfer"]["csw_std"] ** 2 + s["onestep_balanced"]["csw_std"] ** 2) ** 0.5
        rows.append(("B5 平衡更保真 (cSW)", "balanced<Chamfer",
                     "%.4f vs %.4f" % (s["onestep_balanced"]["csw"], s["onestep_chamfer"]["csw"]),
                     "合并 σ=%.4f（%.1fσ 分离）" % (comb, abs(d_csw) / comb), n,
                     "%s（判据取方向）" % ("成立" if d_csw < 0 else "不成立")))
    mm = load(os.path.join("results", "method_map.json"))
    if mm:
        n = len(mm["params"]["seeds"])
        g = mm["summary"]
        rows.append(("C1 独立耦合@1 塌缩", "ρ<0.05", "%.5f" % g["cfm_indep@1"]["rho"],
                     "%.5f" % g["cfm_indep@1"]["rho_std"], n,
                     ratios(0.05 - g["cfm_indep@1"]["rho"], g["cfm_indep@1"]["rho_std"])))
        rows.append(("C2 同网络@32 恢复", "ρ>0.75", "%.4f" % g["cfm_indep@32"]["rho"],
                     "%.5f" % g["cfm_indep@32"]["rho_std"], n,
                     ratios(g["cfm_indep@32"]["rho"] - 0.75, g["cfm_indep@32"]["rho_std"])))
        rows.append(("C3 OT 一步恢复", "ρ>0.50", "%.4f" % g["cfm_ot@1"]["rho"],
                     "%.5f" % g["cfm_ot@1"]["rho_std"], n,
                     ratios(g["cfm_ot@1"]["rho"] - 0.50, g["cfm_ot@1"]["rho_std"])))
        rows.append(("C4 reflow 一步恢复", "ρ>0.50", "%.4f" % g["reflow@1"]["rho"],
                     "%.5f" % g["reflow@1"]["rho_std"], n,
                     ratios(g["reflow@1"]["rho"] - 0.50, g["reflow@1"]["rho_std"])))
        rows.append(("C5 蒸馏一步恢复", "ρ>0.50", "%.4f" % g["distill@1"]["rho"],
                     "%.5f" % g["distill@1"]["rho_std"], n,
                     ratios(g["distill@1"]["rho"] - 0.50, g["distill@1"]["rho_std"])))
        rows.append(("C6 逐点 L2 地板", "ρ<0.05", "%.5f" % g["onestep_l2@1"]["rho"],
                     "%.5f" % g["onestep_l2@1"]["rho_std"], n,
                     ratios(0.05 - g["onestep_l2@1"]["rho"], g["onestep_l2@1"]["rho_std"])))
        rows.append(("C7 min-of-M 逃逸", "ρ>0.20", "%.4f" % g["onestep_minM@1"]["rho"],
                     "%.5f" % g["onestep_minM@1"]["rho_std"], n,
                     ratios(g["onestep_minM@1"]["rho"] - 0.20, g["onestep_minM@1"]["rho_std"])))

    lines = ["| 判据 | 阈值（方向） | 观测（5 种子均值） | 种子 σ | 余量/σ |",
             "|---|---|---|---|---|"]
    for name, thr, obs, sd, n, r in rows:
        lines.append("| %s | %s | %s | %s | %s |" % (name, thr, obs, sd, r))
    out = "\n".join(lines)
    print(out)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print("\nwrote %s" % sys.argv[1])


if __name__ == "__main__":
    main()
