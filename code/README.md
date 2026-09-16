# code/ 目录说明

本目录是论文《Whether One Step Is Enough Is Not Determined by the Data: Coupling,
Loss Form, and an Impossibility Theorem》（ICLR 2027 投稿）的全部代码。

**产物不在本目录**：实验原始结果、图、运行日志都写在项目根的同名目录下
（`../results`、`../figures`、`../logs`）。

---

## 一、当前结构（2026-09-16 第三轮重写后）

```
code/
├── README.md               ← 本文件
├── requirements.txt / requirements-lock.txt
├── _run_pipeline.py        唯一入口：从已有结果重建宏、图与论文（不跑训练）
│
├── experiments/            ① 跑实验 → ../results/*.json
│   ├── run_method_map.py       跨方法族失效地图（8 方法族 × NFE∈{1,32} × 5 种子）
│   │                           同时被下面两个脚本 import，勿移走
│   ├── run_chamfer_supplement.py  表 1：第 7 族双向 Chamfer（**整批池化**指派）× 5 种子
│   ├── run_loss_ladder.py      损失阶梯 L1 / L2b（逐条件 Chamfer）/ L3（逐条件平衡）× 5 种子
│   └── run_chamfer_pooled.py   损失阶梯的池化对照 L2a（整批 argmin）× 5 种子
│                               四个脚本都支持 `--seeds a,b --merge`（续跑、不重算已算种子）
│
├── analysis/              ② 验证、汇总与检查
│   ├── verify_multistep_theory.py     多步欧拉（闭式精确速度场）→ logs/*.json
│   ├── verify_conditional_theory.py   条件受控：同 D 不同耦合
│   ├── verify_coupling_theory.py      耦合对照的最早一版（保留作旁证）
│   ├── verify_impossibility.py        **不可能性定理·构造版**（α 族，ρ*(α)=α²）
│   ├── verify_minimax.py              **不可能性定理·最小最大版**（7 个数据侧统计量）
│   ├── validate_deficit.py            切片 W₂ 高斯亏损 Λ 的估计量与判别力（含 R6 负面结果）
│   ├── run_falsification.py           Λ 的预注册否决实验（假阳性率）
│   ├── margin_vs_noise.py             算每条判据的「余量/种子噪声」比（证据强度审计 §2.4）
│   ├── make_theory_macros.py          logs/ + results/ → ../paper/numbers_theory.tex
│   ├── make_fig_impossibility.py      → ../figures/fig_impossibility.pdf（论文图 1）
│   ├── make_fig_ladder.py             → ../figures/fig_loss_ladder.pdf（论文图 2）
│   ├── check_macros.py                编译前置：宏缺失 / 非法宏名 / 死宏（--strict 供流水线用）
│   └── build_paper.py                 四遍编译 + 错误汇总（页数 / overfull / 未定义引用）
│
├── src/                   ③ 库
│   ├── deficit.py          **在用**：切片 W₂ 高斯亏损 Λ（纯 numpy，含 Acklam 反正态 CDF）
│   ├── common.py           \
│   ├── tasks.py             |  旧版论文的任务/流/度量库；当前流水线只用 deficit.py，
│   ├── flows.py             |  其余四个保留作参考，被 code/archive/stale_protocol/ 里的
│   ├── metrics.py           |  旧实验脚本使用
│   └── discriminant.py     /
│
├── tests/                 ④ 回归测试
│   └── test_audit_guards.py    审计期间的护栏测试（估计器偏置、条件度量口径）
│
├── data/                  数据缓存目录（当前流水线**不使用**：全部实验为合成族，无外部下载）
└── archive/               ⑤ 留档，**不参与流水线**
    ├── stale_protocol/        旧协议实验（run_toy/rl/mnist/ablations）——结果已作废，
    │                          保留以便日后改写成新协议时复用数据生成代码
    ├── _reorg_code_tree.py    目录整理脚本（一次性，已执行）
    ├── _probe_mnist.py        边缘 vs 条件统计量的逐档比对（历史取证）
    ├── _probe_minm_degeneracy.py  min-of-M 旧实现与点式 L2 逐比特等价的取证
    ├── apply_macro_alias.py   含数字宏名 → 纯字母宏名（含 --check 干跑）
    ├── compress_main{,2,3}.py 把正文压到 9 页的搬移脚本（旧版）
    └── rewrite_{abl,claims,mnist_conv}.py  已应用完毕的正文改写脚本
```

被移到项目根 `_archive_20260916/` 的脚本（旧版流水线，产物已不被正文引用）：
`make_tables.py`、`make_figures.py`、`make_vis.py`、`audit_format.py`、
`_final_check.py`、`_verify_estimator_bias.py`、`_verify_kappa.py`。

---

## 二、怎么跑

```bash
PY=E:/anaconda/python.exe     # 需 numpy / scipy / matplotlib / torch(cpu)

# ① 一键重建（不跑训练）：make_theory_macros → 两张图 → check_macros → build_paper
$PY -u code/_run_pipeline.py          # 日志落 logs/pipeline.log，收尾看 "PIPELINE OK"

# ② 分步跑实验（每个都会写 results/*.json）
$PY -u code/experiments/run_method_map.py          # ~75 min（CPU，14 线程；含修正后的 min-of-8）
$PY -u code/experiments/run_chamfer_supplement.py  # ~5 min（表 1 池化 Chamfer 行）
$PY -u code/experiments/run_loss_ladder.py         # ~20 min（表 2 + 图 2：L1/L2b/L3）
$PY -u code/experiments/run_chamfer_pooled.py      # ~7 min（表 2 + 图 2：L2a 池化对照）

# ③ 分步跑验证（每个都会写 logs/*.json）
$PY -u code/analysis/verify_multistep_theory.py
$PY -u code/analysis/verify_conditional_theory.py
$PY -u code/analysis/verify_impossibility.py
$PY -u code/analysis/verify_minimax.py             # ~10 min
$PY -u code/analysis/validate_deficit.py

# ④ 回归测试
$PY -m unittest discover -s code/tests -v
```

环境变量 `PIPELINE_PY` 可覆盖 `_run_pipeline.py` 使用的解释器。

---

## 三、三条约定（改动前请先读）

**1）路径锚点必须用自愈式，不要写死层数。**
所有脚本统一向上找到**同时含 `code/` 与 `paper/` 的那一级**：

```python
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_root(d):
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d

_ROOT_ = _find_root(_HERE)
_CODE_ = os.path.join(_ROOT_, "code")
```

**新增脚本请照抄这一段。** 写得死的 `dirname` 次数在目录一动就崩。

**2）`code/` 下不要再放产物目录。**
`configs/ figures/ logs/ results/ tables/` 曾经存在于 `code/` 下且全为空，真正的产物
一直在项目根——这类空目录只会让人找错地方。新增输出一律写到项目根同名目录。

**3）论文正文不许出现手写数字。**
正文所有实验数值都来自 `../paper/numbers_theory.tex`，由
`analysis/make_theory_macros.py` 从 `../logs/*.json` 与 `../results/*.json` 生成，
**每个宏的行末都带 `% 来源: 文件名.字段` 注释**。改数先改实验、再重跑流水线，
不要直接编辑 `numbers_theory.tex`（它会被覆盖）。
另有两条 LaTeX 硬约束：**宏名只能含字母**（`\RhoAlOne` 合法，`\rho1` 是致命错误）；
`\times 10^{-16}` 这类含科学计数法的值**必须处在数学模式里**，否则报 `Missing $ inserted`。

> **2026-09-16 修**：`sigfigs` 用 `%.4g`，对 `abs(v) < 1e-4` 的值会自动切科学计数法——
> 5 种子重跑后损失阶梯的 $\rho$ 从 `1.039e-4` 变成 `9.716e-5`，越过阈值，于是 Table 1
> 的**文本模式**单元格立刻编译失败。现在 `make_theory_macros.py` 把这类值输出成
> `\ensuremath{...}`，在文本模式与数学模式里都安全；表格里的数值列仍**推荐整体包进**
> 数学定界符，排版更好看。

> `check_macros.py` 会把「已定义但正文没引用」的宏列成 `[DEAD]`（当前 183 条里 74 条）。
> 这是**信息项，不判失败**：它们记录了论文没引用的中间统计量（各统计量的 ANOVA
> $F/p/\eta^2$、样本量阶梯各档信噪比、$\Lambda$ 十分布逐项值等），是可追溯性的证据。
> 只有 `[ILLEGAL]` 与 `[MISSING]` 会让 `--strict` 返回非零。

---

## 四、结果口径版本

**当前论文只引用四个结果文件**，其余均为旧协议产物、已归档：

| 文件 | 对应结论 |
|---|---|
| `../results/method_map.json` | 表 1：跨方法族失效地图 |
| `../results/method_map_chamfer.json` | 表 1 最后一行：双向 Chamfer（整批池化） |
| `../results/loss_ladder.json` | 表 2 + 图 2：L1 / L2b / L3 |
| `../results/chamfer_pooled.json` | 表 2 + 图 2：L2a 池化对照 |

`../logs/` 下的 JSON 是验证脚本的报告（`verify_*.json`、`validate_deficit.json`、
`falsification_lambda_stress.json`）。

**两条口径红线**（不要回退）：

1. **单步惩罚一律用条件统计量 cSW（条件 sliced $W_2$）判定**，不用边缘 sliced $W_2$——
   后者把"一步地板"与"模式塌缩"混在一起，会给欠拟合的场也报出惩罚。
2. **Λ 不得称为多模态检测器**。它已降级为描述性指数：实测会把单峰 banana 排到
   8 模态环之上（`validate_deficit.py` 的 R6 是**预期判负**的负面结果）。
3. **min-of-M 的 M 个候选必须来自互相独立的源噪声**。一步网络是确定性映射，
   若用 `np.repeat` 复制同一个 `x0`，M 个候选在任何训练状态下都完全相同，
   `argmin` 恒为 0，损失**逐比特等于点式 L2**（取证：`archive/_probe_minm_degeneracy.py`
   在小规模训练下测得 `max|Δw| = 0`）。旧版实现踩过这个坑，`run_method_map.py`
   的 `onestep_minM` 用 `rng.normal(size=(bs*M, DIM))` 独立抽噪声。
4. **非平衡 Chamfer 的两次 `argmin` 的"范围"必须与平衡版一致。** 论文 Thm 4(L2) 明确
   要求 `argmin` **逐条件**取；初版 L2 取**整批**，与 L3 的逐条件匈牙利差了两个变量
   （是否平衡 AND 是否分条件）。修正后 `run_loss_ladder.py::train_chamfer` 逐条件取
   `argmin`，池化版单独由 `run_chamfer_pooled.py` 复现。实测结论因此被改写：
   **压平条件展布的是"池化"，不是"非平衡"**——L2b（逐条件非平衡）基本回到对角线，
   L2a（池化）才会出现"支撑恢复、权重自由"的指纹。

旧版 MNIST 结果必须含 `schema_version=2` 与
`metric_protocol=condition-knn-v2-no-label-leakage`；旧版用真实类别标签构造"条件参考
分布"，已作废。汇总脚本遇到旧版 JSON 直接报错，防止旧数值再次进入表格或论文。
