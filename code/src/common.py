# -*- coding: utf-8 -*-
"""公共工具：随机种子、计时、结果落盘。

约定（写作与实验共用，保证论文中每个数字都能回溯）：
  - 所有随机性由 seed 决定：python / numpy / torch 全部播种；
  - 每个实验单元的结果写成一个 json（results/*.json），字段自解释；
  - 不做任何"事后挑选"：所有种子、所有单元的结果全部落盘，
    汇报时只做"先固定汇报配置、再跑重复种子"的合法流程。
"""
import json
import os
import random
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
RESULTS_LEGACY = os.path.join(ROOT, "code", "results")   # 早期一次路径 bug 的落盘位置
FIGDIR = os.path.join(ROOT, "figures")
TABDIR = os.path.join(ROOT, "tables")
LOGDIR = os.path.join(ROOT, "logs")

for _d in (RESULTS, FIGDIR, TABDIR, LOGDIR):
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_SCHEMA_VERSION = 2
DISCRIMINANT_PROTOCOL = "validation-d-v2+gaussian-ch-lambda-v2"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng(seed):
    return np.random.default_rng(seed)


class Timer(object):
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0


def save_result(name, obj):
    # 结果文件必须携带科学口径版本。代码改过而旧 JSON 仍可被汇总，是比运行报错
    # 更危险的失败方式；下游脚本会拒绝协议不匹配的历史结果。
    obj["result_schema_version"] = RESULT_SCHEMA_VERSION
    obj["discriminant_protocol"] = DISCRIMINANT_PROTOCOL
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


def _scan(d):
    out = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
            try:
                out.append(json.load(f))
            except Exception:
                pass
    return out


def load_results(prefix=""):
    out = []
    for d in (RESULTS, RESULTS_LEGACY):
        for r in _scan(d):
            if prefix and not str(r.get("tag", "")).startswith(prefix):
                # 兼容：按文件名前缀过滤在调用侧做，这里按内容不过滤
                pass
            out.append(r)
    return out


def load_result_files(prefix=""):
    """按**文件名**前缀读取（更可靠），返回 [(名字, 内容), ...]。"""
    out = []
    for d in (RESULTS, RESULTS_LEGACY):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json") or not fn.startswith(prefix):
                continue
            with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                try:
                    out.append((fn[:-5], json.load(f)))
                except Exception:
                    pass
    return out


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def bootstrap_ci(xs, n_boot=2000, alpha=0.05, seed=0):
    """对均值做自助置信区间（论文中所有误差棒/区间都用同一个函数）。"""
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), float(xs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    boots = xs[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi
