# -*- coding: utf-8 -*-
"""Common utilities: random seeds, timing, and result persistence.

Conventions (shared by both writing and experiments, so every number in the
paper can be traced back):
  - All randomness is determined by seed: python / numpy / torch are all seeded;
  - Each experiment unit writes one json (results/*.json) with self-explanatory fields;
  - No "post-hoc selection": results for all seeds and all units are persisted, and
    reporting only follows the legitimate flow of "fix the reporting config first,
    then run repeated seeds".
"""
import json
import os
import random
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
RESULTS_LEGACY = os.path.join(ROOT, "code", "results")   # Disk location from an early path-bug incident
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
    # Result files must carry the scientific schema version. Code changes that
    # still let old JSON be aggregated is a more dangerous failure mode than a
    # runtime error; downstream scripts reject historical results whose protocol
    # does not match.
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
                # Compatibility: filename-prefix filtering is done on the caller
                # side; no content filtering is applied here.
                pass
            out.append(r)
    return out


def load_result_files(prefix=""):
    """Read by **filename** prefix (more robust); returns [(name, content), ...]."""
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
    """Bootstrap confidence interval for the mean (every error bar / interval in the
    paper uses this same function)."""
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), float(xs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    boots = xs[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi
