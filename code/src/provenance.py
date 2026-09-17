# -*- coding: utf-8 -*-
"""Minimal traceable metadata for experiment artifacts.

Result files used to store only summary numbers; ``--merge`` could therefore mix
seeds from different code versions. This module attaches the generating script and
the SHA-256 fingerprints of its dependencies to each artifact, and enforces a check
before merging.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import sys

RESULT_SCHEMA_VERSION = "single-step-coupling-v3"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol_fingerprint(root: str, relative_paths: list[str]) -> tuple[str, dict[str, str]]:
    """Return the joint fingerprint of the dependency files and the per-file fingerprints."""
    hashes: dict[str, str] = {}
    joint = hashlib.sha256()
    for rel in sorted(relative_paths):
        norm = rel.replace("\\", "/")
        digest = sha256_file(os.path.join(root, *norm.split("/")))
        hashes[norm] = digest
        joint.update(norm.encode("utf-8"))
        joint.update(b"\0")
        joint.update(digest.encode("ascii"))
        joint.update(b"\0")
    return joint.hexdigest(), hashes


def attach_provenance(payload: dict, root: str, generator: str,
                      protocol_files: list[str], *, merged: bool) -> dict:
    protocol_id, source_hashes = protocol_fingerprint(root, protocol_files)
    payload["result_schema_version"] = RESULT_SCHEMA_VERSION
    payload["provenance"] = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "generator": generator.replace("\\", "/"),
        "protocol_id": protocol_id,
        "source_sha256": source_hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "merged_seed_cache": bool(merged),
    }
    return payload


def require_merge_compatible(existing: dict, expected_protocol_id: str, path: str) -> None:
    """Reject merging caches of unknown origin or a different code version."""
    schema = existing.get("result_schema_version")
    got = existing.get("provenance", {}).get("protocol_id")
    if schema != RESULT_SCHEMA_VERSION or got != expected_protocol_id:
        raise RuntimeError(
            "refusing to merge result %s of unknown origin or mismatched protocol; "
            "rerun in full without --merge. expected schema=%s protocol=%s, "
            "got schema=%r protocol=%r"
            % (path, RESULT_SCHEMA_VERSION, expected_protocol_id, schema, got)
        )
