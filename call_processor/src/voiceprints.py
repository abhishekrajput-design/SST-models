"""Utilities for resolving and reporting enrolled agent voiceprints."""
from __future__ import annotations

import json
import ntpath
import os
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VOICEPRINT_DIR = os.path.join(DATA_DIR, "agent_voiceprints")
AGENTS_INDEX_PATH = os.path.join(VOICEPRINT_DIR, "agents.json")
LEGACY_ENROLLED_PATH = os.path.join(DATA_DIR, "enrolled_agent.npy")
LEGACY_NAME_PATH = os.path.join(DATA_DIR, "enrolled_agent_name.txt")


def voiceprint_basename(path: str | None) -> str:
    raw = str(path or "").strip()
    return ntpath.basename(raw)


def resolve_voiceprint_path(raw_path: str | None, index_path: str | None = None) -> str:
    """Resolve a voiceprint path across machines.

    Older agents.json files may contain absolute Windows paths. On Linux/live,
    the .npy files are expected beside agents.json, so fall back to that basename
    when the stored absolute path does not exist.
    """
    raw = str(raw_path or "").strip()
    if not raw:
        return ""

    if os.path.isfile(raw):
        return raw

    base_dir = os.path.dirname(index_path or AGENTS_INDEX_PATH)
    candidates = []
    if not os.path.isabs(raw):
        candidates.extend([
            os.path.join(PROJECT_ROOT, raw),
            os.path.join(os.getcwd(), raw),
        ])
    base_name = voiceprint_basename(raw)
    candidates.append(os.path.join(base_dir, base_name))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return raw


def load_agents_index(index_path: str | None = None) -> Dict[str, Any]:
    path = index_path or AGENTS_INDEX_PATH
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def voiceprint_inventory(index_path: str | None = None) -> Dict[str, Any]:
    path = index_path or AGENTS_INDEX_PATH
    agents = load_agents_index(path)
    dims: Dict[str, int] = {}
    names = []
    usable = 0
    missing = 0

    for slug, info in agents.items():
        if not isinstance(info, dict):
            continue
        vp_path = resolve_voiceprint_path(
            info.get("voiceprint_path") or info.get("voiceprint") or info.get("path"),
            path,
        )
        if not os.path.isfile(vp_path):
            missing += 1
            continue
        try:
            vp = np.load(vp_path).squeeze()
        except Exception:
            missing += 1
            continue
        if vp.ndim != 1:
            missing += 1
            continue
        dim = str(int(vp.shape[0]))
        dims[dim] = dims.get(dim, 0) + 1
        usable += 1
        names.append(str(info.get("agent_name") or info.get("name") or slug))

    legacy = os.path.isfile(LEGACY_ENROLLED_PATH)
    legacy_name = ""
    if os.path.isfile(LEGACY_NAME_PATH):
        try:
            with open(LEGACY_NAME_PATH, "r", encoding="utf-8") as fh:
                legacy_name = fh.read().strip()
        except Exception:
            legacy_name = ""

    return {
        "agents_index": path,
        "agent_count": usable,
        "missing_count": missing,
        "voiceprint_dims": dims,
        "agent_names": sorted(names),
        "legacy_enrolled": legacy,
        "legacy_agent_name": legacy_name,
        "enrolled": usable > 0 or legacy,
    }
