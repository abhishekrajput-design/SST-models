"""Import a voiceprint bundle on a live server."""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

from src.voiceprints import (
    AGENTS_INDEX_PATH,
    PROJECT_ROOT,
    voiceprint_basename,
    voiceprint_inventory,
)


def _safe_members(tar: tarfile.TarFile):
    root = Path(PROJECT_ROOT).resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe tar member path: {member.name}")
        yield member


def _sanitize_agents_json() -> int:
    if not os.path.isfile(AGENTS_INDEX_PATH):
        return 0
    with open(AGENTS_INDEX_PATH, "r", encoding="utf-8") as fh:
        agents = json.load(fh)
    if not isinstance(agents, dict):
        raise ValueError(f"Invalid agents index: {AGENTS_INDEX_PATH}")

    changed = 0
    vp_dir = Path(AGENTS_INDEX_PATH).parent
    for info in agents.values():
        if not isinstance(info, dict):
            continue
        raw = str(info.get("voiceprint_path") or info.get("voiceprint") or info.get("path") or "")
        if not raw:
            continue
        name = voiceprint_basename(raw)
        rel = f"data/agent_voiceprints/{name}"
        if (vp_dir / name).is_file() and info.get("voiceprint_path") != rel:
            info["voiceprint_path"] = rel
            info.pop("voiceprint", None)
            info.pop("path", None)
            changed += 1

    if changed:
        with open(AGENTS_INDEX_PATH, "w", encoding="utf-8") as fh:
            json.dump(agents, fh, indent=2, ensure_ascii=False)
    return changed


def import_bundle(bundle_path: Path) -> dict:
    if not bundle_path.is_file():
        raise FileNotFoundError(str(bundle_path))
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(PROJECT_ROOT, members=_safe_members(tar))
    changed = _sanitize_agents_json()
    inv = voiceprint_inventory()
    return {
        "bundle": str(bundle_path),
        "project_root": PROJECT_ROOT,
        "sanitized_agents": changed,
        "inventory": inv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import deployed agent voiceprints.")
    parser.add_argument("bundle", help="Path to voiceprints_bundle.tar.gz")
    args = parser.parse_args()
    summary = import_bundle(Path(args.bundle))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["inventory"].get("enrolled"):
        raise SystemExit("No usable voiceprints were imported")


if __name__ == "__main__":
    main()
