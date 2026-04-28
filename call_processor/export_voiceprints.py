"""Create a deployable voiceprint bundle without committing biometric data."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.voiceprints import (
    AGENTS_INDEX_PATH,
    DATA_DIR,
    LEGACY_ENROLLED_PATH,
    LEGACY_NAME_PATH,
    PROJECT_ROOT,
    VOICEPRINT_DIR,
    resolve_voiceprint_path,
    voiceprint_basename,
    voiceprint_inventory,
)


def _rel(path: Path) -> str:
    return path.as_posix()


def build_bundle(output_path: Path) -> dict:
    if not os.path.isfile(AGENTS_INDEX_PATH):
        raise FileNotFoundError(f"Missing agents index: {AGENTS_INDEX_PATH}")

    with open(AGENTS_INDEX_PATH, "r", encoding="utf-8") as fh:
        agents = json.load(fh)
    if not isinstance(agents, dict):
        raise ValueError("agents.json must be a JSON object")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle_voiceprints = root / "data" / "agent_voiceprints"
        bundle_voiceprints.mkdir(parents=True, exist_ok=True)

        sanitized = {}
        copied = 0
        missing = []
        for slug, info in agents.items():
            if not isinstance(info, dict):
                continue
            raw = info.get("voiceprint_path") or info.get("voiceprint") or info.get("path")
            src = resolve_voiceprint_path(raw, AGENTS_INDEX_PATH)
            if not src or not os.path.isfile(src):
                missing.append(slug)
                continue
            dst = bundle_voiceprints / voiceprint_basename(src)
            shutil.copy2(src, dst)
            copied += 1

            item = dict(info)
            item["voiceprint_path"] = _rel(Path("data") / "agent_voiceprints" / dst.name)
            item.pop("voiceprint", None)
            item.pop("path", None)
            sanitized[slug] = item

        with open(bundle_voiceprints / "agents.json", "w", encoding="utf-8") as fh:
            json.dump(sanitized, fh, indent=2, ensure_ascii=False)

        if os.path.isfile(LEGACY_ENROLLED_PATH):
            shutil.copy2(LEGACY_ENROLLED_PATH, root / "data" / "enrolled_agent.npy")
        if os.path.isfile(LEGACY_NAME_PATH):
            shutil.copy2(LEGACY_NAME_PATH, root / "data" / "enrolled_agent_name.txt")

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_project": PROJECT_ROOT,
            "agent_count": copied,
            "missing_agents": missing,
            "inventory": voiceprint_inventory(),
        }
        with open(root / "voiceprints_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(root / "data", arcname="data")
            tar.add(root / "voiceprints_manifest.json", arcname="voiceprints_manifest.json")

    return {"output": str(output_path), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export agent voiceprints for live deployment.")
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(DATA_DIR, "voiceprints_bundle.tar.gz"),
        help="Output .tar.gz path",
    )
    args = parser.parse_args()
    summary = build_bundle(Path(args.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
