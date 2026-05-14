"""Append a desk-recording voiceprint from a verified UI result.

This is intentionally conservative: it only trains from speaker clusters where
the target agent is already the top acoustic match with a clear margin. Mixed
or background clusters are skipped so a bad UI result does not poison the
agent's enrollment.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

CALL_PROCESSOR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CALL_PROCESSOR_DIR.parent
sys.path.insert(0, str(CALL_PROCESSOR_DIR))

from src.diar_multi import _load_voiceprints  # noqa: E402
from src.embedding_campp import get_model  # noqa: E402

VP_DIR = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
AGENTS_JSON = VP_DIR / "agents.json"


def _resolve_result_path(raw: str) -> Path:
    path = Path(raw)
    candidates = [
        path,
        REPO_ROOT / path,
        CALL_PROCESSOR_DIR / path,
    ]
    if path.name != "result.json":
        candidates.append(CALL_PROCESSOR_DIR / "data" / "processed" / path / "result.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def _resolve_audio_path(result: dict[str, Any], result_path: Path) -> Path:
    for key in ("diarization_audio_file", "asr_audio_file", "playback_audio_file", "audio_file"):
        raw = str(result.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        candidates = [
            path,
            REPO_ROOT / path,
            CALL_PROCESSOR_DIR / path,
            result_path.parent / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError("No usable audio path found in result.json")


def _speaker_match_gate(
    cluster_report: dict[str, Any],
    speaker: str,
    agent_slug: str,
    min_similarity: float,
    min_margin: float,
    allow_target_only: bool,
) -> dict[str, Any]:
    matches = cluster_report.get(speaker) or {}
    if not isinstance(matches, dict) or agent_slug not in matches:
        return {"ok": False, "reason": "missing_target_match"}
    scored = sorted(
        (
            (slug, float((data or {}).get("similarity") or 0.0))
            for slug, data in matches.items()
            if isinstance(data, dict)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if not scored:
        return {"ok": False, "reason": "empty_match_table"}
    top_slug, _ = scored[0]
    target_sim = float((matches.get(agent_slug) or {}).get("similarity") or 0.0)
    other_scores = [sim for slug, sim in scored if slug != agent_slug]
    if not other_scores and not allow_target_only:
        return {
            "ok": False,
            "top_slug": top_slug,
            "target_similarity": round(target_sim, 4),
            "best_other_similarity": 0.0,
            "margin": 0.0,
            "reason": "target_only_match_table",
        }
    best_other = max(other_scores, default=0.0)
    margin = target_sim - best_other
    ok = top_slug == agent_slug and target_sim >= min_similarity and margin >= min_margin
    return {
        "ok": ok,
        "top_slug": top_slug,
        "target_similarity": round(target_sim, 4),
        "best_other_similarity": round(best_other, 4),
        "margin": round(margin, 4),
        "reason": "" if ok else "not_clear_target_cluster",
    }


def _segment_ok(seg: dict[str, Any], min_seconds: float) -> bool:
    seconds = max(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 0.0)
    if seconds < min_seconds:
        return False
    words = str(seg.get("text") or "").split()
    if len(words) < 4:
        return False
    return True


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _validate_against_agents(agent_slug: str, centroid: np.ndarray) -> dict[str, Any]:
    voiceprints = _load_voiceprints(str(AGENTS_JSON))
    scored = []
    for slug, (name, stack) in voiceprints.items():
        if stack.ndim != 2 or stack.shape[1] != centroid.shape[0]:
            continue
        sim = float(np.max(stack @ centroid)) if len(stack) else 0.0
        scored.append({"slug": slug, "name": name, "similarity": round(sim, 4)})
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    target = next((item for item in scored if item["slug"] == agent_slug), None)
    best_other = next((item for item in scored if item["slug"] != agent_slug), None)
    margin = (target["similarity"] if target else 0.0) - (best_other["similarity"] if best_other else 0.0)
    return {
        "target": target,
        "best_other": best_other,
        "margin": round(float(margin), 4),
        "top5": scored[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a verified desk voiceprint from a UI result")
    parser.add_argument("--result", required=True, help="result.json path or result id")
    parser.add_argument("--agent-slug", required=True)
    parser.add_argument("--min-cluster-similarity", type=float, default=0.72)
    parser.add_argument("--min-cluster-margin", type=float, default=0.18)
    parser.add_argument("--min-segment-seconds", type=float, default=1.5)
    parser.add_argument("--max-segments", type=int, default=80)
    parser.add_argument("--activation-margin", type=float, default=0.08)
    parser.add_argument("--source", default="desk_result_verified_segment_voiceprint")
    parser.add_argument("--allow-target-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result_path = _resolve_result_path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    audio_path = _resolve_audio_path(result, result_path)
    agents = json.loads(AGENTS_JSON.read_text(encoding="utf-8"))
    agent_info = agents.get(args.agent_slug)
    if not isinstance(agent_info, dict):
        raise RuntimeError(f"agent not found in agents.json: {args.agent_slug}")

    cluster_report = result.get("speaker_id_cluster_report") or result.get("cluster_match_table") or {}
    accepted_speakers = {}
    for speaker in sorted({str(seg.get("speaker") or "") for seg in result.get("segments") or []}):
        gate = _speaker_match_gate(
            cluster_report,
            speaker,
            args.agent_slug,
            args.min_cluster_similarity,
            args.min_cluster_margin,
            args.allow_target_only,
        )
        if gate["ok"]:
            accepted_speakers[speaker] = gate

    if not accepted_speakers:
        raise RuntimeError("no clear target speaker clusters passed the voiceprint gates")

    candidate_segments = [
        seg for seg in result.get("segments") or []
        if str(seg.get("speaker") or "") in accepted_speakers
        and _segment_ok(seg, args.min_segment_seconds)
    ]
    candidate_segments.sort(
        key=lambda seg: max(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 0.0),
        reverse=True,
    )
    candidate_segments = candidate_segments[: args.max_segments]
    if len(candidate_segments) < 3:
        raise RuntimeError(f"not enough clean segments to train: {len(candidate_segments)}")

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    embedder = get_model(force_cpu=True)
    embeddings = []
    used = []
    for seg in candidate_segments:
        start = max(int(float(seg["start"]) * sr), 0)
        end = min(int(float(seg["end"]) * sr), len(audio))
        if end <= start:
            continue
        emb = embedder.embed_chunk(audio[start:end], sr=sr)
        if emb is None or not np.isfinite(emb).all():
            continue
        embeddings.append(_normalize(np.asarray(emb, dtype=np.float32)))
        used.append({
            "speaker": seg.get("speaker"),
            "start": round(float(seg.get("start", 0.0)), 2),
            "end": round(float(seg.get("end", 0.0)), 2),
            "seconds": round(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 2),
            "text": str(seg.get("text") or "")[:180],
        })

    if len(embeddings) < 3:
        raise RuntimeError(f"not enough embeddings to train: {len(embeddings)}")

    centroid = _normalize(np.mean(np.stack(embeddings), axis=0).astype(np.float32))
    validation = _validate_against_agents(args.agent_slug, centroid)
    target = validation.get("target") or {}
    best_other = validation.get("best_other") or {}
    if target.get("slug") != args.agent_slug or validation["margin"] < args.activation_margin:
        raise RuntimeError(
            "new voiceprint failed activation validation: "
            f"target={target} best_other={best_other} margin={validation['margin']}"
        )

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    source_result_id = result_path.parent.name
    file_name = f"{args.agent_slug}_desk_verified_{timestamp}.npy"
    report = {
        "agent_slug": args.agent_slug,
        "agent_name": agent_info.get("agent_name") or agent_info.get("name") or args.agent_slug,
        "result": str(result_path),
        "source_result_id": source_result_id,
        "audio": str(audio_path),
        "voiceprint_file": file_name,
        "accepted_speakers": accepted_speakers,
        "segments_used": used,
        "embedding_count": len(embeddings),
        "validation": validation,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    backup = VP_DIR / f"agents.backup.{args.agent_slug}.desk_verified.{timestamp}.json"
    shutil.copy2(AGENTS_JSON, backup)
    np.save(VP_DIR / file_name, centroid.astype(np.float32))

    voiceprints = agent_info.setdefault("voiceprints", [])
    voiceprints.append({
        "path": file_name,
        "source": args.source,
        "embedding_model": "cam++",
        "embedding_dim": int(centroid.shape[0]),
        "source_result": source_result_id,
        "source_result_file": result_path.name,
        "source_audio": audio_path.name,
        "source_speakers": sorted(accepted_speakers),
        "source_segment_count": len(used),
        "validation_target_similarity": target.get("similarity"),
        "validation_best_other_slug": best_other.get("slug"),
        "validation_best_other_similarity": best_other.get("similarity"),
        "validation_margin": validation["margin"],
        "created_at": now.isoformat(),
    })
    agent_info["voiceprint_path"] = agent_info.get("voiceprint_path") or file_name
    agent_info["n_voiceprints"] = len(voiceprints)
    agent_info["embedding_model"] = "cam++"
    agent_info["updated_at"] = now.isoformat()

    AGENTS_JSON.write_text(json.dumps(agents, indent=2) + "\n", encoding="utf-8")
    report_path = VP_DIR / f"{args.agent_slug}_desk_verified_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "agents_backup": str(backup), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
