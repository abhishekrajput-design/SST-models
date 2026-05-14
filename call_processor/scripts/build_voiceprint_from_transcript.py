"""Build segment-role voiceprints from a verified speaker-labeled transcript.

The transcript is used only for enrollment. Runtime role assignment remains
voiceprint-only: each transcript segment is embedded and compared to the
selected agent's enrolled voiceprints.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

CALL_PROCESSOR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CALL_PROCESSOR_DIR.parent
sys.path.insert(0, str(CALL_PROCESSOR_DIR))

from src.diar_multi import _load_voiceprints  # noqa: E402
from src.embedding_campp import get_model, l2_norm  # noqa: E402

VP_DIR = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
AGENTS_JSON = VP_DIR / "agents.json"

_STOPWORDS = set(
    "the a an and or but so you i im is are was were to of for in on it this "
    "that with if then now right yeah yes no just can do not dont we he she "
    "they my your me our there here like know look because as at from by will "
    "would have has had be been being get go going give take want wants only"
    .split()
)


def _resolve_result_path(raw: str) -> Path:
    path = Path(raw)
    candidates = [path, REPO_ROOT / path, CALL_PROCESSOR_DIR / path]
    if path.name != "result.json":
        candidates.append(CALL_PROCESSOR_DIR / "data" / "processed" / path / "result.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def _resolve_audio_path(result: dict[str, Any], result_path: Path, explicit: str | None) -> Path:
    raw_values = []
    if explicit:
        raw_values.append(explicit)
    raw_values.extend(
        str(result.get(key) or "")
        for key in ("diarization_audio_file", "asr_audio_file", "audio_file", "playback_audio_file")
    )
    for raw in raw_values:
        raw = str(raw or "").strip()
        if not raw:
            continue
        path = Path(raw)
        candidates = [path, REPO_ROOT / path, CALL_PROCESSOR_DIR / path, result_path.parent / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError("No usable audio path found")


def _fix_mojibake(text: str) -> str:
    fixes = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "Â£": "pounds",
        "Â": "",
        "â€¦": "...",
        "\ufeff": "",
    }
    for old, new in fixes.items():
        text = text.replace(old, new)
    return text


def _parse_labeled_transcript(path: Path) -> list[dict[str, str]]:
    raw = _fix_mojibake(path.read_text(encoding="utf-8", errors="replace"))
    turns: list[dict[str, str]] = []
    label_re = re.compile(r"^([^:\n]{1,80}):\s*(.*)$")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.lower() == "conversation transcript":
            continue
        match = label_re.match(line)
        if match:
            label = match.group(1).strip()
            text = match.group(2).strip()
            turns.append({"label": label, "text": text})
        elif turns and not line.startswith("("):
            turns[-1]["text"] += " " + line
    return turns


def _norm(text: str) -> str:
    text = _fix_mojibake(text).lower().replace("£", " pounds ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alignment_score(segment_text: str, turn_text: str) -> float:
    seg = _norm(segment_text)
    turn = _norm(turn_text)
    if not seg or not turn:
        return 0.0
    if len(seg) >= 12 and seg in turn:
        return 1.0
    seg_tokens = [t for t in seg.split() if t not in _STOPWORDS]
    turn_tokens = [t for t in turn.split() if t not in _STOPWORDS]
    if not seg_tokens or not turn_tokens:
        return SequenceMatcher(None, seg, turn).ratio() * 0.7
    remaining = list(turn_tokens)
    overlap = 0
    for token in seg_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(seg_tokens)
    recall = overlap / min(len(turn_tokens), max(len(seg_tokens), 1))
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return max(f1, SequenceMatcher(None, seg, turn).ratio() * 0.8)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != 16000:
        import torchaudio.functional as F_ta

        audio = F_ta.resample(torch.from_numpy(audio.astype(np.float32)), sr, 16000).numpy()
        sr = 16000
    return audio, sr


def _embed_segment(model, audio: np.ndarray, sr: int, start_s: float, end_s: float, pad_s: float):
    pad = int(max(pad_s, 0.0) * sr)
    start = max(0, int(start_s * sr) - pad)
    end = min(len(audio), int(end_s * sr) + pad)
    if end <= start:
        return None
    emb = model.embed_chunk(audio[start:end].astype(np.float32), sr=sr)
    if emb is None or not np.isfinite(emb).all():
        return None
    return l2_norm(np.asarray(emb, dtype=np.float32))


def _evaluate(
    rows: list[dict[str, Any]],
    agent_slug: str,
    centroids: list[np.ndarray],
    threshold: float,
    margin: float,
) -> dict[str, Any]:
    voiceprints = _load_voiceprints(str(AGENTS_JSON))
    others = [
        (slug, stack)
        for slug, (_name, stack) in voiceprints.items()
        if slug != agent_slug and stack.ndim == 2 and stack.shape[1] == centroids[0].shape[0]
    ]
    correct = wrong = 0
    seconds_correct = seconds_wrong = 0.0
    for row in rows:
        emb = row.get("embedding")
        if emb is None:
            continue
        target_sim = max(float(c @ emb) for c in centroids)
        best_other = max((float(np.max(stack @ emb)) for _slug, stack in others), default=0.0)
        pred = "AGENT" if target_sim >= threshold and (target_sim - best_other) >= margin else "CUSTOMER"
        dur = max(float(row["end"]) - float(row["start"]), 0.0)
        if pred == row["expected"]:
            correct += 1
            seconds_correct += dur
        else:
            wrong += 1
            seconds_wrong += dur
    total = correct + wrong
    seconds_total = seconds_correct + seconds_wrong
    return {
        "threshold": threshold,
        "margin": margin,
        "count_accuracy": round((correct / total) * 100, 2) if total else 0.0,
        "duration_accuracy": round((seconds_correct / seconds_total) * 100, 2) if seconds_total else 0.0,
        "correct": correct,
        "wrong": wrong,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build agent voiceprints from a verified transcript")
    parser.add_argument("--result", required=True, help="result id or result.json path")
    parser.add_argument("--transcript", required=True, help="speaker-labeled transcript markdown/text")
    parser.add_argument("--agent-slug", required=True)
    parser.add_argument("--agent-label", required=True, help="label in transcript, e.g. 'Salesperson 1'")
    parser.add_argument("--audio", default=None, help="optional audio override")
    parser.add_argument("--min-align", type=float, default=0.72)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--min-centroid-segments", type=int, default=3)
    parser.add_argument("--pad-seconds", type=float, default=0.12)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument(
        "--replace-segment-role",
        action="store_true",
        help="deactivate older segment-role voiceprints for this agent before adding the new verified set",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result_path = _resolve_result_path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    audio_path = _resolve_audio_path(result, result_path, args.audio)
    turns = _parse_labeled_transcript(Path(args.transcript))
    if not turns:
        raise RuntimeError("No labeled transcript turns found")

    agents = json.loads(AGENTS_JSON.read_text(encoding="utf-8"))
    agent_info = agents.get(args.agent_slug)
    if not isinstance(agent_info, dict):
        raise RuntimeError(f"agent not found in agents.json: {args.agent_slug}")

    rows: list[dict[str, Any]] = []
    for seg in result.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        best_score = -1.0
        best_turn = None
        for turn in turns:
            score = _alignment_score(text, turn["text"])
            if score > best_score:
                best_score = score
                best_turn = turn
        if not best_turn or best_score < args.min_align:
            continue
        expected = "AGENT" if best_turn["label"] == args.agent_label else "CUSTOMER"
        rows.append({
            "segment": seg,
            "speaker": str(seg.get("speaker") or ""),
            "start": float(seg.get("start", 0.0) or 0.0),
            "end": float(seg.get("end", 0.0) or 0.0),
            "text": text,
            "matched_label": best_turn["label"],
            "expected": expected,
            "alignment": round(float(best_score), 4),
        })

    audio, sr = _load_audio(audio_path)
    model = get_model(force_cpu=True)
    for row in rows:
        row["embedding"] = _embed_segment(model, audio, sr, row["start"], row["end"], args.pad_seconds)

    training_rows = [
        row for row in rows
        if row["expected"] == "AGENT"
        and row.get("embedding") is not None
        and max(row["end"] - row["start"], 0.0) >= args.min_seconds
    ]
    if len(training_rows) < args.min_centroid_segments:
        raise RuntimeError(f"not enough agent training segments: {len(training_rows)}")

    centroid_items: list[tuple[str, np.ndarray, list[dict[str, Any]]]] = []
    all_centroid = l2_norm(np.mean(np.stack([row["embedding"] for row in training_rows]), axis=0).astype(np.float32))
    centroid_items.append(("all", all_centroid, training_rows))
    for speaker in sorted({row["speaker"] for row in training_rows if row["speaker"]}):
        speaker_rows = [row for row in training_rows if row["speaker"] == speaker]
        if len(speaker_rows) < args.min_centroid_segments:
            continue
        centroid = l2_norm(np.mean(np.stack([row["embedding"] for row in speaker_rows]), axis=0).astype(np.float32))
        centroid_items.append((speaker.lower(), centroid, speaker_rows))

    eval_rows = [row for row in rows if row.get("embedding") is not None]
    evaluation = _evaluate(eval_rows, args.agent_slug, [item[1] for item in centroid_items], args.threshold, args.margin)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report = {
        "agent_slug": args.agent_slug,
        "agent_label": args.agent_label,
        "result_id": result_path.parent.name,
        "result": str(result_path),
        "transcript": str(Path(args.transcript).resolve()),
        "audio": str(audio_path),
        "aligned_segments": len(rows),
        "training_segments": len(training_rows),
        "training_seconds": round(sum(max(row["end"] - row["start"], 0.0) for row in training_rows), 2),
        "centroids": [
            {"bucket": bucket, "segments": len(bucket_rows), "seconds": round(sum(max(r["end"] - r["start"], 0.0) for r in bucket_rows), 2)}
            for bucket, _centroid, bucket_rows in centroid_items
        ],
        "evaluation": evaluation,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    backup = VP_DIR / f"agents.backup.{args.agent_slug}.verified_transcript.{timestamp}.json"
    shutil.copy2(AGENTS_JSON, backup)
    voiceprints = agent_info.setdefault("voiceprints", [])
    replaced_count = 0
    if args.replace_segment_role:
        for entry in voiceprints:
            if not isinstance(entry, dict):
                continue
            uses_segment_role = (
                entry.get("use_for_segment_role")
                or entry.get("segment_role_voiceprint")
                or entry.get("source") == "verified_transcript_labels"
            )
            if not uses_segment_role:
                continue
            entry["use_for_segment_role"] = False
            entry["segment_role_replaced_by"] = timestamp
            replaced_count += 1
    written = []
    for bucket, centroid, bucket_rows in centroid_items:
        file_name = f"{args.agent_slug}_verified_transcript_{timestamp}_{bucket}.npy"
        np.save(VP_DIR / file_name, centroid.astype(np.float32))
        entry = {
            "path": file_name,
            "source": "verified_transcript_labels",
            "embedding_model": "cam++",
            "embedding_dim": int(centroid.shape[0]),
            "use_for_segment_role": True,
            "segment_role_min_similarity": args.threshold,
            "segment_role_min_margin": args.margin,
            "source_result": result_path.parent.name,
            "source_result_file": result_path.name,
            "source_transcript": str(Path(args.transcript).resolve()),
            "source_audio": audio_path.name,
            "bucket": bucket,
            "source_segment_count": len(bucket_rows),
            "source_seconds": round(sum(max(r["end"] - r["start"], 0.0) for r in bucket_rows), 2),
            "created_at": now.isoformat(),
        }
        voiceprints.append(entry)
        written.append(entry)

    agent_info["n_voiceprints"] = len(voiceprints)
    agent_info["embedding_model"] = "cam++"
    agent_info["embedding_dim"] = int(centroid_items[0][1].shape[0])
    agent_info["updated_at"] = now.isoformat()
    AGENTS_JSON.write_text(json.dumps(agents, indent=2) + "\n", encoding="utf-8")

    report["written_voiceprints"] = written
    report["replaced_segment_role_voiceprints"] = replaced_count
    report["agents_backup"] = str(backup)
    report_path = VP_DIR / f"{args.agent_slug}_verified_transcript_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
