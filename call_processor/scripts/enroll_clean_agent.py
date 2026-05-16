#!/usr/bin/env python
"""Enroll a clean single-speaker agent into agents.json using CAM++ voiceprints."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR))

from src.diar_multi import _load_voiceprints  # noqa: E402
from src.embedding_campp import EmbeddingModel, l2_norm  # noqa: E402

TARGET_SR = 16000
VOICEPRINT_DIR = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
AGENTS_JSON = VOICEPRINT_DIR / "agents.json"


@dataclass
class SourceStats:
    path: Path
    duration_s: float
    windows_seen: int
    windows_used: int
    rms_p20: float
    rms_p95: float
    speech_threshold: float
    centroid: np.ndarray | None


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "agent"


def load_audio(path: Path) -> np.ndarray:
    from pydub import AudioSegment

    sound = AudioSegment.from_file(path).set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
    scale = float(1 << (8 * sound.sample_width - 1))
    if scale > 0:
        samples /= scale
    return samples.astype(np.float32)


def frame_rms(audio: np.ndarray, frame_s: float = 0.05, hop_s: float = 0.025) -> np.ndarray:
    frame = max(int(frame_s * TARGET_SR), 1)
    hop = max(int(hop_s * TARGET_SR), 1)
    if len(audio) < frame:
        return np.asarray([], dtype=np.float32)
    vals = []
    for start in range(0, len(audio) - frame + 1, hop):
        chunk = audio[start:start + frame]
        vals.append(float(np.sqrt(np.mean(np.square(chunk)))))
    return np.asarray(vals, dtype=np.float32)


def speech_threshold(audio: np.ndarray) -> tuple[float, float, float]:
    vals = frame_rms(audio)
    if vals.size == 0:
        return 0.004, 0.0, 0.0
    p20 = float(np.percentile(vals, 20))
    p95 = float(np.percentile(vals, 95))
    threshold = max(0.004, p20 + (p95 - p20) * 0.18)
    return threshold, p20, p95


def speech_ratio(chunk: np.ndarray, threshold: float) -> float:
    vals = frame_rms(chunk)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals >= threshold))


def embed_clean_file(
    model: EmbeddingModel,
    path: Path,
    window_s: float,
    stride_s: float,
    min_speech_ratio: float,
) -> tuple[SourceStats, list[np.ndarray]]:
    audio = load_audio(path)
    threshold, p20, p95 = speech_threshold(audio)
    duration_s = len(audio) / TARGET_SR
    win = max(int(window_s * TARGET_SR), int(1.5 * TARGET_SR))
    stride = max(int(stride_s * TARGET_SR), 1)
    embs: list[np.ndarray] = []
    seen = 0

    if len(audio) >= win:
        offsets = range(0, len(audio) - win + 1, stride)
    else:
        offsets = [0]

    for offset in offsets:
        chunk = audio[offset:offset + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        seen += 1
        if speech_ratio(chunk, threshold) < min_speech_ratio:
            continue
        emb = model.embed_chunk(chunk.astype(np.float32), TARGET_SR)
        if emb is not None:
            embs.append(l2_norm(np.asarray(emb, dtype=np.float32)))

    centroid = l2_norm(np.mean(np.stack(embs), axis=0)) if embs else None
    stats = SourceStats(
        path=path,
        duration_s=duration_s,
        windows_seen=seen,
        windows_used=len(embs),
        rms_p20=p20,
        rms_p95=p95,
        speech_threshold=threshold,
        centroid=centroid,
    )
    return stats, embs


def best_sim(embedding: np.ndarray, stacks: Iterable[np.ndarray]) -> float:
    best = 0.0
    for stack in stacks:
        if getattr(stack, "ndim", 0) != 2 or not len(stack) or stack.shape[1] != embedding.shape[0]:
            continue
        sim = float(np.max(stack @ embedding))
        if sim > best:
            best = sim
    return best


def best_other(embedding: np.ndarray, voiceprints: dict, agent_slug: str) -> tuple[str | None, float]:
    best_slug = None
    best_score = 0.0
    for slug, (_name, stack) in voiceprints.items():
        if slug == agent_slug:
            continue
        if getattr(stack, "ndim", 0) != 2 or not len(stack) or stack.shape[1] != embedding.shape[0]:
            continue
        sim = float(np.max(stack @ embedding))
        if sim > best_score:
            best_slug = slug
            best_score = sim
    return best_slug, best_score


def update_agents_json(
    agent_name: str,
    agent_slug: str,
    centroid_paths: list[str],
    report_name: str,
    source_files: list[Path],
    mean_inside_sim: float,
    max_outside_sim: float,
    min_margin: float,
) -> Path:
    VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
    agents = json.loads(AGENTS_JSON.read_text(encoding="utf-8")) if AGENTS_JSON.exists() else {}
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = VOICEPRINT_DIR / f"agents.backup.{agent_slug}.clean_campp.{timestamp}.json"
    if AGENTS_JSON.exists():
        shutil.copy2(AGENTS_JSON, backup)

    agents[agent_slug] = {
        "agent_name": agent_name,
        "voiceprint_path": centroid_paths[0],
        "voiceprints": [
            {
                "path": path,
                "source": "clean_single_speaker_recording",
                "embedding_model": "cam++",
                "embedding_dim": 512,
                "use_for_segment_role": True,
                "segment_role_min_similarity": round(max(0.40, max_outside_sim + min_margin), 4),
                "segment_role_min_margin": round(min_margin, 4),
            }
            for path in centroid_paths
        ],
        "n_voiceprints": len(centroid_paths),
        "embedding_model": "cam++",
        "embedding_dim": 512,
        "source": "clean_single_speaker_recording",
        "source_files": [str(path) for path in source_files],
        "purity_report": report_name,
        "mean_inside_sim": round(mean_inside_sim, 4),
        "max_outside_sim": round(max_outside_sim, 4),
        "updated_at_epoch": int(time.time()),
    }
    AGENTS_JSON.write_text(json.dumps(agents, indent=2), encoding="utf-8")
    return backup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--agent-slug", default="")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--min-speech-ratio", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("audio", nargs="+")
    args = parser.parse_args()

    agent_slug = args.agent_slug.strip() or slugify(args.agent_name)
    source_files = [Path(path).resolve() for path in args.audio]
    missing = [str(path) for path in source_files if not path.exists()]
    if missing:
        raise FileNotFoundError("missing audio files: " + ", ".join(missing))

    model = EmbeddingModel()
    model.load(force_cpu=True)
    all_window_embs: list[np.ndarray] = []
    source_stats: list[SourceStats] = []
    try:
        for path in source_files:
            stats, embs = embed_clean_file(
                model,
                path,
                window_s=args.window_seconds,
                stride_s=args.stride_seconds,
                min_speech_ratio=args.min_speech_ratio,
            )
            source_stats.append(stats)
            all_window_embs.extend(embs)
    finally:
        model.unload()

    file_centroids = [item.centroid for item in source_stats if item.centroid is not None]
    if not file_centroids or len(all_window_embs) < 2:
        raise RuntimeError("not enough speech windows to create a voiceprint")

    aggregate = l2_norm(np.mean(np.stack(all_window_embs), axis=0))
    centroids = [aggregate] + [centroid for centroid in file_centroids]
    centroid_names = [f"{agent_slug}_clean_campp_all.npy"] + [
        f"{agent_slug}_clean_campp_{idx}.npy"
        for idx in range(1, len(file_centroids) + 1)
    ]

    VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
    for name, centroid in zip(centroid_names, centroids):
        np.save(VOICEPRINT_DIR / name, centroid.astype(np.float32))

    voiceprints = _load_voiceprints(str(AGENTS_JSON))
    new_stack = np.stack(centroids).astype(np.float32)
    inside_sims = [float(np.max(new_stack @ emb)) for emb in all_window_embs]
    outside_pairs = [best_other(centroid, voiceprints, agent_slug) for centroid in centroids]
    max_outside = max((score for _slug, score in outside_pairs), default=0.0)
    mean_inside = float(np.mean(inside_sims)) if inside_sims else 0.0
    min_inside = float(np.min(inside_sims)) if inside_sims else 0.0

    window_results = []
    correct = 0
    for emb in all_window_embs:
        target_score = float(np.max(new_stack @ emb))
        other_slug, other_score = best_other(emb, voiceprints, agent_slug)
        predicted_target = target_score >= max(0.40, other_score + args.min_margin)
        correct += int(predicted_target)
        window_results.append({
            "target_similarity": round(target_score, 4),
            "best_other_slug": other_slug,
            "best_other_similarity": round(other_score, 4),
            "margin": round(target_score - other_score, 4),
            "accepted": predicted_target,
        })

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_name = f"{agent_slug}_clean_campp_enrollment_{timestamp}.json"
    backup = None
    if args.activate:
        backup = update_agents_json(
            args.agent_name,
            agent_slug,
            centroid_names,
            report_name,
            source_files,
            mean_inside,
            max_outside,
            args.min_margin,
        )

    report = {
        "agent_name": args.agent_name,
        "agent_slug": agent_slug,
        "activated": bool(args.activate),
        "agents_json_backup": str(backup) if backup else None,
        "embedding_model": "cam++",
        "embedding_dim": int(centroids[0].shape[0]),
        "voiceprints": [str(VOICEPRINT_DIR / name) for name in centroid_names],
        "source_files": [
            {
                "path": str(item.path),
                "duration_s": round(float(item.duration_s), 2),
                "windows_seen": item.windows_seen,
                "windows_used": item.windows_used,
                "rms_p20": round(float(item.rms_p20), 5),
                "rms_p95": round(float(item.rms_p95), 5),
                "speech_threshold": round(float(item.speech_threshold), 5),
            }
            for item in source_stats
        ],
        "window_count": len(all_window_embs),
        "mean_inside_sim": round(mean_inside, 4),
        "min_inside_sim": round(min_inside, 4),
        "max_outside_sim": round(max_outside, 4),
        "min_margin": round(float(args.min_margin), 4),
        "same_source_accuracy": round(correct / max(len(all_window_embs), 1) * 100.0, 2),
        "centroid_best_other": [
            {
                "voiceprint": name,
                "best_other_slug": slug,
                "best_other_similarity": round(score, 4),
                "margin_to_mean_inside": round(mean_inside - score, 4),
            }
            for name, (slug, score) in zip(centroid_names, outside_pairs)
        ],
        "weakest_windows": sorted(window_results, key=lambda row: row["margin"])[:20],
    }
    report_path = VOICEPRINT_DIR / report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "agent": args.agent_name,
        "slug": agent_slug,
        "activated": bool(args.activate),
        "report": str(report_path),
        "voiceprints": centroid_names,
        "window_count": len(all_window_embs),
        "same_source_accuracy": report["same_source_accuracy"],
        "mean_inside_sim": report["mean_inside_sim"],
        "min_inside_sim": report["min_inside_sim"],
        "max_outside_sim": report["max_outside_sim"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
