"""Train desk-recording voiceprint centroids for selected agents.

The normal phone-call voiceprints are clean and should stay intact. Desk
recordings are far-field, noisy, and often below the phone-call presence floor.
This script derives additional CAM++ centroids from known desk recordings and
appends them to agents.json without replacing the phone-call voiceprints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf

CALL_PROCESSOR_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CALL_PROCESSOR_DIR.parent
sys.path.insert(0, str(CALL_PROCESSOR_DIR))
os.chdir(str(CALL_PROCESSOR_DIR))

from src.diar_clean import (  # noqa: E402
    AGENT_PRESENCE_FLOOR,
    SortformerDiarizer,
    _renumber_speakers_to_canonical,
    compute_cluster_centroids,
    match_agent_to_cluster,
)
from src.diar_multi import _load_voiceprints  # noqa: E402
from src.embedding_campp import get_model  # noqa: E402
from src.voiceprints import resolve_voiceprint_path  # noqa: E402


AGENTS = {
    "zak": {
        "slug": "zak_raissi_barnet",
        "name": "Zak Raissi Barnet",
        "glob": "zak_raissi_barnet_*.mp3",
    },
    "hussein": {
        "slug": "hussein_mohamed",
        "name": "Hussein Mohamed",
        "glob": "hussein_mohamed_*.mp3",
    },
}

VP_DIR = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
AGENTS_JSON = VP_DIR / "agents.json"
DESK_CACHE = CALL_PROCESSOR_DIR / "data" / "desk_recordings_cache"
OUT_DIR = CALL_PROCESSOR_DIR / "data" / "processed" / "ds_diagnose"
CLIP_DIR = OUT_DIR / "clips"
REPORT_PATH = VP_DIR / "stable_ds_voiceprint_report.json"
DS_VOICEPRINT_SOURCE = "desk_recordings_campp_stable_ds"


def _run_ffmpeg(src: Path, dst: Path, seconds: int, offset: int = 0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if offset > 0:
        cmd += ["-ss", str(offset)]
    cmd += [
        "-i",
        str(src),
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return float(out)
    except Exception:
        return 0.0


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)


def _parse_offsets(args: argparse.Namespace) -> List[int]:
    raw = str(getattr(args, "offsets", "") or "").strip()
    if not raw:
        return [int(args.offset)]
    offsets: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        offsets.append(max(int(float(part)), 0))
    return sorted(set(offsets)) or [int(args.offset)]


def _parse_csv_set(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def _best_target_match(
    match_table: Dict[str, Dict[str, Dict[str, Any]]],
    target_slug: str,
) -> Tuple[Optional[str], float]:
    best_spk: Optional[str] = None
    best_sim = -1.0
    for spk, matches in match_table.items():
        sim = float((matches.get(target_slug) or {}).get("similarity") or 0.0)
        if sim > best_sim:
            best_spk = spk
            best_sim = sim
    return best_spk, max(best_sim, 0.0)


def _summarize_all_agent_match(
    centroids: Dict[str, np.ndarray],
    voiceprints: Dict[str, Tuple[str, np.ndarray]],
) -> Dict[str, Any]:
    spk, slug, sim, table = match_agent_to_cluster(
        centroids,
        voiceprints,
        presence_floor=AGENT_PRESENCE_FLOOR,
    )
    top_by_speaker: Dict[str, List[Dict[str, Any]]] = {}
    for speaker, matches in table.items():
        ranked = sorted(
            (
                {"slug": slug0, "name": data["name"], "similarity": data["similarity"]}
                for slug0, data in matches.items()
            ),
            key=lambda item: -float(item["similarity"]),
        )
        top_by_speaker[speaker] = ranked[:5]
    return {
        "speaker": spk,
        "slug": slug,
        "similarity": round(float(sim), 4),
        "top_by_speaker": top_by_speaker,
    }


def _load_voiceprints_excluding_sources(excluded_sources: set[str]) -> Dict[str, Tuple[str, np.ndarray]]:
    """Load enrollment voiceprints while excluding generated DS centroids.

    Candidate DS clips must be scored against trusted base enrollment only. If
    old DS centroids are included here, a rerun can select the same generated
    centroid again with a false 1.0 similarity and poison future training.
    """
    if not AGENTS_JSON.exists():
        return {}
    try:
        agents = json.loads(AGENTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[str, Tuple[str, np.ndarray]] = {}
    for slug, info in agents.items():
        if not isinstance(info, dict):
            continue

        raw_paths: List[str] = []
        vps_field = info.get("voiceprints")
        if isinstance(vps_field, list):
            for entry in vps_field:
                if isinstance(entry, dict):
                    if str(entry.get("source") or "") in excluded_sources:
                        continue
                    pp = entry.get("path") or entry.get("voiceprint_path")
                else:
                    pp = entry
                if pp:
                    raw_paths.append(str(pp))

        if not raw_paths:
            legacy = info.get("voiceprint_path") or info.get("voiceprint")
            if legacy:
                raw_paths.append(str(legacy))

        loaded: List[np.ndarray] = []
        for raw in raw_paths:
            vp_path = resolve_voiceprint_path(raw, str(AGENTS_JSON))
            if not os.path.isfile(vp_path):
                continue
            try:
                vp = np.load(vp_path).astype(np.float32).squeeze()
            except Exception:
                continue
            if vp.ndim != 1:
                continue
            norm = np.linalg.norm(vp)
            if norm > 0:
                vp = vp / norm
            loaded.append(vp)

        if not loaded:
            continue
        dims = {v.shape[0] for v in loaded}
        if len(dims) > 1:
            counts = {d: sum(1 for v in loaded if v.shape[0] == d) for d in dims}
            best_dim = max(counts, key=counts.get)
            loaded = [v for v in loaded if v.shape[0] == best_dim]
        if loaded:
            name = info.get("agent_name") or info.get("name") or slug
            out[slug] = (name, np.stack(loaded).astype(np.float32))
    return out


def _append_voiceprints(agent_slug: str, agent_name: str, saved: List[Dict[str, Any]]) -> None:
    if AGENTS_JSON.exists():
        agents = json.loads(AGENTS_JSON.read_text(encoding="utf-8"))
    else:
        agents = {}

    entry = agents.get(agent_slug)
    if not isinstance(entry, dict):
        entry = {"agent_name": agent_name}

    existing = entry.get("voiceprints")
    if not isinstance(existing, list):
        existing = []
        legacy = entry.get("voiceprint_path") or entry.get("voiceprint")
        if legacy:
            existing.append({"path": legacy, "source": entry.get("source", "legacy")})

    existing = [
        vp
        for vp in existing
        if not (
            isinstance(vp, dict)
            and str(vp.get("source") or "") == DS_VOICEPRINT_SOURCE
        )
    ]
    existing.extend(saved)

    entry["agent_name"] = agent_name
    entry["voiceprints"] = existing
    entry["n_voiceprints"] = len(existing)
    entry["embedding_model"] = "cam++"
    entry["embedding_dim"] = 512
    entry["has_desk_voiceprints"] = True
    entry["desk_voiceprints"] = [vp["path"] for vp in saved]
    entry["stable_ds_report"] = REPORT_PATH.name
    agents[agent_slug] = entry

    AGENTS_JSON.write_text(json.dumps(agents, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_agent_files(agent_key: str, max_files: int) -> List[Path]:
    cfg = AGENTS[agent_key]
    files = sorted(DESK_CACHE.glob(cfg["glob"]))
    if max_files > 0:
        files = files[:max_files]
    return files


def _filter_files(files: List[Path], only_files: set[str]) -> List[Path]:
    if not only_files:
        return files
    return [path for path in files if path.name in only_files]


def process_file(
    audio_path: Path,
    agent_slug: str,
    seconds: int,
    offset: int,
    diarizer: SortformerDiarizer,
    embedder: Any,
    source_voiceprints: Dict[str, Tuple[str, np.ndarray]],
) -> Dict[str, Any]:
    clip = CLIP_DIR / f"{audio_path.stem}_{offset}s_{seconds}s.wav"
    if not clip.exists():
        _run_ffmpeg(audio_path, clip, seconds=seconds, offset=offset)

    speaker_segments = diarizer.diarize(str(clip), max_speakers=4)
    speaker_segments = _renumber_speakers_to_canonical(speaker_segments)
    audio, sr = _read_audio(clip)
    centroids, counts = compute_cluster_centroids(speaker_segments, audio, sr, embedder)

    _, _, target_best_sim, target_table = match_agent_to_cluster(
        centroids,
        source_voiceprints,
        presence_floor=0.0,
        target_agent_slug=agent_slug,
    )
    target_spk, target_sim = _best_target_match(target_table, agent_slug)
    all_agent = _summarize_all_agent_match(centroids, source_voiceprints)

    return {
        "file": str(audio_path),
        "file_name": audio_path.name,
        "clip": str(clip),
        "offset": offset,
        "seconds": seconds,
        "speaker_count": len({s["speaker"] for s in speaker_segments}),
        "speaker_segments": len(speaker_segments),
        "cluster_segment_counts": counts,
        "target_speaker": target_spk,
        "target_similarity": round(float(max(target_sim, target_best_sim)), 4),
        "target_centroid": centroids.get(target_spk) if target_spk else None,
        "all_agent_before": all_agent,
    }


def train_agent(
    agent_key: str,
    args: argparse.Namespace,
    diarizer: SortformerDiarizer,
    embedder: Any,
) -> Dict[str, Any]:
    cfg = AGENTS[agent_key]
    agent_slug = cfg["slug"]
    agent_name = cfg["name"]
    files = _filter_files(_iter_agent_files(agent_key, args.max_files), _parse_csv_set(args.only_files))
    offsets = _parse_offsets(args)
    source_voiceprints = _load_voiceprints_excluding_sources({DS_VOICEPRINT_SOURCE})
    if not source_voiceprints:
        raise RuntimeError("no trusted base voiceprints available for DS scoring")

    processed: List[Dict[str, Any]] = []
    train_items: List[Dict[str, Any]] = []
    skipped_candidates: List[Dict[str, Any]] = []
    holdout_files = _parse_csv_set(args.holdout_files)
    for path in files:
        duration = _probe_duration(path)
        print(f"[ds] {agent_name}: {path.name} ({duration:.1f}s)", flush=True)
        for offset in offsets:
            if duration and offset >= duration - args.min_clip_seconds:
                skipped_candidates.append({
                    "file": path.name,
                    "offset": offset,
                    "reason": "offset_too_late",
                })
                continue
            try:
                item = process_file(
                    path,
                    agent_slug,
                    seconds=args.seconds,
                    offset=offset,
                    diarizer=diarizer,
                    embedder=embedder,
                    source_voiceprints=source_voiceprints,
                )
            except Exception as exc:
                skipped_candidates.append({
                    "file": path.name,
                    "offset": offset,
                    "reason": f"processing_failed:{exc}",
                })
                print(f"      {offset:>4}s skip processing_failed={exc}", flush=True)
                continue

            processed.append(item)
            sim = float(item["target_similarity"])
            before_slug = item["all_agent_before"].get("slug")
            before_sim = float(item["all_agent_before"].get("similarity") or 0.0)
            heldout_for_training = path.name in holdout_files
            conflicting_agent = (
                before_slug is not None
                and before_slug != agent_slug
                and before_sim >= args.reject_other_agent_floor
            )
            if (
                item["target_centroid"] is not None
                and sim >= args.min_include_sim
                and not conflicting_agent
                and not heldout_for_training
            ):
                train_items.append(item)
                print(
                    f"      {offset:>4}s include {item['target_speaker']} "
                    f"target_sim={sim:.3f}",
                    flush=True,
                )
            else:
                reason = "low_target_sim"
                if item["target_centroid"] is None:
                    reason = "no_target_centroid"
                elif conflicting_agent:
                    reason = f"conflicting_agent:{before_slug}:{before_sim:.3f}"
                elif heldout_for_training:
                    reason = "heldout_for_testing"
                skipped_candidates.append({
                    "file": path.name,
                    "offset": offset,
                    "reason": reason,
                    "target_similarity": item["target_similarity"],
                    "before_slug": before_slug,
                    "before_similarity": round(before_sim, 4),
                })
                print(
                    f"      {offset:>4}s skip target_sim={sim:.3f} reason={reason}",
                    flush=True,
                )

    ranked_all = sorted(train_items, key=lambda x: -float(x["target_similarity"]))
    ranked_train: List[Dict[str, Any]] = []
    per_file_used: Dict[str, int] = {}
    for item in ranked_all:
        fname = Path(item["file"]).name
        used = per_file_used.get(fname, 0)
        if used >= args.max_per_file:
            continue
        ranked_train.append(item)
        per_file_used[fname] = used + 1
        if len(ranked_train) >= args.train_limit:
            break
    ranked_train = ranked_train[: args.train_limit]

    saved_vps: List[Dict[str, Any]] = []
    for idx, item in enumerate(ranked_train, start=1):
        centroid = np.asarray(item["target_centroid"], dtype=np.float32)
        name = f"{agent_slug}_desk_campp_v{idx}.npy"
        np.save(VP_DIR / name, centroid)
        saved_vps.append(
            {
                "path": name,
                "source": DS_VOICEPRINT_SOURCE,
                "embedding_model": "cam++",
                "embedding_dim": 512,
                "source_file": Path(item["file"]).name,
                "source_offset": item["offset"],
                "source_seconds": item["seconds"],
                "source_speaker": item["target_speaker"],
                "source_similarity_to_phone_vp": item["target_similarity"],
            }
        )

    if saved_vps:
        _append_voiceprints(agent_slug, agent_name, saved_vps)

    updated_voiceprints = _load_voiceprints()
    evaluations = []
    for item in processed:
        # Reuse centroids from the processing pass; evaluate with updated agents.json.
        clip = Path(item["clip"])
        audio, sr = _read_audio(clip)
        speaker_segments = diarizer.diarize(str(clip), max_speakers=4)
        speaker_segments = _renumber_speakers_to_canonical(speaker_segments)
        centroids, counts = compute_cluster_centroids(speaker_segments, audio, sr, embedder)
        after = _summarize_all_agent_match(centroids, updated_voiceprints)
        target_spk, target_slug, target_sim, _ = match_agent_to_cluster(
            centroids,
            updated_voiceprints,
            presence_floor=args.target_presence_floor,
            target_agent_slug=agent_slug,
        )
        evaluations.append(
            {
                "file": Path(item["file"]).name,
                "offset": item["offset"],
                "trained_from_this_file": any(
                    Path(item["file"]).name == vp.get("source_file") for vp in saved_vps
                ),
                "trained_from_this_clip": any(
                    Path(item["file"]).name == vp.get("source_file")
                    and int(item["offset"]) == int(vp.get("source_offset", -1))
                    for vp in saved_vps
                ),
                "before_slug": item["all_agent_before"].get("slug"),
                "before_similarity": item["all_agent_before"].get("similarity"),
                "before_target_similarity": item["target_similarity"],
                "after_slug": after.get("slug"),
                "after_similarity": after.get("similarity"),
                "correct_after": after.get("slug") == agent_slug,
                "target_after_speaker": target_spk,
                "target_after_slug": target_slug,
                "target_after_similarity": round(float(target_sim), 4),
                "correct_target_after": target_slug == agent_slug,
                "cluster_segment_counts": counts,
            }
        )

    correct = sum(1 for ev in evaluations if ev["correct_after"])
    target_correct = sum(1 for ev in evaluations if ev["correct_target_after"])
    before_target_values = [float(ev["before_target_similarity"]) for ev in evaluations]
    after_target_values = [float(ev["target_after_similarity"]) for ev in evaluations]
    heldout = [ev for ev in evaluations if not ev["trained_from_this_file"]]
    heldout_before = [float(ev["before_target_similarity"]) for ev in heldout]
    heldout_after = [float(ev["target_after_similarity"]) for ev in heldout]

    def _avg(values: List[float]) -> float:
        return round(float(np.mean(values)), 4) if values else 0.0

    def _gain(before: float, after: float) -> float:
        if before <= 0:
            return 0.0
        return round(((after - before) / before) * 100.0, 1)

    avg_before = _avg(before_target_values)
    avg_after = _avg(after_target_values)
    held_avg_before = _avg(heldout_before)
    held_avg_after = _avg(heldout_after)
    return {
        "agent_slug": agent_slug,
        "agent_name": agent_name,
        "files_seen": len(files),
        "candidate_clips_seen": len(processed),
        "offsets": offsets,
        "holdout_files": sorted(holdout_files),
        "train_voiceprints_saved": saved_vps,
        "train_files_used": [Path(item["file"]).name for item in ranked_train],
        "train_clip_offsets_used": [
            {"file": Path(item["file"]).name, "offset": item["offset"], "similarity": item["target_similarity"]}
            for item in ranked_train
        ],
        "skipped_candidates": skipped_candidates,
        "evaluations": evaluations,
        "correct_after": correct,
        "accuracy_after": round(correct / max(len(evaluations), 1), 4),
        "correct_target_after": target_correct,
        "target_accuracy_after": round(target_correct / max(len(evaluations), 1), 4),
        "avg_before_target_similarity": avg_before,
        "avg_after_target_similarity": avg_after,
        "avg_target_similarity_gain_pct": _gain(avg_before, avg_after),
        "heldout_clip_count": len(heldout),
        "heldout_avg_before_target_similarity": held_avg_before,
        "heldout_avg_after_target_similarity": held_avg_after,
        "heldout_target_similarity_gain_pct": _gain(held_avg_before, held_avg_after),
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="zak,hussein", help="comma list: zak,hussein")
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--offsets", default="", help="comma-separated offsets in seconds; overrides --offset")
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--only-files", default="", help="comma list of exact recording names to process after max-files")
    parser.add_argument("--train-limit", type=int, default=2)
    parser.add_argument("--max-per-file", type=int, default=1)
    parser.add_argument("--holdout-files", default="", help="comma list of recording file names to evaluate but not train from")
    parser.add_argument("--min-clip-seconds", type=int, default=30)
    parser.add_argument("--min-include-sim", type=float, default=0.24)
    parser.add_argument("--target-presence-floor", type=float, default=0.24)
    parser.add_argument("--reject-other-agent-floor", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    selected = [a.strip().lower() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in selected if a not in AGENTS]
    if unknown:
        raise SystemExit(f"unknown agents: {', '.join(unknown)}")

    VP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    diarizer = SortformerDiarizer()
    embedder = get_model(force_cpu=True)
    report = {
        "seconds": args.seconds,
        "offset": args.offset,
        "offsets": _parse_offsets(args),
        "max_files": args.max_files,
        "train_limit": args.train_limit,
        "max_per_file": args.max_per_file,
        "min_include_sim": args.min_include_sim,
        "target_presence_floor": args.target_presence_floor,
        "reject_other_agent_floor": args.reject_other_agent_floor,
        "agents": {},
    }
    try:
        for agent_key in selected:
            report["agents"][agent_key] = train_agent(agent_key, args, diarizer, embedder)
    finally:
        diarizer.unload()

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ds] report -> {REPORT_PATH}", flush=True)
    for key, data in report["agents"].items():
        print(
            f"[ds] {data['agent_name']}: all-agent {data['correct_after']}/{data['candidate_clips_seen']} "
            f"({data['accuracy_after'] * 100:.1f}%), target "
            f"{data['correct_target_after']}/{data['candidate_clips_seen']} "
            f"({data['target_accuracy_after'] * 100:.1f}%), heldout gain "
            f"{data['heldout_target_similarity_gain_pct']:.1f}%",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
