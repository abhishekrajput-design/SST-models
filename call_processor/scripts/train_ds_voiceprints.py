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


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)


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
            and str(vp.get("source") or "") == "desk_recordings_campp_stable_ds"
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
    agents[agent_slug] = entry

    AGENTS_JSON.write_text(json.dumps(agents, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_agent_files(agent_key: str, max_files: int) -> List[Path]:
    cfg = AGENTS[agent_key]
    files = sorted(DESK_CACHE.glob(cfg["glob"]))
    if max_files > 0:
        files = files[:max_files]
    return files


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
        "clip": str(clip),
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
    files = _iter_agent_files(agent_key, args.max_files)
    source_voiceprints = _load_voiceprints()

    processed: List[Dict[str, Any]] = []
    train_items: List[Dict[str, Any]] = []
    for path in files:
        print(f"[ds] {agent_name}: {path.name}", flush=True)
        item = process_file(
            path,
            agent_slug,
            seconds=args.seconds,
            offset=args.offset,
            diarizer=diarizer,
            embedder=embedder,
            source_voiceprints=source_voiceprints,
        )
        processed.append(item)
        sim = float(item["target_similarity"])
        if item["target_centroid"] is not None and sim >= args.min_include_sim:
            train_items.append(item)
            print(f"      include {item['target_speaker']} target_sim={sim:.3f}", flush=True)
        else:
            print(f"      skip target_sim={sim:.3f}", flush=True)

    ranked_train = sorted(train_items, key=lambda x: -float(x["target_similarity"]))
    ranked_train = ranked_train[: args.train_limit]

    saved_vps: List[Dict[str, Any]] = []
    for idx, item in enumerate(ranked_train, start=1):
        centroid = np.asarray(item["target_centroid"], dtype=np.float32)
        name = f"{agent_slug}_desk_campp_v{idx}.npy"
        np.save(VP_DIR / name, centroid)
        saved_vps.append(
            {
                "path": name,
                "source": "desk_recordings_campp_stable_ds",
                "embedding_model": "cam++",
                "embedding_dim": 512,
                "source_file": Path(item["file"]).name,
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
                "trained_from_this_file": any(
                    Path(item["file"]).name == vp.get("source_file") for vp in saved_vps
                ),
                "before_slug": item["all_agent_before"].get("slug"),
                "before_similarity": item["all_agent_before"].get("similarity"),
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
    return {
        "agent_slug": agent_slug,
        "agent_name": agent_name,
        "files_seen": len(processed),
        "train_voiceprints_saved": saved_vps,
        "train_files_used": [Path(item["file"]).name for item in ranked_train],
        "evaluations": evaluations,
        "correct_after": correct,
        "accuracy_after": round(correct / max(len(evaluations), 1), 4),
        "correct_target_after": target_correct,
        "target_accuracy_after": round(target_correct / max(len(evaluations), 1), 4),
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="zak,hussein", help="comma list: zak,hussein")
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=2)
    parser.add_argument("--min-include-sim", type=float, default=0.24)
    parser.add_argument("--target-presence-floor", type=float, default=0.24)
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
        "max_files": args.max_files,
        "train_limit": args.train_limit,
        "min_include_sim": args.min_include_sim,
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
            f"[ds] {data['agent_name']}: all-agent {data['correct_after']}/{data['files_seen']} "
            f"({data['accuracy_after'] * 100:.1f}%), target "
            f"{data['correct_target_after']}/{data['files_seen']} "
            f"({data['target_accuracy_after'] * 100:.1f}%)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
