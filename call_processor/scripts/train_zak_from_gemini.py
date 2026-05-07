#!/usr/bin/env python
"""
Train Zak CAM++ voiceprints from Gemini-labelled local calls.

The voiceprint is built only from segments labelled speaker=agent. Customer
segments are used only for threshold statistics and accuracy reporting.
"""
from __future__ import annotations

import argparse
import json
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

from src.embedding_campp import EmbeddingModel, l2_norm  # noqa: E402

TARGET_SR = 16000
DEFAULT_AGENT_SLUG = "zak_local_20260423"
DEFAULT_AGENT_NAME = "Zak Raissi"
BACKCHANNELS = {
    "hello",
    "hi",
    "yeah",
    "yes",
    "yep",
    "ok",
    "okay",
    "sure",
    "right",
    "alright",
    "no worries",
    "thank you",
    "thanks",
}


@dataclass
class CallData:
    name: str
    call_id: str
    audio_path: Path
    audio: np.ndarray
    sr: int
    segments: list[dict]
    label_offset_s: float


@dataclass
class SegmentEmbedding:
    call_name: str
    call_id: str
    index: int
    speaker: str
    start: float
    end: float
    duration: float
    text: str
    embedding: np.ndarray
    used_for_training: bool


def norm_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch == "'" else " " for ch in text).split())


def is_backchannel(text: str) -> bool:
    words = norm_text(text).split()
    return " ".join(words) in BACKCHANNELS or (len(words) <= 1 and bool(words))


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    from pydub import AudioSegment

    sound = AudioSegment.from_file(path)
    sound = sound.set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
    scale = float(1 << (8 * sound.sample_width - 1))
    if scale > 0:
        samples = samples / scale
    return samples.astype(np.float32), TARGET_SR


def detect_leading_speech_offset(audio: np.ndarray, sr: int) -> float:
    frame = max(int(0.05 * sr), 1)
    hop = max(int(0.01 * sr), 1)
    if len(audio) <= frame:
        return 0.0
    rms = []
    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start:start + frame]
        rms.append(float(np.sqrt(np.mean(np.square(chunk)))))
    if not rms:
        return 0.0
    values = np.asarray(rms, dtype=np.float32)
    floor = float(np.percentile(values, 20))
    peak = float(np.percentile(values, 95))
    threshold = max(0.004, floor + (peak - floor) * 0.15)
    for i, value in enumerate(values):
        if value >= threshold:
            return round(i * hop / sr, 3)
    return 0.0


def find_audio_file(call_dir: Path) -> Path:
    preferred = call_dir / "audio_16k.wav"
    if preferred.exists():
        return preferred
    candidates = sorted(
        p for p in call_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
    )
    if not candidates:
        raise FileNotFoundError(f"No audio file found in {call_dir}")
    return candidates[0]


def load_calls(data_dir: Path) -> list[CallData]:
    calls: list[CallData] = []
    for call_dir in sorted(data_dir.glob("call_*")):
        label_path = call_dir / "data.json"
        if not label_path.exists():
            continue
        data = json.loads(label_path.read_text(encoding="utf-8"))
        audio_path = find_audio_file(call_dir)
        audio, sr = load_audio(audio_path)
        label_offset_s = detect_leading_speech_offset(audio, sr)
        calls.append(
            CallData(
                name=call_dir.name,
                call_id=str(data.get("call_id") or call_dir.name),
                audio_path=audio_path,
                audio=audio,
                sr=sr,
                segments=list(data.get("segments") or []),
                label_offset_s=label_offset_s,
            )
        )
    return calls


def segment_window(call: CallData, segment: dict) -> tuple[np.ndarray, float, float, float] | None:
    start = float(segment.get("start") or 0.0) + call.label_offset_s
    end = float(segment.get("end") or 0.0) + call.label_offset_s
    if end <= start:
        return None
    audio_end = len(call.audio) / call.sr
    start = max(0.0, min(start, audio_end))
    end = max(start, min(end, audio_end))
    duration = end - start
    if duration <= 0:
        return None
    start_i = int(start * call.sr)
    end_i = int(end * call.sr)
    chunk = call.audio[start_i:end_i]
    if chunk.size == 0:
        return None
    return chunk, start, end, duration


def extract_embeddings(
    calls: Iterable[CallData],
    model: EmbeddingModel,
    min_eval_dur: float,
    min_train_dur: float,
    max_train_dur: float,
) -> list[SegmentEmbedding]:
    rows: list[SegmentEmbedding] = []
    for call in calls:
        print(f"[extract] {call.name}: {call.audio_path.name} ({len(call.audio) / call.sr:.1f}s)")
        for idx, seg in enumerate(call.segments, start=1):
            speaker = str(seg.get("speaker") or "").strip().lower()
            if speaker not in {"agent", "customer"}:
                continue
            window = segment_window(call, seg)
            if window is None:
                continue
            chunk, start, end, duration = window
            if duration < min_eval_dur:
                continue
            emb = model.embed_chunk(chunk, sr=call.sr)
            if emb is None or np.isnan(emb).any():
                continue
            text = str(seg.get("text") or "")
            used_for_training = (
                speaker == "agent"
                and min_train_dur <= duration <= max_train_dur
                and not is_backchannel(text)
            )
            rows.append(
                SegmentEmbedding(
                    call_name=call.name,
                    call_id=call.call_id,
                    index=idx,
                    speaker=speaker,
                    start=start,
                    end=end,
                    duration=duration,
                    text=text,
                    embedding=l2_norm(np.asarray(emb, dtype=np.float32).squeeze()),
                    used_for_training=used_for_training,
                )
            )
    return rows


def extract_whole_call_agent_embeddings(
    calls: Iterable[CallData],
    model: EmbeddingModel,
    window_s: float,
    step_s: float,
    rms_floor: float,
) -> tuple[list[np.ndarray], list[dict]]:
    embeddings: list[np.ndarray] = []
    stats: list[dict] = []
    for call in calls:
        win = max(int(window_s * call.sr), 1)
        step = max(int(step_s * call.sr), 1)
        count = 0
        used_seconds = 0.0
        print(
            f"[whole-agent] {call.name}: using full audio as Zak enrollment "
            f"({len(call.audio) / call.sr:.1f}s)"
        )
        for start_i in range(0, max(len(call.audio) - win + 1, 1), step):
            chunk = call.audio[start_i:start_i + win]
            if len(chunk) < int(1.0 * call.sr):
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            if rms < rms_floor:
                continue
            emb = model.embed_chunk(chunk, sr=call.sr)
            if emb is None or np.isnan(emb).any():
                continue
            embeddings.append(l2_norm(np.asarray(emb, dtype=np.float32).squeeze()))
            count += 1
            used_seconds += len(chunk) / call.sr
        stats.append(
            {
                "call": call.name,
                "call_id": call.call_id,
                "audio": str(call.audio_path),
                "windows": count,
                "window_seconds_total": round(used_seconds, 2),
            }
        )
        print(f"[whole-agent] {call.name}: {count} windows kept")
    return embeddings, stats


def build_centroids(agent_embeddings: list[np.ndarray], n_clusters: int) -> list[np.ndarray]:
    if not agent_embeddings:
        raise ValueError("No agent embeddings available")
    X = np.stack([l2_norm(e) for e in agent_embeddings]).astype(np.float32)
    k = min(max(1, n_clusters), len(X))
    if k == 1:
        return [l2_norm(np.mean(X, axis=0)).astype(np.float32)]
    try:
        from sklearn.cluster import KMeans
    except Exception:
        print("[warn] sklearn not available; saving one centroid")
        return [l2_norm(np.mean(X, axis=0)).astype(np.float32)]

    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
    centroids: list[np.ndarray] = []
    for cluster_id in range(k):
        cluster = X[labels == cluster_id]
        if len(cluster) == 0:
            continue
        centroids.append(l2_norm(np.mean(cluster, axis=0)).astype(np.float32))
    return centroids


def best_sim(embedding: np.ndarray, centroids: list[np.ndarray]) -> float:
    e = l2_norm(embedding)
    return float(max(float(np.dot(c, e)) for c in centroids))


def production_threshold(max_outside_sim: float) -> float:
    return float(min(max(0.25, max_outside_sim + 0.04), 0.36))


def score_rows(rows: list[SegmentEmbedding], centroids: list[np.ndarray], threshold: float) -> dict:
    total = correct = 0
    agent_total = agent_correct = 0
    customer_total = customer_correct = 0
    errors = []
    for row in rows:
        sim = best_sim(row.embedding, centroids)
        pred = "agent" if sim >= threshold else "customer"
        ok = pred == row.speaker
        total += 1
        correct += int(ok)
        if row.speaker == "agent":
            agent_total += 1
            agent_correct += int(ok)
        else:
            customer_total += 1
            customer_correct += int(ok)
        if not ok and len(errors) < 20:
            errors.append(
                {
                    "call": row.call_name,
                    "segment": row.index,
                    "time": [round(row.start, 2), round(row.end, 2)],
                    "gemini": row.speaker,
                    "predicted": pred,
                    "similarity": round(sim, 4),
                    "text": row.text[:100],
                }
            )
    return {
        "segments": total,
        "overall_accuracy": round(correct / total * 100, 2) if total else 0.0,
        "agent_accuracy": round(agent_correct / agent_total * 100, 2) if agent_total else 0.0,
        "customer_accuracy": round(customer_correct / customer_total * 100, 2) if customer_total else 0.0,
        "agent_correct": agent_correct,
        "agent_total": agent_total,
        "customer_correct": customer_correct,
        "customer_total": customer_total,
        "errors": errors,
    }


def optimize_threshold(rows: list[SegmentEmbedding], centroids: list[np.ndarray]) -> dict:
    if not rows:
        return {"threshold": 0.0, "overall_accuracy": 0.0}

    best_threshold = 0.25
    best_accuracy = -1.0
    for threshold in np.arange(0.10, 0.991, 0.005):
        correct = 0
        for row in rows:
            pred = "agent" if best_sim(row.embedding, centroids) >= threshold else "customer"
            correct += int(pred == row.speaker)
        accuracy = correct / len(rows) * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    scored = score_rows(rows, centroids, best_threshold)
    scored["threshold"] = round(best_threshold, 4)
    return scored


def threshold_from_rows(rows: list[SegmentEmbedding], centroids: list[np.ndarray]) -> tuple[float, float]:
    customer_sims = [
        best_sim(row.embedding, centroids)
        for row in rows
        if row.speaker == "customer"
    ]
    max_outside = float(np.percentile(customer_sims, 95)) if customer_sims else 0.30
    return max_outside, production_threshold(max_outside)


def leave_one_call_out(rows: list[SegmentEmbedding], n_clusters: int) -> list[dict]:
    results = []
    for call_name in sorted({row.call_name for row in rows}):
        train_rows = [row for row in rows if row.call_name != call_name]
        test_rows = [row for row in rows if row.call_name == call_name]
        train_agent = [row.embedding for row in train_rows if row.used_for_training]
        if len(train_agent) < 3 or not test_rows:
            continue
        centroids = build_centroids(train_agent, n_clusters=n_clusters)
        max_outside, threshold = threshold_from_rows(train_rows, centroids)
        scored = score_rows(test_rows, centroids, threshold)
        scored.update(
            {
                "call": call_name,
                "training_agent_segments": len(train_agent),
                "threshold": round(threshold, 4),
                "max_outside_sim": round(max_outside, 4),
            }
        )
        results.append(scored)
    return results


def update_agents_json(
    agent_slug: str,
    agent_name: str,
    centroids: list[np.ndarray],
    rows: list[SegmentEmbedding],
    training_embeddings: list[np.ndarray],
    max_outside_sim: float,
    source_data_dir: Path,
    extra_training_stats: list[dict],
) -> dict:
    voiceprint_dir = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
    voiceprint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    paths = []
    for i, centroid in enumerate(centroids, start=1):
        file_name = f"{agent_slug}_gemini_campp_v{i}.npy"
        np.save(voiceprint_dir / file_name, centroid.astype(np.float32))
        paths.append(file_name)

    agents_path = voiceprint_dir / "agents.json"
    agents = {}
    if agents_path.exists():
        agents = json.loads(agents_path.read_text(encoding="utf-8"))
        backup_path = voiceprint_dir / f"agents.backup.{agent_slug}.gemini_campp.{timestamp}.json"
        shutil.copy2(agents_path, backup_path)
    else:
        backup_path = None

    train_rows = [row for row in rows if row.used_for_training]
    train_sims = [best_sim(embedding, centroids) for embedding in training_embeddings]
    mean_inside = float(np.mean(train_sims)) if train_sims else 0.0
    extra_calls = [str(item["call"]) for item in extra_training_stats]
    extra_seconds = sum(float(item.get("window_seconds_total") or 0.0) for item in extra_training_stats)

    agents[agent_slug] = {
        "agent_name": agent_name,
        "voiceprint_path": paths[0],
        "voiceprints": [
            {
                "path": path,
                "source": "gemini_agent_segments",
                "embedding_model": "cam++",
                "embedding_dim": 512,
            }
            for path in paths
        ],
        "n_voiceprints": len(paths),
        "embedding_model": "cam++",
        "embedding_dim": 512,
        "source": "gemini_agent_only_segments",
        "source_data_dir": str(source_data_dir),
        "used_calls": sorted(set([row.call_name for row in train_rows] + extra_calls)),
        "whole_call_agent_training": extra_training_stats,
        "n_training_segments": len(training_embeddings),
        "total_training_seconds": round(sum(row.duration for row in train_rows) + extra_seconds, 2),
        "mean_inside_sim": mean_inside,
        "max_outside_sim": float(min(max(max_outside_sim, 0.0), 0.50)),
        "updated_at_epoch": timestamp,
    }

    agents_path.write_text(json.dumps(agents, indent=2), encoding="utf-8")
    np.save(CALL_PROCESSOR_DIR / "data" / "enrolled_agent.npy", centroids[0].astype(np.float32))
    (CALL_PROCESSOR_DIR / "data" / "enrolled_agent_name.txt").write_text(agent_name, encoding="utf-8")

    return {
        "agents_json": str(agents_path),
        "backup": str(backup_path) if backup_path else None,
        "voiceprints": [str(voiceprint_dir / p) for p in paths],
        "legacy_enrolled_agent": str(CALL_PROCESSOR_DIR / "data" / "enrolled_agent.npy"),
        "legacy_enrolled_name": str(CALL_PROCESSOR_DIR / "data" / "enrolled_agent_name.txt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "traning_data" / "zak_raissi"))
    parser.add_argument("--agent-slug", default=DEFAULT_AGENT_SLUG)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--min-eval-dur", type=float, default=0.3)
    parser.add_argument("--min-train-dur", type=float, default=1.5)
    parser.add_argument("--max-train-dur", type=float, default=30.0)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--exclude-call", action="append", default=[])
    parser.add_argument("--whole-call-agent", action="append", default=[])
    parser.add_argument("--whole-window-s", type=float, default=5.0)
    parser.add_argument("--whole-step-s", type=float, default=2.5)
    parser.add_argument("--whole-rms-floor", type=float, default=0.003)
    parser.add_argument("--no-update", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    print(f"[load] data={data_dir}")
    all_calls = load_calls(data_dir)
    exclude_calls = set(args.exclude_call)
    whole_call_names = set(args.whole_call_agent)
    calls = [call for call in all_calls if call.name not in exclude_calls]
    for call_name in sorted(exclude_calls):
        print(f"[skip] {call_name}: excluded from labelled segment train/eval")
    whole_calls = [call for call in all_calls if call.name in whole_call_names]
    missing_whole = sorted(whole_call_names - {call.name for call in whole_calls})
    for call_name in missing_whole:
        print(f"[warn] whole-call agent source not found: {call_name}")
    if not calls:
        print("[error] No calls found")
        return 1

    for call in calls:
        agent_count = sum(1 for seg in call.segments if str(seg.get("speaker", "")).lower() == "agent")
        customer_count = sum(1 for seg in call.segments if str(seg.get("speaker", "")).lower() == "customer")
        print(
            f"[data] {call.name}: {agent_count} agent, {customer_count} customer, "
            f"audio={call.audio_path.name}, offset={call.label_offset_s:.2f}s"
        )

    print("[model] loading CAM++ embedding model")
    model = EmbeddingModel()
    model.load(force_cpu=True)
    print(f"[model] loaded {model.model_name} dim={model.dim}")

    try:
        rows = extract_embeddings(
            calls,
            model,
            min_eval_dur=args.min_eval_dur,
            min_train_dur=args.min_train_dur,
            max_train_dur=args.max_train_dur,
        )
        whole_agent_embeddings, whole_agent_stats = extract_whole_call_agent_embeddings(
            whole_calls,
            model,
            window_s=args.whole_window_s,
            step_s=args.whole_step_s,
            rms_floor=args.whole_rms_floor,
        )
    finally:
        model.unload()

    train_agent = [row.embedding for row in rows if row.used_for_training] + whole_agent_embeddings
    eval_agent = [row for row in rows if row.speaker == "agent"]
    eval_customer = [row for row in rows if row.speaker == "customer"]
    print(
        f"[summary] embedded={len(rows)} eval_agent={len(eval_agent)} "
        f"eval_customer={len(eval_customer)} train_agent_only={len(train_agent)} "
        f"whole_call_windows={len(whole_agent_embeddings)}"
    )
    if len(train_agent) < 3:
        print("[error] Need at least 3 usable Zak agent segments")
        return 1

    centroids = build_centroids(train_agent, n_clusters=args.clusters)
    max_outside, threshold = threshold_from_rows(rows, centroids)
    same_data = score_rows(rows, centroids, threshold)
    best_same_data = optimize_threshold(rows, centroids)
    same_data.update(
        {
            "threshold": round(threshold, 4),
            "max_outside_sim": round(max_outside, 4),
            "training_agent_segments": len(train_agent),
            "voiceprint_count": len(centroids),
        }
    )
    loo = leave_one_call_out(rows, n_clusters=args.clusters)

    report = {
        "agent_slug": args.agent_slug,
        "agent_name": args.agent_name,
        "data_dir": str(data_dir),
        "embedding_model": "cam++",
        "embedding_dim": 512,
        "calls": [
            {
                "call": call.name,
                "call_id": call.call_id,
                "audio": str(call.audio_path),
                "audio_duration_s": round(len(call.audio) / call.sr, 2),
                "label_offset_s": call.label_offset_s,
                "segments": len(call.segments),
            }
            for call in calls
        ],
        "same_data_accuracy": same_data,
        "best_same_data_threshold": best_same_data,
        "leave_one_call_out": loo,
        "whole_call_agent_training": whole_agent_stats,
        "note": "Voiceprints trained from Gemini speaker=agent segments plus any CLI whole-call Zak enrollment sources; customer segments used only for evaluation and threshold statistics.",
    }

    artifacts = {}
    if not args.no_update:
        artifacts = update_agents_json(
            args.agent_slug,
            args.agent_name,
            centroids,
            rows,
            train_agent,
            max_outside,
            data_dir,
            whole_agent_stats,
        )
        report["artifacts"] = artifacts

    report_path = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / "zak_gemini_campp_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[accuracy] same-data voiceprint check")
    print(
        f"  overall={same_data['overall_accuracy']}% "
        f"agent={same_data['agent_accuracy']}% ({same_data['agent_correct']}/{same_data['agent_total']}) "
        f"customer={same_data['customer_accuracy']}% ({same_data['customer_correct']}/{same_data['customer_total']}) "
        f"threshold={threshold:.4f}"
    )
    print(
        f"[accuracy] best same-data threshold={best_same_data['threshold']} "
        f"overall={best_same_data['overall_accuracy']}% "
        f"agent={best_same_data['agent_accuracy']}% "
        f"customer={best_same_data['customer_accuracy']}%"
    )
    print("[accuracy] leave-one-call-out")
    for item in loo:
        print(
            f"  {item['call']}: overall={item['overall_accuracy']}% "
            f"agent={item['agent_accuracy']}% customer={item['customer_accuracy']}% "
            f"threshold={item['threshold']}"
        )
    if artifacts:
        print("[saved]")
        print(f"  agents_json={artifacts['agents_json']}")
        print(f"  backup={artifacts['backup']}")
        for path in artifacts["voiceprints"]:
            print(f"  voiceprint={path}")
    print(f"  report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
