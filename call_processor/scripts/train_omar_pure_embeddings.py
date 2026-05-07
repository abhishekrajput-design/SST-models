#!/usr/bin/env python
"""Train a production Omar CAM++ voiceprint from pure labelled agent speech."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR))

from src.embedding_campp import EmbeddingModel, l2_norm  # noqa: E402

TARGET_SR = 16000
AGENT_SLUG = "omar_el_harchaoui"
AGENT_NAME = "Omar El Harchaoui"

BACKCHANNELS = {
    "hello", "hi", "yeah", "yes", "yep", "ok", "okay", "right",
    "sure", "no worries", "thank you", "thanks", "bye", "bye bye",
}

# General sales agent cues
AGENT_TEXT_CUES = (
    "calling you from car planet",
    "from car planet",
    "sold you the car",
    "how many years warranty",
    "years warranty",
    "give me one second",
    "sort this out",
    "bear with me",
    "leave a note",
    "you should receive an email",
    "warranty has been refunded",
    "direct refund",
    "won't affect the finance",
    "payment for the warranty",
    "finance company",
    "make an overpayment",
    "confirm your email",
    "refund usually takes",
    "has been processed",
    "service plan",
    "won't affect your service plan",
    "refund for the five years",
    "thank you for calling",
    "how can i help",
)

CUSTOMER_TEXT_CUES = (
    "i just been transfer",
    "i spoke with one of your",
    "one question please",
    "affect my finance",
    "my finance",
    "i tried to call",
    "very bad experience",
    "same payment method",
    "i'm thinking to pay",
    "i am thinking to pay",
    "i have an appointment",
    "this kind of inspection",
    "you should have my email",
    "i don't finish work",
    "i don't think we'll make it",
    "can you have a quick look",
)


@dataclass
class CallLabels:
    call_name: str
    call_id: str
    label_path: Path
    audio_path: Path
    audio: np.ndarray
    sr: int
    offset_s: float
    segments: list[dict]


@dataclass
class Row:
    call_name: str
    segment_idx: int
    speaker: str
    start: float
    end: float
    duration: float
    text: str
    embedding: np.ndarray
    source: str
    used_for_training: bool = False
    similarity: float = 0.0


def norm_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch == "'" else " " for ch in text).split())


def is_backchannel(text: str) -> bool:
    words = norm_text(text).split()
    joined = " ".join(words)
    return joined in BACKCHANNELS or (len(words) <= 1 and bool(words))


def has_any(text: str, cues: tuple[str, ...]) -> bool:
    norm = norm_text(text)
    return any(cue in norm for cue in cues)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    from pydub import AudioSegment

    sound = AudioSegment.from_file(path).set_channels(1).set_frame_rate(TARGET_SR)
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
    values = []
    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start:start + frame]
        values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    floor = float(np.percentile(arr, 20))
    peak = float(np.percentile(arr, 95))
    threshold = max(0.004, floor + (peak - floor) * 0.15)
    for i, value in enumerate(arr):
        if value >= threshold:
            return round(i * hop / sr, 3)
    return 0.0


def load_labelled_calls(data_dir: Path) -> tuple[list[CallLabels], list[dict]]:
    calls: list[CallLabels] = []
    skipped: list[dict] = []
    
    for label_path in sorted(data_dir.glob("*/data.json")):
        audio_path = label_path.parent / "audio_16k.wav"
        
        data = json.loads(label_path.read_text(encoding="utf-8"))
        call_id = str(data.get("call_id") or "").strip()
        
        if not audio_path.exists():
            skipped.append({"label": str(label_path), "call_id": call_id, "reason": "no matching audio"})
            continue
            
        audio, sr = load_audio(audio_path)
        offset_s = 0.0  # Our timestamps are already absolute for these calls
        calls.append(
            CallLabels(
                call_name=label_path.parent.name,
                call_id=call_id,
                label_path=label_path,
                audio_path=audio_path,
                audio=audio,
                sr=sr,
                offset_s=offset_s,
                segments=list(data.get("segments") or []),
            )
        )
    return calls, skipped


def speech_ratio(chunk: np.ndarray, sr: int) -> float:
    if chunk.size < int(sr * 0.05):
        return 0.0
    win = int(sr * 0.025)
    n = chunk.size // win
    if n < 4:
        return 0.0
    frames = chunk[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    floor = max(float(np.median(rms)) * 1.5, 0.005)
    return float((rms > floor).mean())


def extract_label_rows(
    calls: list[CallLabels],
    model: EmbeddingModel,
    min_eval_dur: float,
    min_train_dur: float,
    max_train_dur: float,
) -> tuple[list[Row], list[dict]]:
    rows: list[Row] = []
    skipped: list[dict] = []
    for call in calls:
        audio_end = len(call.audio) / call.sr
        for idx, seg in enumerate(call.segments, start=1):
            speaker = str(seg.get("speaker") or "").strip().lower()
            if speaker not in {"agent", "customer"}:
                continue
            start = float(seg.get("start") or 0.0) + call.offset_s
            end = float(seg.get("end") or 0.0) + call.offset_s
            if end <= start:
                continue
            if start >= audio_end or end > audio_end:
                skipped.append({
                    "call": call.call_name,
                    "segment": idx,
                    "reason": "outside audio",
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "audio_end": round(audio_end, 2),
                })
                continue
            duration = end - start
            if duration < min_eval_dur:
                continue
            chunk = call.audio[int(start * call.sr):int(end * call.sr)]
            if speech_ratio(chunk, call.sr) < 0.18:
                continue
            emb = model.embed_chunk(chunk, call.sr)
            if emb is None or np.isnan(emb).any():
                continue
            text = str(seg.get("text") or "")
            
            # Use stricter criteria: Not backchannel, within durations.
            train_ok = (
                speaker == "agent"
                and min_train_dur <= duration <= max_train_dur
                and not is_backchannel(text)
            )
            rows.append(
                Row(
                    call_name=call.call_name,
                    segment_idx=idx,
                    speaker=speaker,
                    start=start,
                    end=end,
                    duration=duration,
                    text=text,
                    embedding=l2_norm(np.asarray(emb, dtype=np.float32).squeeze()),
                    source="api_label",
                    used_for_training=train_ok,
                )
            )
    return rows, skipped


def build_centroids(embeddings: list[np.ndarray], n_clusters: int) -> list[np.ndarray]:
    if not embeddings:
        raise ValueError("No embeddings available")
    x = np.stack([l2_norm(e) for e in embeddings]).astype(np.float32)
    k = min(max(1, n_clusters), len(x))
    if k == 1:
        return [l2_norm(x.mean(axis=0)).astype(np.float32)]
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(x)
    centroids: list[np.ndarray] = []
    for cid in range(k):
        part = x[labels == cid]
        if len(part):
            centroids.append(l2_norm(part.mean(axis=0)).astype(np.float32))
    return centroids


def best_sim(embedding: np.ndarray, centroids: list[np.ndarray]) -> float:
    e = l2_norm(embedding)
    return float(max(float(np.dot(c, e)) for c in centroids))


def score(rows: list[Row], centroids: list[np.ndarray], threshold: float) -> dict:
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
        if not ok and len(errors) < 30:
            errors.append({
                "call": row.call_name,
                "segment": row.segment_idx,
                "speaker": row.speaker,
                "predicted": pred,
                "similarity": round(sim, 4),
                "text": row.text[:120],
            })
    return {
        "threshold": round(float(threshold), 4),
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


def best_threshold(rows: list[Row], centroids: list[np.ndarray]) -> dict:
    best = None
    for threshold in np.arange(0.10, 0.951, 0.005):
        item = score(rows, centroids, float(threshold))
        if best is None or item["overall_accuracy"] > best["overall_accuracy"]:
            best = item
    return best or score(rows, centroids, 0.5)


def update_agents_json(
    centroids: list[np.ndarray],
    train_rows: list[Row],
    customer_rows: list[Row],
    labels_dir: Path,
    report_name: str,
    dry_run: bool,
    activate: bool,
) -> dict:
    voiceprint_dir = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
    voiceprint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    paths = []
    for idx, centroid in enumerate(centroids, start=1):
        name = f"{AGENT_SLUG}_pure_campp_v{idx}.npy"
        path = voiceprint_dir / name
        if not dry_run:
            np.save(path, centroid.astype(np.float32))
        paths.append(name)

    customer_sims = [best_sim(row.embedding, centroids) for row in customer_rows]
    agent_sims = [best_sim(row.embedding, centroids) for row in train_rows]
    max_outside = float(np.percentile(customer_sims, 95)) if customer_sims else 0.50
    mean_inside = float(np.mean(agent_sims)) if agent_sims else 0.0

    backup = None
    agents_path = voiceprint_dir / "agents.json"
    if not dry_run and activate:
        backup = voiceprint_dir / f"agents.backup.{AGENT_SLUG}.pure_campp.{timestamp}.json"
        agents = json.loads(agents_path.read_text(encoding="utf-8")) if agents_path.exists() else {}
        if agents_path.exists():
            shutil.copy2(agents_path, backup)
            
        agents[AGENT_SLUG] = {
            "agent_name": AGENT_NAME,
            "voiceprint_path": paths[0],
            "voiceprints": [
                {
                    "path": path,
                    "source": "pure_api_agent_segments",
                    "embedding_model": "cam++",
                    "embedding_dim": 512,
                }
                for path in paths
            ],
            "n_voiceprints": len(paths),
            "embedding_model": "cam++",
            "embedding_dim": 512,
            "source": "pure_api_agent_segments",
            "source_labels_dir": str(labels_dir),
            "purity_report": report_name,
            "n_training_segments": len(train_rows),
            "total_training_seconds": round(sum(row.duration for row in train_rows), 2),
            "mean_inside_sim": round(mean_inside, 4),
            "max_outside_sim": round(max_outside, 4),
            "updated_at_epoch": timestamp,
        }
        agents_path.write_text(json.dumps(agents, indent=2), encoding="utf-8")

    return {
        "voiceprints": [str(voiceprint_dir / p) for p in paths],
        "agents_json": str(agents_path),
        "agents_backup": str(backup) if backup else None,
        "activated": bool(activate and not dry_run),
        "mean_inside_sim": round(mean_inside, 4),
        "max_outside_sim": round(max_outside, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "traning_data" / "omar_el_harchaoui"))
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--min-eval-dur", type=float, default=0.8)
    parser.add_argument("--min-train-dur", type=float, default=1.5)
    parser.add_argument("--max-train-dur", type=float, default=18.0)
    parser.add_argument("--activate", action="store_true",
                        help="replace the production agents.json entry only if validation meets the activation gate")
    parser.add_argument("--min-activation-accuracy", type=float, default=85.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    calls, skipped_calls = load_labelled_calls(data_dir)
    print(f"[data] calls={len(calls)} skipped_calls={len(skipped_calls)}")
    for call in calls:
        print(
            f"[data] {call.call_name}: call_id={call.call_id} audio={call.audio_path.name} "
            f"segments={len(call.segments)} offset={call.offset_s:.2f}s"
        )

    model = EmbeddingModel()
    print("[model] loading CAM++")
    model.load(force_cpu=True)
    try:
        label_rows, skipped_segments = extract_label_rows(
            calls,
            model,
            min_eval_dur=args.min_eval_dur,
            min_train_dur=args.min_train_dur,
            max_train_dur=args.max_train_dur,
        )
    finally:
        model.unload()

    agent_candidates = [row for row in label_rows if row.used_for_training]
    customer_rows = [row for row in label_rows if row.speaker == "customer"]
    
    # Use all valid customer rows for calibration
    customer_calibration_rows = customer_rows
    
    eval_rows = [row for row in label_rows if row.speaker in {"agent", "customer"}]
    anchor_rows = agent_candidates
    
    if len(anchor_rows) < 3:
        print("[error] Not enough pure embeddings")
        return 1

    initial = build_centroids([row.embedding for row in anchor_rows], n_clusters=args.clusters)
    for row in agent_candidates:
        row.similarity = best_sim(row.embedding, initial)
    sims = [row.similarity for row in agent_candidates]
    purity_floor = float(np.percentile(sims, 20)) if sims else 0.0
    purity_floor = max(0.18, min(purity_floor, 0.55))
    
    # Filter outliers
    pure_label_rows = [row for row in agent_candidates if row.similarity >= purity_floor]
    train_rows = pure_label_rows

    final_centroids = build_centroids([row.embedding for row in train_rows], n_clusters=args.clusters)
    customer_sims = [best_sim(row.embedding, final_centroids) for row in customer_calibration_rows]
    customer_p95 = float(np.percentile(customer_sims, 95)) if customer_sims else 0.50
    calibrated_threshold = min(max(customer_p95 + 0.06, 0.34), 0.92)
    same_data = score(eval_rows, final_centroids, calibrated_threshold)
    best_same = best_threshold(eval_rows, final_centroids)
    
    activation_eligible = (
        same_data["overall_accuracy"] >= args.min_activation_accuracy
        and same_data["agent_accuracy"] >= args.min_activation_accuracy
        and same_data["customer_accuracy"] >= args.min_activation_accuracy
    )
    activate = bool(args.activate and activation_eligible)

    report_name = "omar_pure_campp_training_report.json"
    artifacts = update_agents_json(
        final_centroids,
        train_rows,
        customer_calibration_rows,
        data_dir,
        report_name,
        dry_run=args.dry_run,
        activate=activate,
    )

    report = {
        "agent_slug": AGENT_SLUG,
        "agent_name": AGENT_NAME,
        "data_dir": str(data_dir),
        "calls": [
            {
                "call": call.call_name,
                "call_id": call.call_id,
                "audio": str(call.audio_path),
                "segments": len(call.segments),
                "offset_s": call.offset_s,
                "audio_duration_s": round(len(call.audio) / call.sr, 2),
            }
            for call in calls
        ],
        "skipped_calls": skipped_calls,
        "skipped_segments": skipped_segments[:100],
        "label_rows": len(label_rows),
        "eval_rows": len(eval_rows),
        "agent_candidates": len(agent_candidates),
        "customer_calibration_rows": len(customer_calibration_rows),
        "pure_label_rows": len(pure_label_rows),
        "training_rows": len(train_rows),
        "purity_floor": round(purity_floor, 4),
        "customer_p95": round(customer_p95, 4),
        "calibrated_threshold": round(calibrated_threshold, 4),
        "same_data_accuracy": same_data,
        "best_same_data_threshold": best_same,
        "activation_requested": bool(args.activate),
        "activation_eligible": bool(activation_eligible),
        "activation_min_accuracy": float(args.min_activation_accuracy),
        "activated": bool(activate and not args.dry_run),
        "artifacts": artifacts,
        "dry_run": args.dry_run,
    }
    report_path = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / report_name
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))

    print("[summary]")
    print(f"  label_rows={len(label_rows)}")
    print(f"  pure_label_rows={len(pure_label_rows)}/{len(agent_candidates)} training_rows={len(train_rows)}")
    print(f"  threshold={calibrated_threshold:.4f} customer_p95={customer_p95:.4f}")
    print(
        f"  same-data overall={same_data['overall_accuracy']}% "
        f"agent={same_data['agent_accuracy']}% customer={same_data['customer_accuracy']}%"
    )
    print(
        f"  best-threshold={best_same['threshold']} overall={best_same['overall_accuracy']}% "
        f"agent={best_same['agent_accuracy']}% customer={best_same['customer_accuracy']}%"
    )
    print(
        "  activation="
        f"{'yes' if activate and not args.dry_run else 'no'} "
        f"(eligible={activation_eligible}, requested={args.activate}, min={args.min_activation_accuracy:.1f}%)"
    )
    print(f"  report={report_path if not args.dry_run else '<dry-run>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
