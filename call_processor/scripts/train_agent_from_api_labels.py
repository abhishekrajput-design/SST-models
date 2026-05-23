#!/usr/bin/env python
"""Train a production CAM++ voiceprint for ANY agent from API-labelled calls.

Generalized from train_omar_pure_embeddings.py. Works for any agent_name
found in the Audiofy API speaker_json ground truth.

Usage:
  python train_agent_from_api_labels.py \
      --agent-slug omar_el_harchaoui \
      --agent-name "Omar El Harchaoui" \
      --data-dir traning_data/omar_el_harchaoui \
      --activate

  python train_agent_from_api_labels.py \
      --agent-slug hussein_mohamed \
      --agent-name "Hussein Mohamed" \
      --data-dir traning_data/hussein_labels \
      --clusters 4 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR))

from src.embedding_campp import EmbeddingModel, l2_norm  # noqa: E402

TARGET_SR = 16000

BACKCHANNELS = {
    "hello", "hi", "yeah", "yes", "yep", "ok", "okay", "right",
    "sure", "no worries", "thank you", "thanks", "bye", "bye bye",
}

# Call-level poisoning gates (mirror of daily_training_daemon.py defaults).
# A call is dropped from training before embeddings are computed if any gate
# trips — protects the trained voiceprint from Audiofy mis-labels.
POISON_MIN_CUSTOMER = 2
POISON_MIN_AGENT_MEAN_SCORE = 0.50
POISON_MIN_HQ_RATIO = 0.25
POISON_HQ_SCORE = 0.60

VOICEMAIL_OR_SYSTEM_CUES = (
    "voicemail",
    "not available",
    "leave a message",
    "record your message",
    "after the tone",
    "press the hash key",
    "press 1",
    "vodafone voicemail",
    "ee voicemail",
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
    avg_score: float
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


def is_voicemail_or_system_text(text: str) -> bool:
    normalized = norm_text(text)
    return any(cue in normalized for cue in VOICEMAIL_OR_SYSTEM_CUES)


def has_enough_training_content(text: str) -> bool:
    return sum(1 for ch in str(text) if ch.isalpha()) >= 6


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    from pydub import AudioSegment
    sound = AudioSegment.from_file(path).set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
    scale = float(1 << (8 * sound.sample_width - 1))
    if scale > 0:
        samples = samples / scale
    return samples.astype(np.float32), TARGET_SR


def poison_check(segments: list[dict]) -> tuple[bool, str]:
    """Reject a call whose data.json labels look unreliable.

    Mirrors the daemon's call_quality_check so manual training runs from
    on-disk data dirs get the same protection. Returns (ok, reason).
    """
    agents = [s for s in segments if str(s.get("speaker") or "").lower() == "agent"]
    customers = [s for s in segments if str(s.get("speaker") or "").lower() == "customer"]
    if len(customers) < POISON_MIN_CUSTOMER:
        return False, f"only {len(customers)} customer segment(s)"
    if not agents:
        return False, "no agent segments"
    scores = [float(s.get("avg_score") if s.get("avg_score") is not None else 0.85) for s in agents]
    mean = sum(scores) / len(scores)
    if mean < POISON_MIN_AGENT_MEAN_SCORE:
        return False, f"agent mean avg_score {mean:.3f} < {POISON_MIN_AGENT_MEAN_SCORE}"
    high_q = sum(1 for x in scores if x >= POISON_HQ_SCORE)
    ratio = high_q / len(agents)
    if ratio < POISON_MIN_HQ_RATIO:
        return False, f"only {high_q}/{len(agents)} ({ratio:.0%}) agent phrases >= {POISON_HQ_SCORE}"
    return True, ""


def load_labelled_calls(data_dir: Path) -> tuple[list[CallLabels], list[dict]]:
    """Load call data from directories containing data.json + audio_16k.wav."""
    calls: list[CallLabels] = []
    skipped: list[dict] = []

    for label_path in sorted(data_dir.glob("*/data.json")):
        if label_path.parent.name.startswith("_"):
            skipped.append({
                "label": str(label_path),
                "call_id": "",
                "reason": "excluded folder",
            })
            continue
        audio_path = label_path.parent / "audio_16k.wav"
        data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        call_id = str(data.get("call_id") or "").strip()
        call_segments = list(data.get("segments") or [])

        # Call-level poisoning gate — drop calls with unreliable labels before
        # spending CPU/GPU embedding their segments.
        ok, reason = poison_check(call_segments)
        if not ok:
            print(f"  [poison-skip] {label_path.parent.name} ({call_id[:12]}): {reason}", flush=True)
            skipped.append({
                "label": str(label_path),
                "call_id": call_id,
                "reason": f"poisoning gate: {reason}",
            })
            continue

        if not audio_path.exists():
            # Try other common audio names
            for alt in ("audio.mp3", "audio.wav", "call.mp3"):
                alt_path = label_path.parent / alt
                if alt_path.exists():
                    audio_path = alt_path
                    break
            else:
                audio_candidates = sorted(
                    p for p in label_path.parent.iterdir()
                    if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
                )
                if len(audio_candidates) == 1:
                    audio_path = audio_candidates[0]
                else:
                    skipped.append({"label": str(label_path), "call_id": call_id, "reason": "no matching audio"})
                    continue

        audio, sr = load_audio(audio_path)
        calls.append(
            CallLabels(
                call_name=label_path.parent.name,
                call_id=call_id,
                label_path=label_path,
                audio_path=audio_path,
                audio=audio,
                sr=sr,
                offset_s=0.0,
                segments=call_segments,
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
    min_train_score: float = 0.60,
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
                    "call": call.call_name, "segment": idx,
                    "reason": "outside audio",
                    "start": round(start, 2), "end": round(end, 2),
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
            avg_score = float(seg.get("avg_score") if seg.get("avg_score") is not None else 0.85)
            train_ok = (
                speaker == "agent"
                and min_train_dur <= duration <= max_train_dur
                and avg_score >= min_train_score
                and not is_backchannel(text)
                and not is_voicemail_or_system_text(text)
                and has_enough_training_content(text)
            )
            rows.append(
                Row(
                    call_name=call.call_name,
                    segment_idx=idx,
                    speaker=speaker,
                    start=start, end=end, duration=duration,
                    text=text,
                    avg_score=avg_score,
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
                "call": row.call_name, "segment": row.segment_idx,
                "speaker": row.speaker, "predicted": pred,
                "similarity": round(sim, 4), "text": row.text[:120],
            })
    return {
        "threshold": round(float(threshold), 4),
        "segments": total,
        "overall_accuracy": round(correct / total * 100, 2) if total else 0.0,
        "agent_accuracy": round(agent_correct / agent_total * 100, 2) if agent_total else 0.0,
        "customer_accuracy": round(customer_correct / customer_total * 100, 2) if customer_total else 0.0,
        "agent_correct": agent_correct, "agent_total": agent_total,
        "customer_correct": customer_correct, "customer_total": customer_total,
        "errors": errors,
    }


def best_threshold(rows: list[Row], centroids: list[np.ndarray]) -> dict:
    best = None
    for threshold in np.arange(0.10, 0.951, 0.005):
        item = score(rows, centroids, float(threshold))
        if best is None or item["overall_accuracy"] > best["overall_accuracy"]:
            best = item
    return best or score(rows, centroids, 0.5)


def leave_one_call_out(
    rows: list[Row],
    n_clusters: int,
    threshold_margin: float = 0.06,
) -> dict:
    """Leave-one-call-out cross-validation. Returns aggregate accuracy."""
    call_names = sorted(set(row.call_name for row in rows))
    if len(call_names) < 2:
        return {"error": "need >= 2 calls for LOCO", "n_calls": len(call_names)}

    all_correct = 0
    all_total = 0
    per_call = []

    for held_out in call_names:
        train_rows = [r for r in rows if r.call_name != held_out and r.used_for_training]
        eval_rows = [r for r in rows if r.call_name == held_out]
        if len(train_rows) < 3 or not eval_rows:
            continue

        centroids = build_centroids([r.embedding for r in train_rows], n_clusters)
        customer_rows = [r for r in rows if r.call_name != held_out and r.speaker == "customer"]
        customer_sims = [best_sim(r.embedding, centroids) for r in customer_rows]
        customer_p95 = float(np.percentile(customer_sims, 95)) if customer_sims else 0.50
        threshold = min(max(customer_p95 + threshold_margin, 0.34), 0.92)

        result = score(eval_rows, centroids, threshold)
        per_call.append({
            "held_out": held_out,
            "threshold": round(threshold, 4),
            "accuracy": result["overall_accuracy"],
            "agent_accuracy": result["agent_accuracy"],
            "customer_accuracy": result["customer_accuracy"],
            "segments": result["segments"],
        })
        all_correct += int(result["overall_accuracy"] * result["segments"] / 100)
        all_total += result["segments"]

    return {
        "n_calls": len(call_names),
        "n_folds": len(per_call),
        "overall_accuracy": round(all_correct / all_total * 100, 2) if all_total else 0.0,
        "per_call": per_call,
    }


def update_agents_json(
    agent_slug: str,
    agent_name: str,
    centroids: list[np.ndarray],
    train_rows: list[Row],
    customer_rows: list[Row],
    labels_dir: Path,
    report_name: str,
    dry_run: bool,
    activate: bool,
    compare_existing: bool = True,
) -> dict:
    voiceprint_dir = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
    voiceprint_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = voiceprint_dir / "_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())

    # Compute quality metrics first so we know whether to write to production.
    customer_sims = [best_sim(row.embedding, centroids) for row in customer_rows]
    agent_sims = [best_sim(row.embedding, centroids) for row in train_rows]
    max_outside = float(np.percentile(customer_sims, 95)) if customer_sims else 0.50
    mean_inside = float(np.mean(agent_sims)) if agent_sims else 0.0

    # Compare with existing voiceprint quality
    existing_better = False
    existing_stats = {}
    backup = None
    agents_path = voiceprint_dir / "agents.json"

    if agents_path.exists():
        agents = json.loads(agents_path.read_text(encoding="utf-8"))
        existing_entry = agents.get(agent_slug, {})
        if compare_existing and existing_entry:
            existing_inside = float(existing_entry.get("mean_inside_sim") or 0)
            existing_outside = float(existing_entry.get("max_outside_sim") or 1)
            existing_stats = {
                "existing_mean_inside": round(existing_inside, 4),
                "existing_max_outside": round(existing_outside, 4),
            }
            # Only activate if strictly better: higher inside AND lower outside
            if mean_inside <= existing_inside and max_outside >= existing_outside:
                existing_better = True
    else:
        agents = {}

    should_activate = activate and not existing_better and not dry_run

    # Write candidate .npy files. CRITICAL: only write to the production path
    # (which is what agents.json references) when we are activating. Otherwise
    # write to _candidates/ so debugging/inspection is still possible but the
    # production voiceprints are never silently overwritten by a rejected run.
    paths: list[str] = []
    for idx, centroid in enumerate(centroids, start=1):
        prod_name = f"{agent_slug}_pure_campp_v{idx}.npy"
        if should_activate:
            target = voiceprint_dir / prod_name
        elif not dry_run:
            target = candidate_dir / f"{agent_slug}_pure_campp_v{idx}.{timestamp}.npy"
        else:
            target = None
        if target is not None:
            np.save(target, centroid.astype(np.float32))
        # agents.json (when activated) always points at the production filename
        paths.append(prod_name)

    if should_activate:
        backup = voiceprint_dir / f"agents.backup.{agent_slug}.auto.{timestamp}.json"
        shutil.copy2(agents_path, backup)

        agents[agent_slug] = {
            "agent_name": agent_name,
            "voiceprint_path": paths[0],
            "voiceprints": [
                {
                    "path": path,
                    "source": "daily_auto_api_labels",
                    "embedding_model": "cam++",
                    "embedding_dim": 512,
                }
                for path in paths
            ],
            "n_voiceprints": len(paths),
            "embedding_model": "cam++",
            "embedding_dim": 512,
            "source": "daily_auto_api_labels",
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
        "activated": should_activate,
        "blocked_by_existing": existing_better,
        "mean_inside_sim": round(mean_inside, 4),
        "max_outside_sim": round(max_outside, 4),
        **existing_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CAM++ voiceprint for any agent")
    parser.add_argument("--agent-slug", required=True, help="e.g. omar_el_harchaoui")
    parser.add_argument("--agent-name", required=True, help="e.g. 'Omar El Harchaoui'")
    parser.add_argument("--data-dir", required=True, help="Path to labelled call data")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--min-eval-dur", type=float, default=0.8)
    parser.add_argument("--min-train-dur", type=float, default=1.5)
    parser.add_argument("--max-train-dur", type=float, default=18.0)
    parser.add_argument("--min-train-score", type=float, default=0.60,
                        help="Minimum API avg_score for agent segments used in enrollment")
    parser.add_argument("--min-activation-accuracy", type=float, default=85.0)
    parser.add_argument("--threshold-margin", type=float, default=0.06)
    parser.add_argument("--compare-existing", action="store_true", default=True,
                        help="Only activate if strictly better than current voiceprint")
    parser.add_argument("--no-compare", action="store_true",
                        help="Disable comparison with existing voiceprint")
    parser.add_argument("--activate", action="store_true",
                        help="Replace production agents.json if gates pass")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-out",
                        help="Optional path to write the JSON report, including dry-run reports")
    parser.add_argument("--loco", action="store_true", default=True,
                        help="Run leave-one-call-out validation")
    parser.add_argument("--skip-loco", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"[error] data-dir not found: {data_dir}")
        return 1

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
            calls, model,
            min_eval_dur=args.min_eval_dur,
            min_train_dur=args.min_train_dur,
            max_train_dur=args.max_train_dur,
            min_train_score=args.min_train_score,
        )
    finally:
        model.unload()

    agent_candidates = [row for row in label_rows if row.used_for_training]
    customer_rows = [row for row in label_rows if row.speaker == "customer"]
    eval_rows = [row for row in label_rows if row.speaker in {"agent", "customer"}]

    if len(agent_candidates) < 3:
        print(f"[error] Not enough pure agent embeddings ({len(agent_candidates)})")
        return 1

    # ── Leave-one-call-out validation ────────────────────────────────────────
    loco_result = {}
    if not args.skip_loco and len(set(r.call_name for r in label_rows)) >= 2:
        print("[loco] Running leave-one-call-out validation...")
        loco_result = leave_one_call_out(label_rows, args.clusters, args.threshold_margin)
        print(f"[loco] overall_accuracy={loco_result.get('overall_accuracy', 0)}%")
        for fold in loco_result.get("per_call", []):
            print(f"  held_out={fold['held_out']} accuracy={fold['accuracy']}%")

    # ── Purity filtering ─────────────────────────────────────────────────────
    initial = build_centroids([row.embedding for row in agent_candidates], n_clusters=args.clusters)
    for row in agent_candidates:
        row.similarity = best_sim(row.embedding, initial)
    sims = [row.similarity for row in agent_candidates]
    purity_floor = float(np.percentile(sims, 20)) if sims else 0.0
    purity_floor = max(0.18, min(purity_floor, 0.55))
    pure_label_rows = [row for row in agent_candidates if row.similarity >= purity_floor]
    train_rows = pure_label_rows

    # ── Build final centroids ────────────────────────────────────────────────
    final_centroids = build_centroids([row.embedding for row in train_rows], n_clusters=args.clusters)
    customer_sims = [best_sim(row.embedding, final_centroids) for row in customer_rows]
    customer_p95 = float(np.percentile(customer_sims, 95)) if customer_sims else 0.50
    agent_sims = [best_sim(row.embedding, final_centroids) for row in train_rows]
    mean_inside = float(np.mean(agent_sims)) if agent_sims else 0.0
    calibrated_threshold = min(max(customer_p95 + args.threshold_margin, 0.34), 0.92)
    same_data = score(eval_rows, final_centroids, calibrated_threshold)
    best_same = best_threshold(eval_rows, final_centroids)

    # ── Activation gate ──────────────────────────────────────────────────────
    loco_ok = True
    if loco_result and int(loco_result.get("n_folds") or 0) > 0:
        loco_ok = float(loco_result.get("overall_accuracy") or 0.0) >= args.min_activation_accuracy
    activation_eligible = (
        same_data["overall_accuracy"] >= args.min_activation_accuracy
        and same_data["agent_accuracy"] >= args.min_activation_accuracy
        and same_data["customer_accuracy"] >= args.min_activation_accuracy
        and mean_inside >= 0.60
        and (customer_p95 <= 0.42 or args.no_compare)
        and loco_ok
    )
    activate = bool(args.activate and activation_eligible)
    compare_existing = not args.no_compare

    report_name = f"{args.agent_slug}_auto_training_report.json"
    artifacts = update_agents_json(
        args.agent_slug, args.agent_name,
        final_centroids, train_rows, customer_rows,
        data_dir, report_name,
        dry_run=args.dry_run, activate=activate,
        compare_existing=compare_existing,
    )

    report = {
        "agent_slug": args.agent_slug,
        "agent_name": args.agent_name,
        "data_dir": str(data_dir),
        "calls": [
            {
                "call": call.call_name, "call_id": call.call_id,
                "audio": str(call.audio_path),
                "segments": len(call.segments),
                "audio_duration_s": round(len(call.audio) / call.sr, 2),
            }
            for call in calls
        ],
        "skipped_calls": skipped_calls,
        "label_rows": len(label_rows),
        "eval_rows": len(eval_rows),
        "agent_candidates": len(agent_candidates),
        "customer_calibration_rows": len(customer_rows),
        "pure_label_rows": len(pure_label_rows),
        "training_rows": len(train_rows),
        "min_train_score": float(args.min_train_score),
        "purity_floor": round(purity_floor, 4),
        "customer_p95": round(customer_p95, 4),
        "calibrated_threshold": round(calibrated_threshold, 4),
        "same_data_accuracy": same_data,
        "best_same_data_threshold": best_same,
        "loco_result": loco_result,
        "activation_requested": bool(args.activate),
        "activation_eligible": bool(activation_eligible),
        "activation_min_accuracy": float(args.min_activation_accuracy),
        "activated": artifacts.get("activated", False),
        "blocked_by_existing": artifacts.get("blocked_by_existing", False),
        "artifacts": artifacts,
        "dry_run": args.dry_run,
    }
    report_path = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / report_name
    if args.report_out:
        report_out = Path(args.report_out).resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))

    print("[summary]")
    print(f"  agent={args.agent_slug} ({args.agent_name})")
    print(f"  label_rows={len(label_rows)} pure={len(pure_label_rows)}/{len(agent_candidates)} training={len(train_rows)}")
    print(f"  threshold={calibrated_threshold:.4f} customer_p95={customer_p95:.4f}")
    print(
        f"  same-data overall={same_data['overall_accuracy']}% "
        f"agent={same_data['agent_accuracy']}% customer={same_data['customer_accuracy']}%"
    )
    if loco_result and "overall_accuracy" in loco_result:
        print(f"  loco overall={loco_result['overall_accuracy']}%")
    print(
        f"  activation={'yes' if artifacts.get('activated') else 'no'} "
        f"(eligible={activation_eligible}, requested={args.activate})"
    )
    if artifacts.get("blocked_by_existing"):
        print("  ⚠ blocked: existing voiceprint has equal or better metrics")
    print(f"  report={report_path if not args.dry_run else '<dry-run>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
