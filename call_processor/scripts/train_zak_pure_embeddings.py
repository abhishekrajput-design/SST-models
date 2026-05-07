#!/usr/bin/env python
"""Train a production Zak CAM++ voiceprint from pure labelled agent speech.

This intentionally does not create speaker labels for a call. It only builds
agent voiceprints from Gemini-labelled agent segments plus trusted clean Zak
clips, then tests those voiceprints against labelled customer segments.
"""
from __future__ import annotations

import argparse
import hashlib
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
AGENT_SLUG = "zak_raissi_barnet"
AGENT_NAME = "Zak Raissi Barnet"

BACKCHANNELS = {
    "hello", "hi", "yeah", "yes", "yep", "ok", "okay", "right",
    "sure", "no worries", "thank you", "thanks", "bye", "bye bye",
}

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
    raw_start: float = 0.0
    raw_end: float = 0.0
    purity_reason: str = ""


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


def audio_candidates(audio_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in audio_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac"}:
            continue
        if path.name.startswith("audio_16k"):
            parent_audio = next(
                (
                    p for p in path.parent.iterdir()
                    if p.is_file()
                    and p != path
                    and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
                ),
                None,
            )
            if parent_audio:
                out[parent_audio.stem] = path
        out[path.stem] = path
    return out


def primary_audio_in_folder(folder: Path) -> Path | None:
    audio_files = [
        path for path in sorted(folder.iterdir())
        if path.is_file()
        and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
        and not path.name.startswith("audio_16k")
    ]
    if audio_files:
        return audio_files[0]
    audio_files = [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
    ]
    return audio_files[0] if audio_files else None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def label_offset_s(offset_mode: str, detected_offset_s: float, source: str = "") -> float:
    if offset_mode == "none":
        return 0.0
    if offset_mode == "auto-source" and str(source or "").lower().startswith("local_"):
        return 0.0
    return detected_offset_s


def load_folder_data_json_calls(
    audio_root: Path,
    offset_mode: str = "detected",
    skip_call_names: set[str] | None = None,
    seen_audio_hashes: set[str] | None = None,
    include_source_prefix: str = "",
    include_call_names: set[str] | None = None,
    exclude_call_names: set[str] | None = None,
    max_label_overrun_s: float = 3.0,
    max_label_overrun_ratio: float = 0.10,
) -> tuple[list[CallLabels], list[dict], set[str]]:
    calls: list[CallLabels] = []
    skipped: list[dict] = []
    skip_call_names = skip_call_names or set()
    include_call_names = include_call_names or set()
    exclude_call_names = exclude_call_names or set()
    seen_audio_hashes = seen_audio_hashes or set()

    for folder in sorted(audio_root.glob("call_*")):
        if not folder.is_dir():
            continue
        call_name = folder.name
        if call_name in skip_call_names:
            continue
        if include_call_names and call_name not in include_call_names:
            continue
        if call_name in exclude_call_names:
            skipped.append({"call": call_name, "reason": "excluded by request"})
            continue

        label_path = folder / "data.json"
        if not label_path.is_file():
            skipped.append({"label": str(label_path), "call": call_name, "reason": "missing data.json"})
            continue
        try:
            data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            skipped.append({"label": str(label_path), "call": call_name, "reason": f"invalid json: {exc}"})
            continue
        source = str(data.get("source") or "")
        if include_source_prefix and not source.startswith(include_source_prefix):
            skipped.append({
                "label": str(label_path),
                "call": call_name,
                "reason": f"source {source!r} does not match prefix {include_source_prefix!r}",
            })
            continue

        segments = list(data.get("segments") or [])
        if not segments:
            skipped.append({"label": str(label_path), "call": call_name, "reason": "no segments"})
            continue

        audio_path = primary_audio_in_folder(folder)
        if not audio_path:
            skipped.append({"label": str(label_path), "call": call_name, "reason": "no audio in folder"})
            continue

        try:
            audio_hash = file_sha256(audio_path)
        except Exception as exc:
            skipped.append({"label": str(label_path), "call": call_name, "reason": f"cannot hash audio: {exc}"})
            continue
        if audio_hash in seen_audio_hashes:
            skipped.append({
                "label": str(label_path),
                "call": call_name,
                "audio": str(audio_path),
                "reason": "duplicate audio hash",
            })
            continue

        audio, sr = load_audio(audio_path)
        audio_duration_s = len(audio) / sr
        max_label_end = max((float(seg.get("end") or 0.0) for seg in segments), default=0.0)
        allowed_overrun_s = max(max_label_overrun_s, audio_duration_s * max_label_overrun_ratio)
        if max_label_end > audio_duration_s + allowed_overrun_s:
            skipped.append({
                "label": str(label_path),
                "call": call_name,
                "audio": str(audio_path),
                "reason": "label timestamps exceed audio duration",
                "max_label_end_s": round(max_label_end, 2),
                "audio_duration_s": round(audio_duration_s, 2),
                "allowed_overrun_s": round(allowed_overrun_s, 2),
            })
            continue

        seen_audio_hashes.add(audio_hash)
        detected_offset_s = detect_leading_speech_offset(audio, sr)
        offset_s = label_offset_s(offset_mode, detected_offset_s, source)
        call_id = str(data.get("call_id") or audio_path.stem).strip()
        calls.append(
            CallLabels(
                call_name=call_name,
                call_id=call_id,
                label_path=label_path,
                audio_path=audio_path,
                audio=audio,
                sr=sr,
                offset_s=offset_s,
                segments=segments,
            )
        )

    return calls, skipped, seen_audio_hashes


def load_labelled_calls(
    labels_dir: Path,
    audio_root: Path,
    offset_mode: str = "detected",
    label_source: str = "both",
    include_source_prefix: str = "",
    include_call_names: set[str] | None = None,
    exclude_call_names: set[str] | None = None,
) -> tuple[list[CallLabels], list[dict]]:
    audio_by_stem = audio_candidates(audio_root)
    calls: list[CallLabels] = []
    skipped: list[dict] = []
    seen_audio_hashes: set[str] = set()
    if label_source in {"training-json", "both"}:
        for label_path in sorted(labels_dir.glob("gemini_labels_zak_call_*.json")):
            data = json.loads(label_path.read_text(encoding="utf-8"))
            call_id = str(data.get("call_id") or "").strip()
            call_name = label_path.stem.replace("gemini_labels_zak_", "")
            include_call_names = include_call_names or set()
            exclude_call_names = exclude_call_names or set()
            if include_call_names and call_name not in include_call_names:
                continue
            if call_name in exclude_call_names:
                skipped.append({"label": str(label_path), "call": call_name, "reason": "excluded by request"})
                continue
            audio_path = audio_by_stem.get(call_id)
            if not audio_path:
                skipped.append({"label": str(label_path), "call_id": call_id, "reason": "no matching audio"})
                continue
            audio, sr = load_audio(audio_path)
            for sibling in audio_path.parent.iterdir():
                if not sibling.is_file() or sibling.suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac"}:
                    continue
                try:
                    seen_audio_hashes.add(file_sha256(sibling))
                except Exception:
                    pass
            detected_offset_s = detect_leading_speech_offset(audio, sr)
            offset_s = label_offset_s(offset_mode, detected_offset_s, "gemini")
            calls.append(
                CallLabels(
                    call_name=call_name,
                    call_id=call_id,
                    label_path=label_path,
                    audio_path=audio_path,
                    audio=audio,
                    sr=sr,
                    offset_s=offset_s,
                    segments=list(data.get("segments") or []),
                )
            )
    if label_source in {"folder-data", "both"}:
        folder_calls, folder_skipped, seen_audio_hashes = load_folder_data_json_calls(
            audio_root,
            offset_mode=offset_mode,
            skip_call_names={call.call_name for call in calls},
            seen_audio_hashes=seen_audio_hashes,
            include_source_prefix=include_source_prefix,
            include_call_names=include_call_names,
            exclude_call_names=exclude_call_names,
        )
        calls.extend(folder_calls)
        skipped.extend(folder_skipped)
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


def segment_bounds(seg: dict, offset_s: float) -> tuple[float, float]:
    start = float(seg.get("start") or 0.0) + offset_s
    end = float(seg.get("end") or 0.0) + offset_s
    return start, end


def has_opposite_speaker_overlap(
    start: float,
    end: float,
    opposite_bounds: list[tuple[float, float]],
    gap_s: float,
) -> bool:
    for opp_start, opp_end in opposite_bounds:
        if start < (opp_end + gap_s) and end > (opp_start - gap_s):
            return True
    return False


def extract_label_rows(
    calls: list[CallLabels],
    model: EmbeddingModel,
    min_eval_dur: float,
    min_train_dur: float,
    max_train_dur: float,
    agent_filter: str = "cue",
    train_guard_s: float = 0.35,
    opposite_gap_s: float = 0.05,
) -> tuple[list[Row], list[dict]]:
    rows: list[Row] = []
    skipped: list[dict] = []
    for call in calls:
        audio_end = len(call.audio) / call.sr
        opposite_by_speaker = {"agent": [], "customer": []}
        for seg in call.segments:
            speaker = str(seg.get("speaker") or "").strip().lower()
            if speaker not in {"agent", "customer"}:
                continue
            start, end = segment_bounds(seg, call.offset_s)
            if end > start:
                opposite_by_speaker[speaker].append((start, end))
        for idx, seg in enumerate(call.segments, start=1):
            speaker = str(seg.get("speaker") or "").strip().lower()
            if speaker not in {"agent", "customer"}:
                continue
            raw_start, raw_end = segment_bounds(seg, call.offset_s)
            start, end = raw_start, raw_end
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
            raw_duration = end - start
            if raw_duration < min_eval_dur:
                continue
            text = str(seg.get("text") or "")
            purity_reason = ""
            train_ok = False
            if speaker == "agent":
                start = min(end, start + max(0.0, train_guard_s))
                end = max(start, end - max(0.0, train_guard_s))
                duration = end - start
                if duration < min_train_dur:
                    purity_reason = "too_short_after_guard"
                elif duration > max_train_dur:
                    purity_reason = "too_long"
                elif is_backchannel(text):
                    purity_reason = "backchannel"
                elif agent_filter != "all" and not has_any(text, AGENT_TEXT_CUES):
                    purity_reason = "missing_agent_text_cue"
                elif has_opposite_speaker_overlap(
                    start,
                    end,
                    opposite_by_speaker["customer"],
                    max(0.0, opposite_gap_s),
                ):
                    purity_reason = "touches_or_overlaps_customer"
                else:
                    train_ok = True
                    purity_reason = "pure_agent_guarded"
            else:
                duration = raw_duration

            if duration < min_eval_dur:
                continue
            chunk = call.audio[int(start * call.sr):int(end * call.sr)]
            if speech_ratio(chunk, call.sr) < 0.18:
                continue
            emb = model.embed_chunk(chunk, call.sr)
            if emb is None or np.isnan(emb).any():
                continue
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
                    source="gemini_label",
                    used_for_training=train_ok,
                    raw_start=raw_start,
                    raw_end=raw_end,
                    purity_reason=purity_reason,
                )
            )
    return rows, skipped


def extract_clean_clip_rows(clean_dirs: list[Path], model: EmbeddingModel) -> list[Row]:
    rows: list[Row] = []
    for clean_dir in clean_dirs:
        if not clean_dir.exists():
            continue
        for path in sorted(clean_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac"}:
                continue
            audio, sr = load_audio(path)
            window = int(2.0 * sr)
            stride = int(1.0 * sr)
            for n, start_i in enumerate(range(0, max(len(audio) - window + 1, 0), stride), start=1):
                chunk = audio[start_i:start_i + window]
                if speech_ratio(chunk, sr) < 0.18:
                    continue
                emb = model.embed_chunk(chunk, sr)
                if emb is None or np.isnan(emb).any():
                    continue
                start = start_i / sr
                end = (start_i + window) / sr
                rows.append(
                    Row(
                        call_name=clean_dir.name,
                        segment_idx=n,
                        speaker="agent",
                        start=start,
                        end=end,
                        duration=end - start,
                        text=path.name,
                        embedding=l2_norm(np.asarray(emb, dtype=np.float32).squeeze()),
                        source="clean_clip",
                        used_for_training=True,
                    )
                )
    return rows


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


def row_manifest(row: Row) -> dict:
    return {
        "source": row.source,
        "call": row.call_name,
        "segment": row.segment_idx,
        "speaker": row.speaker,
        "raw_start": round(float(row.raw_start), 3),
        "raw_end": round(float(row.raw_end), 3),
        "start": round(float(row.start), 3),
        "end": round(float(row.end), 3),
        "duration": round(float(row.duration), 3),
        "text": row.text,
        "purity_reason": row.purity_reason,
    }


def export_clean_training_clips(train_rows: list[Row], calls: list[CallLabels], out_dir: Path) -> list[dict]:
    import soundfile as sf

    calls_by_name = {call.call_name: call for call in calls}
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for idx, row in enumerate(train_rows, start=1):
        if row.source != "gemini_label":
            continue
        call = calls_by_name.get(row.call_name)
        if not call:
            continue
        start_i = max(0, int(row.start * call.sr))
        end_i = min(len(call.audio), int(row.end * call.sr))
        if end_i <= start_i:
            continue
        name = f"{idx:04d}_{row.call_name}_seg{row.segment_idx:03d}_{row.start:.2f}_{row.end:.2f}.wav"
        path = out_dir / name
        sf.write(path, call.audio[start_i:end_i].astype(np.float32), call.sr)
        exported.append({
            "path": str(path),
            "call": row.call_name,
            "segment": row.segment_idx,
            "start": round(float(row.start), 3),
            "end": round(float(row.end), 3),
            "duration": round(float(row.duration), 3),
        })
    return exported


def update_agents_json(
    centroids: list[np.ndarray],
    train_rows: list[Row],
    customer_rows: list[Row],
    labels_dir: Path,
    report_name: str,
    agent_slug: str,
    agent_name: str,
    dry_run: bool,
    activate: bool,
) -> dict:
    voiceprint_dir = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
    voiceprint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    paths = []
    for idx, centroid in enumerate(centroids, start=1):
        name = f"{agent_slug}_pure_campp_v{idx}.npy"
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
        backup = voiceprint_dir / f"agents.backup.{agent_slug}.pure_campp.{timestamp}.json"
        agents = json.loads(agents_path.read_text(encoding="utf-8")) if agents_path.exists() else {}
        if agents_path.exists():
            shutil.copy2(agents_path, backup)
        agents[agent_slug] = {
            "agent_name": agent_name,
            "voiceprint_path": paths[0],
            "voiceprints": [
                {
                    "path": path,
                    "source": "pure_agent_segments_plus_clean_clips",
                    "embedding_model": "cam++",
                    "embedding_dim": 512,
                }
                for path in paths
            ],
            "n_voiceprints": len(paths),
            "embedding_model": "cam++",
            "embedding_dim": 512,
            "source": "pure_agent_segments_plus_clean_clips",
            "source_labels_dir": str(labels_dir),
            "purity_report": report_name,
            "validation_report": report_name,
            "n_training_segments": len(train_rows),
            "total_training_seconds": round(sum(row.duration for row in train_rows), 2),
            "mean_inside_sim": round(mean_inside, 4),
            "max_outside_sim": round(max_outside, 4),
            "updated_at_epoch": timestamp,
        }
        agents_path.write_text(json.dumps(agents, indent=2), encoding="utf-8")
        np.save(CALL_PROCESSOR_DIR / "data" / "enrolled_agent.npy", centroids[0].astype(np.float32))
        (CALL_PROCESSOR_DIR / "data" / "enrolled_agent_name.txt").write_text(agent_name, encoding="utf-8")

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
    parser.add_argument("--agent-slug", default=AGENT_SLUG)
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--report-name", default="")
    parser.add_argument("--labels-dir", default=str(CALL_PROCESSOR_DIR / "data" / "training"))
    parser.add_argument("--audio-root", default=str(REPO_ROOT / "traning_data" / "zak_raissi"))
    parser.add_argument(
        "--label-source",
        choices=("training-json", "folder-data", "both"),
        default="both",
        help="Load legacy gemini_labels files, per-call data.json files, or both.",
    )
    parser.add_argument(
        "--include-source-prefix",
        default="",
        help="When loading folder data.json labels, keep only sources with this prefix.",
    )
    parser.add_argument("--include-call", action="append", default=[])
    parser.add_argument("--exclude-call", action="append", default=[])
    parser.add_argument("--clean-dir", action="append", default=None)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument(
        "--offset-mode",
        choices=("detected", "none", "auto-source"),
        default="detected",
        help="Use detected leading-speech offset, raw timestamps, or source-aware folder offsets.",
    )
    parser.add_argument("--min-eval-dur", type=float, default=0.8)
    parser.add_argument("--min-train-dur", type=float, default=1.5)
    parser.add_argument("--max-train-dur", type=float, default=18.0)
    parser.add_argument(
        "--train-guard-s",
        type=float,
        default=0.35,
        help="Trim this many seconds from both sides of Gemini agent segments before training.",
    )
    parser.add_argument(
        "--opposite-gap-s",
        type=float,
        default=0.05,
        help="Reject guarded agent segments that overlap or sit within this gap of customer labels.",
    )
    parser.add_argument(
        "--export-clean-clips",
        default="",
        help="Optional folder to write the final guarded agent training clips for audit.",
    )
    parser.add_argument(
        "--agent-filter",
        choices=("all", "cue"),
        default="all",
        help="Train from all Gemini speaker=agent rows, or only rows with strong agent text cues.",
    )
    parser.add_argument("--activate", action="store_true",
                        help="replace the production Zak agents.json entry only if validation meets the activation gate")
    parser.add_argument("--min-activation-accuracy", type=float, default=96.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.clean_dir is None:
        if args.agent_slug == AGENT_SLUG:
            args.clean_dir = [
                str(CALL_PROCESSOR_DIR / "data" / "agent_clean_clips" / "zak_local_train_20260423"),
                str(CALL_PROCESSOR_DIR / "data" / "agent_clean_clips" / "zak_raissi_barnet"),
            ]
        else:
            args.clean_dir = []

    labels_dir = Path(args.labels_dir).resolve()
    audio_root = Path(args.audio_root).resolve()
    clean_dirs = [Path(p).resolve() for p in args.clean_dir]

    calls, skipped_calls = load_labelled_calls(
        labels_dir,
        audio_root,
        offset_mode=args.offset_mode,
        label_source=args.label_source,
        include_source_prefix=args.include_source_prefix,
        include_call_names=set(args.include_call),
        exclude_call_names=set(args.exclude_call),
    )
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
            agent_filter=args.agent_filter,
            train_guard_s=args.train_guard_s,
            opposite_gap_s=args.opposite_gap_s,
        )
        clean_rows = extract_clean_clip_rows(clean_dirs, model)
    finally:
        model.unload()

    agent_candidates = [row for row in label_rows if row.used_for_training]
    customer_rows = [row for row in label_rows if row.speaker == "customer"]
    customer_calibration_rows = [
        row for row in customer_rows
        if has_any(row.text, CUSTOMER_TEXT_CUES) and not has_any(row.text, AGENT_TEXT_CUES)
    ]
    if len(customer_calibration_rows) < 10:
        customer_calibration_rows = [
            row for row in customer_rows
            if not has_any(row.text, AGENT_TEXT_CUES)
        ]
    eval_rows = [row for row in label_rows if row.speaker in {"agent", "customer"}]
    anchor_rows = clean_rows + agent_candidates
    if len(anchor_rows) < 3:
        print(f"[error] Not enough pure {args.agent_name} embeddings")
        return 1

    initial = build_centroids([row.embedding for row in anchor_rows], n_clusters=args.clusters)
    for row in agent_candidates:
        row.similarity = best_sim(row.embedding, initial)
    sims = [row.similarity for row in agent_candidates]
    purity_floor = float(np.percentile(sims, 20)) if sims else 0.0
    purity_floor = max(0.18, min(purity_floor, 0.55))
    pure_label_rows = [row for row in agent_candidates if row.similarity >= purity_floor]
    train_rows = clean_rows + pure_label_rows

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

    report_name = args.report_name or f"{args.agent_slug}_pure_campp_training_report.json"
    artifacts = update_agents_json(
        final_centroids,
        train_rows,
        customer_calibration_rows,
        labels_dir,
        report_name,
        args.agent_slug,
        args.agent_name,
        dry_run=args.dry_run,
        activate=activate,
    )
    exported_clean_clips = []
    if args.export_clean_clips and not args.dry_run:
        exported_clean_clips = export_clean_training_clips(train_rows, calls, Path(args.export_clean_clips).resolve())

    report = {
        "agent_slug": args.agent_slug,
        "agent_name": args.agent_name,
        "labels_dir": str(labels_dir),
        "audio_root": str(audio_root),
        "label_source": args.label_source,
        "include_source_prefix": args.include_source_prefix,
        "include_calls": args.include_call,
        "exclude_calls": args.exclude_call,
        "offset_mode": args.offset_mode,
        "agent_filter": args.agent_filter,
        "train_guard_s": args.train_guard_s,
        "opposite_gap_s": args.opposite_gap_s,
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
        "clean_clip_rows": len(clean_rows),
        "agent_candidates": len(agent_candidates),
        "customer_calibration_rows": len(customer_calibration_rows),
        "pure_label_rows": len(pure_label_rows),
        "training_rows": len(train_rows),
        "training_segments": [row_manifest(row) for row in train_rows],
        "exported_clean_clips": exported_clean_clips,
        "customer_calibration_segments": [row_manifest(row) for row in customer_calibration_rows],
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
        "note": (
            "Production voiceprints are trained from clean clips plus Gemini-labelled "
            "agent segments after short/backchannel/outside-audio/outlier filtering. "
            "Customer segments are used only for calibration and testing."
        ),
    }
    report_path = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / report_name
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))

    print("[summary]")
    print(f"  label_rows={len(label_rows)} clean_rows={len(clean_rows)}")
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
