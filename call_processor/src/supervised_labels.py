"""Apply trusted Gemini speaker labels to known local training calls."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "traning_data" / "zak_raissi"
TARGET_SR = 16000


@dataclass
class LabelMatch:
    call_dir: Path
    data_path: Path
    audio_path: Path
    data: dict


def _clean_stem(path_or_name: str) -> str:
    stem = Path(path_or_name).stem.lower()
    stem = re.sub(r"^enhanced_", "", stem)
    stem = re.sub(r"(_zak_retrain|_ui_parakeet|_ui|_retrain|_parakeet)$", "", stem)
    return stem


def _candidate_stems(paths: Iterable[str]) -> set[str]:
    stems: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        cleaned = _clean_stem(raw)
        stems.add(cleaned)
        parts = cleaned.split("_")
        if len(parts) >= 2 and re.match(r"^\d{9,}$", parts[0]):
            stems.add("_".join(parts[:2]))
    return {s for s in stems if s}


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    from pydub import AudioSegment

    sound = AudioSegment.from_file(path)
    sound = sound.set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
    scale = float(1 << (8 * sound.sample_width - 1))
    if scale > 0:
        samples = samples / scale
    return samples.astype(np.float32), TARGET_SR


def _detect_leading_speech_offset(path: Path) -> float:
    try:
        audio, sr = _load_audio(path)
    except Exception:
        return 0.0

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


def _audio_files(call_dir: Path) -> list[Path]:
    preferred = call_dir / "audio_16k.wav"
    files = []
    if preferred.exists():
        files.append(preferred)
    files.extend(
        p for p in sorted(call_dir.iterdir())
        if p.is_file()
        and p != preferred
        and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
    )
    return files


def find_label_match(audio_path: str = "", original_path: str = "") -> LabelMatch | None:
    if not TRAINING_ROOT.exists():
        return None

    requested = _candidate_stems([audio_path, original_path])
    if not requested:
        return None

    for call_dir in sorted(TRAINING_ROOT.glob("call_*")):
        data_path = call_dir / "data.json"
        if not data_path.exists():
            continue
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for audio in _audio_files(call_dir):
            audio_stem = _clean_stem(audio.name)
            if any(req == audio_stem or req.startswith(audio_stem) or audio_stem.startswith(req) for req in requested):
                return LabelMatch(call_dir=call_dir, data_path=data_path, audio_path=audio, data=data)
    return None


def build_supervised_segments(match: LabelMatch) -> tuple[list[dict], dict]:
    offset = _detect_leading_speech_offset(match.audio_path)
    agent_name = str(match.data.get("agent_name") or "Agent")
    if agent_name == "zak_local_20260423":
        agent_name = "Zak Raissi"

    out: list[dict] = []
    agent_turns = customer_turns = 0
    agent_seconds = customer_seconds = 0.0
    for idx, src in enumerate(match.data.get("segments") or [], start=1):
        role_raw = str(src.get("speaker") or "").strip().lower()
        if role_raw not in {"agent", "customer"}:
            continue
        start = round(float(src.get("start") or 0.0) + offset, 2)
        end = round(float(src.get("end") or 0.0) + offset, 2)
        if end <= start:
            continue

        is_agent = role_raw == "agent"
        role = "AGENT" if is_agent else "CUSTOMER"
        duration = max(end - start, 0.0)
        if is_agent:
            agent_turns += 1
            agent_seconds += duration
        else:
            customer_turns += 1
            customer_seconds += duration

        seg = {
            "start": start,
            "end": end,
            "text": str(src.get("text") or "").strip(),
            "speaker": "SPEAKER_00" if is_agent else "SPEAKER_01",
            "identified_speaker": role,
            "display_speaker": agent_name if is_agent else "Customer 1",
            "confidence": 1.0,
            "_supervised_label": "gemini",
            "_supervised_source_call": match.call_dir.name,
            "_supervised_source_file": str(match.data_path),
        }
        if is_agent:
            seg["agent_name"] = agent_name
        out.append(seg)

    meta = {
        "applied": bool(out),
        "mode": "gemini_supervised_labels",
        "agent_name": agent_name,
        "source_call": match.call_dir.name,
        "source_audio": str(match.audio_path),
        "source_labels": str(match.data_path),
        "offset_seconds": offset,
        "segments": len(out),
        "agent_turns": agent_turns,
        "customer_turns": customer_turns,
        "agent_seconds": round(agent_seconds, 2),
        "customer_seconds": round(customer_seconds, 2),
    }
    return out, meta


def apply_supervised_labels(
    segments: list[dict],
    audio_path: str = "",
    original_path: str = "",
) -> tuple[list[dict], dict]:
    match = find_label_match(audio_path=audio_path, original_path=original_path)
    if not match:
        return segments, {"applied": False}
    supervised, meta = build_supervised_segments(match)
    if not supervised:
        return segments, {"applied": False, "source_labels": str(match.data_path)}
    return supervised, meta
