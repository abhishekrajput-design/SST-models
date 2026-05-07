#!/usr/bin/env python
"""Build Hussein Mohamed candidate data.json labels from local UI results.

The Audiofy metadata says these calls belong to Hussein Mohamed, but current
voiceprint matching may assign the agent speaker to another enrolled agent.
This script therefore maps speaker IDs to agent/customer using transcript cues,
not the current identified_agent field.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
DATA_ROOT = REPO_ROOT / "traning_data" / "hussein_mohamed"
SUMMARY_PATH = CALL_PROCESSOR_DIR / "data" / "hussein_local_results.json"
PROCESSED_ROOT = CALL_PROCESSOR_DIR / "data" / "processed"
EXCLUDED_DIR = DATA_ROOT / "_excluded_unsafe_20260506"

AGENT_CUES = (
    "car planet",
    "carplanet",
    "car plan",
    "my name is hussein",
    "my name's hussein",
    "name is sane",
    "calling from",
    "courtesy call",
    "appointment",
    "viewing",
    "finance",
    "watford branch",
    "barnet branch",
    "how can i help",
    "bear with",
    "no problem",
)

CUSTOMER_CUES = (
    "i had an appointment",
    "i cancelled",
    "i canceled",
    "i purchased",
    "i requested",
    "my car loan",
    "you guys",
    "full payment",
    "full quote",
    "i missed it",
    "i'm all good",
)

VOICEMAIL_CUES = (
    "record your name and reason for calling",
    "person is available",
    "please stay on the line",
    "forwarded to voicemail",
)


def norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch == "'" else " " for ch in text).split())


def cue_score(text: str, cues: tuple[str, ...]) -> int:
    n = norm(text)
    return sum(1 for cue in cues if cue in n)


def move_excluded(folder: Path, reason: str) -> dict:
    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)
    dest = EXCLUDED_DIR / folder.name
    if dest.exists():
        suffix = 2
        while (EXCLUDED_DIR / f"{folder.name}_{suffix}").exists():
            suffix += 1
        dest = EXCLUDED_DIR / f"{folder.name}_{suffix}"
    shutil.move(str(folder), str(dest))
    marker = dest / "EXCLUDED_REASON.txt"
    marker.write_text(reason + "\n", encoding="utf-8")
    return {"folder": folder.name, "excluded_to": str(dest), "reason": reason}


def main() -> int:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    prepared = []
    excluded = []

    for item in summary:
        folder = DATA_ROOT / item["folder"]
        if not folder.is_dir():
            continue
        result_path = PROCESSED_ROOT / item["result_id"] / "result.json"
        if not result_path.is_file():
            excluded.append(move_excluded(folder, "missing local result.json"))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        segments = list(result.get("segments") or [])
        if not segments:
            excluded.append(move_excluded(folder, "no local transcript segments"))
            continue

        all_text = " ".join(str(seg.get("text") or "") for seg in segments)
        if any(cue in norm(all_text) for cue in VOICEMAIL_CUES):
            excluded.append(move_excluded(folder, "voicemail or IVR recording, no clean Hussein conversation"))
            continue

        speaker_scores: dict[str, dict[str, float]] = {}
        for seg in segments:
            speaker = str(seg.get("speaker") or "").strip()
            if not speaker:
                continue
            text = str(seg.get("text") or "")
            dur = max(0.0, float(seg.get("end") or 0.0) - float(seg.get("start") or 0.0))
            bucket = speaker_scores.setdefault(speaker, {"agent": 0.0, "customer": 0.0, "duration": 0.0})
            bucket["agent"] += cue_score(text, AGENT_CUES) * max(1.0, dur)
            bucket["customer"] += cue_score(text, CUSTOMER_CUES) * max(1.0, dur)
            bucket["duration"] += dur

        if len(speaker_scores) < 2:
            excluded.append(move_excluded(folder, "single-speaker diarization; cannot separate agent/customer cleanly"))
            continue

        ranked = sorted(
            speaker_scores.items(),
            key=lambda kv: (kv[1]["agent"] - kv[1]["customer"], kv[1]["agent"], kv[1]["duration"]),
            reverse=True,
        )
        agent_speaker, agent_meta = ranked[0]
        if agent_meta["agent"] <= 0:
            excluded.append(move_excluded(folder, "no reliable agent text cue for speaker mapping"))
            continue

        out_segments = []
        for seg in segments:
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or 0.0)
            if end <= start:
                continue
            speaker = str(seg.get("speaker") or "")
            role = "agent" if speaker == agent_speaker else "customer"
            out_segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": role,
                "text": str(seg.get("text") or "").strip(),
            })

        audio = next((p for p in sorted(folder.iterdir()) if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}), None)
        payload = {
            "call_id": audio.stem if audio else item["audio"],
            "agent_name": "Hussein Mohamed",
            "source": "local_parakeet_textcue_candidate",
            "source_result_id": item["result_id"],
            "agent_speaker": agent_speaker,
            "speaker_scores": speaker_scores,
            "training_warning": "Candidate labels inferred from local ASR text cues; verify before treating as human/Gemini truth.",
            "segments": out_segments,
        }
        (folder / "data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        prepared.append({
            "folder": folder.name,
            "result_id": item["result_id"],
            "agent_speaker": agent_speaker,
            "segments": len(out_segments),
            "agent_segments": sum(1 for seg in out_segments if seg["speaker"] == "agent"),
            "customer_segments": sum(1 for seg in out_segments if seg["speaker"] == "customer"),
            "speaker_scores": speaker_scores,
        })

    report = {
        "data_root": str(DATA_ROOT),
        "prepared": prepared,
        "excluded": excluded,
    }
    out = CALL_PROCESSOR_DIR / "data" / "hussein_label_prep_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "prepared": prepared, "excluded": excluded}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
