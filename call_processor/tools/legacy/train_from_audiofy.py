"""
train_from_audiofy.py — Build ECAPA voiceprints from scraped Audiofy agent clips.

Reads from:   data/audiofy/<agent_slug>/agent_clips/*.wav
Saves to:     data/agent_voiceprints/<agent_slug>.npy  (512-dim L2-normed embedding)
              data/agent_voiceprints/agents.json        (name → path index)

Usage:
    python train_from_audiofy.py               # all agents with clips
    python train_from_audiofy.py --agent "Zak" # name substring filter
    python train_from_audiofy.py --min-clips 5 # skip agents with fewer than N clips
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

SCRIPT_DIR     = Path(__file__).parent
AUDIOFY_DIR    = SCRIPT_DIR / "data" / "audiofy"
VP_DIR         = SCRIPT_DIR / "data" / "agent_voiceprints"
VP_DIR.mkdir(parents=True, exist_ok=True)

ECAPA_SAVE_DIR = str(SCRIPT_DIR / "models" / "ecapa")
MIN_DURATION   = 1.5   # seconds
MIN_CLIPS      = 3     # skip agent if fewer usable clips after filtering


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# ECAPA embedding
# ---------------------------------------------------------------------------

_ecapa_model = None

def _load_ecapa():
    global _ecapa_model
    if _ecapa_model is not None:
        return _ecapa_model
    from speechbrain.inference.speaker import SpeakerRecognition
    print("[ECAPA] Loading model on CPU…")
    _ecapa_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ECAPA_SAVE_DIR,
        run_opts={"device": "cpu"},
    )
    print("[ECAPA] Ready.")
    return _ecapa_model


def _embed_clip(model, wav_path: str) -> np.ndarray | None:
    """Return 192-dim ECAPA embedding or None on failure."""
    try:
        import torchaudio, torch
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[1] < int(sr * MIN_DURATION):
            return None
        # Resample to 16 kHz if needed
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        with torch.no_grad():
            emb = model.encode_batch(wav)   # (1, 1, D)
        return emb.squeeze().cpu().numpy()
    except Exception as e:
        print(f"    [embed] {Path(wav_path).name}: {e}")
        return None


def _clip_duration(wav_path: str) -> float:
    try:
        import soundfile as sf
        return sf.info(wav_path).duration
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Train one agent
# ---------------------------------------------------------------------------

def train_agent(agent_name: str, clips_dir: Path) -> np.ndarray | None:
    clips = sorted(clips_dir.glob("*.wav"))
    if not clips:
        print(f"  [skip] No WAV clips in {clips_dir}")
        return None

    print(f"  {len(clips)} clips found")
    model = _load_ecapa()

    embeddings: List[np.ndarray] = []
    for clip in clips:
        dur = _clip_duration(str(clip))
        if dur < MIN_DURATION:
            continue
        emb = _embed_clip(model, str(clip))
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) < MIN_CLIPS:
        print(f"  [skip] Only {len(embeddings)} usable embeddings (need {MIN_CLIPS})")
        return None

    print(f"  {len(embeddings)} embeddings extracted")

    # Remove outliers: keep embeddings with cosine similarity > 0.3 to the mean
    emb_matrix = np.stack(embeddings)
    mean_emb   = emb_matrix.mean(axis=0)
    mean_norm  = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)

    filtered = []
    for e in embeddings:
        e_norm = e / (np.linalg.norm(e) + 1e-9)
        sim    = float(np.dot(e_norm, mean_norm))
        if sim > 0.3:
            filtered.append(e)

    print(f"  {len(filtered)} after outlier removal ({len(embeddings)-len(filtered)} removed)")

    if len(filtered) < MIN_CLIPS:
        print(f"  [skip] Too few after filtering")
        return None

    # Final voiceprint: L2-normalised mean
    voiceprint = np.stack(filtered).mean(axis=0)
    voiceprint = voiceprint / (np.linalg.norm(voiceprint) + 1e-9)
    return voiceprint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",     default="", help="Agent name substring filter")
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS,
                        help=f"Min usable clips needed (default {MIN_CLIPS})")
    args = parser.parse_args()

    agent_filter = args.agent.strip().lower()

    # Load existing agents index
    agents_index_path = VP_DIR / "agents.json"
    agents_index: dict[str, dict] = {}
    if agents_index_path.exists():
        with open(agents_index_path, encoding="utf-8") as f:
            agents_index = json.load(f)

    # Find agent directories
    if not AUDIOFY_DIR.exists():
        print(f"[Error] {AUDIOFY_DIR} not found. Run scrape_audiofy.py first.")
        sys.exit(1)

    manifest_path = AUDIOFY_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        # Build from directory structure
        manifest = {d.name: {"agent_name": d.name, "clips_dir": str(d / "agent_clips")}
                    for d in AUDIOFY_DIR.iterdir() if d.is_dir()}

    trained   = 0
    skipped   = 0
    failed    = 0
    t_start   = time.time()

    for a_slug, info in sorted(manifest.items()):
        agent_name = info.get("agent_name", a_slug)

        if agent_filter and agent_filter not in agent_name.lower():
            continue

        clips_dir = Path(info.get("clips_dir", str(AUDIOFY_DIR / a_slug / "agent_clips")))
        if not clips_dir.exists():
            skipped += 1
            continue

        out_path = VP_DIR / f"{a_slug}.npy"
        print(f"\n[Train] {agent_name}")

        voiceprint = train_agent(agent_name, clips_dir)
        if voiceprint is None:
            failed += 1
            continue

        np.save(str(out_path), voiceprint)
        agents_index[a_slug] = {
            "agent_name":   agent_name,
            "voiceprint":   str(out_path),
            "clips_dir":    str(clips_dir),
            "n_clips":      len(list(clips_dir.glob("*.wav"))),
        }
        print(f"  Saved voiceprint -> {out_path}")
        trained += 1

    # Save index
    with open(agents_index_path, "w", encoding="utf-8") as f:
        json.dump(agents_index, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[Done] trained={trained}  skipped={skipped}  failed={failed}  time={elapsed:.0f}s")
    print(f"       Index -> {agents_index_path}")
    print(f"       {len(agents_index)} agents total in voiceprint library")


if __name__ == "__main__":
    main()
