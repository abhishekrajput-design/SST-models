"""
Agent vs Customer voice identification for call-center recordings.

Two operational modes:
  1. Enrolled  — cosine similarity against data/enrolled_agent.npy
  2. Heuristic — most total speaking time = Agent (works without enrollment)

Enrollment workflow:
  Process a directory of known-agent recordings (e.g. sd_agent_customer_audio).
  For each file: extract ECAPA embeddings from 2-second sliding windows,
  cluster into 2 speakers, pick the larger cluster as agent (agent speaks more
  in dealership calls), average → save to enrolled_agent.npy.
"""
from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR      = 16_000
MIN_CHUNK_S    = 0.5
ENROLLED_PATH  = os.path.join("data", "enrolled_agent.npy")
ECAPA_SAVE_DIR = "./models/spkrec-ecapa"

_FFMPEG_BIN = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)


def _ffmpeg_env() -> dict:
    env = os.environ.copy()
    if sys.platform == "win32" and os.path.isdir(_FFMPEG_BIN):
        if _FFMPEG_BIN not in env.get("PATH", ""):
            env["PATH"] = _FFMPEG_BIN + os.pathsep + env.get("PATH", "")
    return env


_ENV = _ffmpeg_env()


# ── ECAPA helpers ─────────────────────────────────────────────────────────────

def _load_ecapa():
    import torch
    from speechbrain.inference.speaker import SpeakerRecognition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ECAPA_SAVE_DIR,
        run_opts={"device": device},
    )
    return model, device


def _embed(model, chunk: np.ndarray, device: str) -> Optional[np.ndarray]:
    import torch
    if len(chunk) < int(TARGET_SR * MIN_CHUNK_S):
        return None
    try:
        t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_batch(t).squeeze().cpu().numpy()
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb
    except Exception as e:
        logger.warning(f"embed: {e}")
        return None


def _free(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _to_16k_wav(fpath: str) -> np.ndarray:
    """Load any audio file → 16kHz mono float32 array via ffmpeg."""
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", fpath, "-ac", "1", "-ar", str(TARGET_SR), tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_ENV,
        )
        audio, _ = sf.read(tmp, dtype="float32")
        return audio[:, 0] if audio.ndim > 1 else audio
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Agent identification (called per transcription) ───────────────────────────

def identify_agent_speaker(
    segments: List[Dict],
    norm_wav: str,
    spk_time: Dict[str, float],
) -> str:
    """
    Return which SPEAKER_XX is the agent.

    Uses enrolled voiceprint if available, otherwise falls back to
    most-speaking-time heuristic (agent guides the conversation).
    """
    if not spk_time:
        return "SPEAKER_00"

    if os.path.exists(ENROLLED_PATH):
        try:
            result = _match_enrolled(segments, norm_wav, spk_time)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Enrollment match failed ({e}) — using heuristic")

    agent = max(spk_time, key=lambda k: spk_time[k])
    print(f"  [SpeakerID] Heuristic: agent={agent} (speaking time={spk_time})", flush=True)
    return agent


def _match_enrolled(
    segments: List[Dict],
    norm_wav: str,
    spk_time: Dict[str, float],
) -> Optional[str]:
    enrolled_emb = np.load(ENROLLED_PATH)
    speakers     = list(spk_time.keys())

    audio, sr = sf.read(norm_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    model, device = _load_ecapa()
    try:
        best_spk, best_sim = None, -2.0
        for spk in speakers:
            chunks, collected = [], 0
            for seg in segments:
                if seg.get("speaker") != spk:
                    continue
                s = int(float(seg["start"]) * sr)
                e = min(int(float(seg["end"]) * sr), len(audio))
                chunk = audio[s:e]
                if len(chunk) >= int(sr * MIN_CHUNK_S):
                    chunks.append(chunk)
                    collected += len(chunk)
                if collected >= sr * 10:   # 10 s is enough
                    break

            embs = [_embed(model, c, device) for c in chunks]
            embs = [e for e in embs if e is not None]
            if not embs:
                continue

            avg = np.mean(np.stack(embs), axis=0)
            n   = np.linalg.norm(avg)
            avg = avg / n if n > 0 else avg
            sim = float(np.dot(avg, enrolled_emb))
            print(f"  [SpeakerID] {spk}: cosine={sim:.3f}", flush=True)
            if sim > best_sim:
                best_sim, best_spk = sim, spk
    finally:
        _free(model)

    if best_spk and best_sim > 0.25:
        print(f"  [SpeakerID] Agent={best_spk} via enrollment (cosine={best_sim:.3f})", flush=True)
        return best_spk

    print(f"  [SpeakerID] Cosine {best_sim:.3f} below threshold — using heuristic", flush=True)
    return None


# ── Enrollment (one-time, from known-agent recordings) ────────────────────────

def enroll_agent(
    recordings_dir: str,
    progress_cb: Optional[Callable] = None,
) -> str:
    """
    Build an agent voiceprint from all audio files in recordings_dir.

    Steps per file:
      1. Load as 16kHz mono
      2. Slide 2s windows (1s stride), extract ECAPA embeddings
      3. K-Means (n=2) → two speaker clusters
      4. Agent cluster = whichever has more windows
      5. Collect per-file average agent embedding

    Final: average all → L2-normalise → save to ENROLLED_PATH.
    """
    from sklearn.cluster import KMeans

    exts  = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files = sorted([
        os.path.join(recordings_dir, f)
        for f in os.listdir(recordings_dir)
        if os.path.splitext(f.lower())[1] in exts
    ])
    if not files:
        return f"No audio files found in {recordings_dir}"

    model, device  = _load_ecapa()
    agent_embs: List[np.ndarray] = []
    n_ok = 0

    try:
        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            if progress_cb:
                progress_cb(i, len(files), fname)
            print(f"  [Enroll] {i+1}/{len(files)}: {fname}", flush=True)

            try:
                audio = _to_16k_wav(fpath)
            except Exception as e:
                print(f"  [Enroll] Load failed ({e}) — skipping", flush=True)
                continue

            # 2s windows, 1s stride
            window, stride = TARGET_SR * 2, TARGET_SR
            embs: List[np.ndarray] = []
            for start in range(0, len(audio) - window, stride):
                emb = _embed(model, audio[start:start + window], device)
                if emb is not None:
                    embs.append(emb)

            if len(embs) < 6:
                print(f"  [Enroll] Only {len(embs)} windows — skipping", flush=True)
                continue

            emb_matrix = np.stack(embs)
            km     = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = km.fit_predict(emb_matrix)

            c0, c1       = np.sum(labels == 0), np.sum(labels == 1)
            agent_cluster = 0 if c0 >= c1 else 1

            a_embs = emb_matrix[labels == agent_cluster]
            avg    = np.mean(a_embs, axis=0)
            n      = np.linalg.norm(avg)
            agent_embs.append(avg / n if n > 0 else avg)
            n_ok += 1
            print(
                f"  [Enroll] {fname}: cluster={agent_cluster}"
                f" agent_windows={len(a_embs)}/{len(embs)}",
                flush=True,
            )
    finally:
        _free(model)

    if not agent_embs:
        return "Enrollment failed — no valid embeddings extracted."

    final     = np.mean(np.stack(agent_embs), axis=0)
    n         = np.linalg.norm(final)
    final     = final / n if n > 0 else final

    os.makedirs("data", exist_ok=True)
    np.save(ENROLLED_PATH, final)

    msg = (
        f"Agent enrolled from {n_ok}/{len(files)} recordings. "
        f"Voiceprint saved to {ENROLLED_PATH}."
    )
    print(f"  [Enroll] {msg}", flush=True)
    return msg
