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
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR         = 16_000
MIN_CHUNK_S       = 0.5
ENROLLED_PATH     = os.path.join("data", "enrolled_agent.npy")
ECAPA_SAVE_DIR    = "./models/spkrec-ecapa"
VOICEPRINT_DIR    = os.path.join("data", "agent_voiceprints")
AGENTS_INDEX_PATH = os.path.join(VOICEPRINT_DIR, "agents.json")

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

def _load_ecapa(force_cpu: bool = False):
    import torch
    from speechbrain.inference.speaker import SpeakerRecognition
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
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


# ── Keyword-based role detection ─────────────────────────────────────────────
# Phrases strongly associated with the agent role in a car-dealership call
_AGENT_KEYWORDS = [
    "thank you for calling", "how can i help", "how may i help",
    "welcome to", "what brings you in", "let me pull up",
    "i can get you", "let me show you", "what are you looking for",
    "what's your budget", "what is your budget",
    "i'd like to help", "i will help", "can i help you",
    "my name is", "this is", "i'm calling from",
    "we have", "we can offer", "let me check",
]
_CUSTOMER_KEYWORDS = [
    "i'm looking for", "i am looking for", "i need",
    "i want", "my budget is", "i saw online", "i found online",
    "i called because", "interested in", "how much is", "what's the price",
    "i was told", "i'd like to buy", "i'd like to get",
]


def detect_agent_by_keywords(segments: List[Dict]) -> Optional[str]:
    """
    Score each SPEAKER_XX against agent/customer keyword lists.
    Returns the speaker most likely to be the agent, or None if inconclusive.
    A score ≥ 2 means at least one strong agent phrase was matched.
    """
    scores: Dict[str, int] = {}
    for seg in segments:
        spk  = seg.get("speaker", "")
        text = seg.get("text", "").lower()
        if not spk:
            continue
        for kw in _AGENT_KEYWORDS:
            if kw in text:
                scores[spk] = scores.get(spk, 0) + 2
        for kw in _CUSTOMER_KEYWORDS:
            if kw in text:
                scores[spk] = scores.get(spk, 0) - 1

    if not scores:
        return None

    best     = max(scores, key=lambda k: scores[k])
    best_val = scores[best]
    others   = [v for k, v in scores.items() if k != best]
    margin   = best_val - (others[0] if others else 0)

    if best_val >= 2 and margin >= 1:
        print(f"  [SpeakerID] Keyword match: agent={best} (score={best_val})", flush=True)
        return best

    return None   # not confident enough


# ── Agent identification (called per transcription) ───────────────────────────

def _find_best_voiceprint() -> Optional[str]:
    """Return one unambiguous voiceprint for TS-VAD, or None."""
    if os.path.exists(AGENTS_INDEX_PATH):
        try:
            with open(AGENTS_INDEX_PATH, encoding="utf-8") as f:
                idx = json.load(f)
            valid = []
            for info in idx.values():
                if not isinstance(info, dict):
                    continue
                vp = info.get("voiceprint_path") or info.get("voiceprint") or ""
                if vp and os.path.exists(vp):
                    valid.append(vp)
            if len(valid) == 1:
                return valid[0]
            if len(valid) > 1:
                logger.info("Multiple enrolled agents found; skipping single-target TS-VAD")
                return None
        except Exception:
            pass
    # Legacy single-agent fallback
    if os.path.exists(ENROLLED_PATH):
        return ENROLLED_PATH
    return None


def identify_agent_speaker(
    segments: List[Dict],
    norm_wav: str,
    spk_time: Dict[str, float],
) -> str:
    """
    Return which SPEAKER_XX is the agent.

    Priority order:
      0. Target-Speaker VAD  (most accurate — requires CAM++ voiceprint)
      1. Enrolled voiceprint cosine similarity
      2. Keyword detection — agent phrases in transcript
      3. Most total speaking time               (heuristic fallback)
    """
    if not spk_time:
        return "SPEAKER_00"

    # 0 — Target-Speaker VAD (TS-VAD) with CAM++ or ECAPA voiceprint
    vp_path = _find_best_voiceprint()
    if vp_path:
        try:
            from src.target_speaker_vad import TargetSpeakerVAD
            vp = np.load(vp_path)
            tsvad = TargetSpeakerVAD(vp)
            agent_regions = tsvad.agent_segments(norm_wav)
            if agent_regions:
                # Vote: which SPEAKER_XX overlaps most with TS-VAD agent regions
                votes: Dict[str, float] = {s: 0.0 for s in spk_time}
                for seg in segments:
                    spk = seg.get("speaker", "")
                    if spk not in votes:
                        continue
                    seg_s = float(seg["start"]); seg_e = float(seg["end"])
                    for r in agent_regions:
                        overlap = min(seg_e, r["end"]) - max(seg_s, r["start"])
                        if overlap > 0:
                            votes[spk] = votes.get(spk, 0.0) + overlap
                if votes:
                    best = max(votes, key=lambda k: votes[k])
                    if votes[best] > 0:
                        print(f"  [SpeakerID] TS-VAD agent={best} (overlap={votes[best]:.1f}s)", flush=True)
                        return best
        except Exception as e:
            logger.warning(f"TS-VAD failed ({e}) — falling back")

    # 1 — Enrolled voiceprint
    if os.path.exists(ENROLLED_PATH):
        try:
            result = _match_enrolled(segments, norm_wav, spk_time)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Enrollment match failed ({e}) — trying keywords")

    # 2 — Keyword detection
    kw = detect_agent_by_keywords(segments)
    if kw:
        return kw

    # 3 — Heuristic: most speaking time
    agent = max(spk_time, key=lambda k: spk_time[k])
    print(f"  [SpeakerID] Heuristic: agent={agent} times={spk_time}", flush=True)
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

    # Run enrollment on CPU — one-time background task, so GPU speed not critical.
    # This prevents VRAM contention when a transcription request arrives mid-enrollment.
    # EmbeddingModel uses CAM++ (512-dim) if wespeaker is installed, else ECAPA (192-dim).
    from src.embedding_campp import EmbeddingModel
    emb_model = EmbeddingModel()
    emb_model.load(force_cpu=True)
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
                emb = emb_model.embed_chunk(audio[start:start + window], TARGET_SR)
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
                f" agent_windows={len(a_embs)}/{len(embs)} dim={emb_model.dim}",
                flush=True,
            )
    finally:
        emb_model.unload()

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


# ── Multi-agent identification ────────────────────────────────────────────────

def _load_agents_index() -> dict:
    """Load {slug: {agent_name, voiceprint_path}} from agents.json."""
    if not os.path.exists(AGENTS_INDEX_PATH):
        return {}
    try:
        with open(AGENTS_INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _agent_voiceprint_path(info: dict) -> str:
    return str(info.get("voiceprint") or info.get("voiceprint_path") or "")


def _agent_name(info: dict, fallback: str) -> str:
    return str(info.get("agent_name") or info.get("name") or fallback)


def _speaker_embeddings_ecapa(
    segments: List[Dict],
    norm_wav: str,
    speakers: List[str],
) -> Dict[str, Optional[np.ndarray]]:
    audio, sr = sf.read(norm_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    model, device = _load_ecapa()
    try:
        spk_embs: Dict[str, Optional[np.ndarray]] = {}
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
                if collected >= sr * 10:
                    break
            embs = [_embed(model, c, device) for c in chunks]
            embs = [em for em in embs if em is not None]
            if embs:
                avg = np.mean(np.stack(embs), axis=0)
                n = np.linalg.norm(avg)
                spk_embs[spk] = avg / n if n > 0 else avg
            else:
                spk_embs[spk] = None
        return spk_embs
    finally:
        _free(model)


def _speaker_embeddings_campp(
    segments: List[Dict],
    norm_wav: str,
    speakers: List[str],
    expected_dim: int,
) -> Dict[str, Optional[np.ndarray]]:
    from src.embedding_campp import get_model

    audio, sr = sf.read(norm_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    model = get_model(force_cpu=False)
    if model.dim != expected_dim:
        logger.warning(
            "CAM++ speaker matching unavailable: embedding backend=%s dim=%s, voiceprint dim=%s",
            model.model_name,
            model.dim,
            expected_dim,
        )
        return {spk: None for spk in speakers}

    spk_embs: Dict[str, Optional[np.ndarray]] = {}
    for spk in speakers:
        embs, collected = [], 0
        for seg in segments:
            if seg.get("speaker") != spk:
                continue
            s = int(float(seg["start"]) * sr)
            e = min(int(float(seg["end"]) * sr), len(audio))
            chunk = audio[s:e]
            if len(chunk) < int(sr * MIN_CHUNK_S):
                continue
            emb = model.embed_chunk(chunk, sr)
            if emb is not None:
                embs.append(emb)
                collected += len(chunk)
            if collected >= sr * 10:
                break
        if embs:
            avg = np.mean(np.stack(embs), axis=0)
            n = np.linalg.norm(avg)
            spk_embs[spk] = avg / n if n > 0 else avg
        else:
            spk_embs[spk] = None
    return spk_embs


def identify_agent_name(
    segments: List[Dict],
    norm_wav: str,
    spk_time: Dict[str, float],
    threshold: float = 0.25,
) -> tuple[str, str, float]:
    """
    Match speakers against ALL enrolled agent voiceprints.

    Returns:
        (agent_speaker_id, agent_name, best_cosine_similarity)

    Falls back to heuristic (most speaking time) if no voiceprint matches.
    """
    agents = _load_agents_index()
    if not agents:
        # No multi-agent library; fall back to single enrolled + heuristic
        spk = identify_agent_speaker(segments, norm_wav, spk_time)
        return spk, "Unknown Agent", 0.0

    try:
        sf.info(norm_wav)
    except Exception as e:
        logger.warning(f"identify_agent_name: cannot read {norm_wav}: {e}")
        spk = max(spk_time, key=lambda k: spk_time[k]) if spk_time else "SPEAKER_00"
        return spk, "Unknown Agent", 0.0

    speakers = list(spk_time.keys())

    # Compare every speaker × every agent voiceprint
    best_spk, best_name, best_sim = None, "Unknown Agent", -2.0
    spk_embs_by_dim: Dict[int, Dict[str, Optional[np.ndarray]]] = {}

    for a_slug, info in agents.items():
        if not isinstance(info, dict):
            continue
        vp_path = _agent_voiceprint_path(info)
        if not os.path.exists(vp_path):
            continue
        try:
            vp = np.load(vp_path)
        except Exception as e:
            logger.warning("Could not load voiceprint %s: %s", vp_path, e)
            continue
        vp = np.asarray(vp, dtype=np.float32).squeeze()
        if vp.ndim != 1:
            logger.warning("Skipping %s voiceprint with invalid shape %s", a_slug, vp.shape)
            continue
        vp_dim = int(vp.shape[0])

        if vp_dim not in spk_embs_by_dim:
            try:
                if vp_dim == 512:
                    spk_embs_by_dim[vp_dim] = _speaker_embeddings_campp(
                        segments, norm_wav, speakers, expected_dim=vp_dim
                    )
                elif vp_dim == 192:
                    spk_embs_by_dim[vp_dim] = _speaker_embeddings_ecapa(
                        segments, norm_wav, speakers
                    )
                else:
                    logger.warning("Skipping unsupported voiceprint dim=%s for %s", vp_dim, a_slug)
                    spk_embs_by_dim[vp_dim] = {spk: None for spk in speakers}
            except Exception as e:
                logger.warning("Could not build %s-dim speaker embeddings: %s", vp_dim, e)
                spk_embs_by_dim[vp_dim] = {spk: None for spk in speakers}

        spk_embs = spk_embs_by_dim[vp_dim]
        agent_label = _agent_name(info, a_slug)
        vp_norm = np.linalg.norm(vp)
        vp = vp / vp_norm if vp_norm > 0 else vp

        for spk, emb in spk_embs.items():
            if emb is None:
                continue
            if emb.shape[0] != vp_dim:
                logger.warning(
                    "Skipping %s vs %s: embedding dim %s != voiceprint dim %s",
                    spk,
                    agent_label,
                    emb.shape[0],
                    vp_dim,
                )
                continue
            sim = float(np.dot(emb, vp))
            print(f"  [MultiID] {spk} vs {agent_label}: cosine={sim:.3f}", flush=True)
            if sim > best_sim:
                best_sim = sim
                best_spk = spk
                best_name = agent_label

    if best_spk and best_sim >= threshold:
        print(f"  [MultiID] Agent={best_spk} -> {best_name} (cosine={best_sim:.3f})", flush=True)
        return best_spk, best_name, best_sim

    # Below threshold: fall back to most speaking time
    fallback = max(spk_time, key=lambda k: spk_time[k]) if spk_time else "SPEAKER_00"
    print(f"  [MultiID] No confident match (best={best_sim:.3f}) - heuristic: {fallback}", flush=True)
    return fallback, "Unknown Agent", best_sim


def enroll_agent_clean(
    recordings_dir: str,
    progress_cb: Optional[Callable] = None,
) -> str:
    """
    Build an agent voiceprint from CLEAN single-speaker recordings.

    Unlike enroll_agent(), no KMeans clustering is needed — every window in
    every file belongs to the same speaker (the agent).  Use this when you
    have a folder of recordings that contain ONLY the agent's voice (no customer).

    Steps per file:
      1. Load as 16kHz mono
      2. Slide 2s windows (1s stride), extract ECAPA embeddings
      3. Average all windows → per-file embedding
    Final: average all files → L2-normalise → save to ENROLLED_PATH.
    """
    exts  = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files = sorted([
        os.path.join(recordings_dir, f)
        for f in os.listdir(recordings_dir)
        if os.path.splitext(f.lower())[1] in exts
    ])
    if not files:
        return f"No audio files found in {recordings_dir}"

    from src.embedding_campp import EmbeddingModel
    emb_model = EmbeddingModel()
    emb_model.load(force_cpu=True)
    agent_embs: List[np.ndarray] = []
    n_ok = 0

    try:
        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            if progress_cb:
                progress_cb(i, len(files), fname)
            print(f"  [EnrollClean] {i+1}/{len(files)}: {fname}", flush=True)

            try:
                audio = _to_16k_wav(fpath)
            except Exception as e:
                print(f"  [EnrollClean] Load failed ({e}) — skipping", flush=True)
                continue

            window, stride = TARGET_SR * 2, TARGET_SR
            embs: List[np.ndarray] = []
            for start in range(0, len(audio) - window, stride):
                emb = emb_model.embed_chunk(audio[start:start + window], TARGET_SR)
                if emb is not None:
                    embs.append(emb)

            if len(embs) < 2:
                print(f"  [EnrollClean] Only {len(embs)} windows — skipping", flush=True)
                continue

            avg = np.mean(np.stack(embs), axis=0)
            n   = np.linalg.norm(avg)
            agent_embs.append(avg / n if n > 0 else avg)
            n_ok += 1
            print(f"  [EnrollClean] {fname}: {len(embs)} windows dim={emb_model.dim}", flush=True)
    finally:
        emb_model.unload()

    if not agent_embs:
        return "Enrollment failed — no valid embeddings extracted."

    final = np.mean(np.stack(agent_embs), axis=0)
    n     = np.linalg.norm(final)
    final = final / n if n > 0 else final

    os.makedirs("data", exist_ok=True)
    np.save(ENROLLED_PATH, final)

    msg = (
        f"Agent (clean) enrolled from {n_ok}/{len(files)} recordings. "
        f"Voiceprint saved to {ENROLLED_PATH}."
    )
    print(f"  [EnrollClean] {msg}", flush=True)
    return msg
