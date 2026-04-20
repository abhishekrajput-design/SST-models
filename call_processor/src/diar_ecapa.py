"""
Per-segment speaker diarization using ECAPA-TDNN embeddings + K-Means.

More robust than pyannote on short call-center utterances because each
Whisper segment gets its own voiceprint, and K-Means forces exactly N
speaker clusters (typically 2 for agent + customer calls).

Usage:
    from src.diar_ecapa import diarize_segments_ecapa
    segments = diarize_segments_ecapa(segments, norm_wav_path, num_speakers=2)
"""
from __future__ import annotations
import gc
import logging
from typing import List, Dict, Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MIN_SEG_SAMPLES = int(TARGET_SR * 0.5)   # need ≥ 500 ms for a reliable embedding
# Shorter segments get their speaker from nearest-neighbor rather than
# contributing a noisy embedding to the K-Means centroid.


def _load_model(model_dir: str = "./models/spkrec-ecapa"):
    from speechbrain.inference.speaker import SpeakerRecognition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=model_dir,
        run_opts={"device": device},
    )


def _embed_chunk(model, chunk: np.ndarray) -> Optional[np.ndarray]:
    """Extract a 192-dim ECAPA embedding from a 16 kHz mono chunk."""
    if len(chunk) < MIN_SEG_SAMPLES:
        return None
    try:
        t = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, T]
        device = next(model.mods.parameters()).device
        t = t.to(device)
        with torch.no_grad():
            emb = model.encode_batch(t).squeeze().cpu().numpy()
        return emb
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def diarize_segments_ecapa(
    segments: List[Dict],
    norm_wav_path: str,
    num_speakers: int = 2,
    pyannote_turns: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Assign speaker labels to segments using ECAPA embeddings + K-Means.

    If `pyannote_turns` is provided, Whisper segments are first SPLIT at
    pyannote turn boundaries (which are accurate in time) before labeling.
    This prevents multi-speaker blocks from being merged into one turn.

    Args:
        segments: list of dicts with 'start' and 'end' in seconds (Whisper output)
        norm_wav_path: path to the 16 kHz mono WAV used for transcription
        num_speakers: expected number of speakers (default 2 for call-center)
        pyannote_turns: optional list of {start, end, speaker} from pyannote

    Returns:
        Same segments with 'speaker' and 'identified_speaker' set.
    """
    if not segments:
        return segments

    from sklearn.cluster import KMeans

    # Step 1: only split a Whisper segment if it clearly contains multiple
    # speaker turns with meaningful duration (not a split mid-word).
    # Minimum split threshold: both sides must be ≥ 1.0s of audio AND the
    # word count in each side must be ≥ 2 (so "Hello, August." stays whole).
    MIN_SPLIT_DUR = 1.0
    MIN_SPLIT_WORDS = 2
    if pyannote_turns:
        split_segments = []
        for seg in segments:
            s_start = float(seg["start"])
            s_end   = float(seg["end"])
            words = seg.get("text", "").split()
            # Only consider splitting if segment is long enough and has enough
            # words that we could meaningfully divide it
            if (s_end - s_start) < 2 * MIN_SPLIT_DUR or len(words) < 2 * MIN_SPLIT_WORDS:
                split_segments.append(seg)
                continue
            # Find pyannote boundaries that land at least MIN_SPLIT_DUR from
            # both ends of this Whisper segment.
            splits = sorted({
                t["start"] for t in pyannote_turns
                if s_start + MIN_SPLIT_DUR < t["start"] < s_end - MIN_SPLIT_DUR
            })
            if not splits:
                split_segments.append(seg)
                continue
            dur = max(s_end - s_start, 0.01)
            cuts = [s_start] + splits + [s_end]
            sub_segs = []
            ok = True
            for i in range(len(cuts) - 1):
                sub_start, sub_end = cuts[i], cuts[i+1]
                w_start = int(len(words) * (sub_start - s_start) / dur)
                w_end   = int(len(words) * (sub_end   - s_start) / dur)
                sub_text = " ".join(words[w_start:w_end]).strip()
                if len(sub_text.split()) < MIN_SPLIT_WORDS:
                    # Not enough words per side — abort split, keep original
                    ok = False
                    break
                sub_segs.append({
                    **seg,
                    "start": round(sub_start, 2),
                    "end":   round(sub_end, 2),
                    "text":  sub_text,
                })
            split_segments.extend(sub_segs if ok else [seg])
        if len(split_segments) != len(segments):
            logger.info(f"Split {len(segments)} -> {len(split_segments)} by pyannote turns")
        segments = split_segments

    audio, sr = sf.read(norm_wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import torchaudio.functional as F_ta
        audio = F_ta.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
        sr = TARGET_SR

    model = _load_model()

    embeddings: List[Optional[np.ndarray]] = []
    valid_idx: List[int] = []
    for i, seg in enumerate(segments):
        s = int(float(seg["start"]) * sr)
        e = int(float(seg["end"]) * sr)
        e = min(e, len(audio))
        chunk = audio[s:e]
        emb = _embed_chunk(model, chunk)
        embeddings.append(emb)
        if emb is not None:
            valid_idx.append(i)

    # Free model memory
    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if len(valid_idx) < num_speakers:
        logger.warning(f"Only {len(valid_idx)} valid segments — skip clustering")
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
            seg["identified_speaker"] = "SPEAKER_00"
        return segments

    # L2-normalise embeddings for cosine-style clustering
    embs_valid = np.stack([embeddings[i] for i in valid_idx])
    norms = np.linalg.norm(embs_valid, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_valid = embs_valid / norms

    kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embs_valid)

    # Map valid segment index → cluster id
    idx_to_cluster = {vi: int(cid) for vi, cid in zip(valid_idx, cluster_ids)}

    # Assign labels. For segments too short to embed, use the nearest valid
    # segment's cluster (by time mid-point).
    valid_mids = [
        (float(segments[vi]["start"]) + float(segments[vi]["end"])) / 2
        for vi in valid_idx
    ]
    for i, seg in enumerate(segments):
        if i in idx_to_cluster:
            cid = idx_to_cluster[i]
        else:
            seg_mid = (float(seg["start"]) + float(seg["end"])) / 2
            nearest_pos = int(np.argmin([abs(seg_mid - m) for m in valid_mids]))
            cid = idx_to_cluster[valid_idx[nearest_pos]]
        spk = f"SPEAKER_{cid:02d}"
        seg["speaker"] = spk
        seg["identified_speaker"] = spk

    return segments
