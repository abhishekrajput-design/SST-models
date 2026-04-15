"""
Multi-model ASR consensus for Tier 3 & 4 audio.

Runs three ASR models in parallel (threading) and picks the result with
the highest average word/segment confidence:
  1. Deepgram Nova-3      — cloud API, best noise robustness + native diarization
  2. Whisper large-v3-turbo — faster-whisper, robust, hallucination-resistant
  3. Cohere Transcribe   — best WER on Open ASR Leaderboard (5.42%)

Consensus rules:
  - Compare average_confidence per model
  - Pick result with highest avg confidence
  - If ALL results have avg_confidence < REVIEW_THRESHOLD → needs_human_review=True

Confidence normalisation:
  - Deepgram:   0–1 native utterance confidence
  - Whisper:    avg_logprob (negative log prob) → exp(avg_logprob) → 0–1
  - Cohere:     uses segment confidence if available, else 0.5 default
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# If ALL models score below this, flag the call for human review
REVIEW_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #

def transcribe_consensus(
    audio_path: str,
    device: str = "cuda",
    model_dir: str = "models",
    language: str = "en",
) -> dict:
    """
    Run all three ASR models and return the best transcript.

    Args:
        audio_path: Path to enhanced WAV (16 kHz mono recommended).
        device:     "cuda" or "cpu"
        model_dir:  Local model cache root.
        language:   BCP-47 language code.

    Returns:
        dict with keys:
            segments          (list)  — winning transcript
            model_used        (str)   — name of winning model
            avg_confidence    (float) — winning model's average confidence
            models_compared   (int)   — how many models ran successfully
            needs_human_review (bool)
            review_reasons    (list[str])
            all_results       (dict)  — {model_name: {segments, avg_confidence}}
    """
    results: dict = {}
    errors:  dict = {}

    def _run(name: str, fn):
        try:
            segs = fn()
            avg  = _avg_confidence(segs)
            results[name] = {"segments": segs, "avg_confidence": round(avg, 4)}
            logger.info(f"[MultiASR] {name}: {len(segs)} segs, avg_conf={avg:.3f}")
        except Exception as exc:
            logger.warning(f"[MultiASR] {name} failed: {exc}")
            errors[name] = str(exc)

    # ── Build callables ───────────────────────────────────────────────────────
    tasks = {
        "deepgram-nova-3":         lambda: _run_deepgram(audio_path, "nova-3"),
        "whisper-large-v3-turbo":  lambda: _run_whisper(audio_path, device, model_dir, language),
        "cohere-transcribe":       lambda: _run_cohere(audio_path, device, model_dir),
    }

    # Run in parallel threads
    threads = []
    for name, fn in tasks.items():
        t = threading.Thread(target=_run, args=(name, fn), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=600)  # 10-minute max per model

    if not results:
        logger.error("[MultiASR] All models failed.")
        return _empty_result(review_reasons=["All ASR models failed"] + list(errors.values()))

    # ── Pick winner ───────────────────────────────────────────────────────────
    winner_name = max(results, key=lambda k: results[k]["avg_confidence"])
    winner      = results[winner_name]

    # ── Human review flags ───────────────────────────────────────────────────
    review_reasons: List[str] = []
    all_low = all(r["avg_confidence"] < REVIEW_THRESHOLD for r in results.values())
    if all_low:
        review_reasons.append(
            f"All {len(results)} ASR models have avg confidence < {REVIEW_THRESHOLD}"
        )

    needs_human_review = bool(review_reasons)

    return {
        "segments":           winner["segments"],
        "model_used":         winner_name,
        "avg_confidence":     winner["avg_confidence"],
        "models_compared":    len(results),
        "needs_human_review": needs_human_review,
        "review_reasons":     review_reasons,
        "all_results": {
            k: {"avg_confidence": v["avg_confidence"], "segments": len(v["segments"])}
            for k, v in results.items()
        },
    }


# --------------------------------------------------------------------------- #
#  Per-model runners
# --------------------------------------------------------------------------- #

def _run_deepgram(audio_path: str, model: str = "nova-3") -> List[Dict[str, Any]]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.transcribers.deepgram_asr import DeepgramTranscriber

    t = DeepgramTranscriber(model=model)
    t.load()
    try:
        return t.transcribe(audio_path)
    finally:
        t.unload()


def _run_whisper(
    audio_path: str,
    device: str,
    model_dir: str,
    language: str,
) -> List[Dict[str, Any]]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.transcribers.whisper_turbo import WhisperTurboTranscriber

    t = WhisperTurboTranscriber(
        model_size="large-v3-turbo",
        device=device,
        model_dir=os.path.join(model_dir, "faster-whisper"),
    )
    t.load()
    try:
        return t.transcribe(audio_path, language=language)
    finally:
        t.unload()


def _run_cohere(audio_path: str, device: str, model_dir: str) -> List[Dict[str, Any]]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.transcribers.cohere import CohereTranscriber

    t = CohereTranscriber(
        device=device,
        model_dir=os.path.join(model_dir, "hf"),
    )
    t.load()
    try:
        return t.transcribe(audio_path)
    finally:
        t.unload()


# --------------------------------------------------------------------------- #
#  Confidence helpers
# --------------------------------------------------------------------------- #

def _avg_confidence(segments: List[Dict]) -> float:
    """
    Compute normalised average confidence across all segments.
    Handles Deepgram (0–1), Whisper (avg_logprob < 0), and Cohere formats.
    """
    if not segments:
        return 0.0

    scores = []
    for seg in segments:
        conf = seg.get("confidence", None)
        if conf is None:
            # Whisper avg_logprob → probability
            logp = seg.get("avg_logprob", None)
            if logp is not None:
                import math
                conf = math.exp(max(logp, -10.0))  # clamp to avoid exp(-inf)
            else:
                conf = 0.5  # no confidence info → neutral
        scores.append(float(conf))

    return sum(scores) / len(scores)


def _empty_result(review_reasons: Optional[List[str]] = None) -> dict:
    return {
        "segments":            [],
        "model_used":          "none",
        "avg_confidence":      0.0,
        "models_compared":     0,
        "needs_human_review":  True,
        "review_reasons":      review_reasons or ["No ASR results available"],
        "all_results":         {},
    }
