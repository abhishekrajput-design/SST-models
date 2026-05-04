"""
test_voiceprints_vs_gemini.py — Compare diarization against Gemini transcription.

Gemini transcription includes speaker labels (in 'speakers' field of API result).
This is more realistic than speaker_json (which may have errors).

Usage:
  python test_voiceprints_vs_gemini.py --top 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent
os.chdir(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from enroll_all_from_api import (
    INDEX_PATH, AUDIO_DIR, load_mp3_mono_16k, TARGET_SR, slug, ts2s,
)
from test_voiceprints_api import load_voiceprint_stacks

THRESHOLD_DEFAULT = 0.35
MIN_PHRASE_S = 0.4

def held_out_calls_with_gemini(max_calls: int) -> List[dict]:
    """Calls that have Gemini transcription with speaker labels."""
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    held: List[dict] = []
    for rec in index:
        if not rec.get("_id"):
            continue
        # Check for Gemini result (has 'speakers' field, not just speaker_json)
        gemini = rec.get("gemini_result", {})
        if not gemini or not gemini.get("speakers"):
            continue
        if rec.get("n_agent_phrases", 0) < 3:
            continue
        held.append(rec)

    held.sort(key=lambda r: r.get("connect_time", ""), reverse=True)
    return held[:max_calls]


def evaluate_call_vs_gemini(
    rec: dict,
    voiceprints: Dict[str, Tuple[str, np.ndarray]],
    model,
    threshold: float,
    expected_slug: str,
) -> Optional[dict]:
    """Score one call using Gemini transcription as ground truth."""
    rid = rec["_id"]
    mp3 = AUDIO_DIR / f"{rid}.mp3"
    if not (mp3.exists() and mp3.stat().st_size > 1000):
        return None
    try:
        audio, sr = load_mp3_mono_16k(mp3)
    except Exception:
        return None

    # Get Gemini speaker labels
    gemini = rec.get("gemini_result", {})
    speakers = gemini.get("speakers", [])
    if not speakers:
        return None

    # Track confusion matrix
    tp = fp = tn = fn = 0
    correct_agent_hits = 0
    total_agent_phrases = 0
    agent_sim_sum: Dict[str, float] = {}
    agent_sim_n: Dict[str, int] = {}

    slugs = list(voiceprints.keys())
    stacks = [voiceprints[s][1] for s in slugs]

    # Iterate Gemini speakers (ground truth)
    for spk in speakers:
        if not isinstance(spk, dict):
            continue
        s = ts2s(spk.get("start"))
        e = ts2s(spk.get("end"))
        if e - s < MIN_PHRASE_S:
            continue

        si = max(0, int(s * sr))
        ei = min(int(e * sr), len(audio))
        if ei - si < int(sr * MIN_PHRASE_S):
            continue

        chunk = audio[si:ei]
        emb = model.embed_chunk(chunk, sr)
        if emb is None:
            continue
        n = np.linalg.norm(emb)
        if n == 0:
            continue
        emb = emb / n

        # Cosine to every agent (max across stacks)
        sims = np.array([float(np.max(stacks[j] @ emb))
                          for j in range(len(slugs))], dtype=np.float32)
        if sims.size == 0:
            continue
        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])
        pred_slug = slugs[best_j]

        is_agent_pred = best_sim >= threshold
        speaker = (spk.get("speaker") or "").strip()
        is_agent_truth = bool(speaker) and speaker.lower() != "customer"

        if is_agent_truth:
            total_agent_phrases += 1
            if pred_slug == expected_slug and is_agent_pred:
                correct_agent_hits += 1

        if is_agent_pred and is_agent_truth:
            tp += 1
        elif is_agent_pred and not is_agent_truth:
            fp += 1
        elif not is_agent_pred and is_agent_truth:
            fn += 1
        else:
            tn += 1

        agent_sim_sum[pred_slug] = agent_sim_sum.get(pred_slug, 0.0) + best_sim
        agent_sim_n[pred_slug] = agent_sim_n.get(pred_slug, 0) + 1

    if not agent_sim_n:
        return None

    call_agent_slug = max(
        agent_sim_sum, key=lambda k: agent_sim_sum[k] / max(agent_sim_n[k], 1))
    correct_call = (call_agent_slug == expected_slug)

    return {
        "_id": rid,
        "expected_slug": expected_slug,
        "predicted_slug": call_agent_slug,
        "correct_call": correct_call,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "agent_phrases": total_agent_phrases,
        "agent_hits": correct_agent_hits,
    }


def summarise(rows: List[dict], label: str) -> None:
    if not rows:
        print(f"[{label}] no scored calls")
        return
    n_calls = len(rows)
    correct = sum(1 for r in rows if r["correct_call"])
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n=== {label} ===")
    print(f"  calls scored:           {n_calls}")
    print(f"  call-level agent ID:    {correct}/{n_calls} ({100.0*correct/n_calls:.1f}%)")
    print(f"  segment AGENT P/R/F1:   {prec:.3f} / {rec:.3f} / {f1:.3f}")
    print(f"  segment counts:         tp={tp} fp={fp} tn={tn} fn={fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20,
                    help="Max held-out calls to score")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    args = ap.parse_args()

    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[gemini-test] {model.model_name} ready (dim={model.dim})")

    vps = load_voiceprint_stacks(multi=True, target_dim=model.dim)
    if not vps:
        sys.exit("[ERROR] no voiceprints loaded")
    n_centroids = sum(s.shape[0] for _, s in vps.values())
    print(f"[gemini-test] {len(vps)} agents / {n_centroids} centroids")

    held = held_out_calls_with_gemini(args.top)
    print(f"[gemini-test] {len(held)} calls with Gemini transcription")
    if not held:
        sys.exit("[gemini-test] no calls with Gemini transcription")

    rows: List[dict] = []
    try:
        for i, rec in enumerate(held, 1):
            expected_slug = slug(rec["agent_name"])
            if expected_slug not in vps:
                continue
            r = evaluate_call_vs_gemini(rec, vps, model, args.threshold, expected_slug)
            if r is None:
                continue
            rows.append(r)
            verdict = "ok" if r["correct_call"] else "WRONG"
            print(f"  [{i:3d}] {rec['_id'][:8]} ({rec['agent_name'][:30]:>30}) "
                  f"{verdict} ({i*1.0:.0f}s)", flush=True)
    finally:
        model.unload()

    summarise(rows, "GEMINI GROUND TRUTH")


if __name__ == "__main__":
    main()
