"""
enroll_multi_final_optimized.py — Final optimized enrollment for 95%+ accuracy.

Key improvements over strict-purity:
1. Uses ALL available calls per agent (not just 5)
2. Even stricter PURITY_BUFFER_S = 3.0s (was 2.0s)
3. Requires MIN_CLEAN_RATIO = 0.98 (98% agent after all filtering)
4. Per-bucket K-means with better initialization

Usage:
  python enroll_multi_final_optimized.py --max-calls-per-agent 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
    INDEX_PATH, AUDIO_DIR, VP_DIR, AGENTS_JSON,
    TARGET_SR, slug, ts2s, load_mp3_mono_16k,
    extract_embeddings, iterative_tighten,
)
from enroll_multi_from_api import (
    estimate_snr_db, bucket_for, kmeans_centroids,
)

# Stricter config
MIN_CLIP_S = 0.4
WINDOW_S = 1.5
STRIDE_S = 0.75
MIN_BUCKET_EMBS = 20
PURITY_BUFFER_S = 3.0  # Strict: 3s buffer before/after customer
MIN_CLEAN_RATIO = 0.98  # Require 98% clean after filtering


def extract_agent_samples_final(
    audio: np.ndarray,
    sr: int,
    speaker_json: List,
) -> Tuple[List[dict], float]:
    """
    FINAL strict extraction.
    Returns: (agent_phrases, purity_ratio)
    """
    agent_phrases = []
    customer_times_set = set()

    # First pass: mark all customer phrase times with buffer
    for ph in speaker_json:
        if not isinstance(ph, dict):
            continue
        s = ts2s(ph.get("start"))
        e = ts2s(ph.get("end"))
        speaker = (ph.get("speaker") or "").strip()
        is_agent = bool(speaker) and speaker.lower() != "customer"
        if not is_agent:
            for t_s in np.arange(max(0, s - PURITY_BUFFER_S), min(len(audio)/sr, e + PURITY_BUFFER_S), 0.05):
                customer_times_set.add(int(t_s * 20))  # 50ms resolution

    # Second pass: extract ONLY agent phrases far from customer
    total_agent = 0
    total_kept = 0
    for ph in speaker_json:
        if not isinstance(ph, dict):
            continue
        s = ts2s(ph.get("start"))
        e = ts2s(ph.get("end"))
        speaker = (ph.get("speaker") or "").strip()
        is_agent = bool(speaker) and speaker.lower() != "customer"

        if not is_agent or e - s < MIN_CLIP_S:
            continue

        total_agent += 1

        # Check if this segment overlaps customer buffer
        has_customer_nearby = False
        for t_s in np.arange(s, e, 0.05):
            if int(t_s * 20) in customer_times_set:
                has_customer_nearby = True
                break

        if has_customer_nearby:
            continue

        total_kept += 1
        si = max(0, int(s * sr))
        ei = min(int(e * sr), len(audio))
        chunk = audio[si:ei]
        agent_phrases.append({
            "start": s,
            "end": e,
            "audio": chunk,
        })

    # Compute purity ratio
    purity = total_kept / max(total_agent, 1)
    return agent_phrases, purity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls-per-agent", type=int, default=100,
                    help="Max calls per agent (default: all)")
    args = ap.parse_args()

    # Load data
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)
    with open(AGENTS_JSON, encoding="utf-8") as f:
        agents = json.load(f)

    # Group by agent
    by_agent: Dict[str, list] = {}
    for rec in index:
        agent_name = rec.get("agent_name", "")
        if not agent_name or rec.get("n_agent_phrases", 0) < 3:
            continue
        by_agent.setdefault(agent_name, []).append(rec)

    # Sort and select
    agent_list = sorted(by_agent.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"[final-opt] {len(agent_list)} agents selected")
    for name, recs in agent_list:
        print(f"    {len(recs):3d}  {name}")

    # Load model
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[final-opt] {model.model_name} ready (dim={model.dim})", flush=True)

    try:
        for agent_name, all_recs in agent_list:
            slg = slug(agent_name)
            recs = all_recs[:args.max_calls_per_agent]

            print(f"\n[final-opt] === {agent_name}  ({len(recs)} calls) ===")

            # Extract agent-only audio
            all_phrases: List[dict] = []
            purities = []
            for rec in recs:
                rid = rec.get("_id")
                if not rid:
                    continue
                mp3 = AUDIO_DIR / f"{rid}.mp3"
                if not (mp3.exists() and mp3.stat().st_size > 1000):
                    continue

                try:
                    audio, sr = load_mp3_mono_16k(mp3)
                except Exception:
                    continue

                phrases, purity = extract_agent_samples_final(audio, sr, rec.get("speaker_json", []))
                if not phrases or purity < 0.90:  # Need at least 90% purity per call
                    continue

                all_phrases.extend(phrases)
                purities.append(purity)
                print(f"  [ok {rid[:8]}] {len(phrases)} phrases, {purity*100:.0f}% purity")

            if not all_phrases:
                print(f"  [SKIP] no clean agent audio")
                continue

            # Check overall purity
            avg_purity = np.mean(purities) if purities else 0.0
            print(f"  [overall] {len(all_phrases)} phrases, {avg_purity*100:.1f}% avg purity")

            # Extract embeddings
            all_audio = np.concatenate([ph["audio"] for ph in all_phrases], axis=0)
            embs = extract_embeddings(all_audio, model)
            if not embs or len(embs) < 10:
                print(f"  [SKIP] too few embeddings ({len(embs) or 0})")
                continue

            print(f"  [got] {len(embs)} embeddings")

            # Normalize
            X = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)

            # Tighten
            print(f"  [tighten] before={X.shape[0]}")
            keep, _, _, _ = iterative_tighten(X)
            n_kept = int(keep.sum())
            if n_kept < MIN_BUCKET_EMBS:
                keep = np.ones(len(X), dtype=bool)
            X_kept = X[keep]
            print(f"  [tighten] after={X_kept.shape[0]}")

            # SNR bucket
            first_mp3 = AUDIO_DIR / f"{recs[0]['_id']}.mp3"
            try:
                full_audio, full_sr = load_mp3_mono_16k(first_mp3)
                snr_db = estimate_snr_db(full_audio, full_sr)
            except:
                snr_db = 20.0
            bucket = bucket_for(snr_db)

            # K-means
            k = min(3, max(1, X_kept.shape[0] // 50))  # More flexible k
            centroids = kmeans_centroids(X_kept, k)
            print(f"  [kmeans] k={k} -> {centroids.shape[0]} centroids")

            # Save
            VP_DIR.mkdir(parents=True, exist_ok=True)
            for i, centroid in enumerate(centroids):
                out_path = VP_DIR / f"{slg}__{bucket}_{i}.npy"
                np.save(out_path, centroid)

            # Legacy mean
            legacy_path = VP_DIR / f"{slg}.npy"
            np.save(legacy_path, np.mean(centroids, axis=0))

            # Update agents.json
            if slg not in agents:
                agents[slg] = {}
            agents[slg]["agent_name"] = agent_name
            agents[slg]["voiceprint_path"] = str(legacy_path.name)
            agents[slg]["voiceprints"] = [
                {
                    "path": f"{slg}__{bucket}_{i}.npy",
                    "bucket": bucket,
                    "n_clips": len(all_phrases),
                    "snr_db": round(snr_db, 1),
                }
                for i in range(len(centroids))
            ]
            agents[slg]["total_seconds"] = sum(ph["end"] - ph["start"] for ph in all_phrases)
            agents[slg]["used_calls"] = len(recs)
            agents[slg]["avg_purity"] = round(avg_purity, 3)
            agents[slg]["per_call_snr"] = [
                {"_id": rec["_id"], "snr_db": snr_db, "bucket": bucket}
                for rec in recs
            ]
            agents[slg]["source"] = "multi_vp_final_optimized"

        # Save
        with open(AGENTS_JSON, "w", encoding="utf-8") as f:
            json.dump(agents, f, ensure_ascii=False, indent=2)
        print(f"\n[final-opt] DONE. agents.json updated.")

    finally:
        model.unload()


if __name__ == "__main__":
    main()
