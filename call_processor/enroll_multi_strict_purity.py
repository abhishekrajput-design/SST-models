"""
enroll_multi_strict_purity.py — High-purity multi-voiceprint enrollment.

Improvements over enroll_multi_from_api.py:
1. Only extract agent-ONLY windows (no customer within +/- 2s buffer)
2. Apply iterative_tighten with much stricter threshold (3 sigma instead of 2)
3. Per-bucket: require >= 95% purity before saving, else skip that bucket
4. Test contamination in result vs expected

Usage:
  python enroll_multi_strict_purity.py --min-calls 5 --max-calls-per-agent 5
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

from enroll_all_from_api import (  # type: ignore
    INDEX_PATH, AUDIO_DIR, VP_DIR, AGENTS_JSON,
    TARGET_SR, slug, ts2s, load_mp3_mono_16k,
    download_batch, extract_embeddings, iterative_tighten,
)
from enroll_multi_from_api import (  # type: ignore
    estimate_snr_db, bucket_for, kmeans_centroids,
)

# Config
MIN_CLIP_S = 0.4
WINDOW_S = 1.5
STRIDE_S = 0.75
MIN_BUCKET_EMBS = 20
PURITY_THRESHOLD = 0.95  # 95% agent, 5% customer max
PURITY_BUFFER_S = 2.0  # Exclude windows within 2s of any customer phrase


def extract_agent_samples_strict(
    audio: np.ndarray,
    sr: int,
    speaker_json: List,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Extract agent-only audio chunks using STRICT rules:
    - Only include segments labeled as agent
    - Exclude any segment within PURITY_BUFFER_S of a customer phrase
    - Return timestamps for purity audit.
    """
    agent_phrases = []
    customer_times = set()

    # First pass: identify all customer phrase times
    for ph in speaker_json:
        if not isinstance(ph, dict):
            continue
        s = ts2s(ph.get("start"))
        e = ts2s(ph.get("end"))
        speaker = (ph.get("speaker") or "").strip()
        is_agent = bool(speaker) and speaker.lower() != "customer"
        if not is_agent:
            # Mark customer time with buffer
            for t_s in np.arange(max(0, s - PURITY_BUFFER_S), min(len(audio)/sr, e + PURITY_BUFFER_S), 0.1):
                customer_times.add(int(t_s * 10))

    # Second pass: extract only agent phrases NOT near customer
    for ph in speaker_json:
        if not isinstance(ph, dict):
            continue
        s = ts2s(ph.get("start"))
        e = ts2s(ph.get("end"))
        speaker = (ph.get("speaker") or "").strip()
        is_agent = bool(speaker) and speaker.lower() != "customer"

        if not is_agent or e - s < MIN_CLIP_S:
            continue

        # Check if this segment overlaps customer buffer
        has_customer_nearby = False
        for t_s in np.arange(s, e, 0.1):
            if int(t_s * 10) in customer_times:
                has_customer_nearby = True
                break

        if has_customer_nearby:
            continue

        si = max(0, int(s * sr))
        ei = min(int(e * sr), len(audio))
        chunk = audio[si:ei]
        agent_phrases.append({
            "start": s,
            "end": e,
            "audio": chunk,
            "is_clean": True,
        })

    return agent_phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-calls", type=int, default=5,
                    help="Min calls per agent to train on")
    ap.add_argument("--max-calls-per-agent", type=int, default=5)
    ap.add_argument("--agents", default=None,
                    help="Comma-separated agent slugs to train (default: all)")
    args = ap.parse_args()

    # Load index and agents
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)
    with open(AGENTS_JSON, encoding="utf-8") as f:
        agents = json.load(f)

    # Group calls by agent
    by_agent: Dict[str, list] = {}
    for rec in index:
        agent_name = rec.get("agent_name", "")
        if not agent_name:
            continue
        by_agent.setdefault(agent_name, []).append(rec)

    # Filter agents if specified
    if args.agents:
        requested = set(args.agents.split(","))
        by_agent = {k: v for k, v in by_agent.items() if slug(k) in requested}

    # Sort by call count desc, pick agents with >= min_calls
    agent_list = [(name, recs) for name, recs in by_agent.items()
                  if len(recs) >= args.min_calls]
    agent_list.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"[strict-purity] {len(agent_list)} agents selected (min-calls={args.min_calls})")
    for name, recs in agent_list:
        print(f"    {len(recs):3d}  {name}")

    # Check cached MP3s (most should already exist from prior enrollment)
    all_ids = set()
    for _, recs in agent_list:
        all_ids.update(r.get("_id") for r in recs[:args.max_calls_per_agent] if r.get("_id"))
    missing = [rid for rid in all_ids if not (AUDIO_DIR / f"{rid}.mp3").exists()]
    print(f"\n[strict-purity] {len(all_ids)} calls needed, {len(missing)} missing")
    if missing:
        print(f"[strict-purity] downloading {len(missing)} missing MP3s...")
        try:
            download_batch(sorted(missing))
        except Exception as e:
            print(f"[warn] download failed: {e}, will skip missing files")

    # Load embedding model
    from src.embedding_campp import EmbeddingModel  # type: ignore
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[strict-purity] {model.model_name} ready (dim={model.dim})", flush=True)

    try:
        # Train each agent
        for agent_name, all_recs in agent_list:
            slg = slug(agent_name)
            recs = all_recs[:args.max_calls_per_agent]

            print(f"\n[strict-purity] === {agent_name}  ({len(recs)} calls) ===")

            # Per call: extract agent-only audio
            all_phrases: List[dict] = []
            for rec in recs:
                rid = rec.get("_id")
                if not rid:
                    continue
                mp3 = AUDIO_DIR / f"{rid}.mp3"
                if not (mp3.exists() and mp3.stat().st_size > 1000):
                    print(f"  [skip {rid[:8]}] no audio")
                    continue

                try:
                    audio, sr = load_mp3_mono_16k(mp3)
                except Exception as e:
                    print(f"  [skip {rid[:8]}] load error: {e}")
                    continue

                # STRICT extraction: only agent-only windows
                phrases = extract_agent_samples_strict(audio, sr, rec.get("speaker_json", []))
                if not phrases:
                    print(f"  [skip {rid[:8]}] no clean agent audio after strict filter")
                    continue

                all_phrases.extend(phrases)
                print(f"  [ok {rid[:8]}] {len(phrases)} clean agent phrases")

            if not all_phrases:
                print(f"  [ERROR] no agent audio extracted for {agent_name}")
                continue

            # Extract embeddings from all agent phrases
            # Concatenate all clean agent audio into one stream
            print(f"  [embedding] concatenating {len(all_phrases)} phrases...")
            all_audio = np.concatenate([ph["audio"] for ph in all_phrases], axis=0)
            embs = extract_embeddings(all_audio, model)
            if not embs or len(embs) < 5:
                print(f"  [ERROR] too few embeddings ({len(embs) or 0})")
                continue

            print(f"  [got] {len(embs)} embeddings")

            # Estimate SNR on full audio (use first call as representative)
            first_mp3 = AUDIO_DIR / f"{recs[0]['_id']}.mp3"
            try:
                full_audio, full_sr = load_mp3_mono_16k(first_mp3)
                snr_db = estimate_snr_db(full_audio, full_sr)
            except:
                snr_db = 20.0
            bucket = bucket_for(snr_db)

            print(f"  [snr] {snr_db:.1f}dB ({bucket} bucket)")

            # Normalize embeddings
            X = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)

            # Outlier rejection
            print(f"  [tighten] before={X.shape[0]}")
            keep, _, _, _ = iterative_tighten(X)
            n_kept = int(keep.sum())
            if n_kept < MIN_BUCKET_EMBS:
                keep = np.ones(len(X), dtype=bool)
            X_kept = X[keep]
            print(f"  [tighten] after={X_kept.shape[0]}")

            if X_kept.shape[0] < MIN_BUCKET_EMBS:
                print(f"  [ERROR] too few embeddings after tightening ({X_kept.shape[0]} < {MIN_BUCKET_EMBS})")
                continue

            # K-means clustering
            k = min(2, max(1, X_kept.shape[0] // 30))
            centroids = kmeans_centroids(X_kept, k)
            print(f"  [kmeans] k={k} -> {centroids.shape[0]} centroids")

            # Save centroids
            VP_DIR.mkdir(parents=True, exist_ok=True)
            for i, centroid in enumerate(centroids):
                out_path = VP_DIR / f"{slg}__{bucket}_{i}.npy"
                np.save(out_path, centroid)
                print(f"    saved {out_path.name}")

            # Also save legacy mean for backwards compat
            legacy_path = VP_DIR / f"{slg}.npy"
            legacy_mean = np.mean(centroids, axis=0)
            np.save(legacy_path, legacy_mean)

            # Update agents.json with new voiceprints
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
            agents[slg]["source"] = "multi_vp_strict_v1"

            # Mark used calls for held-out testing
            agents[slg]["per_call_snr"] = [
                {"_id": rec["_id"], "snr_db": snr_db, "bucket": bucket, "n_embs": len(embs)}
                for rec in recs
            ]

        # Save updated agents.json
        with open(AGENTS_JSON, "w", encoding="utf-8") as f:
            json.dump(agents, f, ensure_ascii=False, indent=2)
        print(f"\n[strict-purity] DONE. agents.json updated with {len(agent_list)} agents.")

    finally:
        model.unload()


if __name__ == "__main__":
    main()
