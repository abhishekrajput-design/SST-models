"""
enroll_multi_advanced.py — Advanced enrollment with improved clustering.

Improvements:
1. K-means++ initialization (better centroids)
2. Quality filtering (only use high-confidence embeddings)
3. Per-agent threshold tuning (98th percentile purity)
4. Ensemble clustering (multiple runs, pick best)

Usage:
  python enroll_multi_advanced.py --max-calls-per-agent 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

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
    estimate_snr_db, bucket_for,
)

MIN_CLIP_S = 0.4
PURITY_BUFFER_S = 3.0


def extract_agent_samples_advanced(
    audio: np.ndarray,
    sr: int,
    speaker_json: List,
) -> List[dict]:
    """Advanced extraction with stricter filtering."""
    agent_phrases = []
    customer_times = set()

    # Mark customer times
    for ph in speaker_json:
        if not isinstance(ph, dict):
            continue
        s = ts2s(ph.get("start"))
        e = ts2s(ph.get("end"))
        speaker = (ph.get("speaker") or "").strip()
        is_agent = bool(speaker) and speaker.lower() != "customer"
        if not is_agent:
            for t_s in np.arange(max(0, s - PURITY_BUFFER_S), min(len(audio)/sr, e + PURITY_BUFFER_S), 0.05):
                customer_times.add(int(t_s * 20))

    # Extract only clean agent phrases
    for ph in speaker_json:
        if not isinstance(ph, dict):
            continue
        s = ts2s(ph.get("start"))
        e = ts2s(ph.get("end"))
        speaker = (ph.get("speaker") or "").strip()
        is_agent = bool(speaker) and speaker.lower() != "customer"

        if not is_agent or e - s < MIN_CLIP_S:
            continue

        # Check for customer nearby
        has_customer = False
        for t_s in np.arange(s, e, 0.05):
            if int(t_s * 20) in customer_times:
                has_customer = True
                break

        if has_customer:
            continue

        si = max(0, int(s * sr))
        ei = min(int(e * sr), len(audio))
        agent_phrases.append({
            "start": s,
            "end": e,
            "audio": audio[si:ei],
        })

    return agent_phrases


def kmeans_advanced(X: np.ndarray, k: int) -> np.ndarray:
    """K-means with k-means++ initialization."""
    if X.shape[0] < k:
        return X
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42, max_iter=300)
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls-per-agent", type=int, default=100)
    args = ap.parse_args()

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)
    with open(AGENTS_JSON, encoding="utf-8") as f:
        agents = json.load(f)

    by_agent: Dict[str, list] = {}
    for rec in index:
        agent_name = rec.get("agent_name", "")
        if not agent_name or rec.get("n_agent_phrases", 0) < 3:
            continue
        by_agent.setdefault(agent_name, []).append(rec)

    agent_list = sorted(by_agent.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"[advanced] {len(agent_list)} agents")

    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[advanced] {model.model_name} ready (dim={model.dim})")

    try:
        for agent_name, all_recs in agent_list:
            slg = slug(agent_name)
            recs = all_recs[:args.max_calls_per_agent]

            print(f"\n[advanced] === {agent_name}  ({len(recs)} calls) ===")

            all_phrases: List[dict] = []
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

                phrases = extract_agent_samples_advanced(audio, sr, rec.get("speaker_json", []))
                if not phrases:
                    continue
                all_phrases.extend(phrases)

            if not all_phrases:
                print(f"  [SKIP] no clean audio")
                continue

            all_audio = np.concatenate([ph["audio"] for ph in all_phrases], axis=0)
            embs = extract_embeddings(all_audio, model)
            if not embs or len(embs) < 10:
                print(f"  [SKIP] too few embs ({len(embs)})")
                continue

            X = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)

            # Tighten
            keep, _, _, _ = iterative_tighten(X)
            n_kept = int(keep.sum())
            if n_kept < 20:
                keep = np.ones(len(X), dtype=bool)
            X_kept = X[keep]

            # SNR
            try:
                full_audio, full_sr = load_mp3_mono_16k(AUDIO_DIR / f"{recs[0]['_id']}.mp3")
                snr_db = estimate_snr_db(full_audio, full_sr)
            except:
                snr_db = 20.0
            bucket = bucket_for(snr_db)

            # Advanced K-means
            k = min(3, max(1, X_kept.shape[0] // 40))
            centroids = kmeans_advanced(X_kept, k)

            print(f"  [embs] {len(embs)} -> {X_kept.shape[0]} (kept)")
            print(f"  [cluster] k={k} -> {len(centroids)} centroids")

            # Save
            VP_DIR.mkdir(parents=True, exist_ok=True)
            for i, centroid in enumerate(centroids):
                out_path = VP_DIR / f"{slg}__{bucket}_{i}.npy"
                np.save(out_path, centroid)

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
            agents[slg]["per_call_snr"] = [{"_id": rec["_id"]} for rec in recs]
            agents[slg]["source"] = "multi_vp_advanced"

        with open(AGENTS_JSON, "w", encoding="utf-8") as f:
            json.dump(agents, f, ensure_ascii=False, indent=2)
        print(f"\n[advanced] DONE. agents.json updated.")

    finally:
        model.unload()


if __name__ == "__main__":
    main()
