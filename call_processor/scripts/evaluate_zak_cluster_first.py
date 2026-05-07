#!/usr/bin/env python
"""Evaluate call-level cluster-first Zak role identification.

This tests a production-like role strategy on labelled calls:
1. Embed each labelled utterance.
2. Cluster utterance embeddings inside each call.
3. Map the cluster closest to Zak's voiceprint as AGENT.
4. Score against local data.json roles.

It does not use labels to choose the agent cluster.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR))
sys.path.insert(0, str(CALL_PROCESSOR_DIR / "scripts"))

from src.embedding_campp import EmbeddingModel, l2_norm  # noqa: E402
from src.voiceprints import resolve_voiceprint_path  # noqa: E402
from train_zak_pure_embeddings import (  # noqa: E402
    CALL_PROCESSOR_DIR as TRAIN_CALL_PROCESSOR_DIR,
    extract_label_rows,
    load_labelled_calls,
)


def load_stack(paths: list[str], voiceprint_dir: Path) -> np.ndarray:
    vectors = []
    for raw in paths:
        path = Path(resolve_voiceprint_path(raw, str(voiceprint_dir / "agents.json")))
        if not path.is_absolute():
            path = voiceprint_dir / path
        arr = np.load(path).astype(np.float32).squeeze()
        if arr.ndim != 1 or arr.shape[0] != 512:
            continue
        vectors.append(l2_norm(arr))
    if not vectors:
        raise ValueError(f"No 512-dim voiceprints loaded from {paths}")
    return np.stack(vectors).astype(np.float32)


def best_sim(embedding: np.ndarray, stack: np.ndarray) -> float:
    return float(np.max(stack @ l2_norm(embedding)))


def centroid(mat: np.ndarray) -> np.ndarray:
    return l2_norm(mat.mean(axis=0)).astype(np.float32)


def score_predictions(items: list[dict]) -> dict:
    total = correct = 0
    agent_total = agent_correct = 0
    customer_total = customer_correct = 0
    dur_total = dur_correct = 0.0
    dur_by_role = {"agent": 0.0, "customer": 0.0}
    dur_correct_by_role = {"agent": 0.0, "customer": 0.0}
    for item in items:
        truth = item["truth"]
        pred = item["predicted"]
        dur = float(item["duration"])
        ok = pred == truth
        total += 1
        correct += int(ok)
        dur_total += dur
        dur_correct += dur if ok else 0.0
        if truth == "agent":
            agent_total += 1
            agent_correct += int(ok)
        elif truth == "customer":
            customer_total += 1
            customer_correct += int(ok)
        if truth in dur_by_role:
            dur_by_role[truth] += dur
            dur_correct_by_role[truth] += dur if ok else 0.0

    def pct(num: float, den: float) -> float:
        return round((num / den * 100.0) if den else 0.0, 2)

    return {
        "segments": total,
        "overall_accuracy": pct(correct, total),
        "agent_accuracy": pct(agent_correct, agent_total),
        "customer_accuracy": pct(customer_correct, customer_total),
        "duration_accuracy": pct(dur_correct, dur_total),
        "duration_agent_accuracy": pct(dur_correct_by_role["agent"], dur_by_role["agent"]),
        "duration_customer_accuracy": pct(dur_correct_by_role["customer"], dur_by_role["customer"]),
        "agent_correct": agent_correct,
        "agent_total": agent_total,
        "customer_correct": customer_correct,
        "customer_total": customer_total,
    }


def cluster_call(rows, stack: np.ndarray, n_clusters: int, min_cluster_rows: int) -> tuple[list[dict], dict]:
    from sklearn.cluster import KMeans

    valid = [row for row in rows if row.speaker in {"agent", "customer"}]
    if len(valid) < max(2, min_cluster_rows):
        return [], {"reason": "not enough rows", "rows": len(valid)}
    x = np.stack([l2_norm(row.embedding) for row in valid]).astype(np.float32)
    k = min(max(2, n_clusters), len(valid))
    labels = KMeans(n_clusters=k, random_state=42, n_init=25).fit_predict(x)
    cluster_sims = {}
    for cid in sorted(set(int(v) for v in labels)):
        part = x[labels == cid]
        cluster_sims[cid] = best_sim(centroid(part), stack)
    agent_cluster = max(cluster_sims, key=cluster_sims.get)
    predictions = []
    for row, cid in zip(valid, labels):
        pred = "agent" if int(cid) == int(agent_cluster) else "customer"
        predictions.append({
            "call": row.call_name,
            "segment": row.segment_idx,
            "truth": row.speaker,
            "predicted": pred,
            "cluster": int(cid),
            "duration": round(float(row.duration), 3),
            "text": row.text[:140],
        })
    return predictions, {
        "rows": len(valid),
        "n_clusters": k,
        "agent_cluster": int(agent_cluster),
        "cluster_sims": {str(k): round(v, 4) for k, v in cluster_sims.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default=str(CALL_PROCESSOR_DIR / "data" / "training"))
    parser.add_argument("--audio-root", default=str(REPO_ROOT / "traning_data" / "zak_raissi"))
    parser.add_argument("--label-source", choices=("training-json", "folder-data", "both"), default="folder-data")
    parser.add_argument("--out", default=str(CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / "zak_cluster_first_comparison.json"))
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--min-cluster-rows", type=int, default=8)
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir).resolve()
    audio_root = Path(args.audio_root).resolve()
    calls, skipped_calls = load_labelled_calls(labels_dir, audio_root, label_source=args.label_source)
    print(f"[data] calls={len(calls)} skipped={len(skipped_calls)}")

    model = EmbeddingModel()
    print("[model] loading CAM++")
    model.load(force_cpu=True)
    try:
        rows, skipped_segments = extract_label_rows(
            calls,
            model,
            min_eval_dur=0.8,
            min_train_dur=1.5,
            max_train_dur=60.0,
            agent_filter="all",
            train_guard_s=0.0,
            opposite_gap_s=0.0,
        )
    finally:
        model.unload()

    voiceprint_dir = TRAIN_CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
    pure_paths = [p.name for p in sorted(voiceprint_dir.glob("zak_raissi_barnet_pure_campp_v*.npy"))]
    candidates = {
        "restored_phone_campp": ["zak_raissi_barnet.npy"],
        "pure_gemini_campp": pure_paths,
        "combined_restored_plus_pure": ["zak_raissi_barnet.npy", *pure_paths],
    }

    by_call: dict[str, list] = {}
    for row in rows:
        by_call.setdefault(row.call_name, []).append(row)

    results = {}
    for name, paths in candidates.items():
        stack = load_stack(paths, voiceprint_dir)
        predictions = []
        per_call = {}
        for call_name, call_rows in sorted(by_call.items()):
            preds, meta = cluster_call(call_rows, stack, args.clusters, args.min_cluster_rows)
            if preds:
                predictions.extend(preds)
                per_call[call_name] = {**meta, **score_predictions(preds)}
            else:
                per_call[call_name] = meta
        results[name] = {
            "paths": paths,
            "n_voiceprints": int(stack.shape[0]),
            "overall": score_predictions(predictions),
            "per_call": per_call,
            "errors": [p for p in predictions if p["truth"] != p["predicted"]][:80],
        }

    winner = max(results, key=lambda k: (
        results[k]["overall"]["overall_accuracy"],
        results[k]["overall"]["customer_accuracy"],
        results[k]["overall"]["agent_accuracy"],
    ))
    report = {
        "label_source": args.label_source,
        "audio_root": str(audio_root),
        "calls": len(calls),
        "skipped_calls": skipped_calls,
        "skipped_segments": skipped_segments[:100],
        "cluster_strategy": "KMeans per call; cluster closest to Zak voiceprint stack is AGENT",
        "clusters": args.clusters,
        "winner": winner,
        "results": results,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "winner": winner,
        "results": {k: v["overall"] for k, v in results.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
