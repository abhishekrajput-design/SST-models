#!/usr/bin/env python
"""Search hybrid Zak role-ID rules on labelled calls.

The rule is intentionally production-shaped:
  - compare each utterance to the available Zak voiceprint stack;
  - cluster utterances inside each call without using labels;
  - mark a row as AGENT only when direct similarity or cluster evidence is high.

Labels are used only for scoring and threshold search.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Rule:
    strong_sim: float
    cluster_sim: float
    cluster_margin: float
    weak_sim: float


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
        raise ValueError(f"No valid 512-dim voiceprints loaded from {paths}")
    return np.stack(vectors).astype(np.float32)


def best_sim(embedding: np.ndarray, stack: np.ndarray) -> float:
    return float(np.max(stack @ l2_norm(embedding)))


def centroid(mat: np.ndarray) -> np.ndarray:
    return l2_norm(mat.mean(axis=0)).astype(np.float32)


def build_items(rows, stack: np.ndarray, n_clusters: int, min_cluster_rows: int) -> tuple[list[dict], dict]:
    from sklearn.cluster import KMeans

    valid = [row for row in rows if row.speaker in {"agent", "customer"}]
    by_call: dict[str, list] = {}
    for row in valid:
        by_call.setdefault(row.call_name, []).append(row)

    items: list[dict] = []
    per_call: dict[str, dict] = {}
    for call_name, call_rows in sorted(by_call.items()):
        x = np.stack([l2_norm(row.embedding) for row in call_rows]).astype(np.float32)
        row_sims = np.asarray([best_sim(row.embedding, stack) for row in call_rows], dtype=np.float32)
        k = min(max(2, n_clusters), len(call_rows))
        if len(call_rows) < max(2, min_cluster_rows):
            labels = np.zeros(len(call_rows), dtype=np.int32)
            cluster_sims = {0: float(best_sim(centroid(x), stack))}
        else:
            labels = KMeans(n_clusters=k, random_state=42, n_init=25).fit_predict(x)
            cluster_sims = {}
            for cid in sorted(set(int(v) for v in labels)):
                cluster_sims[cid] = float(best_sim(centroid(x[labels == cid]), stack))
        sorted_clusters = sorted(cluster_sims.items(), key=lambda item: item[1], reverse=True)
        top_cluster, top_sim = sorted_clusters[0]
        second_sim = sorted_clusters[1][1] if len(sorted_clusters) > 1 else -1.0
        margin = top_sim - second_sim
        per_call[call_name] = {
            "rows": len(call_rows),
            "n_clusters": len(cluster_sims),
            "top_cluster": int(top_cluster),
            "top_cluster_sim": round(top_sim, 4),
            "second_cluster_sim": round(second_sim, 4),
            "cluster_margin": round(margin, 4),
            "cluster_sims": {str(k): round(v, 4) for k, v in cluster_sims.items()},
        }
        for row, cid, sim in zip(call_rows, labels, row_sims):
            items.append({
                "call": row.call_name,
                "segment": row.segment_idx,
                "truth": row.speaker,
                "duration": float(row.duration),
                "sim": float(sim),
                "cluster": int(cid),
                "cluster_is_top": int(cid) == int(top_cluster),
                "top_cluster_sim": float(top_sim),
                "cluster_margin": float(margin),
                "text": row.text[:160],
            })
    return items, per_call


def predict(item: dict, rule: Rule) -> str:
    if item["sim"] >= rule.strong_sim:
        return "agent"
    if (
        item["cluster_is_top"]
        and item["top_cluster_sim"] >= rule.cluster_sim
        and item["cluster_margin"] >= rule.cluster_margin
        and item["sim"] >= rule.weak_sim
    ):
        return "agent"
    return "customer"


def score_items(items: list[dict], rule: Rule) -> dict:
    total = correct = 0
    agent_total = agent_correct = 0
    customer_total = customer_correct = 0
    dur_total = dur_correct = 0.0
    dur_by_role = {"agent": 0.0, "customer": 0.0}
    dur_correct_by_role = {"agent": 0.0, "customer": 0.0}
    errors = []
    per_call: dict[str, dict[str, int]] = {}
    for item in items:
        truth = item["truth"]
        pred = predict(item, rule)
        ok = pred == truth
        total += 1
        correct += int(ok)
        dur = float(item["duration"])
        dur_total += dur
        dur_correct += dur if ok else 0.0
        bucket = per_call.setdefault(
            item["call"],
            {"total": 0, "correct": 0, "agent_total": 0, "agent_correct": 0, "customer_total": 0, "customer_correct": 0},
        )
        bucket["total"] += 1
        bucket["correct"] += int(ok)
        if truth == "agent":
            agent_total += 1
            agent_correct += int(ok)
            bucket["agent_total"] += 1
            bucket["agent_correct"] += int(ok)
        elif truth == "customer":
            customer_total += 1
            customer_correct += int(ok)
            bucket["customer_total"] += 1
            bucket["customer_correct"] += int(ok)
        if truth in dur_by_role:
            dur_by_role[truth] += dur
            dur_correct_by_role[truth] += dur if ok else 0.0
        if not ok and len(errors) < 60:
            errors.append({
                "call": item["call"],
                "segment": item["segment"],
                "truth": truth,
                "predicted": pred,
                "sim": round(float(item["sim"]), 4),
                "cluster": item["cluster"],
                "cluster_is_top": item["cluster_is_top"],
                "top_cluster_sim": round(float(item["top_cluster_sim"]), 4),
                "cluster_margin": round(float(item["cluster_margin"]), 4),
                "text": item["text"],
            })

    def pct(num: float, den: float) -> float:
        return round((num / den * 100.0) if den else 0.0, 2)

    per_call_out = {}
    for call, bucket in per_call.items():
        per_call_out[call] = {
            **bucket,
            "overall_accuracy": pct(bucket["correct"], bucket["total"]),
            "agent_accuracy": pct(bucket["agent_correct"], bucket["agent_total"]),
            "customer_accuracy": pct(bucket["customer_correct"], bucket["customer_total"]),
        }
    return {
        "rule": asdict(rule),
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
        "per_call": per_call_out,
        "errors": errors,
    }


def rule_grid() -> list[Rule]:
    rules = []
    for strong_sim in np.arange(0.58, 0.901, 0.02):
        for cluster_sim in np.arange(0.62, 0.901, 0.04):
            for cluster_margin in np.arange(0.02, 0.451, 0.04):
                for weak_sim in np.arange(0.34, 0.741, 0.04):
                    if weak_sim > strong_sim:
                        continue
                    rules.append(Rule(
                        strong_sim=round(float(strong_sim), 3),
                        cluster_sim=round(float(cluster_sim), 3),
                        cluster_margin=round(float(cluster_margin), 3),
                        weak_sim=round(float(weak_sim), 3),
                    ))
    return rules


def choose_best(items: list[dict], rules: list[Rule], min_agent: float, min_customer: float) -> dict:
    best = None
    for rule in rules:
        scored = score_items(items, rule)
        if scored["agent_accuracy"] < min_agent or scored["customer_accuracy"] < min_customer:
            continue
        key = (
            scored["overall_accuracy"],
            min(scored["agent_accuracy"], scored["customer_accuracy"]),
            scored["customer_accuracy"],
            scored["agent_accuracy"],
        )
        if best is None or key > best["_key"]:
            scored["_key"] = key
            best = scored
    if best is None:
        for rule in rules:
            scored = score_items(items, rule)
            key = (
                scored["overall_accuracy"],
                min(scored["agent_accuracy"], scored["customer_accuracy"]),
                scored["customer_accuracy"],
                scored["agent_accuracy"],
            )
            if best is None or key > best["_key"]:
                scored["_key"] = key
                best = scored
    best.pop("_key", None)
    return best


def leave_one_call_out(items: list[dict], rules: list[Rule], min_agent: float, min_customer: float) -> dict:
    calls = sorted({item["call"] for item in items})
    heldout_scores = []
    predictions = []
    for call in calls:
        train_items = [item for item in items if item["call"] != call]
        test_items = [item for item in items if item["call"] == call]
        best = choose_best(train_items, rules, min_agent, min_customer)
        rule = Rule(**best["rule"])
        scored = score_items(test_items, rule)
        heldout_scores.append({
            "call": call,
            "selected_rule": best["rule"],
            **{k: v for k, v in scored.items() if k not in {"per_call", "errors", "rule"}},
        })
        for item in test_items:
            predictions.append({**item, "predicted": predict(item, rule)})

    cv_rule = Rule(strong_sim=1.0, cluster_sim=1.0, cluster_margin=1.0, weak_sim=1.0)
    total = correct = 0
    agent_total = agent_correct = 0
    customer_total = customer_correct = 0
    errors = []
    for item in predictions:
        truth = item["truth"]
        pred = item["predicted"]
        ok = pred == truth
        total += 1
        correct += int(ok)
        if truth == "agent":
            agent_total += 1
            agent_correct += int(ok)
        else:
            customer_total += 1
            customer_correct += int(ok)
        if not ok and len(errors) < 60:
            errors.append({
                "call": item["call"],
                "segment": item["segment"],
                "truth": truth,
                "predicted": pred,
                "sim": round(float(item["sim"]), 4),
                "cluster": item["cluster"],
                "cluster_is_top": item["cluster_is_top"],
                "text": item["text"],
            })

    def pct(num: int, den: int) -> float:
        return round((num / den * 100.0) if den else 0.0, 2)

    return {
        "rule": asdict(cv_rule),
        "segments": total,
        "overall_accuracy": pct(correct, total),
        "agent_accuracy": pct(agent_correct, agent_total),
        "customer_accuracy": pct(customer_correct, customer_total),
        "agent_correct": agent_correct,
        "agent_total": agent_total,
        "customer_correct": customer_correct,
        "customer_total": customer_total,
        "heldout_calls": heldout_scores,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default=str(CALL_PROCESSOR_DIR / "data" / "training"))
    parser.add_argument("--audio-root", default=str(REPO_ROOT / "traning_data" / "zak_raissi"))
    parser.add_argument("--label-source", choices=("training-json", "folder-data", "both"), default="folder-data")
    parser.add_argument("--out", default=str(CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / "zak_hybrid_comparison.json"))
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--min-cluster-rows", type=int, default=8)
    parser.add_argument("--min-agent-accuracy", type=float, default=50.0)
    parser.add_argument("--min-customer-accuracy", type=float, default=50.0)
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

    rules = rule_grid()
    results = {}
    for name, paths in candidates.items():
        stack = load_stack(paths, voiceprint_dir)
        items, cluster_meta = build_items(rows, stack, args.clusters, args.min_cluster_rows)
        best_same_data = choose_best(items, rules, args.min_agent_accuracy, args.min_customer_accuracy)
        cv = leave_one_call_out(items, rules, args.min_agent_accuracy, args.min_customer_accuracy)
        results[name] = {
            "paths": paths,
            "n_voiceprints": int(stack.shape[0]),
            "cluster_meta": cluster_meta,
            "best_same_data": best_same_data,
            "leave_one_call_out": cv,
        }

    winner = max(
        results,
        key=lambda name: (
            results[name]["leave_one_call_out"]["overall_accuracy"],
            min(
                results[name]["leave_one_call_out"]["agent_accuracy"],
                results[name]["leave_one_call_out"]["customer_accuracy"],
            ),
            results[name]["leave_one_call_out"]["customer_accuracy"],
        ),
    )
    report = {
        "label_source": args.label_source,
        "audio_root": str(audio_root),
        "calls": len(calls),
        "skipped_calls": skipped_calls,
        "skipped_segments": skipped_segments[:100],
        "clusters": args.clusters,
        "rule": "agent if sim>=strong_sim OR top-cluster evidence plus sim>=weak_sim",
        "min_agent_accuracy": args.min_agent_accuracy,
        "min_customer_accuracy": args.min_customer_accuracy,
        "winner": winner,
        "results": results,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "winner": winner,
        "results": {
            name: {
                "best_same_data": item["best_same_data"],
                "leave_one_call_out": {
                    k: v for k, v in item["leave_one_call_out"].items()
                    if k not in {"heldout_calls", "errors"}
                },
            }
            for name, item in results.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
