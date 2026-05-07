#!/usr/bin/env python
"""Leave-one-call-out verification for labelled agent voiceprints.

This script is intentionally generic so it can check Zak, Hussein, or any
other agent folder without mixing training sources across agents.
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

from src.embedding_campp import EmbeddingModel  # noqa: E402
from train_zak_pure_embeddings import (  # noqa: E402
    build_centroids,
    best_sim,
    extract_clean_clip_rows,
    extract_label_rows,
    load_labelled_calls,
)


def pct(num: int, den: int) -> float:
    return round((num / den * 100.0) if den else 0.0, 2)


def score_with_threshold(rows, centroids, threshold: float) -> dict:
    total = correct = 0
    agent_total = agent_correct = 0
    customer_total = customer_correct = 0
    errors = []
    for row in rows:
        sim = best_sim(row.embedding, centroids)
        predicted = "agent" if sim >= threshold else "customer"
        ok = predicted == row.speaker
        total += 1
        correct += int(ok)
        if row.speaker == "agent":
            agent_total += 1
            agent_correct += int(ok)
        elif row.speaker == "customer":
            customer_total += 1
            customer_correct += int(ok)
        if not ok and len(errors) < 40:
            errors.append({
                "call": row.call_name,
                "segment": row.segment_idx,
                "truth": row.speaker,
                "predicted": predicted,
                "similarity": round(float(sim), 4),
                "text": row.text[:160],
            })
    return {
        "threshold": round(float(threshold), 4),
        "segments": total,
        "overall_accuracy": pct(correct, total),
        "agent_accuracy": pct(agent_correct, agent_total),
        "customer_accuracy": pct(customer_correct, customer_total),
        "correct": correct,
        "total": total,
        "agent_correct": agent_correct,
        "agent_total": agent_total,
        "customer_correct": customer_correct,
        "customer_total": customer_total,
        "errors": errors,
    }


def choose_best_train_threshold(rows, centroids, min_agent: float, min_customer: float) -> dict:
    best = None
    fallback = None
    for threshold in np.arange(0.10, 0.951, 0.005):
        scored = score_with_threshold(rows, centroids, float(threshold))
        key = (
            scored["overall_accuracy"],
            min(scored["agent_accuracy"], scored["customer_accuracy"]),
            scored["customer_accuracy"],
            scored["agent_accuracy"],
        )
        if fallback is None or key > fallback["_key"]:
            scored["_key"] = key
            fallback = scored
        if scored["agent_accuracy"] < min_agent or scored["customer_accuracy"] < min_customer:
            continue
        if best is None or key > best["_key"]:
            scored["_key"] = key
            best = scored
    chosen = best or fallback
    chosen = dict(chosen or {})
    chosen.pop("_key", None)
    chosen["met_role_floor"] = best is not None
    return chosen


def aggregate(folds: list[dict], key: str) -> dict:
    total = correct = 0
    agent_total = agent_correct = 0
    customer_total = customer_correct = 0
    errors = []
    per_call = {}
    for fold in folds:
        item = fold[key]
        total += item["total"]
        correct += item["correct"]
        agent_total += item["agent_total"]
        agent_correct += item["agent_correct"]
        customer_total += item["customer_total"]
        customer_correct += item["customer_correct"]
        per_call[fold["heldout_call"]] = {
            k: item[k]
            for k in (
                "threshold",
                "segments",
                "overall_accuracy",
                "agent_accuracy",
                "customer_accuracy",
                "correct",
                "total",
                "agent_correct",
                "agent_total",
                "customer_correct",
                "customer_total",
            )
        }
        errors.extend(item.get("errors", [])[:8])
    return {
        "overall_accuracy": pct(correct, total),
        "agent_accuracy": pct(agent_correct, agent_total),
        "customer_accuracy": pct(customer_correct, customer_total),
        "correct": correct,
        "total": total,
        "agent_correct": agent_correct,
        "agent_total": agent_total,
        "customer_correct": customer_correct,
        "customer_total": customer_total,
        "per_call": per_call,
        "errors": errors[:60],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-slug", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--labels-dir", default=str(CALL_PROCESSOR_DIR / "data" / "training"))
    parser.add_argument("--label-source", choices=("training-json", "folder-data", "both"), default="folder-data")
    parser.add_argument("--offset-mode", choices=("detected", "none", "auto-source"), default="detected")
    parser.add_argument("--include-source-prefix", default="")
    parser.add_argument("--include-call", action="append", default=[])
    parser.add_argument("--exclude-call", action="append", default=[])
    parser.add_argument("--clean-dir", action="append", default=[])
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--min-eval-dur", type=float, default=0.8)
    parser.add_argument("--min-train-dur", type=float, default=1.5)
    parser.add_argument("--max-train-dur", type=float, default=45.0)
    parser.add_argument("--train-guard-s", type=float, default=0.35)
    parser.add_argument("--opposite-gap-s", type=float, default=0.05)
    parser.add_argument("--agent-filter", choices=("all", "cue"), default="all")
    parser.add_argument("--threshold-margin", type=float, default=0.06)
    parser.add_argument("--min-agent-accuracy", type=float, default=50.0)
    parser.add_argument("--min-customer-accuracy", type=float, default=50.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    audio_root = Path(args.audio_root).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    clean_dirs = [Path(p).resolve() for p in args.clean_dir]

    calls, skipped_calls = load_labelled_calls(
        labels_dir,
        audio_root,
        offset_mode=args.offset_mode,
        label_source=args.label_source,
        include_source_prefix=args.include_source_prefix,
        include_call_names=set(args.include_call),
        exclude_call_names=set(args.exclude_call),
    )
    print(f"[data] calls={len(calls)} skipped={len(skipped_calls)}")

    model = EmbeddingModel()
    print("[model] loading CAM++")
    model.load(force_cpu=True)
    try:
        rows, skipped_segments = extract_label_rows(
            calls,
            model,
            min_eval_dur=args.min_eval_dur,
            min_train_dur=args.min_train_dur,
            max_train_dur=args.max_train_dur,
            agent_filter=args.agent_filter,
            train_guard_s=args.train_guard_s,
            opposite_gap_s=args.opposite_gap_s,
        )
        clean_rows = extract_clean_clip_rows(clean_dirs, model)
    finally:
        model.unload()

    eval_rows = [row for row in rows if row.speaker in {"agent", "customer"}]
    call_names = sorted({row.call_name for row in eval_rows})
    folds = []
    for heldout in call_names:
        train_rows = [row for row in rows if row.call_name != heldout and row.used_for_training]
        if clean_rows:
            train_rows = clean_rows + train_rows
        train_eval = [
            row for row in eval_rows
            if row.call_name != heldout and row.speaker in {"agent", "customer"}
        ]
        test_rows = [row for row in eval_rows if row.call_name == heldout]
        if not train_rows or not train_eval or not test_rows:
            folds.append({
                "heldout_call": heldout,
                "skipped": True,
                "reason": "missing train/eval/test rows",
            })
            continue
        centroids = build_centroids([row.embedding for row in train_rows], n_clusters=args.clusters)
        customer_train = [row for row in train_eval if row.speaker == "customer"]
        customer_sims = [best_sim(row.embedding, centroids) for row in customer_train]
        customer_p95 = float(np.percentile(customer_sims, 95)) if customer_sims else 0.0
        calibrated_threshold = min(max(customer_p95 + args.threshold_margin, 0.34), 0.92)
        calibrated = score_with_threshold(test_rows, centroids, calibrated_threshold)
        best_train = choose_best_train_threshold(
            train_eval,
            centroids,
            min_agent=args.min_agent_accuracy,
            min_customer=args.min_customer_accuracy,
        )
        tuned = score_with_threshold(test_rows, centroids, best_train["threshold"])
        folds.append({
            "heldout_call": heldout,
            "train_segments": len(train_rows),
            "train_eval_segments": len(train_eval),
            "test_segments": len(test_rows),
            "customer_p95": round(customer_p95, 4),
            "calibrated_threshold": round(float(calibrated_threshold), 4),
            "best_train_threshold": best_train,
            "calibrated": calibrated,
            "train_tuned": tuned,
        })

    scored_folds = [fold for fold in folds if not fold.get("skipped")]
    report = {
        "agent_slug": args.agent_slug,
        "audio_root": str(audio_root),
        "label_source": args.label_source,
        "offset_mode": args.offset_mode,
        "include_source_prefix": args.include_source_prefix,
        "include_calls": args.include_call,
        "exclude_calls": args.exclude_call,
        "calls": len(calls),
        "eval_rows": len(eval_rows),
        "clean_rows": len(clean_rows),
        "skipped_calls": skipped_calls,
        "skipped_segments": skipped_segments[:100],
        "folds": folds,
        "calibrated_leave_one_call_out": aggregate(scored_folds, "calibrated"),
        "train_tuned_leave_one_call_out": aggregate(scored_folds, "train_tuned"),
    }

    out = Path(args.out).resolve() if args.out else (
        CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / f"{args.agent_slug}_leave_one_call_out.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "agent_slug": args.agent_slug,
        "calls": report["calls"],
        "eval_rows": report["eval_rows"],
        "clean_rows": report["clean_rows"],
        "calibrated_leave_one_call_out": report["calibrated_leave_one_call_out"],
        "train_tuned_leave_one_call_out": report["train_tuned_leave_one_call_out"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
