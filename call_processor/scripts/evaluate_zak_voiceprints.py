#!/usr/bin/env python
"""Compare Zak voiceprint candidates against Gemini-labelled Zak calls."""
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

from src.voiceprints import resolve_voiceprint_path  # noqa: E402
from train_zak_pure_embeddings import (  # noqa: E402
    CALL_PROCESSOR_DIR as TRAIN_CALL_PROCESSOR_DIR,
    TARGET_SR,
    best_threshold,
    extract_label_rows,
    load_labelled_calls,
)
from src.embedding_campp import EmbeddingModel, l2_norm  # noqa: E402


GLOBAL_THRESHOLD = 0.34
PER_AGENT_MARGIN = 0.06
PER_AGENT_THRESH_CAP = 0.92


def load_voiceprint_stack(paths: list[str], base: Path) -> np.ndarray:
    vectors = []
    for raw in paths:
        path = Path(resolve_voiceprint_path(raw, str(base / "agents.json")))
        if not path.is_absolute():
            path = base / path
        vp = np.load(path).astype(np.float32).squeeze()
        if vp.ndim != 1:
            raise ValueError(f"Invalid voiceprint shape for {path}: {vp.shape}")
        vectors.append(l2_norm(vp))
    if not vectors:
        raise ValueError("No voiceprints loaded")
    return np.stack(vectors).astype(np.float32)


def best_sim(embedding: np.ndarray, stack: np.ndarray) -> float:
    emb = l2_norm(embedding)
    return float(np.max(stack @ emb))


def threshold_from_max_outside(max_outside: float | None) -> float:
    if max_outside is None:
        return GLOBAL_THRESHOLD
    return float(min(max(GLOBAL_THRESHOLD, float(max_outside) + PER_AGENT_MARGIN), PER_AGENT_THRESH_CAP))


def score_rows(rows, stack: np.ndarray, threshold: float) -> dict:
    total = correct = 0
    agent_total = agent_correct = 0
    customer_total = customer_correct = 0
    errors = []
    per_call: dict[str, dict[str, int]] = {}
    for row in rows:
        sim = best_sim(row.embedding, stack)
        pred = "agent" if sim >= threshold else "customer"
        ok = pred == row.speaker
        total += 1
        correct += int(ok)
        bucket = per_call.setdefault(
            row.call_name,
            {"total": 0, "correct": 0, "agent_total": 0, "agent_correct": 0, "customer_total": 0, "customer_correct": 0},
        )
        bucket["total"] += 1
        bucket["correct"] += int(ok)
        if row.speaker == "agent":
            agent_total += 1
            agent_correct += int(ok)
            bucket["agent_total"] += 1
            bucket["agent_correct"] += int(ok)
        else:
            customer_total += 1
            customer_correct += int(ok)
            bucket["customer_total"] += 1
            bucket["customer_correct"] += int(ok)
        if not ok and len(errors) < 25:
            errors.append({
                "call": row.call_name,
                "segment": row.segment_idx,
                "speaker": row.speaker,
                "predicted": pred,
                "similarity": round(sim, 4),
                "text": row.text[:120],
            })

    def pct(num: int, den: int) -> float:
        return round((num / den * 100.0) if den else 0.0, 2)

    per_call_out = {}
    for call, item in per_call.items():
        per_call_out[call] = {
            **item,
            "overall_accuracy": pct(item["correct"], item["total"]),
            "agent_accuracy": pct(item["agent_correct"], item["agent_total"]),
            "customer_accuracy": pct(item["customer_correct"], item["customer_total"]),
        }

    return {
        "threshold": round(float(threshold), 4),
        "segments": total,
        "overall_accuracy": pct(correct, total),
        "agent_accuracy": pct(agent_correct, agent_total),
        "customer_accuracy": pct(customer_correct, customer_total),
        "agent_correct": agent_correct,
        "agent_total": agent_total,
        "customer_correct": customer_correct,
        "customer_total": customer_total,
        "per_call": per_call_out,
        "errors": errors,
    }


def score_best_threshold(rows, stack: np.ndarray) -> dict:
    pseudo_centroids = [stack[i] for i in range(stack.shape[0])]
    return best_threshold(rows, pseudo_centroids)


def candidate_paths(info: dict) -> list[str]:
    raw = []
    for item in info.get("voiceprints") or []:
        if isinstance(item, dict) and item.get("path"):
            raw.append(str(item["path"]))
        elif isinstance(item, str):
            raw.append(item)
    if not raw and info.get("voiceprint_path"):
        raw.append(str(info["voiceprint_path"]))
    return raw


def glob_candidate_paths(base: Path, pattern: str) -> list[str]:
    return [p.name for p in sorted(base.glob(pattern))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default=str(CALL_PROCESSOR_DIR / "data" / "training"))
    parser.add_argument("--audio-root", default=str(REPO_ROOT / "traning_data" / "zak_raissi"))
    parser.add_argument(
        "--label-source",
        choices=("training-json", "folder-data", "both"),
        default="both",
        help="Load legacy gemini_labels files, per-call data.json files, or both.",
    )
    parser.add_argument("--out", default=str(CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / "zak_voiceprint_comparison.json"))
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir).resolve()
    audio_root = Path(args.audio_root).resolve()
    voiceprint_dir = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
    agents_path = voiceprint_dir / "agents.json"
    agents = json.loads(agents_path.read_text(encoding="utf-8"))

    calls, skipped_calls = load_labelled_calls(labels_dir, audio_root, label_source=args.label_source)
    model = EmbeddingModel()
    print("[model] loading CAM++")
    model.load(force_cpu=True)
    try:
        rows, skipped_segments = extract_label_rows(
            calls,
            model,
            min_eval_dur=0.8,
            min_train_dur=1.5,
            max_train_dur=18.0,
        )
    finally:
        model.unload()

    eval_rows = [row for row in rows if row.speaker in {"agent", "customer"}]

    pure_paths = glob_candidate_paths(voiceprint_dir, "zak_raissi_barnet_pure_campp_v*.npy")
    candidates = {
        "restored_phone_campp": {
            "agent_name": "Zak Raissi Barnet",
            "paths": ["zak_raissi_barnet.npy"],
            "max_outside_sim": 0.541,
            "source": "phone_calls_kmeans_campp_restored_for_random_calls",
        },
        "pure_gemini_campp": {
            "agent_name": "Zak Raissi Barnet",
            "paths": pure_paths or [f"zak_raissi_barnet_pure_campp_v{i}.npy" for i in range(1, 6)],
            "max_outside_sim": 0.87,
            "source": "pure_gemini_agent_segments_plus_clean_clips",
        },
    }
    if "zak_local_20260423" in agents:
        local_info = agents["zak_local_20260423"]
        candidates["local_clean_campp"] = {
            "agent_name": local_info.get("agent_name", "Zak Raissi"),
            "paths": candidate_paths(local_info),
            "max_outside_sim": local_info.get("max_outside_sim"),
            "source": local_info.get("source", "local_clean_clips"),
        }

    results = {}
    expected_dim = int(eval_rows[0].embedding.shape[0]) if eval_rows else 0
    for name, info in candidates.items():
        stack = load_voiceprint_stack(info["paths"], voiceprint_dir)
        if expected_dim and int(stack.shape[1]) != expected_dim:
            results[name] = {
                "agent_name": info["agent_name"],
                "source": info.get("source"),
                "paths": info["paths"],
                "n_voiceprints": int(stack.shape[0]),
                "skipped": True,
                "reason": f"embedding dimension {int(stack.shape[1])} != eval dimension {expected_dim}",
            }
            continue
        threshold = threshold_from_max_outside(info.get("max_outside_sim"))
        operating = score_rows(eval_rows, stack, threshold)
        best = score_best_threshold(eval_rows, stack)
        results[name] = {
            "agent_name": info["agent_name"],
            "source": info.get("source"),
            "paths": info["paths"],
            "n_voiceprints": int(stack.shape[0]),
            "threshold_mode": "max_outside_plus_margin",
            "max_outside_sim": info.get("max_outside_sim"),
            "operating": operating,
            "best_threshold": best,
        }

    winner = max(
        (name for name, item in results.items() if not item.get("skipped")),
        key=lambda k: (
            results[k]["operating"]["overall_accuracy"],
            results[k]["operating"]["customer_accuracy"],
            results[k]["operating"]["agent_accuracy"],
        ),
    )

    report = {
        "labels_dir": str(labels_dir),
        "audio_root": str(audio_root),
        "label_source": args.label_source,
        "calls": [
            {
                "call": call.call_name,
                "call_id": call.call_id,
                "audio": str(call.audio_path),
                "offset_s": call.offset_s,
                "segments": len(call.segments),
            }
            for call in calls
        ],
        "skipped_calls": skipped_calls,
        "skipped_segments": skipped_segments[:100],
        "eval_rows": len(eval_rows),
        "agent_rows": sum(1 for row in eval_rows if row.speaker == "agent"),
        "customer_rows": sum(1 for row in eval_rows if row.speaker == "customer"),
        "winner": winner,
        "winner_basis": "highest operating overall accuracy, then customer accuracy, then agent accuracy",
        "results": results,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "winner": winner,
        "eval_rows": report["eval_rows"],
        "results": {
            name: (
                {"skipped": True, "reason": item.get("reason")}
                if item.get("skipped")
                else {
                    "operating": item["operating"],
                    "best_threshold": item["best_threshold"],
                }
            )
            for name, item in results.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
