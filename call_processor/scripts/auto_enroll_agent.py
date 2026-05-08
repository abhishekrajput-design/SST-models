#!/usr/bin/env python
"""Safe automation wrapper for pure agent voiceprint enrollment.

This script runs the same gated flow for Zak, Hussein, or a custom agent:

1. Leave-one-call-out validation on labelled agent/customer data.
2. Pure CAM++ voiceprint training from agent-only segments.
3. Optional activation only when validation gates pass.

It does not upload audio anywhere and it does not train on customer-labelled
speech. Customer rows are used only for rejection/calibration checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
VP_DIR = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
CLEAN_CLIP_DIR = CALL_PROCESSOR_DIR / "data" / "agent_clean_clips"


@dataclass(frozen=True)
class AgentPreset:
    slug: str
    name: str
    audio_root: Path
    label_source: str
    offset_mode: str
    clusters: int
    train_guard_s: float
    opposite_gap_s: float
    min_train_dur: float
    max_train_dur: float
    min_loco_accuracy: float = 95.0
    min_activation_accuracy: float = 96.0


PRESETS: dict[str, AgentPreset] = {
    "zak": AgentPreset(
        slug="zak_raissi_barnet",
        name="Zak Raissi Barnet",
        audio_root=REPO_ROOT / "traning_data" / "zak_raissi",
        label_source="training-json",
        offset_mode="detected",
        clusters=5,
        train_guard_s=0.35,
        opposite_gap_s=0.05,
        min_train_dur=1.5,
        max_train_dur=18.0,
    ),
    "hussein": AgentPreset(
        slug="hussein_mohamed",
        name="Hussein Mohamed",
        audio_root=REPO_ROOT / "traning_data" / "hussein_mohamed_pure_candidate_20260508",
        label_source="folder-data",
        offset_mode="none",
        clusters=4,
        train_guard_s=0.45,
        opposite_gap_s=0.12,
        min_train_dur=1.5,
        max_train_dur=12.0,
    ),
}


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_calls(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out


def _pct_ok(metrics: dict[str, Any], floor: float) -> bool:
    return (
        float(metrics.get("overall_accuracy") or 0.0) >= floor
        and float(metrics.get("agent_accuracy") or 0.0) >= floor
        and float(metrics.get("customer_accuracy") or 0.0) >= floor
    )


def _run(cmd: list[str], *, no_run: bool) -> int:
    print("[auto] " + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    if no_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    return int(proc.returncode)


def _append_repeated(cmd: list[str], flag: str, values: list[str]) -> None:
    for value in values:
        cmd.extend([flag, value])


def _build_common_args(args: argparse.Namespace, preset: AgentPreset) -> dict[str, Any]:
    include_calls = _parse_calls(args.include_call)
    exclude_calls = _parse_calls(args.exclude_call)
    audio_root = Path(args.audio_root).resolve() if args.audio_root else preset.audio_root.resolve()
    agent_slug = args.agent_slug or preset.slug
    agent_name = args.agent_name or preset.name
    label_source = args.label_source or preset.label_source
    offset_mode = args.offset_mode or preset.offset_mode
    clusters = args.clusters if args.clusters is not None else preset.clusters
    train_guard_s = args.train_guard_s if args.train_guard_s is not None else preset.train_guard_s
    opposite_gap_s = args.opposite_gap_s if args.opposite_gap_s is not None else preset.opposite_gap_s
    min_train_dur = args.min_train_dur if args.min_train_dur is not None else preset.min_train_dur
    max_train_dur = args.max_train_dur if args.max_train_dur is not None else preset.max_train_dur
    min_loco_accuracy = args.min_loco_accuracy if args.min_loco_accuracy is not None else preset.min_loco_accuracy
    min_activation_accuracy = (
        args.min_activation_accuracy
        if args.min_activation_accuracy is not None
        else preset.min_activation_accuracy
    )
    return {
        "agent_slug": agent_slug,
        "agent_name": agent_name,
        "audio_root": audio_root,
        "label_source": label_source,
        "offset_mode": offset_mode,
        "clusters": clusters,
        "train_guard_s": train_guard_s,
        "opposite_gap_s": opposite_gap_s,
        "min_train_dur": min_train_dur,
        "max_train_dur": max_train_dur,
        "min_loco_accuracy": min_loco_accuracy,
        "min_activation_accuracy": min_activation_accuracy,
        "include_calls": include_calls,
        "exclude_calls": exclude_calls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="hussein")
    parser.add_argument("--agent-slug", default="")
    parser.add_argument("--agent-name", default="")
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--labels-dir", default=str(CALL_PROCESSOR_DIR / "data" / "training"))
    parser.add_argument("--label-source", choices=("training-json", "folder-data", "both"), default="")
    parser.add_argument("--offset-mode", choices=("detected", "none", "auto-source"), default="")
    parser.add_argument("--include-source-prefix", default="")
    parser.add_argument("--include-call", action="append", default=[])
    parser.add_argument("--exclude-call", action="append", default=[])
    parser.add_argument("--clean-dir", action="append", default=[])
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--train-guard-s", type=float, default=None)
    parser.add_argument("--opposite-gap-s", type=float, default=None)
    parser.add_argument("--min-eval-dur", type=float, default=0.8)
    parser.add_argument("--min-train-dur", type=float, default=None)
    parser.add_argument("--max-train-dur", type=float, default=None)
    parser.add_argument("--agent-filter", choices=("all", "cue"), default="all")
    parser.add_argument("--min-loco-accuracy", type=float, default=None)
    parser.add_argument("--min-activation-accuracy", type=float, default=None)
    parser.add_argument("--threshold-margin", type=float, default=0.06)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--skip-loco", action="store_true")
    parser.add_argument("--train-on-loco-fail", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="Print commands and write a plan report only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    cfg = _build_common_args(args, preset)

    VP_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_CLIP_DIR.mkdir(parents=True, exist_ok=True)

    tag = _now_tag()
    loco_report = VP_DIR / f"{cfg['agent_slug']}_auto_loco_{tag}.json"
    train_report_name = f"{cfg['agent_slug']}_auto_training_{tag}.json"
    automation_report = VP_DIR / f"{cfg['agent_slug']}_auto_enrollment_{tag}.json"
    clean_export = CLEAN_CLIP_DIR / f"{cfg['agent_slug']}_auto_{tag}"

    report: dict[str, Any] = {
        "agent_slug": cfg["agent_slug"],
        "agent_name": cfg["agent_name"],
        "audio_root": str(cfg["audio_root"]),
        "label_source": cfg["label_source"],
        "offset_mode": cfg["offset_mode"],
        "activate_requested": bool(args.activate),
        "min_loco_accuracy": cfg["min_loco_accuracy"],
        "min_activation_accuracy": cfg["min_activation_accuracy"],
        "loco_report": str(loco_report),
        "training_report": str(VP_DIR / train_report_name),
        "clean_clip_export": str(clean_export),
        "commands": {},
        "loco_passed": False,
        "training_started": False,
        "activation_requested_to_trainer": False,
    }

    if not cfg["audio_root"].exists():
        report["error"] = f"audio root not found: {cfg['audio_root']}"
        automation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[auto] error: {report['error']}", file=sys.stderr)
        return 2

    common_flags = [
        "--agent-slug", cfg["agent_slug"],
        "--audio-root", str(cfg["audio_root"]),
        "--labels-dir", str(Path(args.labels_dir).resolve()),
        "--label-source", cfg["label_source"],
        "--offset-mode", cfg["offset_mode"],
        "--clusters", str(cfg["clusters"]),
        "--min-eval-dur", str(args.min_eval_dur),
        "--min-train-dur", str(cfg["min_train_dur"]),
        "--max-train-dur", str(cfg["max_train_dur"]),
        "--train-guard-s", str(cfg["train_guard_s"]),
        "--opposite-gap-s", str(cfg["opposite_gap_s"]),
        "--agent-filter", args.agent_filter,
    ]
    if args.include_source_prefix:
        common_flags.extend(["--include-source-prefix", args.include_source_prefix])
    _append_repeated(common_flags, "--include-call", cfg["include_calls"])
    _append_repeated(common_flags, "--exclude-call", cfg["exclude_calls"])
    _append_repeated(common_flags, "--clean-dir", [str(Path(p).resolve()) for p in args.clean_dir])

    loco_passed = bool(args.skip_loco)
    if args.skip_loco:
        report["loco_skipped"] = True
    else:
        loco_cmd = [
            sys.executable,
            str(CALL_PROCESSOR_DIR / "scripts" / "evaluate_agent_loco.py"),
            *common_flags,
            "--threshold-margin", str(args.threshold_margin),
            "--min-agent-accuracy", str(cfg["min_loco_accuracy"]),
            "--min-customer-accuracy", str(cfg["min_loco_accuracy"]),
            "--out", str(loco_report),
        ]
        report["commands"]["loco"] = loco_cmd
        code = _run(loco_cmd, no_run=args.no_run)
        if code != 0:
            report["error"] = f"leave-one-call-out failed with exit code {code}"
            automation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return code
        if args.no_run:
            report["loco_passed"] = None
            report["plan_only_assumes_loco_pass_for_training_command"] = True
            loco_passed = True
        else:
            loco_data = json.loads(loco_report.read_text(encoding="utf-8"))
            calibrated = loco_data.get("calibrated_leave_one_call_out") or {}
            tuned = loco_data.get("train_tuned_leave_one_call_out") or {}
            report["loco_calibrated"] = calibrated
            report["loco_train_tuned"] = tuned
            loco_passed = _pct_ok(calibrated, cfg["min_loco_accuracy"])
            report["loco_passed"] = bool(loco_passed)

    train_allowed = bool(loco_passed or args.train_on_loco_fail)
    report["training_allowed"] = train_allowed
    if not train_allowed:
        report["decision"] = "blocked: leave-one-call-out accuracy did not meet gate"
        automation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[auto] blocked: LOCO did not meet {cfg['min_loco_accuracy']:.1f}% gate")
        print(f"[auto] report -> {automation_report}")
        return 2

    train_cmd = [
        sys.executable,
        str(CALL_PROCESSOR_DIR / "scripts" / "train_zak_pure_embeddings.py"),
        *common_flags,
        "--agent-name", cfg["agent_name"],
        "--report-name", train_report_name,
        "--export-clean-clips", str(clean_export),
        "--min-activation-accuracy", str(cfg["min_activation_accuracy"]),
    ]
    trainer_activation = bool(args.activate and loco_passed)
    if trainer_activation:
        train_cmd.append("--activate")
    report["commands"]["train"] = train_cmd
    report["activation_requested_to_trainer"] = trainer_activation
    report["training_started"] = True
    code = _run(train_cmd, no_run=args.no_run)
    if code != 0:
        report["error"] = f"training failed with exit code {code}"
        automation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return code

    if not args.no_run:
        train_data = json.loads((VP_DIR / train_report_name).read_text(encoding="utf-8"))
        report["training_same_data_accuracy"] = train_data.get("same_data_accuracy")
        report["training_best_same_data_threshold"] = train_data.get("best_same_data_threshold")
        report["activated"] = bool(train_data.get("activated"))
        report["activation_eligible"] = bool(train_data.get("activation_eligible"))
        report["decision"] = "activated" if report["activated"] else "trained candidate only"
    else:
        report["decision"] = "plan only"
    automation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[auto] report -> {automation_report}")
    if args.no_run:
        print("[auto] plan only; no validation/training was run")
    elif args.activate:
        print(f"[auto] activation={'yes' if report.get('activated') else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
