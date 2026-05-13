"""End-to-end identification test on a fresh desk recording from Audiofy.

Pulls a recent desk recording for a chosen agent (outside the recent training
window), uploads it to the local UI pipeline, then compares the produced
role assignments against Audiofy's ground-truth speaker_json. Reports
agent / customer / overall accuracy.

Usage:
  python verify_identification_e2e.py "Hussein Mohamed"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "call_processor" / "scripts"
sys.path.insert(0, str(SCRIPTS))

env_path = REPO / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import daily_training_daemon as dtd  # noqa: E402

UI = "http://localhost:8080"
UI_AUTH = (
    (os.environ.get("CALLPROC_USER", ""), os.environ.get("CALLPROC_PASS", ""))
    if os.environ.get("CALLPROC_AUTH_REQUIRED", "").strip().lower() in {"1", "true", "yes", "on"}
    else None
)


def _recording_quality(rec: dict, agent: str) -> tuple[float, dict]:
    sj = rec.get("speaker_json") or []
    duration = float(rec.get("duration") or 0.0)
    agent_rows = [
        s for s in sj
        if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), agent) == "agent"
    ]
    customer_rows = [
        s for s in sj
        if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), agent) == "customer"
    ]
    agent_scores = [float(s.get("avg_score") or 0.0) for s in agent_rows]
    customer_scores = [float(s.get("avg_score") or 0.0) for s in customer_rows]
    mean_agent_score = sum(agent_scores) / len(agent_scores) if agent_scores else 0.0
    mean_customer_score = sum(customer_scores) / len(customer_scores) if customer_scores else 0.0
    balance = min(len(agent_rows), len(customer_rows)) / max(max(len(agent_rows), len(customer_rows)), 1)
    duration_score = 1.0 - min(abs(duration - 420.0) / 420.0, 1.0)
    quality = (
        len(agent_rows) * 0.8
        + len(customer_rows) * 0.8
        + mean_agent_score * 20.0
        + mean_customer_score * 10.0
        + balance * 20.0
        + duration_score * 10.0
    )
    details = {
        "duration": round(duration, 2),
        "agent_segments": len(agent_rows),
        "customer_segments": len(customer_rows),
        "mean_agent_score": round(mean_agent_score, 3),
        "mean_customer_score": round(mean_customer_score, 3),
        "balance": round(balance, 3),
        "quality": round(quality, 2),
    }
    return quality, details


def fetch_one_outside_recent(agent: str, days_old_min: int = 30, days_old_max: int = 60, max_calls: int = 150) -> tuple[dict, dict]:
    """Find a desk recording older than days_old_min but within days_old_max
    so it is not part of our recent training scrapes."""
    end = datetime.now(timezone.utc) - timedelta(days=days_old_min)
    start = end - timedelta(days=days_old_max - days_old_min)
    recs = dtd.fetch_api_recordings(
        days=days_old_max,
        max_calls=max_calls,
        start_time=start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        end_time=end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        user_name=agent,
    )
    if not recs:
        raise SystemExit(f"no {agent} recordings in {days_old_min}-{days_old_max}d window")
    candidates = []
    for rec in recs:
        url = rec.get("horizon_call_s3_url")
        sj = rec.get("speaker_json") or []
        duration = float(rec.get("duration") or 0.0)
        if not url or not sj or duration < 90 or duration > 900:
            continue
        ok, reason = dtd.call_quality_check(rec, agent)
        if not ok:
            continue
        score, details = _recording_quality(rec, agent)
        candidates.append((score, rec, details))
    if not candidates:
        raise SystemExit(f"no quality {agent} recordings in {days_old_min}-{days_old_max}d window")
    _, rec, details = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return rec, details


def overlap(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))


def score_against_ground_truth(result: dict, gt_segments: list, agent_name: str) -> dict:
    """For each ground-truth segment, find the produced segment with most overlap
    and check whether the predicted role matches the ground-truth role."""
    produced = result.get("segments") or result.get("transcription_json") or []
    if not produced:
        return {"error": "no produced segments"}

    agent_correct = agent_total = 0
    cust_correct = cust_total = 0
    mismatches = []

    for gt in gt_segments:
        gt_role = dtd.speaker_role_from_api_label(str(gt.get("speaker") or ""), agent_name)
        if gt_role not in ("agent", "customer"):
            continue
        # Audiofy returns timestamps as 'HH:MM:SS.ms' strings — convert via daemon helper.
        gs = float(dtd.ts2s(gt.get("start") or 0))
        ge = float(dtd.ts2s(gt.get("end") or 0))
        if ge <= gs:
            continue

        best = None
        best_ov = 0.0
        for p in produced:
            ov = overlap(gs, ge, float(p.get("start") or 0), float(p.get("end") or 0))
            if ov > best_ov:
                best_ov = ov
                best = p
        if best is None or best_ov <= 0:
            continue

        pred_role_raw = str(best.get("identified_speaker") or best.get("speaker") or "").lower()
        if "agent" in pred_role_raw:
            pred = "agent"
        elif "customer" in pred_role_raw:
            pred = "customer"
        else:
            pred = "unknown"

        if gt_role == "agent":
            agent_total += 1
            agent_correct += int(pred == "agent")
            if pred != "agent" and len(mismatches) < 10:
                mismatches.append({
                    "gt": "agent", "pred": pred,
                    "time": f"{gs:.1f}-{ge:.1f}",
                    "text_gt": (gt.get("phrase") or gt.get("text") or "")[:80],
                    "text_pred": (best.get("text") or "")[:80],
                })
        else:
            cust_total += 1
            cust_correct += int(pred == "customer")
            if pred != "customer" and len(mismatches) < 10:
                mismatches.append({
                    "gt": "customer", "pred": pred,
                    "time": f"{gs:.1f}-{ge:.1f}",
                    "text_gt": (gt.get("phrase") or gt.get("text") or "")[:80],
                    "text_pred": (best.get("text") or "")[:80],
                })

    total = agent_total + cust_total
    correct = agent_correct + cust_correct
    return {
        "agent_total": agent_total,
        "agent_correct": agent_correct,
        "agent_acc": round(agent_correct / agent_total * 100, 2) if agent_total else 0.0,
        "customer_total": cust_total,
        "customer_correct": cust_correct,
        "customer_acc": round(cust_correct / cust_total * 100, 2) if cust_total else 0.0,
        "overall_total": total,
        "overall_correct": correct,
        "overall_acc": round(correct / total * 100, 2) if total else 0.0,
        "mismatch_samples": mismatches,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("agent", help='e.g. "Hussein Mohamed"')
    p.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    p.add_argument("--days-old-min", type=int, default=30)
    p.add_argument("--days-old-max", type=int, default=60)
    p.add_argument("--max-calls", type=int, default=150)
    args = p.parse_args()

    pick, quality = fetch_one_outside_recent(args.agent, args.days_old_min, args.days_old_max, args.max_calls)
    call_id = str(pick.get("_id") or "")[:12]
    sj = pick.get("speaker_json") or []
    duration = pick.get("duration", "?")
    print(f"[pick] {args.agent}  call_id={call_id}  duration={duration}s  sj_segments={len(sj)}")
    print(f"[pick-quality] {quality}")

    agents = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), args.agent) == "agent")
    customers = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), args.agent) == "customer")
    print(f"[gt] agent={agents}  customer={customers}")

    slug = args.agent.replace(" ", "_").lower()
    fname = f"identtest_{slug}_{call_id}.mp3"
    local = REPO / "call_processor" / "data" / "raw_calls" / fname
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] -> {local.name}")
    with requests.get(pick["horizon_call_s3_url"], stream=True, timeout=300) as g:
        g.raise_for_status()
        local.write_bytes(g.content)
    print(f"[download] {local.stat().st_size:,} bytes")

    (local.with_suffix(".gt.json")).write_text(
        json.dumps({"agent_name": args.agent, "call_id": call_id, "segments": sj}, indent=2),
        encoding="utf-8",
    )

    print(f"[upload] POST /api/upload?filename={fname}&model={args.model}")
    # IMPORTANT: read into memory so `requests` sends one Content-Length body.
    # Passing the file handle directly triggers chunked transfer encoding which
    # the basic http.server in ui.py doesn't decode — it reads Content-Length
    # bytes (which is missing/0 in chunked mode) and silently truncates.
    body = local.read_bytes()
    r = requests.post(
        f"{UI}/api/upload",
        params={"filename": fname, "model": args.model, "agent_slug": slug},
        data=body,
        auth=UI_AUTH,
        timeout=60,
        headers={"Content-Type": "audio/mpeg"},
    )
    if r.status_code != 200:
        print(f"[upload] HTTP {r.status_code}: {r.text[:300]}")
        sys.exit(1)
    print(f"[upload] {r.json()}")

    print("[poll] pipeline status...")
    started = time.time()
    last_stage = ""
    while True:
        s = requests.get(f"{UI}/api/status", auth=UI_AUTH, timeout=10).json()
        stage = f"{s.get('stage_num')}.{s.get('stage')}"
        if stage != last_stage:
            elapsed = round(time.time() - started, 1)
            print(f"  [+{elapsed}s] {stage}: {s.get('message','')[:80]}")
            last_stage = stage
        if s.get("done"):
            print(f"  done in {round(time.time()-started,1)}s -> result_id={s.get('result_id')}")
            result_id = s.get("result_id")
            break
        if s.get("error"):
            print(f"  ERROR: {s.get('error')}")
            sys.exit(1)
        time.sleep(3)

    r = requests.get(f"{UI}/api/call/{result_id}", auth=UI_AUTH, timeout=30)
    if r.status_code != 200:
        print(f"[result] HTTP {r.status_code}: {r.text[:300]}")
        sys.exit(1)
    result = r.json()
    print(f"[result] identified_agent={result.get('identified_agent')!r}  total_segments={result.get('total_segments')}")

    score = score_against_ground_truth(result, sj, args.agent)
    print()
    print("=" * 70)
    print("IDENTIFICATION ACCURACY  (predicted vs Audiofy ground-truth)")
    print("=" * 70)
    print(f"  agent:      {score['agent_correct']}/{score['agent_total']}  =  {score['agent_acc']}%")
    print(f"  customer:   {score['customer_correct']}/{score['customer_total']}  =  {score['customer_acc']}%")
    print(f"  overall:    {score['overall_correct']}/{score['overall_total']}  =  {score['overall_acc']}%")
    if score.get("mismatch_samples"):
        print()
        print(f"Sample mismatches (up to 10):")
        for m in score["mismatch_samples"]:
            print(f"  [{m['time']}]  gt={m['gt']}  pred={m['pred']}")
            print(f"    gt_text:   {m['text_gt']}")
            print(f"    pred_text: {m['text_pred']}")


if __name__ == "__main__":
    main()
