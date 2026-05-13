"""Download a fresh desk recording and time every stage of the pipeline.

Pulls one recording from Audiofy, uploads to /api/upload, then samples
/api/status at 1Hz capturing every stage transition so we can report a
per-stage breakdown plus the total wall clock.

Usage:
  python time_pipeline.py "Hussein Mohamed"
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

for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("AUDIOFY_USERNAME", "abhishek")
os.environ.setdefault("AUDIOFY_PASSWORD", "123456")

import daily_training_daemon as dtd  # noqa: E402

UI = "http://localhost:8080"
UI_AUTH = ("abhishek", "123456")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("agent", help='e.g. "Hussein Mohamed"')
    p.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    p.add_argument("--days-old-min", type=int, default=20)
    p.add_argument("--days-old-max", type=int, default=40)
    p.add_argument("--target-duration", type=int, default=240,
                   help="Prefer the recording whose duration is closest to this many seconds")
    args = p.parse_args()

    # 1. Find a fresh recording in the window
    end_dt = datetime.now(timezone.utc) - timedelta(days=args.days_old_min)
    start_dt = end_dt - timedelta(days=args.days_old_max - args.days_old_min)
    print(f"[scrape] {start_dt.date()} -> {end_dt.date()} for {args.agent}")
    recs = dtd.fetch_api_recordings(
        days=args.days_old_max, max_calls=40,
        start_time=start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        end_time=end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        user_name=args.agent,
    )
    if not recs:
        raise SystemExit("no recordings found")
    target_dur = args.target_duration
    recs.sort(key=lambda r: abs(float(r.get("duration") or 9999) - target_dur))
    pick = recs[0]
    cid = str(pick.get("_id") or "")[:12]
    audio_dur = float(pick.get("duration") or 0)
    sj_n = len(pick.get("speaker_json") or [])
    print(f"[pick] {cid}  duration={audio_dur:.0f}s  segments_in_gt={sj_n}")

    # 2. Download
    fname = f"timing_{args.agent.replace(' ', '_').lower()}_{cid}.mp3"
    local = REPO / "call_processor" / "data" / "raw_calls" / fname
    local.parent.mkdir(parents=True, exist_ok=True)
    t_download_start = time.time()
    with requests.get(pick["horizon_call_s3_url"], stream=True, timeout=180) as g:
        g.raise_for_status()
        local.write_bytes(g.content)
    t_download = time.time() - t_download_start
    size_mb = local.stat().st_size / (1024 * 1024)
    print(f"[download] {size_mb:.2f} MB in {t_download:.2f}s ({size_mb / t_download:.1f} MB/s)")

    # 3. Upload + start pipeline
    body = local.read_bytes()
    t_upload_start = time.time()
    r = requests.post(
        f"{UI}/api/upload",
        params={"filename": fname, "model": args.model},
        data=body, auth=UI_AUTH, timeout=60,
        headers={"Content-Type": "audio/mpeg"},
    )
    r.raise_for_status()
    t_upload = time.time() - t_upload_start
    print(f"[upload] {size_mb:.2f} MB in {t_upload:.2f}s — pipeline kicked")

    # 4. Sample status at 1Hz, log every stage transition
    t_pipeline_start = time.time()
    stage_log: list[tuple[float, str, str]] = []  # (elapsed, stage_num.stage, message)
    last_key = None
    while True:
        s = requests.get(f"{UI}/api/status", auth=UI_AUTH, timeout=10).json()
        key = f"{s.get('stage_num')}.{s.get('stage')}"
        msg = s.get("message", "")
        if key != last_key:
            stage_log.append((round(time.time() - t_pipeline_start, 2), key, msg))
            print(f"  [+{stage_log[-1][0]:>6.2f}s] {key}  {msg[:80]}")
            last_key = key
        if s.get("done"):
            t_total = time.time() - t_pipeline_start
            print(f"  [done] pipeline {t_total:.2f}s")
            result_id = s.get("result_id")
            break
        if s.get("error"):
            print(f"  ERROR: {s.get('error')}")
            sys.exit(1)
        time.sleep(1)

    # 5. Compute per-stage durations
    durations: list[tuple[str, float]] = []
    for i, (t, key, _msg) in enumerate(stage_log):
        nxt = stage_log[i + 1][0] if i + 1 < len(stage_log) else t_total
        durations.append((key, nxt - t))

    # 6. Pull the final result for sanity
    res = requests.get(f"{UI}/api/call/{result_id}", auth=UI_AUTH, timeout=30).json()
    print()
    print("=" * 70)
    print(f"PIPELINE TIMING — {args.agent}  ({audio_dur:.0f}s audio, {size_mb:.1f} MB)")
    print("=" * 70)
    print(f"{'Step':40s} {'Time (s)':>10s} {'% of total':>12s}")
    print("-" * 70)
    print(f"{'Download from Audiofy':40s} {t_download:>10.2f} {t_download/t_total*100:>11.1f}%")
    print(f"{'Upload to UI':40s} {t_upload:>10.2f} {t_upload/t_total*100:>11.1f}%")
    for key, d in durations:
        if d < 0.01:
            continue
        print(f"{'  ' + key[:38]:40s} {d:>10.2f} {d/t_total*100:>11.1f}%")
    print("-" * 70)
    print(f"{'TOTAL pipeline wall-clock':40s} {t_total:>10.2f}")
    print(f"{'TOTAL with download+upload':40s} {t_download+t_upload+t_total:>10.2f}")
    print()
    print(f"Real-time factor: {t_total/audio_dur:.2f}x  (1.0 = real-time, lower = faster)")
    print(f"Processing speed: {audio_dur/t_total:.2f}x real-time")
    print()
    print(f"Result: identified_agent={res.get('identified_agent')!r}  total_segments={res.get('total_segments')}")
    proc_time = res.get("processing_time_seconds")
    if proc_time:
        print(f"Server-recorded processing_time_seconds: {proc_time}")


if __name__ == "__main__":
    main()
