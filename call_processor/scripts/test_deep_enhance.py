"""Upload one audio file to the local UI and time + describe the result.

Used to A/B test desk-recording processing against the local UI. The script is
kept intentionally thin: it exercises the real /api/upload path and reports the
saved result, without changing production voiceprints.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import requests


REPO = Path(__file__).resolve().parents[2]
RAW_CALLS = REPO / "call_processor" / "data" / "raw_calls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local audio file to upload")
    parser.add_argument("--ui", default="http://localhost:8080")
    parser.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    parser.add_argument("--agent-slug", default="auto",
                        help="Target agent slug, e.g. zak_raissi_barnet or hussein_mohamed")
    parser.add_argument("--target-name",
                        help="Filename to send to /api/upload; defaults to source name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        print(f"[error] source not found: {source}", file=sys.stderr)
        return 2

    target_name = args.target_name or source.name
    RAW_CALLS.mkdir(parents=True, exist_ok=True)
    dest = RAW_CALLS / target_name
    if source != dest.resolve():
        shutil.copy2(source, dest)

    body = dest.read_bytes()
    mb = len(body) / 1e6
    print(f"=== {target_name}  {mb:.2f} MB  agent_slug={args.agent_slug} ===")
    t_up = time.time()
    r = requests.post(
        f"{args.ui}/api/upload",
        params={
            "filename": target_name,
            "model": args.model,
            "agent_slug": args.agent_slug,
        },
        data=body,
        timeout=60,
        headers={"Content-Type": "audio/mpeg"},
    )
    r.raise_for_status()
    print(f"[upload] {round(time.time()-t_up, 1)}s  {r.json()}")

    t0 = time.time()
    last = ""
    while True:
        s = requests.get(f"{args.ui}/api/status", timeout=10).json()
        key = f"{s.get('stage_num')}.{s.get('stage')}"
        if key != last:
            print(f"  [+{round(time.time()-t0,1):>6.1f}s] {key:<22s} {s.get('message','')[:80]}")
            last = key
        if s.get("done"):
            rid = s.get("result_id")
            print(f"  [done] {round(time.time()-t0,1)}s   result_id={rid}")
            result = requests.get(f"{args.ui}/api/call/{rid}", timeout=30).json()
            break
        if s.get("error"):
            print(f"  pipeline error: {s.get('error')}", file=sys.stderr)
            return 1
        time.sleep(2)

    print()
    print(f"identified_agent: {result.get('identified_agent')!r}")
    print(f"total_segments:   {result.get('total_segments')}")
    print(f"transcriber:      {result.get('transcriber_device')} / {result.get('model')}")
    print(f"target_agent_slug:{result.get('target_agent_slug')!r}")

    segs = result.get("segments") or []
    if segs:
        durs = [(float(seg.get("end") or 0) - float(seg.get("start") or 0)) for seg in segs]
        total_speech = sum(durs)
        first_start = float(segs[0].get("start") or 0)
        last_end = float(segs[-1].get("end") or 0)
        spans = last_end - first_start
        print()
        print(f"first segment start: {first_start:.2f}s")
        print(f"last segment end:    {last_end:.2f}s")
        print(f"total speech time:   {total_speech:.2f}s  ({total_speech/max(spans,1)*100:.1f}% of {spans:.0f}s span)")
        by = {}
        for seg in segs:
            role = seg.get("identified_speaker") or seg.get("speaker") or "?"
            by[role] = by.get(role, 0) + 1
        print(f"speaker distribution: {by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
