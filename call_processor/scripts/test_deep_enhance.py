"""Upload one audio file to the local UI and time + describe the result.

Used to A/B test the SST_DEEP_ENHANCE filter chain against the default one.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import requests

UI = "http://localhost:8080"
SOURCE = Path(r"C:\Users\abhis\Downloads\audio_05_01_2026_10_33_30_c2k0vk.mp3")
TARGET_NAME = "deep_test_c2k0vk.mp3"

raw_calls = Path(r"C:\Users\abhis\Desktop\SST-models\call_processor\data\raw_calls")
raw_calls.mkdir(parents=True, exist_ok=True)
dest = raw_calls / TARGET_NAME
shutil.copy2(SOURCE, dest)

body = dest.read_bytes()
mb = len(body) / 1e6
print(f"=== {TARGET_NAME}  {mb:.2f} MB ===")
t_up = time.time()
r = requests.post(
    f"{UI}/api/upload",
    params={"filename": TARGET_NAME, "model": "parakeet-tdt-0.6b-v3"},
    data=body, timeout=60,
    headers={"Content-Type": "audio/mpeg"},
)
r.raise_for_status()
print(f"[upload] {round(time.time()-t_up, 1)}s  {r.json()}")

t0 = time.time()
last = ""
while True:
    s = requests.get(f"{UI}/api/status", timeout=10).json()
    key = f"{s.get('stage_num')}.{s.get('stage')}"
    if key != last:
        print(f"  [+{round(time.time()-t0,1):>6.1f}s] {key:<22s} {s.get('message','')[:80]}")
        last = key
    if s.get("done"):
        rid = s.get("result_id")
        print(f"  [done] {round(time.time()-t0,1)}s   result_id={rid}")
        result = requests.get(f"{UI}/api/call/{rid}", timeout=30).json()
        break
    if s.get("error"):
        sys.exit(f"  pipeline error: {s.get('error')}")
    time.sleep(2)

print()
print(f"identified_agent: {result.get('identified_agent')!r}")
print(f"total_segments:   {result.get('total_segments')}")
print(f"transcriber:      {result.get('transcriber_device')} / {result.get('model')}")

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
    # Speaker tallies
    by = {}
    for seg in segs:
        role = (seg.get("identified_speaker") or seg.get("speaker") or "?")
        by[role] = by.get(role, 0) + 1
    print(f"speaker distribution: {by}")
