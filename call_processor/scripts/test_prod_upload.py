"""Upload local audio files to the production UI, time the pipeline, report.

Run from a workstation that can reach the prod box on port 8080 (e.g. via
its public IP or through SSH port-forward). Per-file flow:
  1. POST /api/upload with the file body (read into memory so the basic
     http.server gets a clean Content-Length and doesn't truncate)
  2. Poll /api/status at 2 Hz, log every stage transition with elapsed time
  3. Fetch /api/call/<result_id> and print identified_agent + segment count
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROD = "http://13.42.127.218:8080"
TIMEOUT_STATUS = 10

FILES = [
    r"C:\Users\abhis\Downloads\audio_05_01_2026_10_33_30_c2k0vk.mp3",
    r"C:\Users\abhis\Downloads\audio_05_01_2026_10_03_30_uzpfxq.mp3",
]


def upload_and_wait(path: Path) -> dict:
    body = path.read_bytes()
    fname = path.name
    mb = len(body) / 1e6
    print(f"\n=== {fname}  ({mb:.2f} MB) ===")
    t_up = time.time()
    r = requests.post(
        f"{PROD}/api/upload",
        params={"filename": fname, "model": "parakeet-tdt-0.6b-v3"},
        data=body,
        timeout=180,
        headers={"Content-Type": "audio/mpeg"},
    )
    r.raise_for_status()
    print(f"[upload] {round(time.time() - t_up, 1)}s — {r.json()}")

    t0 = time.time()
    last = ""
    while True:
        s = requests.get(f"{PROD}/api/status", timeout=TIMEOUT_STATUS).json()
        key = f"{s.get('stage_num')}.{s.get('stage')}"
        if key != last:
            print(f"  [+{round(time.time()-t0,1):>5.1f}s] {key:<22s} {s.get('message','')[:80]}")
            last = key
        if s.get("done"):
            rid = s.get("result_id")
            print(f"  [done] {round(time.time()-t0,1)}s   result_id={rid}")
            r = requests.get(f"{PROD}/api/call/{rid}", timeout=30)
            r.raise_for_status()
            return r.json()
        if s.get("error"):
            raise SystemExit(f"  pipeline error: {s.get('error')}")
        time.sleep(2)


def main():
    print(f"Target: {PROD}")
    print(f"Probing... ", end="", flush=True)
    p = requests.get(f"{PROD}/api/status", timeout=TIMEOUT_STATUS).json()
    print(f"running={p.get('running')} stage={p.get('stage')}")
    if p.get("running"):
        raise SystemExit("prod pipeline is already running; aborting")

    results: list[tuple[str, dict]] = []
    for f in FILES:
        path = Path(f)
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        try:
            res = upload_and_wait(path)
            results.append((path.name, res))
        except Exception as e:
            print(f"  [error] {path.name}: {e}")
            results.append((path.name, {"error": str(e)}))

    print()
    print("=" * 78)
    print(f"{'File':<46s} {'Agent':<28s} {'Segs':>5s}")
    print("=" * 78)
    for fname, res in results:
        if "error" in res:
            print(f"{fname:<46s} ERROR: {res['error']}")
            continue
        ag = str(res.get("identified_agent") or "")[:26]
        segs = res.get("total_segments") or 0
        dev = res.get("transcriber_device", "?")
        print(f"{fname:<46s} {ag:<28s} {segs:>5d}  ({dev})")
    print()


if __name__ == "__main__":
    main()
