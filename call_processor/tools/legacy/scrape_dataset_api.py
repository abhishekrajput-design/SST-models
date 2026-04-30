"""
scrape_dataset_api.py — Pull labeled recordings from cp.audiofy.co.uk.

Endpoint:  POST /api/desk-streamer/get-recording-for-dataset
Body:      {"start_time": "...", "end_time": "...", "limit": N, "skip": K?}
Returns:   {"success": true, "data": [{_id, agent_name, horizon_call_s3_url, speaker_json: [...] }, ...]}

Saves:
  data/audiofy/_dataset/api_response_<window>.json     (raw manifest)
  data/audiofy/_dataset/audio/<_id>.mp3                 (downloaded calls)
  data/audiofy/_dataset/index.json                      (flat per-recording list)

Usage:
  python scrape_dataset_api.py --days 30 --max-calls 200
  python scrape_dataset_api.py --days 60 --max-calls 500 --skip-audio
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

SCRIPT_DIR = Path(__file__).parent
os.chdir(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env
ENV_PATH = SCRIPT_DIR.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_BASE  = os.environ.get("AUDIOFY_API_BASE", "https://cp.audiofy.co.uk")
API_TOKEN = os.environ.get("AUDIOFY_API_TOKEN", "").strip()
OUT_DIR   = SCRIPT_DIR / "data" / "audiofy" / "_dataset"
AUDIO_DIR = OUT_DIR / "audio"
INDEX_PATH = OUT_DIR / "index.json"


def _auth_headers() -> dict:
    if not API_TOKEN:
        sys.exit("[ERROR] AUDIOFY_API_TOKEN not set in .env")
    return {"Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
            "accept": "application/json"}


def fetch_batch(start_iso: str, end_iso: str,
                limit: int = 100, skip: int = 0) -> list[dict]:
    url = f"{API_BASE}/api/desk-streamer/get-recording-for-dataset"
    body = {"start_time": start_iso, "end_time": end_iso,
            "limit": limit, "skip": skip}
    r = requests.post(url, headers=_auth_headers(), json=body, timeout=60)
    if r.status_code != 200:
        print(f"  [batch] HTTP {r.status_code}: {r.text[:300]}", flush=True)
        return []
    try:
        data = r.json()
    except Exception as e:
        print(f"  [batch] JSON decode failed: {e}", flush=True)
        return []
    if not data.get("success"):
        print(f"  [batch] API error: {data.get('message')}", flush=True)
        return []
    return data.get("data", []) or []


def download_audio(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
        return dest.stat().st_size > 1000
    except Exception as e:
        print(f"    [audio] {dest.name}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="Pull recordings from last N days (default 30)")
    ap.add_argument("--max-calls", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--skip-audio", action="store_true",
                    help="Only save metadata, don't download MP3s")
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_iso   = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    print(f"[scrape] window={start_iso} -> {end_iso}")
    print(f"[scrape] max_calls={args.max_calls} batch_size={args.batch_size} skip_audio={args.skip_audio}")

    # ── Paginate ──────────────────────────────────────────────────────────────
    all_records: list[dict] = []
    skip = 0
    while len(all_records) < args.max_calls:
        want = min(args.batch_size, args.max_calls - len(all_records))
        print(f"[scrape] batch skip={skip} limit={want}...", flush=True)
        batch = fetch_batch(start_iso, end_iso, limit=want, skip=skip)
        if not batch:
            break
        all_records.extend(batch)
        skip += len(batch)
        if len(batch) < want:
            break

    print(f"[scrape] got {len(all_records)} records")

    # Save raw manifest
    manifest_path = OUT_DIR / f"raw_{start.strftime('%Y-%m-%d')}_{args.days}d.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"window": {"start": start_iso, "end": end_iso},
                   "count": len(all_records),
                   "data": all_records}, f, ensure_ascii=False, indent=2)
    print(f"[scrape] manifest -> {manifest_path}")

    # ── Filter: need speaker_json + horizon_call_s3_url + agent_name ─────────
    index: list[dict] = []
    agents_seen: dict[str, int] = {}

    for rec in all_records:
        rid = rec.get("_id")
        agent = (rec.get("agent_name") or "").strip()
        url = rec.get("horizon_call_s3_url")
        sj = rec.get("speaker_json") or []
        if not rid or not agent or not url or not sj:
            continue
        # Skip if agent has fewer than 1 phrase in speaker_json
        n_agent_phrases = sum(1 for s in sj if isinstance(s, dict)
                              and s.get("speaker") and s["speaker"] != "Customer")
        if n_agent_phrases == 0:
            continue

        audio_path = AUDIO_DIR / f"{rid}.mp3"
        ok_audio = audio_path.exists() and audio_path.stat().st_size > 1000
        if not ok_audio and not args.skip_audio:
            ok_audio = download_audio(url, audio_path)

        entry = {
            "_id":              rid,
            "agent_name":       agent,
            "horizon_s3":       url,
            "audio_path":       str(audio_path.relative_to(SCRIPT_DIR)) if ok_audio else None,
            "duration":         rec.get("duration"),
            "connect_time":     rec.get("connect_time"),
            "call_type":        rec.get("call_type"),
            "direction":        rec.get("direction"),
            "speaker_json":     sj,
            "n_agent_phrases":  n_agent_phrases,
            "n_total_phrases":  len(sj),
        }
        index.append(entry)
        agents_seen[agent] = agents_seen.get(agent, 0) + 1

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[scrape] index -> {INDEX_PATH}  ({len(index)} usable recordings)")

    print(f"\n[scrape] Agents ({len(agents_seen)} unique):")
    for name, n in sorted(agents_seen.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {name}")


if __name__ == "__main__":
    main()
