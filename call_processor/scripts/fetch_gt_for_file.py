"""Look up Audiofy speaker_json ground truth for a local recording by its
filename pattern (the c2k0vk-style trailing ID from Audiofy is the call id).
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

target_filename = sys.argv[1] if len(sys.argv) > 1 else "audio_05_01_2026_10_33_30_c2k0vk.mp3"
# The trailing 6-char ID is the suffix of Audiofy's S3 URL filename.
m = re.search(r"_([a-zA-Z0-9]{6})\.mp3$", target_filename)
trail = m.group(1) if m else ""
print(f"[lookup] filename={target_filename!r}  trail={trail!r}")

# Scrape a wide window and search.
end = datetime.now(timezone.utc)
start = end - timedelta(days=180)
print(f"[scrape] {start.date()} -> {end.date()}  (no user_name filter)")
recs = dtd.fetch_api_recordings(
    days=180, max_calls=2000,
    start_time=start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    end_time=end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
)
print(f"[scrape] {len(recs)} records")

# Match by URL containing the trail.
matches = [r for r in recs if trail and trail in str(r.get("horizon_call_s3_url") or "")]
if not matches:
    # Fall back to file size or duration heuristic
    print("[warn] no URL match; trying duration heuristic (1800s ± 60)")
    matches = [r for r in recs if 1740 <= float(r.get("duration") or 0) <= 1860]
    matches = matches[:5]
    for r in matches:
        url = str(r.get("horizon_call_s3_url") or "")
        print(f"  candidate: {url[-50:]}  agent={r.get('agent_name')!r}  dur={r.get('duration')}")
    sys.exit("no exact match; manual selection needed")

pick = matches[0]
agent = pick.get("agent_name")
sj = pick.get("speaker_json") or []
print(f"[match] agent={agent!r}  segments={len(sj)}  dur={pick.get('duration')}s")

# Count agent/customer
a = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), agent or "") == "agent")
c = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), agent or "") == "customer")
print(f"[gt] agent_segments={a}  customer_segments={c}")

out = REPO / "call_processor" / "data" / "raw_calls" / (target_filename + ".gt.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps({"agent_name": agent, "call_id": pick.get("_id"), "duration": pick.get("duration"), "segments": sj}, indent=2),
    encoding="utf-8",
)
print(f"[gt-saved] {out}")
