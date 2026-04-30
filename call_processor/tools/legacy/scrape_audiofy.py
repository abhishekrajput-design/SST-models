"""
scrape_audiofy.py — Download labeled call data from beta.audiofy.co.uk

API endpoints (discovered via network interception):
  POST /public/login
  GET  /api/user/get-all-users
  POST /api/desk-streamer/get-horizon-call-recordings
  GET  /api/desk-streamer/get-horizon-call-by-id?recording_id=<id>
  GET  /api/desk-streamer/get-transcripts-by-parent-id?parent_id=<id>

Usage:
    python scrape_audiofy.py                    # all Sales agents, 1 year history
    python scrape_audiofy.py --agent "Zak"      # name substring filter
    python scrape_audiofy.py --pages 2          # first 2 pages per agent (test)
    python scrape_audiofy.py --days 730         # 2-year window
    python scrape_audiofy.py --no-audio         # metadata + transcripts only, skip download
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL   = "https://beta.audiofy.co.uk"
SCRIPT_DIR = Path(__file__).parent
OUT_DIR    = SCRIPT_DIR / "data" / "audiofy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

DEFAULT_USER = os.getenv("AUDIOFY_USERNAME", "")
DEFAULT_PASS = os.getenv("AUDIOFY_PASSWORD", "")
PAGE_LIMIT   = 25

# Only scrape agents in these departments (skip IT, HR, Finance, etc.)
TARGET_DEPTS = {
    "sales", "call center - sales", "call center",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def ts_to_sec(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except Exception:
        return 0.0


def cut_clip(audio_path: str, start: float, end: float, out_path: str) -> bool:
    dur = end - start
    if dur < 0.5:
        return False
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", audio_path,
             "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
             "-ac", "1", "-ar", "16000", out_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        print(f"    [clip] FFmpeg error: {e}")
        return False


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class AudiofyClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept":       "application/json",
            "Content-Type": "application/json",
            "Origin":       BASE_URL,
            "Referer":      f"{BASE_URL}/history",
        })

    def login(self, username: str, password: str) -> str:
        r = self.session.post(
            f"{BASE_URL}/public/login",
            json={"username": username, "password": password},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        token = data.get("accessToken") or data.get("token") or ""
        nested = data.get("data")
        if not token and isinstance(nested, dict):
            token = (
                nested.get("accessToken")
                or nested.get("access_token")
                or nested.get("token")
                or nested.get("jwt")
                or ""
            )
        if not token:
            msg = data.get("message") if isinstance(data, dict) else ""
            raise RuntimeError(f"No token in login response: keys={list(data.keys())} message={msg!r}")
        self.session.headers["Authorization"] = f"Bearer {token}"
        print(f"[Auth] Logged in as {username}")
        return token

    def get_all_users(self) -> list[dict]:
        r = self.session.get(f"{BASE_URL}/api/user/get-all-users", timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        for key in ("users", "data", "result", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return []

    def get_recordings_page(self, page: int, user_id: str,
                            start_dt: str, end_dt: str) -> dict:
        body = {
            "start_datetime":      start_dt,
            "end_datetime":        end_dt,
            "page":                page,
            "limit":               PAGE_LIMIT,
            "user":                user_id,
            "external_user":       user_id,
            "sortBy":              [],
            "route_status":        "",          # empty = all statuses
            "platform_name":       "",
            "call_type":           "",
            "start_duration":      0,
            "end_duration":        99999,
            "search":              "",
            "department_id":       None,
            "branch":              "",
            "isCommentsCalls":     None,
            "comments_start_date": None,
            "comments_end_date":   None,
            "commentedBy":         [],
            "byComments":          False,
            "conversion_calls":    "",
            "compliance_calls":    "",
            "recording_source":    "",
            "mis_identified":      False,
            "agent_attempted_switch": "",
        }
        r = self.session.post(
            f"{BASE_URL}/api/desk-streamer/get-horizon-call-recordings",
            json=body, timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def get_call_detail(self, recording_id: str) -> dict:
        r = self.session.get(
            f"{BASE_URL}/api/desk-streamer/get-horizon-call-by-id",
            params={"recording_id": recording_id},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def get_transcripts(self, parent_id: str) -> list[dict]:
        r = self.session.get(
            f"{BASE_URL}/api/desk-streamer/get-transcripts-by-parent-id",
            params={"parent_id": parent_id},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            nested = data["data"].get("data")
            if isinstance(nested, list):
                return nested
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        for key in ("transcripts", "segments", "data", "result", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]

        r = self.session.get(
            f"{BASE_URL}/api/desk-streamer/get-transcription-by-parent-id",
            params={"parent_id": parent_id},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            payload = data.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("conversation"), list):
                return payload["conversation"]
            if isinstance(payload, list):
                return payload
        return []

    def download_audio(self, url: str, dest_path: str) -> bool:
        if not url:
            return False
        if url.startswith("/"):
            url = BASE_URL + url
        # S3/CDN URLs must not carry the Audiofy Bearer token — AWS rejects it.
        # Use a plain request for any URL outside our own API host.
        import urllib.parse
        is_own_host = urllib.parse.urlparse(url).netloc in (
            "beta.audiofy.co.uk", "audiofy.co.uk"
        )
        getter = self.session.get if is_own_host else requests.get
        try:
            with getter(url, stream=True, timeout=120) as r:
                if r.status_code in (200, 206):
                    with open(dest_path, "wb") as f:
                        for chunk in r.iter_content(65536):
                            f.write(chunk)
                    size_kb = os.path.getsize(dest_path) // 1024
                    return size_kb > 1  # reject empty files
                print(f"    [dl] HTTP {r.status_code}")
        except Exception as e:
            print(f"    [dl] Error: {e}")
        return False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_history(data: dict) -> tuple[list[dict], int]:
    rows: list[dict] = []
    if isinstance(data, list):
        rows = data
    else:
        payload = data.get("data") if isinstance(data, dict) else None
        if isinstance(payload, dict):
            nested_rows = payload.get("data")
            if isinstance(nested_rows, list):
                rows = nested_rows
            elif isinstance(nested_rows, dict) and isinstance(nested_rows.get("data"), list):
                rows = nested_rows["data"]
        for key in ("recordings", "calls", "data", "result", "records", "history"):
            if not rows and key in data and isinstance(data[key], list):
                rows = data[key]
                break
    total = 1
    total_sources = [data]
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        total_sources.append(data["data"])
    for source in total_sources:
        if not isinstance(source, dict):
            continue
        for key in ("totalPages", "total_pages", "pages", "pageCount", "total_page"):
            if key not in source:
                continue
            try:
                total = max(1, int(source[key]))
                break
            except Exception:
                pass
        if total != 1:
            break
    return rows, total


def _parse_segments(raw: list[dict]) -> list[dict]:
    segments = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        start = ts_to_sec(seg.get("start") or seg.get("startTime") or seg.get("start_time") or 0)
        end   = ts_to_sec(seg.get("end")   or seg.get("endTime")   or seg.get("end_time")   or 0)
        spk   = str(seg.get("speaker") or seg.get("speakerName") or seg.get("name") or "").strip()
        text  = str(seg.get("text")    or seg.get("content")     or seg.get("transcript")
                    or seg.get("phrase") or seg.get("original") or "").strip()
        if end > start:
            segments.append({"start": round(start, 3), "end": round(end, 3),
                             "speaker": spk, "text": text})
    return segments


def _find_audio_url(obj: dict) -> str:
    keys = ("audioUrl", "audio_url", "recordingUrl", "recording_url",
            "fileUrl", "file_url", "horizon_call_s3_url", "url", "audio", "recording")
    for k in keys:
        v = obj.get(k)
        if v and isinstance(v, str) and v.startswith("http"):
            return v
    # Check one level of nesting
    for sub in ("recording", "call", "data", "result", "file"):
        child = obj.get(sub)
        if isinstance(child, dict):
            for k in keys:
                v = child.get(k)
                if v and isinstance(v, str) and v.startswith("http"):
                    return v
    xbees = obj.get("xBees_json_file")
    if isinstance(xbees, dict):
        for attachment in xbees.get("attachments", []) or []:
            if not isinstance(attachment, dict):
                continue
            recording = attachment.get("recording")
            if isinstance(recording, dict):
                url = recording.get("url")
                if isinstance(url, str) and url.startswith("http"):
                    return url
    return ""


# ---------------------------------------------------------------------------
# Main scrape
# ---------------------------------------------------------------------------

def scrape(client: AudiofyClient, args) -> list[dict]:
    agent_filter = (args.agent or "").strip().lower()
    max_pages    = args.pages or 999
    max_calls_pa = args.max_calls
    days         = args.days

    now      = datetime.now(timezone.utc)
    end_dt   = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    start_dt = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    print(f"[Range] {start_dt}  to  {end_dt}  ({days} days)")

    # --- Fetch all users ---
    print("[Users] Fetching agent list…")
    all_users = client.get_all_users()
    print(f"[Users] {len(all_users)} users total")

    # Filter to Sales / Call Center agents only
    targets: list[tuple[str, str, list[str]]] = []
    for u in all_users:
        fn   = (u.get("firstName") or u.get("first_name") or "").strip()
        ln   = (u.get("lastName")  or u.get("last_name")  or "").strip()
        name = f"{fn} {ln}".strip() or u.get("username") or "Unknown"
        uid  = str(u.get("_id") or u.get("id") or "")
        username = str(u.get("username") or "").strip()
        email = str(u.get("email") or "").strip()
        if not uid:
            continue

        dept_obj = u.get("department")
        dept = ""
        if isinstance(dept_obj, dict):
            dept = dept_obj.get("name", "").lower()
        elif isinstance(dept_obj, str):
            dept = dept_obj.lower()

        if agent_filter:
            if agent_filter not in name.lower():
                continue
        else:
            # Skip non-sales departments
            if not any(d in dept for d in TARGET_DEPTS):
                continue
            # Skip obvious system / test / office accounts
            if any(x in name.lower() for x in
                   ("office", "reception", "test", "admin", "site", "account", "team")):
                continue

        external_values = []
        for candidate in (uid, username, email, name):
            if candidate and candidate not in external_values:
                external_values.append(candidate)

        targets.append((uid, name, external_values))

    print(f"[Targets] {len(targets)} agents:")
    for uid, name, _external_values in targets:
        print(f"  {name}")

    all_calls: list[dict] = []
    debug_printed = False

    for user_id, user_name, external_values in targets:
        print(f"\n=== {user_name} ===")
        agent_count = 0

        for pg in range(1, max_pages + 1):
            print(f"  page {pg}…", end=" ", flush=True)
            try:
                data = None
                for ext_value in external_values:
                    data = client.get_recordings_page(pg, ext_value, start_dt, end_dt)
                    if args.debug_response and not debug_printed:
                        print("\n[Debug] first recordings response:")
                        print(json.dumps(data, indent=2, ensure_ascii=False, default=str)[:4000])
                        debug_printed = True
                    probe_rows, _probe_pages = _parse_history(data)
                    if probe_rows or ext_value == external_values[-1]:
                        break
            except Exception as e:
                print(f"ERROR {e}")
                break

            rows, total_pages = _parse_history(data)
            print(f"{len(rows)} calls  (total_pages={total_pages})")

            for row in rows:
                rec_id = str(row.get("parent_id") or row.get("_id") or row.get("id") or row.get("recording_id") or "")
                recording_id = str(row.get("recording_id") or row.get("horizon_recording_id") or "")
                if not rec_id:
                    continue

                # Get audio URL from detail endpoint
                audio_url = _find_audio_url(row)
                if not audio_url:
                    try:
                        detail = client.get_call_detail(rec_id)
                        audio_url = _find_audio_url(detail)
                    except Exception:
                        pass

                # Get transcript segments
                segments: list[dict] = []
                row_id = str(row.get("_id") or row.get("id") or "")
                for transcript_parent in (rec_id, row_id, recording_id):
                    if not transcript_parent:
                        continue
                    try:
                        raw_segs = client.get_transcripts(transcript_parent)
                        segments = _parse_segments(raw_segs)
                        if segments:
                            break
                    except Exception as e:
                        if transcript_parent == recording_id or not recording_id:
                            print(f"    [transcript error] {rec_id}: {e}")

                duration = str(row.get("duration") or row.get("call_duration") or "")
                all_calls.append({
                    "call_id":    rec_id,
                    "agent_name": user_name,
                    "agent_id":   user_id,
                    "duration":   duration,
                    "audio_url":  audio_url,
                    "segments":   segments,
                })
                agent_count += 1

                if max_calls_pa and agent_count >= max_calls_pa:
                    break

            if pg >= total_pages or (max_calls_pa and agent_count >= max_calls_pa):
                break

        print(f"  -> {agent_count} calls collected")

    return all_calls


# ---------------------------------------------------------------------------
# Save + cut agent clips
# ---------------------------------------------------------------------------

def save_and_cut(client: AudiofyClient, all_calls: list[dict], skip_audio: bool = False):
    # Group by agent
    agents: dict[str, list[dict]] = {}
    for call in all_calls:
        agents.setdefault(call["agent_name"], []).append(call)

    manifest: dict[str, dict] = {}

    for agent_name, calls in agents.items():
        a_slug    = slug(agent_name)
        agent_dir = OUT_DIR / a_slug
        audio_dir = agent_dir / "audio"
        clips_dir = agent_dir / "agent_clips"
        for d in (agent_dir, audio_dir, clips_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n[Save] {agent_name} — {len(calls)} calls")
        total_clips = 0

        for call in calls:
            call_id   = call["call_id"]
            audio_url = call.get("audio_url") or ""
            segments  = call.get("segments", [])

            # Agent-only segments: exclude "Customer" speaker
            agent_segs = [s for s in segments
                          if "customer" not in s.get("speaker", "").lower()]

            audio_path  = str(audio_dir / f"call_{call_id}.mp3")
            downloaded  = os.path.exists(audio_path)

            if not skip_audio and audio_url and not downloaded:
                print(f"  DL {call_id}…", end=" ", flush=True)
                downloaded = client.download_audio(audio_url, audio_path)
                print("OK" if downloaded else "FAIL")

            clips_saved = 0
            if downloaded and agent_segs:
                for j, seg in enumerate(agent_segs):
                    start = float(seg.get("start", 0))
                    end   = float(seg.get("end", 0))
                    clip_path = str(clips_dir / f"clip_{j+1:03d}_{start:.1f}s_{end:.1f}s.wav")
                    if not os.path.exists(clip_path):
                        if cut_clip(audio_path, start, end, clip_path):
                            clips_saved += 1

            total_clips += clips_saved
            status = f"segs={len(segments)} agent_segs={len(agent_segs)} clips+={clips_saved} audio={'yes' if audio_url else 'no-url'}"
            print(f"    {call_id[:16]}  {status}")

        # Save calls.json
        with open(agent_dir / "calls.json", "w", encoding="utf-8") as f:
            json.dump({"agent": agent_name, "calls": calls}, f, indent=2,
                      ensure_ascii=False, default=str)

        print(f"  -> {total_clips} agent clips saved to {clips_dir}")
        manifest[a_slug] = {
            "agent_name":  agent_name,
            "calls_count": len(calls),
            "clips_dir":   str(clips_dir),
            "clips_total": total_clips,
        }

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Done] manifest -> {OUT_DIR / 'manifest.json'}")
    print(f"       {sum(v['clips_total'] for v in manifest.values())} total agent clips across {len(manifest)} agents")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",     default="",  help="Agent name substring filter")
    parser.add_argument("--pages",     type=int,    help="Max pages per agent (default: all)")
    parser.add_argument("--max-calls", type=int,    help="Max calls per agent")
    parser.add_argument("--days",      type=int, default=365, help="History window in days")
    parser.add_argument(
        "--username",
        default=DEFAULT_USER,
        help="Audiofy username. Defaults to AUDIOFY_USERNAME.",
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASS,
        help="Audiofy password. Defaults to AUDIOFY_PASSWORD.",
    )
    parser.add_argument("--no-audio",  action="store_true", help="Skip audio download")
    parser.add_argument("--debug-response", action="store_true", help="Print first recordings API response shape")
    args = parser.parse_args()
    if not args.username:
        parser.error("Audiofy username is required: pass --username or set AUDIOFY_USERNAME")
    if not args.password:
        parser.error("Audiofy password is required: pass --password or set AUDIOFY_PASSWORD")

    client = AudiofyClient()
    try:
        client.login(args.username, args.password)
    except Exception as e:
        print(f"[Auth] Login failed: {e}")
        sys.exit(1)

    all_calls = scrape(client, args)
    print(f"\n[Scrape] {len(all_calls)} total calls collected")

    if not all_calls:
        print("No calls found. Try --days 730 or check credentials.")
        return

    # Summary table
    from collections import Counter
    counts = Counter(c["agent_name"] for c in all_calls)
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        has_url  = sum(1 for c in all_calls if c["agent_name"] == name and c["audio_url"])
        has_segs = sum(1 for c in all_calls if c["agent_name"] == name and c["segments"])
        print(f"  {name:40s} {n:3d} calls  audio={has_url}/{n}  transcripts={has_segs}/{n}")

    save_and_cut(client, all_calls, skip_audio=args.no_audio)


if __name__ == "__main__":
    main()
