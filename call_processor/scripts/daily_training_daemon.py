#!/usr/bin/env python
"""Daily automated agent voiceprint training daemon.

Orchestrates the full pipeline:
  1. Scrape new call recordings from the Audiofy API
  2. Group by agent, prepare labelled training data
  3. Train CAM++ voiceprints per agent
  4. Validate with leave-one-call-out
  5. Activate only if quality improves over existing voiceprint
  6. Track confidence history day-by-day

Usage:
  python daily_training_daemon.py                          # default: last 7 days
  python daily_training_daemon.py --days 14 --dry-run      # preview only
  python daily_training_daemon.py --agents "Omar" "Zak"    # specific agents
  python daily_training_daemon.py --min-calls 3            # lower bar for new hires
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
VP_DIR = CALL_PROCESSOR_DIR / "data" / "agent_voiceprints"
DAILY_REPORTS_DIR = VP_DIR / "daily_reports"
TRAINING_HISTORY_PATH = VP_DIR / "training_history.json"

sys.path.insert(0, str(CALL_PROCESSOR_DIR))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_BASE = os.environ.get("AUDIOFY_API_BASE", "https://cp.audiofy.co.uk")
API_LOGIN_BASE = os.environ.get("AUDIOFY_LOGIN_BASE", API_BASE)
API_TOKEN = os.environ.get("AUDIOFY_API_TOKEN", "").strip()
API_USERNAME = (
    os.environ.get("AUDIOFY_USERNAME")
    or os.environ.get("AUDIOFY_USER")
    or ""
).strip()
API_PASSWORD = (
    os.environ.get("AUDIOFY_PASSWORD")
    or os.environ.get("AUDIOFY_PASS")
    or ""
).strip()
LAST_SCRAPE_ERROR: str | None = None

FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")
TARGET_SR = 16000

# Quality filters
MIN_AVG_SCORE = 0.60          # minimum avg_score per phrase to trust API label
MIN_AGENT_PHRASES = 3         # minimum agent phrases per call
MIN_CALL_DURATION_S = 30      # skip very short calls
MAX_CALL_DURATION_S = 900     # skip extremely long calls (>15min)
MIN_ACTIVATION_ACCURACY = 85.0
N_CLUSTERS = 3

# Call-level poisoning gates — prevent agent voiceprints from being trained on
# calls where Audiofy's speaker_json labels are unreliable (noise mislabeled as
# agent, all-agent labelling, single-side recordings, etc.).
MIN_CUSTOMER_PHRASES = 2          # real phone calls have both speakers; 0-1 customers => labels suspect
MIN_AGENT_MEAN_SCORE = 0.50       # if mean(avg_score) of agent phrases is below this, labels are mostly noise
MIN_HIGH_QUALITY_AGENT_RATIO = 0.25  # at least 25% of agent-labelled phrases must clear MIN_AVG_SCORE

BACKCHANNELS = {
    "hello", "hi", "yeah", "yes", "yep", "ok", "okay", "right",
    "sure", "no worries", "thank you", "thanks", "bye", "bye bye",
}

VOICEMAIL_OR_SYSTEM_CUES = (
    "voicemail",
    "not available",
    "leave a message",
    "record your message",
    "after the tone",
    "press the hash key",
    "press 1",
    "vodafone voicemail",
    "ee voicemail",
)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _norm_name(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def usable_agent_training_text(text: str) -> bool:
    normalized = _norm_name(text)
    if not normalized:
        return False
    if normalized in BACKCHANNELS or (len(normalized.split()) <= 1 and normalized):
        return False
    if any(cue in normalized for cue in VOICEMAIL_OR_SYSTEM_CUES):
        return False
    if sum(1 for ch in str(text) if ch.isalpha()) < 6:
        return False
    return True


def speaker_role_from_api_label(speaker_raw: str, agent_name: str) -> str | None:
    """Map Audiofy speaker labels to agent/customer without poisoning labels.

    Dataset labels may arrive as Customer, Customer_1, Customer_2, or
    "Agent Name_1". Treat every Customer-prefixed label as customer and only
    accept agent labels that match the requested agent name.
    """
    raw = str(speaker_raw or "").strip()
    if not raw:
        return None
    normalized = _norm_name(raw)
    agent_norm = _norm_name(agent_name)
    if normalized.startswith("customer"):
        return "customer"
    if agent_norm and (
        normalized == agent_norm
        or normalized.startswith(f"{agent_norm} ")
        or normalized.startswith(agent_norm)
    ):
        return "agent"
    return None


def infer_local_agent_identity(data_dir: Path, fallback_name: str) -> tuple[str, str]:
    """Infer production agent name/slug for --skip-scrape local folders."""
    agent_name = fallback_name
    for label_path in sorted(data_dir.glob("*/data.json")):
        if label_path.parent.name.startswith("_"):
            continue
        try:
            data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        candidate = str(data.get("agent_name") or "").strip()
        if candidate:
            agent_name = candidate
            break

    folder_slug = data_dir.name
    agents_path = VP_DIR / "agents.json"
    if not agents_path.exists():
        return agent_name, folder_slug

    try:
        agents = json.loads(agents_path.read_text(encoding="utf-8"))
    except Exception:
        return agent_name, folder_slug

    if folder_slug in agents:
        entry = agents.get(folder_slug) or {}
        return entry.get("agent_name") or agent_name, folder_slug

    wanted = _norm_name(agent_name)
    matches = []
    for existing_slug, entry in agents.items():
        existing_name = _norm_name(str(entry.get("agent_name") or existing_slug))
        if wanted and (wanted in existing_name or existing_name in wanted):
            score = int(entry.get("n_voiceprints") or 0)
            if existing_slug.endswith("_local_20260423") or "local" in existing_slug:
                score -= 10
            matches.append((score, existing_slug, entry))
    if matches:
        _, existing_slug, entry = sorted(matches, reverse=True)[0]
        return entry.get("agent_name") or agent_name, existing_slug

    return agent_name, folder_slug


def ts2s(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except Exception:
        return 0.0


def _token_from_login_payload(data: dict) -> str:
    token = (
        data.get("accessToken")
        or data.get("access_token")
        or data.get("token")
        or data.get("jwt")
        or ""
    )
    nested = data.get("data")
    if not token and isinstance(nested, dict):
        token = (
            nested.get("accessToken")
            or nested.get("access_token")
            or nested.get("token")
            or nested.get("jwt")
            or ""
        )
    return str(token or "").strip()


def refresh_api_token_from_login() -> bool:
    """Refresh Audiofy bearer token from runtime credentials, without logging it."""
    global API_TOKEN, LAST_SCRAPE_ERROR
    if not API_USERNAME or not API_PASSWORD:
        return False
    for base in dict.fromkeys([API_LOGIN_BASE, API_BASE, "https://cp.audiofy.co.uk", "https://beta.audiofy.co.uk"]):
        try:
            r = requests.post(
                f"{base.rstrip('/')}/public/login",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Origin": base.rstrip("/"),
                    "Referer": f"{base.rstrip('/')}/leaderboard",
                },
                json={"username": API_USERNAME, "password": API_PASSWORD},
                timeout=30,
            )
            if r.status_code != 200:
                LAST_SCRAPE_ERROR = f"login HTTP {r.status_code}: {r.text[:200]}"
                continue
            data = r.json()
            token = _token_from_login_payload(data if isinstance(data, dict) else {})
            if token:
                API_TOKEN = token
                print(f"[auth] refreshed Audiofy token from {base.rstrip('/')}")
                return True
            LAST_SCRAPE_ERROR = "login succeeded but response did not contain a token"
        except Exception as exc:
            LAST_SCRAPE_ERROR = f"login failed: {exc}"
    return False


def _auth_headers() -> dict:
    if not API_TOKEN:
        if not refresh_api_token_from_login():
            print(
                "[error] AUDIOFY_API_TOKEN not set/valid and AUDIOFY_USERNAME/AUDIOFY_PASSWORD login failed",
                file=sys.stderr,
            )
            sys.exit(1)
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }


def fetch_api_recordings(
    days: int,
    max_calls: int,
    batch_size: int = 100,
    start_time: str | None = None,
    end_time: str | None = None,
    user_name: str | None = None,
) -> list[dict]:
    """Fetch recordings from the Audiofy API."""
    global LAST_SCRAPE_ERROR
    LAST_SCRAPE_ERROR = None
    if start_time and end_time:
        start_iso = start_time
        end_iso = end_time
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_iso = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    url = f"{API_BASE}/api/desk-streamer/get-recording-for-dataset"
    who = f" user_name={user_name}" if user_name else ""
    print(f"[scrape] window={start_iso} -> {end_iso}{who}")

    all_records: list[dict] = []
    skip = 0
    retried_after_auth = False
    while len(all_records) < max_calls:
        want = min(batch_size, max_calls - len(all_records))
        body = {"start_time": start_iso, "end_time": end_iso, "limit": want, "skip": skip}
        if user_name:
            body["user_name"] = user_name
        try:
            r = requests.post(url, headers=_auth_headers(), json=body, timeout=60)
            if r.status_code == 401 and not retried_after_auth and refresh_api_token_from_login():
                retried_after_auth = True
                print("  [batch] token rejected; refreshed login token and retrying")
                continue
            if r.status_code != 200:
                LAST_SCRAPE_ERROR = f"HTTP {r.status_code}: {r.text[:300]}"
                print(f"  [batch] {LAST_SCRAPE_ERROR}")
                break
            data = r.json()
            if not data.get("success"):
                LAST_SCRAPE_ERROR = f"API error: {data.get('message')}"
                print(f"  [batch] {LAST_SCRAPE_ERROR}")
                break
            batch = data.get("data", []) or []
        except Exception as e:
            LAST_SCRAPE_ERROR = f"request failed: {e}"
            print(f"  [batch] {LAST_SCRAPE_ERROR}")
            break

        if not batch:
            break
        all_records.extend(batch)
        skip += len(batch)
        if len(batch) < want:
            break

    print(f"[scrape] got {len(all_records)} records")
    return all_records


def call_quality_check(rec: dict, agent: str) -> tuple[bool, str]:
    """Reject calls whose speaker_json labels look unreliable.

    A real phone call has both speakers and a reasonable agent labelling
    confidence. When Audiofy mis-classifies background noise / voicemail /
    single-side recordings as 100% agent, those segments poison the trained
    voiceprint. Returns (ok, reason). reason is "" when ok.
    """
    sj = rec.get("speaker_json") or []
    all_agent = [
        s for s in sj
        if isinstance(s, dict)
        and speaker_role_from_api_label(str(s.get("speaker") or ""), agent) == "agent"
    ]
    all_customer = [
        s for s in sj
        if isinstance(s, dict)
        and speaker_role_from_api_label(str(s.get("speaker") or ""), agent) == "customer"
    ]
    if len(all_customer) < MIN_CUSTOMER_PHRASES:
        return False, f"only {len(all_customer)} customer phrase(s) — likely mislabelled (real calls have both speakers)"
    if not all_agent:
        return False, "no agent phrases at all"
    agent_scores = [float(s.get("avg_score") if s.get("avg_score") is not None else 0.85) for s in all_agent]
    mean_score = sum(agent_scores) / len(agent_scores)
    if mean_score < MIN_AGENT_MEAN_SCORE:
        return False, f"agent mean avg_score {mean_score:.3f} < {MIN_AGENT_MEAN_SCORE} (most labels are noise)"
    high_q = sum(1 for x in agent_scores if x >= MIN_AVG_SCORE)
    ratio = high_q / len(all_agent)
    if ratio < MIN_HIGH_QUALITY_AGENT_RATIO:
        return False, f"only {high_q}/{len(all_agent)} ({ratio:.0%}) agent phrases score >= {MIN_AVG_SCORE}"
    return True, ""


def filter_and_group(records: list[dict], min_calls: int) -> dict[str, list[dict]]:
    """Filter valid records and group by agent_name."""
    by_agent: dict[str, list[dict]] = {}
    poisoned_skips = 0

    for rec in records:
        agent = (rec.get("agent_name") or "").strip()
        url = rec.get("horizon_call_s3_url")
        sj = rec.get("speaker_json") or []
        duration = rec.get("duration") or 0

        if not agent or not url or not sj:
            continue

        # Filter by call duration
        if duration and (duration < MIN_CALL_DURATION_S or duration > MAX_CALL_DURATION_S):
            continue

        # Call-level poisoning gate — drops calls with bad speaker_json labels
        ok, reason = call_quality_check(rec, agent)
        if not ok:
            poisoned_skips += 1
            call_id = str(rec.get("_id") or rec.get("id") or "")[:12]
            print(f"  [poison-skip] {agent} {call_id}: {reason}")
            continue

        # Count high-quality agent phrases (training-content gate)
        agent_phrases = [
            s for s in sj
            if isinstance(s, dict)
            and speaker_role_from_api_label(str(s.get("speaker") or ""), agent) == "agent"
            and float(s.get("avg_score") if s.get("avg_score") is not None else 0.85) >= MIN_AVG_SCORE
            and usable_agent_training_text(str(s.get("phrase") or ""))
        ]
        if len(agent_phrases) < MIN_AGENT_PHRASES:
            continue

        by_agent.setdefault(agent, []).append(rec)

    if poisoned_skips:
        print(f"[filter] dropped {poisoned_skips} call(s) via poisoning gates")

    # Filter agents with enough calls
    qualified = {
        name: calls for name, calls in by_agent.items()
        if len(calls) >= min_calls
    }

    print(f"[filter] {len(qualified)} agents with >= {min_calls} calls:")
    for name, calls in sorted(qualified.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(calls):4d}  {name}")

    return qualified


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
        print(f"    [download] {dest.name}: {e}")
        return False


def convert_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert MP3 to 16kHz mono WAV."""
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return True
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", str(mp3_path),
             "-ac", "1", "-ar", str(TARGET_SR), str(wav_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=True,
        )
        return wav_path.exists() and wav_path.stat().st_size > 1000
    except Exception as e:
        print(f"    [convert] {mp3_path.name}: {e}")
        return False


def _call_quality_score(rec: dict, agent_name: str) -> tuple[float, float]:
    """Rank-key for call selection: (mean agent avg_score, total agent duration).

    Higher mean score = cleaner Audiofy labels. Duration breaks ties so longer
    calls with the same mean score win. Used to pick the cleanest N calls
    instead of the longest N — diversity of clean audio matters more than raw
    segment count for voiceprint quality.
    """
    sj = rec.get("speaker_json") or []
    scores: list[float] = []
    duration = 0.0
    for s in sj:
        if not isinstance(s, dict):
            continue
        if speaker_role_from_api_label(str(s.get("speaker") or ""), agent_name) != "agent":
            continue
        scores.append(float(s.get("avg_score") if s.get("avg_score") is not None else 0.85))
        try:
            duration += float(ts2s(s.get("end") or 0)) - float(ts2s(s.get("start") or 0))
        except Exception:
            pass
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return (mean_score, duration)


def prepare_training_data(
    agent_name: str,
    agent_slug_str: str,
    calls: list[dict],
    max_calls: int,
    work_dir: Path,
) -> Path:
    """Download calls and prepare data.json files for training.

    Returns the data directory path containing call_XX/data.json + audio_16k.wav.
    """
    data_dir = work_dir / agent_slug_str
    data_dir.mkdir(parents=True, exist_ok=True)
    for old_call_dir in data_dir.glob("call_*"):
        if old_call_dir.is_dir():
            shutil.rmtree(old_call_dir)

    # Dedupe — the paginated scrape can return the same record across batches.
    # Without this, sorting by quality clumps duplicate copies of one call at the
    # top and crowds out distinct calls from the training set.
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in calls:
        key = str(r.get("_id") or r.get("id") or r.get("horizon_call_s3_url") or "")
        if key and key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    if len(uniq) != len(calls):
        print(f"  [dedupe] {len(calls)} -> {len(uniq)} unique calls")
    calls = uniq

    # Rank calls by mean agent avg_score (descending) — pick the cleanest first.
    calls = sorted(calls, key=lambda r: _call_quality_score(r, agent_name), reverse=True)
    if calls:
        top = _call_quality_score(calls[0], agent_name)
        tail = _call_quality_score(calls[min(len(calls) - 1, max_calls - 1)], agent_name)
        print(f"  [rank] selecting top {max_calls}; best mean_score={top[0]:.3f}  worst-of-kept mean_score={tail[0]:.3f}")

    used = 0
    for i, rec in enumerate(calls[:max_calls]):
        rid = rec.get("_id", f"call_{i:02d}")
        sj = rec.get("speaker_json") or []
        url = rec.get("horizon_call_s3_url")

        call_dir = data_dir / f"call_{i:02d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        # Download and convert audio
        mp3_path = call_dir / "audio.mp3"
        wav_path = call_dir / "audio_16k.wav"

        if not download_audio(url, mp3_path):
            print(f"    [skip] {rid[:12]}: download failed")
            continue
        if not convert_to_wav(mp3_path, wav_path):
            print(f"    [skip] {rid[:12]}: convert failed")
            continue

        # Build segments from speaker_json
        segments = []
        for phrase in sj:
            if not isinstance(phrase, dict):
                continue
            speaker_raw = (phrase.get("speaker") or "").strip()
            if not speaker_raw:
                continue
            role = speaker_role_from_api_label(speaker_raw, agent_name)
            if role is None:
                continue
            start = ts2s(phrase.get("start") or 0)
            end = ts2s(phrase.get("end") or 0)
            text = phrase.get("phrase") or ""
            avg_score = float(phrase.get("avg_score") if phrase.get("avg_score") is not None else 0.85)

            if end <= start:
                continue

            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": role,
                "raw_speaker": speaker_raw,
                "text": text,
                "avg_score": round(avg_score, 3),
            })

        if not segments:
            continue

        # Write data.json
        data_json = {
            "call_id": str(rid),
            "agent_name": agent_name,
            "source": "daily_auto_api_scrape",
            "segments": segments,
        }
        (call_dir / "data.json").write_text(
            json.dumps(data_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        used += 1
        print(f"    [ok] call_{i:02d} ({rid[:12]}): {len(segments)} segments")

    print(f"  [prepare] {agent_name}: {used} calls ready in {data_dir}")
    return data_dir


def load_training_history() -> dict:
    if TRAINING_HISTORY_PATH.exists():
        try:
            return json.loads(TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_training_history(history: dict) -> None:
    TRAINING_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_existing_call_ids(agent_slug_str: str) -> set[str]:
    """Get call IDs already used for training this agent."""
    agents_path = VP_DIR / "agents.json"
    if not agents_path.exists():
        return set()
    try:
        agents = json.loads(agents_path.read_text(encoding="utf-8"))
        entry = agents.get(agent_slug_str, {})
        # Check per_call_snr for used call IDs
        used = set()
        for item in entry.get("per_call_snr", []):
            if isinstance(item, dict) and item.get("_id"):
                used.add(item["_id"])
        return used
    except Exception:
        return set()


def train_single_agent(
    agent_name: str,
    agent_slug_str: str,
    data_dir: Path,
    n_clusters: int,
    activate: bool,
    dry_run: bool,
    no_compare: bool = False,
) -> dict:
    """Run the training script for a single agent."""
    train_script = CALL_PROCESSOR_DIR / "scripts" / "train_agent_from_api_labels.py"
    report_out = DAILY_REPORTS_DIR / f"{agent_slug_str}.last_training_report.json"

    cmd = [
        sys.executable, str(train_script),
        "--agent-slug", agent_slug_str,
        "--agent-name", agent_name,
        "--data-dir", str(data_dir),
        "--clusters", str(n_clusters),
        "--min-activation-accuracy", str(MIN_ACTIVATION_ACCURACY),
        "--report-out", str(report_out),
    ]
    if no_compare:
        cmd.append("--no-compare")
    else:
        cmd.append("--compare-existing")
    if activate:
        cmd.append("--activate")
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n[train] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"  [train] FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return {"error": f"exit code {result.returncode}", "stderr": result.stderr[:500]}

    # Try to read the training report
    report_name = f"{agent_slug_str}_auto_training_report.json"
    if report_out.exists():
        try:
            return json.loads(report_out.read_text(encoding="utf-8"))
        except Exception:
            pass
    report_path = VP_DIR / report_name
    if report_path.exists() and not dry_run:
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {"status": "completed", "stdout": result.stdout[-500:]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily automated agent enrollment")
    parser.add_argument("--days", type=int, default=7, help="Fetch recordings from last N days")
    parser.add_argument("--start-time",
                        help="Exact API start_time ISO value, e.g. 2026-03-27T00:12:00.000Z")
    parser.add_argument("--end-time",
                        help="Exact API end_time ISO value, e.g. 2026-03-28T23:59:59.999Z")
    parser.add_argument("--user-name",
                        help="Pass user_name directly to the dataset API")
    parser.add_argument("--max-calls-total", type=int, default=500,
                        help="Max total calls to fetch from API")
    parser.add_argument("--max-calls-per-agent", type=int, default=20,
                        help="Max calls to use per agent for training")
    parser.add_argument("--min-calls", type=int, default=3,
                        help="Min calls required per agent to attempt training")
    parser.add_argument("--agents", nargs="*", default=None,
                        help="Only train these agents (substring match)")
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--activate", action="store_true",
                        help="Actually activate improved voiceprints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only, don't modify anything")
    parser.add_argument("--no-compare", action="store_true",
                        help="Disable comparison with existing voiceprints")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip API scrape, use existing data in work_dir")
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "traning_data" / "_daily_auto"),
                        help="Working directory for downloaded data")
    args = parser.parse_args()

    DAILY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print(f"DAILY AGENT TRAINING DAEMON - {today}")
    print(f"  days={args.days} min_calls={args.min_calls} "
          f"max_per_agent={args.max_calls_per_agent}")
    if args.start_time or args.end_time or args.user_name:
        print(f"  exact_window={args.start_time or '<auto>'} -> {args.end_time or '<auto>'} "
              f"user_name={args.user_name or '<all>'}")
    print(f"  activate={args.activate} dry_run={args.dry_run}")
    print("=" * 80)

    # ── Step 1: Scrape API ───────────────────────────────────────────────────
    if args.skip_scrape:
        print("\n[step 1] SKIPPED (--skip-scrape)")
        records = []
    else:
        print("\n[step 1] Scraping Audiofy API...")
        if bool(args.start_time) != bool(args.end_time):
            print("[error] --start-time and --end-time must be provided together", file=sys.stderr)
            return 1
        records = fetch_api_recordings(
            args.days,
            args.max_calls_total,
            start_time=args.start_time,
            end_time=args.end_time,
            user_name=args.user_name,
        )
        if LAST_SCRAPE_ERROR and not records:
            print(f"[error] Scrape failed: {LAST_SCRAPE_ERROR}", file=sys.stderr)
            return 2

    # ── Step 2: Group and filter ─────────────────────────────────────────────
    print("\n[step 2] Filtering and grouping by agent...")
    if not records and args.skip_scrape:
        # Use existing data dirs
        qualified = {}
        for d in sorted(work_dir.iterdir()):
            if d.is_dir() and list(d.glob("*/data.json")):
                agent_name_guess, agent_slug_guess = infer_local_agent_identity(
                    d, d.name.replace("_", " ").title()
                )
                qualified[agent_name_guess] = [{
                    "_local_dir": str(d),
                    "_agent_slug": agent_slug_guess,
                }]
    else:
        qualified = filter_and_group(records, args.min_calls)

    if args.agents:
        qualified = {
            name: calls for name, calls in qualified.items()
            if any(a.lower() in name.lower() for a in args.agents)
        }
        print(f"[filter] narrowed to {len(qualified)} agents matching: {args.agents}")

    if not qualified:
        print("[done] No agents qualified for training.")
        return 0

    # ── Step 3: Prepare data and train ───────────────────────────────────────
    history = load_training_history()
    daily_report: dict = {
        "date": today,
        "run_tag": run_tag,
        "days_scraped": args.days,
        "total_api_records": len(records),
        "qualified_agents": len(qualified),
        "results": {},
    }

    for agent_name, calls in sorted(qualified.items()):
        agent_slug_str = calls[0].get("_agent_slug") or slug(agent_name)
        print(f"\n{'='*60}")
        print(f"[agent] {agent_name} ({agent_slug_str}) - {len(calls)} calls")
        print(f"{'='*60}")

        # Check for new calls not yet used
        existing_ids = get_existing_call_ids(agent_slug_str)
        new_calls = [c for c in calls if c.get("_id") not in existing_ids]
        if not new_calls and not calls[0].get("_local_dir"):
            print(f"  [skip] No new calls since last training")
            daily_report["results"][agent_slug_str] = {
                "status": "skipped", "reason": "no new calls",
            }
            continue

        # Prepare training data
        if calls[0].get("_local_dir"):
            data_dir = Path(calls[0]["_local_dir"])
        else:
            print(f"  [prepare] Downloading and preparing {len(calls)} calls...")
            data_dir = prepare_training_data(
                agent_name, agent_slug_str, calls,
                args.max_calls_per_agent, work_dir,
            )

        # Train
        print(f"  [train] Starting CAM++ training...")
        result = train_single_agent(
            agent_name, agent_slug_str, data_dir, args.clusters,
            activate=args.activate, dry_run=args.dry_run,
            no_compare=args.no_compare,
        )

        # Update history
        agent_history = history.setdefault(agent_slug_str, {"agent_name": agent_name, "history": []})
        entry = {
            "date": today,
            "n_calls_used": len(calls),
            "activated": result.get("activated", False),
            "activation_eligible": result.get("activation_eligible"),
            "blocked_by_existing": result.get("blocked_by_existing", False),
        }

        # Extract metrics from training report
        artifacts = result.get("artifacts", {})
        same_data = result.get("same_data_accuracy", {})
        loco = result.get("loco_result", {})

        if artifacts:
            entry["mean_inside_sim"] = artifacts.get("mean_inside_sim")
            entry["max_outside_sim"] = artifacts.get("max_outside_sim")
        if same_data:
            entry["same_data_accuracy"] = same_data.get("overall_accuracy")
            entry["agent_accuracy"] = same_data.get("agent_accuracy")
            entry["customer_accuracy"] = same_data.get("customer_accuracy")
        if loco:
            entry["loco_accuracy"] = loco.get("overall_accuracy")
        entry["n_training_segments"] = result.get("training_rows", 0)

        # Track improvement over previous
        prev_entries = agent_history["history"]
        if prev_entries and entry.get("mean_inside_sim"):
            prev = prev_entries[-1]
            prev_inside = prev.get("mean_inside_sim") or 0
            prev_outside = prev.get("max_outside_sim") or 1
            curr_inside = entry.get("mean_inside_sim") or 0
            curr_outside = entry.get("max_outside_sim") or 1
            delta_inside = curr_inside - prev_inside
            delta_outside = curr_outside - prev_outside
            entry["improvement"] = (
                f"{'+'if delta_inside>=0 else ''}{delta_inside:.4f} inside, "
                f"{'+'if delta_outside>=0 else ''}{delta_outside:.4f} outside"
            )

        agent_history["history"].append(entry)
        daily_report["results"][agent_slug_str] = {
            "status": "activated" if entry.get("activated") else
                      "blocked" if entry.get("blocked_by_existing") else
                      "gated" if result.get("activation_eligible") is False else
                      "candidate" if not result.get("error") else "error",
            **entry,
        }

        status = "ACTIVATED" if entry.get("activated") else \
                 "BLOCKED (existing is better)" if entry.get("blocked_by_existing") else \
                 "CANDIDATE SAVED" if not result.get("error") else \
                 f"ERROR: {result.get('error', 'unknown')}"
        if (
            result.get("activation_eligible") is False
            and not entry.get("activated")
            and not entry.get("blocked_by_existing")
            and not result.get("error")
        ):
            status = "GATED (validation below activation threshold)"
        print(f"  [result] {status}")

    # ── Step 4: Save reports ─────────────────────────────────────────────────
    if not args.dry_run:
        save_training_history(history)
        report_path = DAILY_REPORTS_DIR / f"{today}_{run_tag}.json"
        report_path.write_text(
            json.dumps(daily_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[saved] history -> {TRAINING_HISTORY_PATH}")
        print(f"[saved] report  -> {report_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"DAILY TRAINING SUMMARY - {today}")
    print(f"{'='*80}")
    activated = sum(1 for r in daily_report["results"].values() if r.get("status") == "activated")
    blocked = sum(1 for r in daily_report["results"].values() if r.get("status") == "blocked")
    candidates = sum(1 for r in daily_report["results"].values() if r.get("status") == "candidate")
    gated = sum(1 for r in daily_report["results"].values() if r.get("status") == "gated")
    errors = sum(1 for r in daily_report["results"].values() if r.get("status") == "error")
    skipped = sum(1 for r in daily_report["results"].values() if r.get("status") == "skipped")

    print(f"  Activated:  {activated}")
    print(f"  Blocked:    {blocked} (existing voiceprint is better)")
    print(f"  Gated:      {gated} (validation below activation threshold)")
    print(f"  Candidates: {candidates} (saved but not activated)")
    print(f"  Errors:     {errors}")
    print(f"  Skipped:    {skipped} (no new calls)")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
