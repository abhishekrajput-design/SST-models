"""A/B identification test for Amandeep Nandra.

1. Pull a fresh desk recording (outside recent training window)
2. Snapshot baseline -> run pipeline with current production voiceprint -> score
3. Activate the new CAM++ voiceprint via /api/auto-train (activate=true)
4. Re-run pipeline on a copy of the same audio -> score
5. Print before/after diff

Notes:
- The daily training daemon auto-creates a timestamped backup of agents.json
  before activation, so the change is reversible.
- We use two filenames (_pre / _post) so the second run does not overwrite the
  first result.json.
"""
from __future__ import annotations

import json
import os
import shutil
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

AGENT = "Amandeep Nandra"
SLUG = "amandeep_nandra"
UI = "http://localhost:8080"
UI_AUTH = ("abhishek", "123456")
AGENTS_JSON = REPO / "call_processor" / "data" / "agent_voiceprints" / "agents.json"


def overlap(a_s, a_e, b_s, b_e):
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))


def score(result, sj):
    produced = result.get("segments") or []
    a_t = a_c = c_t = c_c = 0
    for g in sj:
        role = dtd.speaker_role_from_api_label(str(g.get("speaker") or ""), AGENT)
        if role not in ("agent", "customer"):
            continue
        gs = float(dtd.ts2s(g.get("start") or 0))
        ge = float(dtd.ts2s(g.get("end") or 0))
        if ge <= gs:
            continue
        best, best_ov = None, 0.0
        for p in produced:
            ov = overlap(gs, ge, float(p.get("start") or 0), float(p.get("end") or 0))
            if ov > best_ov:
                best_ov, best = ov, p
        if best is None:
            continue
        pred_raw = str(best.get("identified_speaker") or best.get("speaker") or "").lower()
        pred = "agent" if "agent" in pred_raw else ("customer" if "customer" in pred_raw else "unknown")
        if role == "agent":
            a_t += 1
            a_c += int(pred == "agent")
        else:
            c_t += 1
            c_c += int(pred == "customer")
    return {
        "agent_total": a_t, "agent_correct": a_c,
        "agent_acc": round(a_c / a_t * 100, 2) if a_t else 0.0,
        "customer_total": c_t, "customer_correct": c_c,
        "customer_acc": round(c_c / c_t * 100, 2) if c_t else 0.0,
        "overall_total": a_t + c_t, "overall_correct": a_c + c_c,
        "overall_acc": round((a_c + c_c) / max(1, a_t + c_t) * 100, 2),
    }


def upload_and_wait(local: Path, label: str):
    fname = local.name
    print(f"\n[{label}] uploading {fname}")
    body = local.read_bytes()
    r = requests.post(
        f"{UI}/api/upload",
        params={"filename": fname, "model": "parakeet-tdt-0.6b-v3", "agent_slug": SLUG},
        data=body, auth=UI_AUTH, timeout=60,
        headers={"Content-Type": "audio/mpeg"},
    )
    r.raise_for_status()
    print(f"[{label}] {r.json()}")

    started = time.time()
    last_stage = ""
    while True:
        s = requests.get(f"{UI}/api/status", auth=UI_AUTH, timeout=10).json()
        stage = f"{s.get('stage_num')}.{s.get('stage')}"
        if stage != last_stage:
            elapsed = round(time.time() - started, 1)
            print(f"  [+{elapsed}s] {stage}: {s.get('message', '')[:80]}")
            last_stage = stage
        if s.get("done"):
            rid = s.get("result_id")
            print(f"  [done] {round(time.time() - started, 1)}s  result_id={rid}")
            r = requests.get(f"{UI}/api/call/{rid}", auth=UI_AUTH, timeout=30)
            r.raise_for_status()
            return r.json()
        if s.get("error"):
            raise RuntimeError(f"pipeline error: {s.get('error')}")
        time.sleep(3)


def activate_amandeep():
    print(f"\n[activate] POST /api/auto-train activate=true")
    r = requests.post(
        f"{UI}/api/auto-train",
        json={
            "agents": [AGENT], "days": 60, "dry_run": False, "activate": True,
            "audiofy_username": "abhishek", "audiofy_password": "123456",
        },
        auth=UI_AUTH, timeout=30,
    )
    r.raise_for_status()
    print(f"[activate] kicked: {r.json()}")
    started = time.time()
    last_msg = ""
    while True:
        s = requests.get(f"{UI}/api/auto-train-status", auth=UI_AUTH, timeout=10).json()
        msg = s.get("message", "")[:80]
        if msg != last_msg:
            elapsed = round(time.time() - started, 1)
            print(f"  [+{elapsed}s] {msg}")
            last_msg = msg
        if not s.get("running"):
            print(f"  [done] {round(time.time() - started, 1)}s  exit={s.get('exit_code')}")
            return
        time.sleep(5)


def main():
    # 1. Pull a fresh Amandeep recording from 30-60 days ago
    end = datetime.now(timezone.utc) - timedelta(days=30)
    start = end - timedelta(days=30)
    print(f"[scrape] window {start.isoformat()} -> {end.isoformat()}")
    recs = dtd.fetch_api_recordings(
        days=30, max_calls=30,
        start_time=start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        end_time=end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        user_name=AGENT,
    )
    if not recs:
        raise SystemExit("no recordings")
    # Pick one with a mix of agent + customer, moderate length
    def quality_key(r):
        sj = r.get("speaker_json") or []
        a = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), AGENT) == "agent")
        c = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), AGENT) == "customer")
        return (a > 5 and c > 5, min(a, c), abs(len(sj) - 100))
    recs.sort(key=quality_key, reverse=True)
    pick = recs[0]
    cid = str(pick.get("_id") or "")[:12]
    sj = pick.get("speaker_json") or []
    a_n = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), AGENT) == "agent")
    c_n = sum(1 for s in sj if dtd.speaker_role_from_api_label(str(s.get("speaker") or ""), AGENT) == "customer")
    print(f"[pick] {cid}  segments={len(sj)}  (agent {a_n}, customer {c_n})  duration={pick.get('duration', '?')}s")

    # 2. Download to disk
    raw_dir = REPO / "call_processor" / "data" / "raw_calls"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pre_file = raw_dir / f"abtest_amandeep_{cid}_pre.mp3"
    post_file = raw_dir / f"abtest_amandeep_{cid}_post.mp3"
    print(f"[download] -> {pre_file.name}")
    with requests.get(pick["horizon_call_s3_url"], stream=True, timeout=300) as g:
        g.raise_for_status()
        pre_file.write_bytes(g.content)
    print(f"[download] {pre_file.stat().st_size:,} bytes")
    shutil.copy2(pre_file, post_file)

    # 3. Snapshot agents.json BEFORE
    agents_pre = json.loads(AGENTS_JSON.read_text(encoding="utf-8"))
    amandeep_pre = agents_pre.get(SLUG, {})
    print(f"\n[before] Amandeep in agents.json:")
    print(f"  model={amandeep_pre.get('embedding_model')}  n_vp={amandeep_pre.get('n_voiceprints')}  inside={amandeep_pre.get('mean_inside_sim')}  outside={amandeep_pre.get('max_outside_sim')}")

    # 4. Run PRE pipeline
    res_pre = upload_and_wait(pre_file, "PRE")
    pre_score = score(res_pre, sj)
    print(f"[PRE]  identified_agent={res_pre.get('identified_agent')!r}")
    print(f"[PRE]  agent={pre_score['agent_correct']}/{pre_score['agent_total']} ({pre_score['agent_acc']}%)  customer={pre_score['customer_correct']}/{pre_score['customer_total']} ({pre_score['customer_acc']}%)  overall={pre_score['overall_acc']}%")

    # 5. Activate the new CAM++ voiceprint
    activate_amandeep()

    # 6. Verify agents.json changed
    agents_post = json.loads(AGENTS_JSON.read_text(encoding="utf-8"))
    amandeep_post = agents_post.get(SLUG, {})
    print(f"\n[after] Amandeep in agents.json:")
    print(f"  model={amandeep_post.get('embedding_model')}  n_vp={amandeep_post.get('n_voiceprints')}  inside={amandeep_post.get('mean_inside_sim')}  outside={amandeep_post.get('max_outside_sim')}")

    # 7. Run POST pipeline on the same audio
    res_post = upload_and_wait(post_file, "POST")
    post_score = score(res_post, sj)
    print(f"[POST] identified_agent={res_post.get('identified_agent')!r}")
    print(f"[POST] agent={post_score['agent_correct']}/{post_score['agent_total']} ({post_score['agent_acc']}%)  customer={post_score['customer_correct']}/{post_score['customer_total']} ({post_score['customer_acc']}%)  overall={post_score['overall_acc']}%")

    # 8. Summary
    print()
    print("=" * 70)
    print(f"A/B TEST RESULT — Amandeep Nandra on unseen desk recording {cid}")
    print("=" * 70)
    print(f"{'Metric':25s} {'BEFORE':>15s} {'AFTER':>15s} {'Δ':>10s}")
    print(f"{'voiceprint model':25s} {amandeep_pre.get('embedding_model','?'):>15s} {amandeep_post.get('embedding_model','?'):>15s}")
    print(f"{'n_voiceprints':25s} {amandeep_pre.get('n_voiceprints','?'):>15} {amandeep_post.get('n_voiceprints','?'):>15}")
    print(f"{'mean_inside_sim':25s} {amandeep_pre.get('mean_inside_sim','?'):>15} {amandeep_post.get('mean_inside_sim','?'):>15}")
    print(f"{'max_outside_sim':25s} {amandeep_pre.get('max_outside_sim','?'):>15} {amandeep_post.get('max_outside_sim','?'):>15}")
    print(f"{'identified_agent':25s} {str(res_pre.get('identified_agent'))[:15]:>15s} {str(res_post.get('identified_agent'))[:15]:>15s}")
    print(f"{'agent_acc':25s} {pre_score['agent_acc']:>14.2f}% {post_score['agent_acc']:>14.2f}% {post_score['agent_acc']-pre_score['agent_acc']:>+10.2f}")
    print(f"{'customer_acc':25s} {pre_score['customer_acc']:>14.2f}% {post_score['customer_acc']:>14.2f}% {post_score['customer_acc']-pre_score['customer_acc']:>+10.2f}")
    print(f"{'overall_acc':25s} {pre_score['overall_acc']:>14.2f}% {post_score['overall_acc']:>14.2f}% {post_score['overall_acc']-pre_score['overall_acc']:>+10.2f}")


if __name__ == "__main__":
    main()
