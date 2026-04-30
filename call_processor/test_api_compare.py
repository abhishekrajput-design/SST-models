"""
test_api_compare.py — Validate our pipeline against Audiofy API ground truth.

For each call in omar_dataset, run Parakeet + diar_multi and compare:
  1. Per-timestamp speaker label (Agent vs Customer) → identification accuracy
  2. Transcript text overlap → WER vs API phrases

Outputs side-by-side numbers, no fluff.
"""
from __future__ import annotations
import gc, json, os, subprocess, sys, tempfile, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(str(SCRIPT_DIR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FFMPEG = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

OMAR_DIR = SCRIPT_DIR / "data" / "audiofy" / "omar_dataset"
API_PATH = OMAR_DIR / "api_response.json"

# Map call_id → local audio file
TEST_CASES = [
    {"id": "69c842366a2041f487a6b158", "audio": "call1_132s.mp3",   "dur": 132},
    {"id": "69c840836a2041f487a6ac20", "audio": "call2_88s.mp3",    "dur": 88},
    {"id": "69c83e186a2041f487a6a4be", "audio": "enroll1_149s.mp3", "dur": 149},
    {"id": "69c839e46a2041f487a695f1", "audio": "enroll2_186s.mp3", "dur": 186},
]

# ── Subprocess-isolated transcribe + diarize ──────────────────────────────────
_TRANSCRIBE = r"""
import gc, json, os, sys
sys.path.insert(0, r"{root}")
os.chdir(r"{root}")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.transcribers import get_transcriber
tr = get_transcriber("parakeet-tdt-0.6b-v3", device="cuda")
tr.load()
segs = tr.transcribe(sys.argv[1], language="en")
tr.unload(); gc.collect()
print("RESULT_START" + json.dumps(segs, ensure_ascii=False) + "RESULT_END")
sys.stdout.flush()
"""

_DIARIZE = r"""
import json, logging, os, sys
sys.path.insert(0, r"{root}")
os.chdir(r"{root}")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
from src.diar_multi import diarize_multi
with open(sys.argv[2], encoding="utf-8") as f:
    segs = json.load(f)
out = diarize_multi(segs, sys.argv[1], force_cpu=True)
print("RESULT_START" + json.dumps(out, ensure_ascii=False, default=str) + "RESULT_END")
sys.stdout.flush()
"""


def _write_tmp(body: str, suffix: str) -> str:
    body = body.replace("{root}", str(SCRIPT_DIR).replace("\\", "\\\\"))
    fd, path = tempfile.mkstemp(suffix=suffix, dir=str(SCRIPT_DIR))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def _extract(stdout: str) -> str:
    a = stdout.find("RESULT_START")
    b = stdout.find("RESULT_END")
    if a < 0 or b < 0:
        return ""
    return stdout[a + len("RESULT_START"):b]


def transcribe(audio: str) -> list[dict]:
    sp = _write_tmp(_TRANSCRIBE, "_tr.py")
    r = subprocess.run([sys.executable, "-u", sp, audio],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600, cwd=str(SCRIPT_DIR))
    os.unlink(sp)
    if r.returncode != 0:
        print(f"[ERR] transcribe failed: {r.stderr[-500:]}")
        return []
    return json.loads(_extract(r.stdout) or "[]")


def diarize(norm_wav: str, segs: list[dict]) -> dict:
    fd, sj = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(segs, f)
    sp = _write_tmp(_DIARIZE, "_di.py")
    r = subprocess.run([sys.executable, "-u", sp, norm_wav, sj],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600, cwd=str(SCRIPT_DIR))
    os.unlink(sp); os.unlink(sj)
    if r.returncode != 0:
        print(f"[ERR] diarize failed: {r.stderr[-500:]}")
        return {}
    return json.loads(_extract(r.stdout) or "{}")


def normalise_audio(src: str, dst: str) -> None:
    subprocess.run([FFMPEG, "-y", "-i", src,
                    "-ar", "16000", "-ac", "1",
                    "-af", "aformat=channel_layouts=mono,aresample=44100,loudnorm=I=-16:TP=-1.5:LRA=11,dynaudnorm=p=0.9:m=100:s=5",
                    dst],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


# ── Time helpers ──────────────────────────────────────────────────────────────

def ts_to_sec(ts: str) -> float:
    p = ts.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])


def label_at(t: float, segs: list[dict], key: str) -> str:
    """Return label of seg containing time t (or nearest)."""
    for s in segs:
        if s["_s"] <= t <= s["_e"]:
            return s[key]
    # nearest fallback
    return min(segs, key=lambda s: min(abs(s["_s"] - t), abs(s["_e"] - t)))[key]


# ── WER ────────────────────────────────────────────────────────────────────────
import re as _re

def _words(s: str) -> list[str]:
    return _re.findall(r"[a-z0-9']+", (s or "").lower())


def _wer(ref: str, hyp: str) -> float:
    a = _words(ref); b = _words(hyp)
    if not a:
        return 0.0 if not b else 1.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m] / n


# ── Compare one call ──────────────────────────────────────────────────────────

def compare_call(case: dict, gt_call: dict) -> dict:
    name = case["audio"]
    print(f"\n  [{name}] dur={case['dur']}s  GT segs={len(gt_call['speaker_json'])}")

    src = str(OMAR_DIR / name)
    out_dir = SCRIPT_DIR / "data" / "processed" / f"_apicmp_{Path(name).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    norm = str(out_dir / "norm.wav")

    print(f"    1) normalising audio …", flush=True)
    normalise_audio(src, norm)

    print(f"    2) Parakeet transcription …", flush=True)
    t0 = time.time()
    our_segs = transcribe(norm)
    t_tr = time.time() - t0
    print(f"       {len(our_segs)} segments in {t_tr:.0f}s", flush=True)

    if not our_segs:
        return {"id": case["id"], "name": name, "skipped": True}

    print(f"    3) diar_multi voiceprint matching …", flush=True)
    t0 = time.time()
    diar = diarize(norm, our_segs)
    t_d = time.time() - t0
    our_segs = diar.get("segments", our_segs)
    print(f"       agent={diar.get('agent_name')}  speaker_mode={diar.get('speaker_mode')}  in {t_d:.0f}s", flush=True)

    # Prepare for comparison
    for s in our_segs:
        s["_s"] = float(s["start"])
        s["_e"] = float(s["end"])
        s["_role"] = "AGENT" if s.get("identified_speaker") == "AGENT" else "CUSTOMER"
    gt_segs = []
    for s in gt_call["speaker_json"]:
        spk = s["speaker"] or ""
        # Audiofy labels are inconsistent: "Customer", "Customer_1", "Customer_2",
        # "Omar El Harchaoui", "Omar El Harchaoui_3" — strip trailing "_N" before
        # checking for the "Customer" prefix.
        spk_base = spk.split("_")[0] if "_" in spk and spk.split("_")[-1].isdigit() else spk
        is_customer = spk_base.lower().startswith("customer")
        gt_segs.append({
            "_s": ts_to_sec(s["start"]),
            "_e": ts_to_sec(s["end"]),
            "_role": "CUSTOMER" if is_customer else "AGENT",
            "phrase": s.get("phrase", ""),
        })

    # Identification accuracy: tick every 0.5s, compare GT role vs our role
    correct = total = 0
    agent_correct = customer_correct = 0
    agent_total = customer_total = 0
    t = 0.0
    end_t = case["dur"]
    while t < end_t:
        # find GT label for this tick
        gt = next((g for g in gt_segs if g["_s"] <= t <= g["_e"]), None)
        if gt is None:
            t += 0.5
            continue
        ours = next((o for o in our_segs if o["_s"] <= t <= o["_e"]), None)
        if ours is None:
            t += 0.5
            continue
        total += 1
        if gt["_role"] == "AGENT":
            agent_total += 1
        else:
            customer_total += 1
        if gt["_role"] == ours["_role"]:
            correct += 1
            if gt["_role"] == "AGENT":
                agent_correct += 1
            else:
                customer_correct += 1
        t += 0.5

    # WER per role: collect all GT agent speech, our agent speech; same for customer
    gt_agent_text = " ".join(g["phrase"] for g in gt_segs if g["_role"] == "AGENT")
    gt_cust_text  = " ".join(g["phrase"] for g in gt_segs if g["_role"] == "CUSTOMER")
    our_agent_text = " ".join(s.get("text", "") for s in our_segs if s["_role"] == "AGENT")
    our_cust_text  = " ".join(s.get("text", "") for s in our_segs if s["_role"] == "CUSTOMER")
    full_gt   = " ".join(g["phrase"] for g in gt_segs)
    full_ours = " ".join(s.get("text", "") for s in our_segs)

    return {
        "id": case["id"],
        "name": name,
        "duration": case["dur"],
        "gt_segs": len(gt_segs),
        "our_segs": len(our_segs),
        "transcribe_s": round(t_tr, 1),
        "diarize_s": round(t_d, 1),
        "agent_identified": diar.get("agent_name"),
        "speaker_mode": diar.get("speaker_mode"),
        "id_acc_overall": round(100 * correct / max(total, 1), 1),
        "id_acc_agent":   round(100 * agent_correct / max(agent_total, 1), 1),
        "id_acc_customer": round(100 * customer_correct / max(customer_total, 1), 1),
        "wer_full":   round(100 * _wer(full_gt, full_ours), 1),
        "wer_agent":  round(100 * _wer(gt_agent_text, our_agent_text), 1),
        "wer_customer": round(100 * _wer(gt_cust_text, our_cust_text), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api = json.load(open(API_PATH, encoding="utf-8"))
    gt_by_id = {c["_id"]: c for c in api["data"]}

    results = []
    for case in TEST_CASES:
        if not (OMAR_DIR / case["audio"]).exists():
            print(f"  [skip] missing {case['audio']}")
            continue
        gt = gt_by_id.get(case["id"])
        if not gt:
            print(f"  [skip] no GT for {case['id']}")
            continue
        try:
            r = compare_call(case, gt)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [err] {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  AUDIOFY API vs OUR PIPELINE — REAL DATA COMPARISON")
    print("═" * 90)
    print(f"  {'audio':24} {'dur':>4} {'GT segs':>7} {'our':>5} {'ID%':>5} {'A%':>5} {'C%':>5} {'WER%':>5} {'A-WER':>5} {'C-WER':>5}")
    print("  " + "─" * 88)
    for r in results:
        if r.get("skipped"):
            continue
        print(f"  {r['name']:24} {r['duration']:>4}s {r['gt_segs']:>7} {r['our_segs']:>5} "
              f"{r['id_acc_overall']:>5.1f} {r['id_acc_agent']:>5.1f} {r['id_acc_customer']:>5.1f} "
              f"{r['wer_full']:>5.1f} {r['wer_agent']:>5.1f} {r['wer_customer']:>5.1f}")
    print()

    # Macro averages
    if results:
        valid = [r for r in results if not r.get("skipped")]
        if valid:
            avg = lambda k: round(sum(r[k] for r in valid) / len(valid), 1)
            print("  ── Macro averages ──")
            print(f"    Overall identification accuracy : {avg('id_acc_overall')}%")
            print(f"    Agent identification accuracy   : {avg('id_acc_agent')}%")
            print(f"    Customer identification accuracy: {avg('id_acc_customer')}%")
            print(f"    Overall WER vs API              : {avg('wer_full')}%")
            print(f"    Agent-text WER                  : {avg('wer_agent')}%")
            print(f"    Customer-text WER               : {avg('wer_customer')}%")

    # Save JSON
    out_path = SCRIPT_DIR / "data" / "processed" / "_apicmp_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
