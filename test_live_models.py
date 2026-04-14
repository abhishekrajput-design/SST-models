"""
test_live_models.py — Test all working ASR models against the live server.

Usage:
    python test_live_models.py                          # all models, default audio
    python test_live_models.py --model whisper-large-v3-turbo
    python test_live_models.py --all-audio              # all models x 3 audio files
    python test_live_models.py --server http://13.42.127.218:8080
"""
import sys, os, time, json, argparse
import urllib.request, urllib.error

SERVER   = "http://13.42.127.218:8080"
RAW_DIR  = os.path.join(os.path.dirname(__file__), "call_processor", "data", "raw_calls")

# 6 working models (Qwen3 + VibeVoice disabled — transformers incompatibility)
MODELS = [
    "whisper-large-v3-turbo",
    "deepgram-nova-3",
    "deepgram-nova-2-phonecall",
    "deepgram-nova-2-meeting",
    "parakeet-tdt-0.6b-v3",
    "cohere-transcribe-03-2026",
]

# 3 representative audio files: short (~1.5 MB), medium (~8.6 MB), long (~11.3 MB)
AUDIO_FILES = {
    "short":  os.path.join(RAW_DIR, "audio_04_12_2026_11_56_45_xcj42i.mp3"),
    "medium": os.path.join(RAW_DIR, "audio_04_12_2026_10_38_45_ldwibu.mp3"),
    "long":   os.path.join(RAW_DIR, "audio_04_12_2026_12_28_59_vrcta2.mp3"),
}
DEFAULT_AUDIO = AUDIO_FILES["short"]

TIMEOUT_S = 900   # 15 min per model on CPU


def api(path):
    req = urllib.request.Request(f"{SERVER}{path}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_fetch_error": str(e)}


def wait_idle(max_wait=180):
    for _ in range(max_wait):
        s = api("/api/status")
        if not s.get("running"):
            return True
        time.sleep(2)
    return False


def upload(filename, model, audio_path):
    url = f"{SERVER}/api/upload?filename={filename}&model={model}"
    with open(audio_path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(url, data=data,
          headers={"Content-Type": "audio/mpeg"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def poll(timeout_s=TIMEOUT_S):
    t0 = time.time()
    last_stage = ""
    seen_running = False
    # Brief pause so server can transition from previous "done" state to "running"
    time.sleep(3)
    while time.time() - t0 < timeout_s:
        s = api("/api/status")
        stage = s.get("stage", "")
        msg   = s.get("message", "")
        if stage != last_stage:
            print(f"    {stage}: {msg}")
            last_stage = stage
        if s.get("running"):
            seen_running = True
        if s.get("done") and seen_running:
            return "PASS", s.get("result_id", "")
        if s.get("error"):
            return "FAIL", s["error"]
        if "_fetch_error" in s:
            return "FAIL", s["_fetch_error"]
        if not s.get("running") and seen_running and time.time() - t0 > 15:
            return "FAIL", "stopped unexpectedly"
        time.sleep(5)
    return "FAIL", f"timeout after {timeout_s}s"


def run_tests(models, audio_path, audio_label=""):
    label = f" [{audio_label}]" if audio_label else ""
    results = {}

    for model in models:
        print(f"\n{'='*50}")
        print(f"  Model: {model}{label}")
        print(f"{'='*50}")

        if not wait_idle(max_wait=60):
            results[model] = ("FAIL", "server busy after 2 min wait")
            print(f"  -> FAIL (server busy)")
            continue

        safe = model.replace('-','_').replace('.','_')
        tag  = audio_label.replace(' ','_') if audio_label else "default"
        fname = f"test_{safe}__{tag}.mp3"
        resp = upload(fname, model, audio_path)
        if "error" in resp:
            results[model] = ("FAIL", resp["error"])
            print(f"  -> FAIL (upload: {resp['error']})")
            continue
        print(f"  Upload: OK  ({fname})")

        status, detail = poll()
        results[model] = (status, detail)
        if status == "PASS":
            print(f"  -> PASS  result_id={detail}")
        else:
            short = detail[:120] + "..." if len(detail) > 120 else detail
            print(f"  -> FAIL: {short}")

    return results


def print_summary(all_results):
    total_pass = sum(1 for r in all_results.values() for s,_ in [r] if s == "PASS")
    total_fail = len(all_results) - total_pass
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  {total_pass} passed / {total_fail} failed")
    print(f"{'='*60}")
    for key, (status, detail) in all_results.items():
        icon  = "OK" if status == "PASS" else "!!"
        short = detail[:55] + "..." if len(detail) > 55 else detail
        print(f"  {icon}  {key:<45} {status if status == 'PASS' else short}")
    print(f"{'='*60}")


def main():
    global SERVER
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--model",  default=None, help="Test one model only")
    parser.add_argument("--all-audio", action="store_true",
                        help="Test all models against short + medium + long audio")
    args = parser.parse_args()
    SERVER = args.server

    s = api("/api/status")
    if "_fetch_error" in s:
        print(f"[ERROR] Server not reachable: {s['_fetch_error']}")
        sys.exit(1)
    print(f"Server: {SERVER}  (status: {s.get('stage','?')})\n")

    models = [args.model] if args.model else MODELS
    all_results = {}

    if args.all_audio:
        for label, path in AUDIO_FILES.items():
            if not os.path.isfile(path):
                print(f"[SKIP] {label}: {path} not found")
                continue
            size_kb = os.path.getsize(path) // 1024
            print(f"\n### Audio: {label} ({size_kb} KB) — {os.path.basename(path)}")
            r = run_tests(models, path, audio_label=label)
            for model, result in r.items():
                all_results[f"{model} [{label}]"] = result
    else:
        audio_path = DEFAULT_AUDIO
        if not os.path.isfile(audio_path):
            print(f"[ERROR] Default audio not found: {audio_path}")
            sys.exit(1)
        size_kb = os.path.getsize(audio_path) // 1024
        print(f"Audio: {os.path.basename(audio_path)} ({size_kb} KB)\n")
        r = run_tests(models, audio_path)
        all_results.update(r)

    print_summary(all_results)
    sys.exit(0 if all(s == "PASS" for s,_ in all_results.values()) else 1)


if __name__ == "__main__":
    main()
