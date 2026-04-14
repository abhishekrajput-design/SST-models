"""
test_live_models.py — Test all 6 ASR models against the live server.

Usage:
    python test_live_models.py
    python test_live_models.py --server http://13.42.127.218:8080
    python test_live_models.py --model whisper-large-v3-turbo
"""
import sys, os, time, json, argparse
import urllib.request, urllib.error

SERVER  = "http://13.42.127.218:8080"
AUDIO   = os.path.join(os.path.dirname(__file__),
          "call_processor", "data", "raw_calls",
          "audio_04_12_2026_11_56_45_xcj42i.mp3")
MODELS  = [
    "whisper-large-v3-turbo",
    "deepgram-nova-3",
    "parakeet-tdt-0.6b-v3",
    "qwen3-asr-1.7b",
    "vibevoice-asr",
    "cohere-transcribe-03-2026",
]
TIMEOUT_S = 300   # 5 min per model on CPU


def api(path):
    req = urllib.request.Request(f"{SERVER}{path}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_fetch_error": str(e)}


def wait_idle(max_wait=60):
    """Return True as soon as server is not running (ignore error/done state)."""
    for _ in range(max_wait):
        s = api("/api/status")
        if not s.get("running"):
            return True
        time.sleep(2)
    return False


def upload(filename, model, audio_path):
    url  = f"{SERVER}/api/upload?filename={filename}&model={model}"
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
    while time.time() - t0 < timeout_s:
        s = api("/api/status")
        stage = s.get("stage", "")
        msg   = s.get("message", "")
        if stage != last_stage:
            print(f"    {stage}: {msg}")
            last_stage = stage
        if s.get("done"):
            return "PASS", s.get("result_id", "")
        if s.get("error"):           # null → falsy, string → truthy
            return "FAIL", s["error"]
        if "_fetch_error" in s:
            return "FAIL", s["_fetch_error"]
        if not s.get("running") and time.time() - t0 > 15:
            return "FAIL", "stopped unexpectedly"
        time.sleep(5)
    return "FAIL", f"timeout after {timeout_s}s"


def main():
    global SERVER
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--model", default=None, help="Test one model only")
    args = parser.parse_args()
    SERVER = args.server

    # Check server is reachable
    s = api("/api/status")
    if "_fetch_error" in s:
        print(f"[ERROR] Server not reachable: {s['_fetch_error']}")
        sys.exit(1)
    print(f"Server: {SERVER}  (status: {s.get('stage','?')})")
    print(f"Audio:  {AUDIO}  ({os.path.getsize(AUDIO)//1024} KB)\n")

    models = [args.model] if args.model else MODELS
    results = {}

    for model in models:
        print(f"{'='*50}")
        print(f"  Model: {model}")
        print(f"{'='*50}")

        # Wait for server to be idle
        if not wait_idle(max_wait=60):
            results[model] = ("FAIL", "server busy after 2 min wait")
            print(f"  -> FAIL (server busy)\n")
            continue

        # Upload
        fname = f"test_{model.replace('-','_').replace('.','_')}.mp3"
        resp = upload(fname, model, AUDIO)
        if "error" in resp:
            results[model] = ("FAIL", resp["error"])
            print(f"  -> FAIL (upload error: {resp['error']})\n")
            continue
        print(f"  Upload: OK  (file={fname})")

        # Poll
        status, detail = poll()
        results[model] = (status, detail)
        if status == "PASS":
            print(f"  -> PASS  result_id={detail}\n")
        else:
            # Truncate long error messages
            short = detail[:120] + "..." if len(detail) > 120 else detail
            print(f"  -> FAIL: {short}\n")

    # Summary table
    passed = sum(1 for s, _ in results.values() if s == "PASS")
    failed = len(results) - passed
    print(f"\n{'='*50}")
    print(f"  RESULTS  {passed} passed / {failed} failed")
    print(f"{'='*50}")
    for model in models:
        status, detail = results.get(model, ("?", "not run"))
        icon = "✓" if status == "PASS" else "✗"
        short = detail[:60] + "..." if len(detail) > 60 else detail
        print(f"  {icon}  {model:<35} {status if status == 'PASS' else short}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
