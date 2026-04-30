"""Diagnose which customer GT segments get mis-labeled as AGENT and why."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR)); os.chdir(str(SCRIPT_DIR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FFMPEG = (r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
          r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe")

OMAR_DIR = SCRIPT_DIR / "data" / "audiofy" / "omar_dataset"
API = json.load(open(OMAR_DIR / "api_response.json", encoding="utf-8"))["data"]
GT_BY_ID = {c["_id"]: c for c in API}

CASES = [
    ("69c842366a2041f487a6b158", "call1_132s.mp3"),
    ("69c840836a2041f487a6ac20", "call2_88s.mp3"),
    ("69c83e186a2041f487a6a4be", "enroll1_149s.mp3"),
    ("69c839e46a2041f487a695f1", "enroll2_186s.mp3"),
]

def ts(s): p = s.split(":"); return int(p[0])*3600 + int(p[1])*60 + float(p[2])

# Reuse the diarised result.json files from previous runs
for case_id, audio in CASES:
    out_dir = SCRIPT_DIR / "data" / "processed" / f"_apicmp_{Path(audio).stem}"
    norm = str(out_dir / "norm.wav")
    if not os.path.exists(norm):
        continue

    # Re-run diar (fast, keeps embeddings) — capture per-segment sim
    fd, sp = tempfile.mkstemp(suffix=".py", dir=str(SCRIPT_DIR))
    os.close(fd)
    body = """
import json, os, sys
sys.path.insert(0, r'{root}')
os.chdir(r'{root}')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import logging; logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
from src.transcribers import get_transcriber
from src.diar_multi import diarize_multi
tr = get_transcriber('parakeet-tdt-0.6b-v3', device='cuda')
tr.load(); segs = tr.transcribe(sys.argv[1], language='en'); tr.unload()
out = diarize_multi(segs, sys.argv[1], force_cpu=True)
print('RESULT_START' + json.dumps(out, ensure_ascii=False, default=str) + 'RESULT_END')
""".replace("{root}", str(SCRIPT_DIR).replace("\\", "\\\\"))
    open(sp, "w", encoding="utf-8").write(body)
    print(f"\n=== {audio} ===", flush=True)
    r = subprocess.run([sys.executable, "-u", sp, norm], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300, cwd=str(SCRIPT_DIR))
    os.unlink(sp)
    if r.returncode != 0:
        print(f"  FAIL: {r.stderr[-300:]}")
        continue
    a, b = r.stdout.find("RESULT_START"), r.stdout.find("RESULT_END")
    diar = json.loads(r.stdout[a + len("RESULT_START"):b])
    our = diar.get("segments", [])

    # GT
    gt = GT_BY_ID[case_id]["speaker_json"]
    def gt_role(t):
        for s in gt:
            if ts(s["start"]) <= t <= ts(s["end"]):
                spk = s["speaker"]
                base = spk.split("_")[0] if "_" in spk and spk.split("_")[-1].isdigit() else spk
                return "CUSTOMER" if base.lower().startswith("customer") else "AGENT"
        return None

    # Find mis-labels: GT=CUSTOMER but our=AGENT
    misC = []  # (mid, sim, dur, our_label, text)
    misA = []
    for s in our:
        mid = (float(s["start"]) + float(s["end"])) / 2.0
        gtr = gt_role(mid)
        ours = "AGENT" if s.get("identified_speaker") == "AGENT" else "CUSTOMER"
        if gtr == "CUSTOMER" and ours == "AGENT":
            misC.append((s["start"], s["end"], s.get("_best_sim", 0), s.get("text", "")))
        elif gtr == "AGENT" and ours == "CUSTOMER":
            misA.append((s["start"], s["end"], s.get("_best_sim", 0), s.get("text", "")))

    print(f"  Customer GT mis-labeled as AGENT: {len(misC)}")
    for st, e, sim, txt in misC[:10]:
        print(f"    {float(st):6.1f}-{float(e):6.1f}s  sim={sim:.3f}  | {txt[:80]}")
    print(f"  Agent GT mis-labeled as CUSTOMER: {len(misA)}")
    for st, e, sim, txt in misA[:5]:
        print(f"    {float(st):6.1f}-{float(e):6.1f}s  sim={sim:.3f}  | {txt[:80]}")
