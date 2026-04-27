"""
enroll_all_from_api.py — Train clean ECAPA voiceprints for all agents.

Workflow:
  1. Load data/audiofy/_dataset/index.json (written by scrape_dataset_api.py)
  2. Group by agent_name; keep agents with >= --min-calls recordings
  3. Pre-download all needed MP3s in parallel
  4. For each agent:
       - Load each MP3 once (one ffmpeg call -> 16 kHz mono numpy array)
       - Slice agent-labeled phrases using speaker_json ground truth
       - Sliding ECAPA embeddings on GPU
       - Iterative tightening: drop windows whose cosine to the provisional
         centroid is below threshold; repeat once with a tighter threshold.
         This removes any customer contamination from the voiceprint.
  5. Save tight centroid -> data/agent_voiceprints/<slug>.npy and update agents.json

Usage:
  python enroll_all_from_api.py                           # all agents with >=5 calls
  python enroll_all_from_api.py --min-calls 3
  python enroll_all_from_api.py --agents "Rajan" "Haris"
  python enroll_all_from_api.py --keep-existing           # skip already-enrolled
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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

INDEX_PATH  = SCRIPT_DIR / "data" / "audiofy" / "_dataset" / "index.json"
AUDIO_DIR   = SCRIPT_DIR / "data" / "audiofy" / "_dataset" / "audio"
VP_DIR      = SCRIPT_DIR / "data" / "agent_voiceprints"
AGENTS_JSON = VP_DIR / "agents.json"

FFMPEG = (r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
          r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe")

TARGET_SR         = 16000
WINDOW_S          = 2.0
STRIDE_S          = 1.0
MIN_CLIP_S        = 0.5
MIN_AGENT_TOTAL_S = 60.0     # refuse agent with < 60 s of agent-labeled audio

# Iterative tightening thresholds
TIGHT_PASS_1 = 0.45
TIGHT_PASS_2 = 0.55
MIN_TIGHT_N  = 15


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


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


def download(url: str, dest: Path, connect_timeout: int = 10,
             read_timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        with requests.get(url, timeout=(connect_timeout, read_timeout),
                           stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        return dest.stat().st_size > 1000
    except Exception as e:
        print(f"    [download] {dest.name}: {e}", flush=True)
        try: dest.unlink()
        except (OSError, FileNotFoundError): pass
        return False


def download_batch(tasks, workers: int = 8) -> None:
    todo = [(u, d) for (u, d) in tasks
            if not (d.exists() and d.stat().st_size > 1000)]
    already = len(tasks) - len(todo)
    if already:
        print(f"    [download-batch] {already} already cached", flush=True)
    if not todo:
        return
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download, u, d): d for (u, d) in todo}
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 5 == 0 or done == len(todo):
                print(f"    [download-batch] {done}/{len(todo)} "
                      f"({time.time()-t0:.0f}s)", flush=True)


def load_mp3_mono_16k(mp3: Path):
    """Single ffmpeg call -> (audio np.ndarray float32, sr=16000)."""
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", str(mp3),
             "-ac", "1", "-ar", str(TARGET_SR), tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=True,
        )
        audio, sr = sf.read(tmp, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    finally:
        try: os.unlink(tmp)
        except OSError: pass


def extract_agent_samples(audio, sr: int, agent_segs):
    """Slice agent-labeled phrases from an in-memory mono array."""
    if not agent_segs:
        return None, 0.0
    parts = []
    total = 0.0
    for seg in agent_segs:
        s = ts2s(seg.get("start"))
        e = ts2s(seg.get("end"))
        if e - s < MIN_CLIP_S:
            continue
        si = int(s * sr); ei = min(int(e * sr), len(audio))
        if ei - si < int(sr * MIN_CLIP_S):
            continue
        parts.append(audio[si:ei])
        total += (ei - si) / sr
    if not parts:
        return None, 0.0
    return np.concatenate(parts), total


def extract_embeddings(samples, model):
    """Sliding-window speaker embeddings using EmbeddingModel (CAM++ or ECAPA fallback)."""
    if samples is None or len(samples) < int(TARGET_SR * WINDOW_S):
        return []
    window = int(TARGET_SR * WINDOW_S)
    stride = int(TARGET_SR * STRIDE_S)
    embs = []
    for start in range(0, len(samples) - window, stride):
        emb = model.embed_chunk(samples[start:start + window], TARGET_SR)
        if emb is not None:
            embs.append(emb)
    return embs


def _centroid(mat):
    c = np.mean(mat, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def iterative_tighten(X: np.ndarray):
    """Return (tight_mask, tight_centroid, inside_mean, outside_max)."""
    c = _centroid(X)
    sims = X @ c
    keep1 = sims >= TIGHT_PASS_1
    if int(keep1.sum()) < MIN_TIGHT_N:
        keep1 = np.ones(len(X), dtype=bool)
    c = _centroid(X[keep1])
    sims2 = X @ c
    keep2 = keep1 & (sims2 >= TIGHT_PASS_2)
    if int(keep2.sum()) < MIN_TIGHT_N:
        keep2 = keep1
    final = _centroid(X[keep2])
    n = np.linalg.norm(final)
    final = final / n if n > 0 else final
    inside  = float((X[keep2] @ final).mean()) if keep2.any() else 0.0
    outside = float((X[~keep2] @ final).max()) if (~keep2).any() else 0.0
    return keep2, final, inside, outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-calls", type=int, default=5)
    ap.add_argument("--agents", nargs="*", default=None)
    ap.add_argument("--max-calls-per-agent", type=int, default=5)
    ap.add_argument("--keep-existing", action="store_true")
    args = ap.parse_args()

    VP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_PATH.exists():
        sys.exit(f"[ERROR] {INDEX_PATH} not found — run scrape_dataset_api.py first")

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    by_agent: dict = {}
    for rec in index:
        by_agent.setdefault(rec["agent_name"], []).append(rec)

    selected: dict = {}
    for name, recs in by_agent.items():
        if args.agents and not any(a.lower() in name.lower() for a in args.agents):
            continue
        if len(recs) < args.min_calls:
            continue
        selected[name] = recs[: args.max_calls_per_agent]

    print(f"[enroll] {len(selected)} agents selected (min-calls={args.min_calls})",
          flush=True)
    for name, recs in sorted(selected.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(recs):3d}  {name}", flush=True)
    if not selected:
        sys.exit("[enroll] no agents matched")

    if AGENTS_JSON.exists():
        with open(AGENTS_JSON, encoding="utf-8") as f:
            agents = json.load(f)
    else:
        agents = {}

    # ── Pre-download all needed MP3s in parallel ─────────────────────────────
    tasks = []
    for name, recs in selected.items():
        if args.keep_existing and (VP_DIR / f"{slug(name)}.npy").exists():
            continue
        for rec in recs:
            tasks.append((rec["horizon_s3"], AUDIO_DIR / f"{rec['_id']}.mp3"))
    print(f"\n[enroll] downloading {len(tasks)} MP3s in parallel ...", flush=True)
    download_batch(tasks, workers=8)

    # ── Load embedding model (CAM++ preferred, ECAPA-TDNN fallback) ─────────
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
        print(f"[enroll] {model.model_name} ready (dim={model.dim})", flush=True)
    except Exception as e:
        print(f"[enroll] GPU load failed ({e}), using CPU", flush=True)
        model.load(force_cpu=True)
        print(f"[enroll] {model.model_name} ready on CPU (dim={model.dim})", flush=True)

    try:
        for name, recs in sorted(selected.items(), key=lambda kv: -len(kv[1])):
            agent_slug = slug(name)
            vp_path = VP_DIR / f"{agent_slug}.npy"
            if args.keep_existing and vp_path.exists():
                print(f"\n[enroll] {name}: voiceprint exists, skipping", flush=True)
                continue

            print(f"\n[enroll] === {name}  ({len(recs)} calls) ===", flush=True)
            all_embs = []
            total_agent_s = 0.0
            used_calls = 0

            for rec in recs:
                rid = rec["_id"]
                mp3 = AUDIO_DIR / f"{rid}.mp3"
                if not (mp3.exists() and mp3.stat().st_size > 1000):
                    print(f"  [skip {rid[:8]}] no MP3", flush=True)
                    continue

                agent_segs = [s for s in rec["speaker_json"]
                              if isinstance(s, dict)
                              and s.get("speaker") and s["speaker"] != "Customer"]
                if not agent_segs:
                    print(f"  [skip {rid[:8]}] no agent phrases", flush=True)
                    continue

                t0 = time.time()
                try:
                    audio, sr = load_mp3_mono_16k(mp3)
                except Exception as e:
                    print(f"  [skip {rid[:8]}] decode failed: {e}", flush=True)
                    continue

                samples, dur = extract_agent_samples(audio, sr, agent_segs)
                if dur < 3.0:
                    print(f"  [skip {rid[:8]}] only {dur:.1f}s agent audio",
                          flush=True)
                    continue

                embs = extract_embeddings(samples, model)
                if len(embs) < 3:
                    print(f"  [skip {rid[:8]}] only {len(embs)} embs",
                          flush=True)
                    continue
                all_embs.extend(embs)
                total_agent_s += dur
                used_calls += 1
                print(f"  [ok {rid[:8]}] {dur:.0f}s agent -> {len(embs)} embs  "
                      f"({time.time()-t0:.1f}s)  "
                      f"[running: {total_agent_s:.0f}s / {len(all_embs)} embs]",
                      flush=True)

            if len(all_embs) < 30 or total_agent_s < MIN_AGENT_TOTAL_S:
                print(f"  [enroll] {name}: only {len(all_embs)} embs, "
                      f"{total_agent_s:.0f}s — INSUFFICIENT, skipping",
                      flush=True)
                continue

            # L2-normalize + iterative tightening
            X = np.stack(all_embs).astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1
            X = X / norms
            keep, final, inside, outside = iterative_tighten(X)
            n_kept = int(keep.sum())

            np.save(vp_path, final)
            agents[agent_slug] = {
                "agent_name":      name,
                "voiceprint_path": str(vp_path),
                "n_clips":         n_kept,
                "n_windows_total": int(len(X)),
                "total_seconds":   round(total_agent_s, 1),
                "used_calls":      used_calls,
                "mean_inside_sim": round(inside, 3),
                "max_outside_sim": round(outside, 3),
                "source":          "audiofy_api_bulk_tight_20260424",
            }
            print(f"  [enroll] {name}: saved (tight {n_kept}/{len(X)}, "
                  f"inside={inside:.3f}, outside_max={outside:.3f}, "
                  f"{total_agent_s:.0f}s)", flush=True)

            # Checkpoint
            with open(AGENTS_JSON, "w", encoding="utf-8") as f:
                json.dump(agents, f, ensure_ascii=False, indent=2)
    finally:
        model.unload()

    print(f"\n[enroll] DONE. agents.json now has {len(agents)} entries:",
          flush=True)
    for k, v in agents.items():
        print(f"  {k}: {v['agent_name']}  {v.get('total_seconds',0)}s / "
              f"{v.get('n_clips', 0)} embs  "
              f"(inside_sim={v.get('mean_inside_sim', 'n/a')})", flush=True)


if __name__ == "__main__":
    main()
