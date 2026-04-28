"""
enroll_from_desk.py — Enroll an agent from clean desk recordings (S3 URLs).

Desk recordings are full-capture audio where the agent's voice dominates.
No speaker_json needed — we process the entire recording.

Usage:
  python enroll_from_desk.py --json Agents-recoding/zak.json   --name "Zak"  --slug zak_raissi_barnet
  python enroll_from_desk.py --json Agents-recoding/hussien.json --name "Hussein Mohamed" --slug hussein_mohamed
  python enroll_from_desk.py --json Agents-recoding/zak.json   --name "Zak"  --slug zak_raissi_barnet --max-files 5
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
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(str(SCRIPT_DIR))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FFMPEG = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

VP_DIR      = SCRIPT_DIR / "data" / "agent_voiceprints"
AGENTS_JSON = VP_DIR / "agents.json"
CACHE_DIR   = SCRIPT_DIR / "data" / "desk_recordings_cache"

TARGET_SR = 16000
WINDOW_S  = 2.0
STRIDE_S  = 1.0

TIGHT_PASS_1 = 0.35   # softer than phone calls — desk audio is cleaner
TIGHT_PASS_2 = 0.45
MIN_TIGHT_N  = 20


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        with requests.get(url, timeout=(10, 120), stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
        return dest.stat().st_size > 1000
    except Exception as e:
        print(f"  [download] {dest.name}: {e}", flush=True)
        try:
            dest.unlink()
        except OSError:
            pass
        return False


def download_batch(tasks: list, workers: int = 6) -> None:
    todo = [(u, d) for u, d in tasks if not (d.exists() and d.stat().st_size > 1000)]
    already = len(tasks) - len(todo)
    if already:
        print(f"  {already} already cached", flush=True)
    if not todo:
        return
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download, u, d): d for u, d in todo}
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            print(f"  downloaded {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)


def load_mp3_16k_mono(mp3: Path) -> np.ndarray:
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", str(mp3), "-ac", "1", "-ar", str(TARGET_SR), tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120, check=True,
        )
        audio, _ = sf.read(tmp, dtype="float32")
        return audio[:, 0] if audio.ndim > 1 else audio
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _centroid(mat: np.ndarray) -> np.ndarray:
    c = np.mean(mat, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def iterative_tighten(X: np.ndarray):
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
    inside  = float((X[keep2] @ final).mean())  if keep2.any()  else 0.0
    outside = float((X[~keep2] @ final).max()) if (~keep2).any() else 0.0
    return keep2, final, inside, outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",      required=True,  help="Path to JSON with s3_url list")
    ap.add_argument("--name",      required=True,  help="Agent display name")
    ap.add_argument("--slug",      default=None,   help="Slug override (default: auto from name)")
    ap.add_argument("--max-files", type=int, default=10)
    args = ap.parse_args()

    agent_slug = args.slug or slug(args.name)
    json_path  = Path(args.json)
    if not json_path.is_absolute():
        json_path = SCRIPT_DIR.parent / json_path  # relative to project root

    print(f"\n{'='*60}")
    print(f"  DESK ENROLL: {args.name}  (slug={agent_slug})")
    print(f"  Source: {json_path}")
    print(f"{'='*60}\n")

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    urls = [r["s3_url"] for r in records if "s3_url" in r][: args.max_files]
    print(f"[enroll] {len(urls)} recordings to process", flush=True)

    # Download
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VP_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, url in enumerate(urls):
        fname = url.split("/")[-1]
        dest  = CACHE_DIR / f"{agent_slug}_{i:02d}_{fname}"
        tasks.append((url, dest))

    print(f"\n[enroll] Downloading {len(tasks)} MP3s ...", flush=True)
    download_batch(tasks)

    # Load embedding model
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    model.load(force_cpu=False)
    print(f"\n[enroll] {model.model_name} ready  dim={model.dim}", flush=True)

    window  = int(TARGET_SR * WINDOW_S)
    stride  = int(TARGET_SR * STRIDE_S)
    all_embs: list = []
    total_s  = 0.0

    try:
        for url, dest in tasks:
            if not dest.exists() or dest.stat().st_size < 1000:
                print(f"  [skip] {dest.name}: not downloaded", flush=True)
                continue

            t0 = time.time()
            try:
                audio = load_mp3_16k_mono(dest)
            except Exception as e:
                print(f"  [skip] {dest.name}: decode failed: {e}", flush=True)
                continue

            dur = len(audio) / TARGET_SR
            file_embs = []
            for start in range(0, len(audio) - window, stride):
                emb = model.embed_chunk(audio[start: start + window], TARGET_SR)
                if emb is not None:
                    file_embs.append(emb)

            all_embs.extend(file_embs)
            total_s += dur
            print(f"  [ok] {dest.name}: {dur:.0f}s → {len(file_embs)} embs  "
                  f"({time.time()-t0:.1f}s)  [total: {total_s:.0f}s / {len(all_embs)} embs]",
                  flush=True)
    finally:
        model.unload()

    print(f"\n[enroll] Total: {len(all_embs)} embedding windows from {total_s:.0f}s audio", flush=True)

    if len(all_embs) < 30:
        print("[enroll] ERROR: not enough embeddings (<30) — aborting", flush=True)
        sys.exit(1)

    # L2-normalise + iterative tightening
    X = np.stack(all_embs).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms

    keep, final, inside, outside = iterative_tighten(X)
    n_kept = int(keep.sum())

    print(f"[enroll] Tightening: kept {n_kept}/{len(X)} windows  "
          f"inside_sim={inside:.3f}  outside_max={outside:.3f}", flush=True)

    vp_path = VP_DIR / f"{agent_slug}.npy"
    np.save(vp_path, final)
    print(f"[enroll] Saved -> {vp_path}  shape={final.shape}", flush=True)

    # Update agents.json
    if AGENTS_JSON.exists():
        with open(AGENTS_JSON, encoding="utf-8") as f:
            agents = json.load(f)
    else:
        agents = {}

    agents[agent_slug] = {
        "agent_name":      args.name,
        "voiceprint_path": str(vp_path),
        "n_clips":         n_kept,
        "n_windows_total": int(len(X)),
        "total_seconds":   round(total_s, 1),
        "source":          "desk_recordings_campp",
        "mean_inside_sim": round(inside, 3),
        "max_outside_sim": round(outside, 3),
    }

    with open(AGENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(agents, f, ensure_ascii=False, indent=2)

    print(f"\n[enroll] DONE — {args.name}  shape={final.shape}  "
          f"inside_sim={inside:.3f}  agents.json updated", flush=True)


if __name__ == "__main__":
    main()
