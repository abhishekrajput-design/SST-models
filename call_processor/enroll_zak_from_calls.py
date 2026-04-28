"""
enroll_zak_from_calls.py — Enroll Zak from real phone call recordings.

Phone calls have 2 speakers (agent + customer), no ground-truth labels.
Uses KMeans(2) to separate, takes larger cluster = Zak (agent speaks more),
then applies iterative tightening to remove customer contamination.

Held-out (NOT used for training): call_69e3afc81bbc87d03ab29ae6.mp3
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

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

# Use 2 calls for enrollment — hold out call_69e3afc8 for testing
TRAIN_CALLS = [
    str(SCRIPT_DIR / "data" / "audiofy" / "zak_raissi_barnet" / "audio"
        / "call_69e3b2051bbc87d03ab2a3b0.mp3"),  # 478s
    str(SCRIPT_DIR / "data" / "audiofy" / "zak_raissi_barnet" / "audio"
        / "call_69e4a4415453871f7cde30b5.mp3"),  # 170s
]

VP_DIR      = SCRIPT_DIR / "data" / "agent_voiceprints"
AGENTS_JSON = VP_DIR / "agents.json"
SLUG        = "zak_raissi_barnet"
NAME        = "Zak Raissi Barnet"

TARGET_SR = 16000
WINDOW_S  = 2.0
STRIDE_S  = 1.0
TIGHT_1   = 0.45
TIGHT_2   = 0.55
MIN_TIGHT = 15


def load_16k_mono(mp3: str) -> np.ndarray:
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", mp3, "-ac", "1", "-ar", str(TARGET_SR), tmp],
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


def centroid(mat: np.ndarray) -> np.ndarray:
    c = np.mean(mat, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def main():
    VP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ENROLL ZAK from phone call recordings (CAM++ 512-dim)")
    print(f"{'='*60}\n")

    # Load CAM++ model
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    model.load(force_cpu=False)
    print(f"[enroll] {model.model_name} ready  dim={model.dim}", flush=True)

    window = int(TARGET_SR * WINDOW_S)
    stride = int(TARGET_SR * STRIDE_S)
    all_embs: list = []

    try:
        for mp3 in TRAIN_CALLS:
            fname = Path(mp3).name
            print(f"\n[enroll] Processing {fname} ...", flush=True)
            t0 = time.time()

            audio = load_16k_mono(mp3)
            dur = len(audio) / TARGET_SR
            print(f"  {dur:.0f}s audio loaded", flush=True)

            # Extract embeddings (whole call — agent + customer mixed)
            windows = []
            for start in range(0, len(audio) - window, stride):
                emb = model.embed_chunk(audio[start: start + window], TARGET_SR)
                if emb is not None:
                    windows.append(emb)

            if len(windows) < 10:
                print(f"  [skip] only {len(windows)} windows", flush=True)
                continue

            X = np.stack(windows).astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

            # KMeans(2): separate agent from customer
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            c0, c1 = int((labels == 0).sum()), int((labels == 1).sum())
            agent_cluster = 0 if c0 >= c1 else 1
            agent_mask = labels == agent_cluster
            print(f"  KMeans: cluster0={c0} cluster1={c1} → agent cluster {agent_cluster} ({agent_mask.sum()} windows)",
                  flush=True)

            all_embs.extend(X[agent_mask].tolist())
            print(f"  Done in {time.time()-t0:.0f}s  [total agent windows: {len(all_embs)}]",
                  flush=True)
    finally:
        model.unload()

    if len(all_embs) < 30:
        print(f"\n[enroll] ERROR: only {len(all_embs)} windows — need at least 30", flush=True)
        sys.exit(1)

    X = np.array(all_embs, dtype=np.float32)
    print(f"\n[enroll] Total agent windows: {len(X)}", flush=True)

    # Iterative tightening to remove customer contamination
    c = centroid(X)
    sims = X @ c
    keep1 = sims >= TIGHT_1
    if int(keep1.sum()) < MIN_TIGHT:
        keep1 = np.ones(len(X), dtype=bool)
    c = centroid(X[keep1])
    sims2 = X @ c
    keep2 = keep1 & (sims2 >= TIGHT_2)
    if int(keep2.sum()) < MIN_TIGHT:
        keep2 = keep1

    final = centroid(X[keep2])
    n = np.linalg.norm(final)
    final = final / n if n > 0 else final

    inside  = float((X[keep2] @ final).mean())
    outside = float((X[~keep2] @ final).max()) if (~keep2).any() else 0.0
    print(f"[enroll] Tightening: kept {keep2.sum()}/{len(X)}  "
          f"inside_sim={inside:.3f}  outside_max={outside:.3f}", flush=True)

    vp_path = VP_DIR / f"{SLUG}.npy"
    np.save(vp_path, final)
    print(f"[enroll] Saved → {vp_path}  shape={final.shape}", flush=True)

    # Update agents.json
    agents = json.load(open(AGENTS_JSON, encoding="utf-8")) if AGENTS_JSON.exists() else {}
    agents[SLUG] = {
        "agent_name":      NAME,
        "voiceprint_path": str(vp_path),
        "n_clips":         int(keep2.sum()),
        "n_windows_total": int(len(X)),
        "source":          "phone_calls_kmeans_campp",
        "mean_inside_sim": round(inside, 3),
        "max_outside_sim": round(outside, 3),
    }
    json.dump(agents, open(AGENTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[enroll] agents.json updated  ({len(agents)} total agents)", flush=True)
    print(f"\n[enroll] DONE — {NAME}  512-dim CAM++  inside_sim={inside:.3f}", flush=True)


if __name__ == "__main__":
    main()
