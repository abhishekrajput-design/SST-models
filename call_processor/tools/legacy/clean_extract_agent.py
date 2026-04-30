"""
clean_extract_agent.py — Master agent voiceprint training script.

Reads labeled Audiofy data (scraped by scrape_audiofy.py) and builds
high-quality 512-dim CAM++ voiceprints per agent.

Input layout:
    data/audiofy/<agent_slug>/agent_clips/*.wav   (16kHz mono, pre-cut by scraper)

Output:
    data/agent_voiceprints/<agent_slug>.npy       (512-dim L2-normalised CAM++ vector)
    data/agent_voiceprints/agents.json            (registry of all trained voiceprints)
    data/agent_clean_clips/<agent_slug>/          (quality-filtered training clips)

Usage:
    python clean_extract_agent.py                     # all agents
    python clean_extract_agent.py --agent "Zak"       # name substring
    python clean_extract_agent.py --min-clips 5       # skip agents with < N good clips
    python clean_extract_agent.py --threshold 0.45    # outlier cosine threshold
    python clean_extract_agent.py --no-copy           # skip copying clean clips
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR       = Path(__file__).parent
AUDIOFY_DIR      = SCRIPT_DIR / "data" / "audiofy"
VOICEPRINT_DIR   = SCRIPT_DIR / "data" / "agent_voiceprints"
CLEAN_CLIPS_DIR  = SCRIPT_DIR / "data" / "agent_clean_clips"

VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Quality-filter thresholds (defaults, can be overridden via CLI)
# ---------------------------------------------------------------------------

MIN_DUR_S      = 1.5    # seconds
MAX_DUR_S      = 20.0   # seconds
MIN_RMS_DBFS   = -40.0  # dBFS  (near-silence threshold)
MAX_SPEC_FLAT  = 0.6    # spectral flatness (> this = noise-like)
MIN_SNR_DB     = 10.0   # estimated SNR in dB


# ---------------------------------------------------------------------------
# Audio quality helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def rms_dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    return float(20.0 * np.log10(rms + 1e-9))


def spectral_flatness(audio: np.ndarray) -> float:
    """
    Wiener entropy / spectral flatness of the signal.
    Value close to 1 = noise-like; close to 0 = tonal / speech.
    Computed over the magnitude spectrum of the whole clip.
    """
    spec = np.abs(np.fft.rfft(audio.astype(np.float64)))
    spec = spec + 1e-10                          # avoid log(0)
    log_mean = float(np.mean(np.log(spec)))      # geometric mean in log domain
    arith_mean = float(np.mean(spec))
    geom_mean = float(np.exp(log_mean))
    return float(geom_mean / (arith_mean + 1e-10))


def snr_estimate(audio: np.ndarray) -> float:
    """
    Quick SNR estimate: 99th-percentile amplitude vs 5th-percentile amplitude.
    Works well enough to reject silence-padded / noise-only clips.
    """
    a = np.abs(audio.astype(np.float64))
    peak  = float(np.percentile(a, 99))
    noise = float(np.percentile(a, 5))
    return float(20.0 * np.log10(peak / (noise + 1e-9)))


def load_wav_mono_16k(path: str) -> Optional[Tuple[np.ndarray, float]]:
    """
    Load a WAV file, mix to mono, return (audio_float32, duration_s).
    Returns None on read failure.
    The scraper already writes 16kHz mono, but we tolerate any SR here.
    """
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"    [read] Cannot read {path}: {exc}")
        return None

    # Mix down to mono
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    # Resample if needed (clips from scraper should already be 16kHz)
    if sr != 16000:
        try:
            import torchaudio.functional as F_ta
            import torch
            audio = F_ta.resample(
                torch.from_numpy(audio), sr, 16000
            ).numpy()
        except Exception:
            # Fallback: naive linear resample (rare)
            target_len = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

    duration = len(audio) / 16000.0
    return audio, duration


# ---------------------------------------------------------------------------
# Per-clip quality filter
# ---------------------------------------------------------------------------

def passes_quality_filter(
    audio: np.ndarray,
    duration: float,
    path: str,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Run all quality checks.  Returns (passed, reason_if_failed).
    """
    # 1. Duration
    if duration < MIN_DUR_S:
        return False, f"too short ({duration:.2f}s < {MIN_DUR_S}s)"
    if duration > MAX_DUR_S:
        return False, f"too long ({duration:.2f}s > {MAX_DUR_S}s)"

    # 2. RMS energy
    rms = rms_dbfs(audio)
    if rms < MIN_RMS_DBFS:
        return False, f"too quiet ({rms:.1f} dBFS < {MIN_RMS_DBFS})"

    # 3. Spectral flatness
    sf_val = spectral_flatness(audio)
    if sf_val > MAX_SPEC_FLAT:
        return False, f"too noisy (flatness={sf_val:.3f} > {MAX_SPEC_FLAT})"

    # 4. SNR estimate
    snr = snr_estimate(audio)
    if snr < MIN_SNR_DB:
        return False, f"low SNR ({snr:.1f} dB < {MIN_SNR_DB})"

    return True, ""


# ---------------------------------------------------------------------------
# Core per-agent pipeline
# ---------------------------------------------------------------------------

def process_agent(
    agent_slug: str,
    agent_name: str,
    clips_dir: Path,
    threshold: float,
    no_copy: bool,
) -> Optional[dict]:
    """
    Run the full pipeline for one agent.
    Returns a results dict or None if not enough clips.
    """
    wav_files = sorted(clips_dir.glob("*.wav"))
    total_clips = len(wav_files)

    if total_clips == 0:
        print(f"  [skip] No WAV files in {clips_dir}")
        return None

    # ------------------------------------------------------------------
    # Step 1+2: Load clips and apply quality filter
    # ------------------------------------------------------------------
    good_clips: List[Tuple[str, np.ndarray]] = []   # (path, audio)

    for wav_path in wav_files:
        result = load_wav_mono_16k(str(wav_path))
        if result is None:
            continue
        audio, duration = result
        passed, reason = passes_quality_filter(audio, duration, str(wav_path))
        if not passed:
            continue
        good_clips.append((str(wav_path), audio))

    n_good = len(good_clips)
    print(f"  Quality filter: {total_clips} total -> {n_good} passed")

    if n_good == 0:
        print("  [skip] No clips passed quality filter.")
        return None

    # ------------------------------------------------------------------
    # Step 3: Extract CAM++ embeddings
    # ------------------------------------------------------------------
    # Lazy import so the model isn't loaded if we bail early
    from src.embedding_campp import get_model, cosine_sim, l2_norm

    model = get_model()
    dim   = model.dim
    print(f"  Extracting embeddings ({model.model_name}, dim={dim})…")

    embeddings: List[np.ndarray] = []
    paths_embedded: List[str]   = []

    for wav_path, _audio in good_clips:
        emb = model.embed_file(wav_path)
        if emb is not None:
            embeddings.append(emb)
            paths_embedded.append(wav_path)

    n_embedded = len(embeddings)
    print(f"  Embedded: {n_embedded} / {n_good}")

    if n_embedded == 0:
        print("  [skip] Embedding failed for all clips.")
        return None

    # ------------------------------------------------------------------
    # Step 4: Outlier removal
    # ------------------------------------------------------------------
    emb_matrix = np.stack(embeddings, axis=0)   # (N, dim)
    mean_emb   = l2_norm(emb_matrix.mean(axis=0))

    sims = np.array([cosine_sim(e, mean_emb) for e in embeddings])
    keep_mask = sims >= threshold

    kept_embeddings = emb_matrix[keep_mask]
    kept_paths      = [p for p, k in zip(paths_embedded, keep_mask) if k]
    n_used          = int(keep_mask.sum())

    # Re-compute mean from kept set
    if n_used > 0:
        mean_emb = l2_norm(kept_embeddings.mean(axis=0))
        final_sims = np.array([cosine_sim(e, mean_emb) for e in kept_embeddings])
    else:
        # Nothing survived – relax to top-half
        half = max(1, n_embedded // 2)
        top_idx     = np.argsort(sims)[-half:]
        kept_embeddings = emb_matrix[top_idx]
        kept_paths      = [paths_embedded[i] for i in top_idx]
        n_used          = len(top_idx)
        mean_emb        = l2_norm(kept_embeddings.mean(axis=0))
        final_sims      = np.array([cosine_sim(e, mean_emb) for e in kept_embeddings])
        print(f"  [warn] threshold too strict; relaxed to top-{n_used} clips")

    mean_cosine = float(final_sims.mean()) if len(final_sims) > 0 else 0.0
    min_cosine  = float(final_sims.min())  if len(final_sims) > 0 else 0.0
    max_cosine  = float(final_sims.max())  if len(final_sims) > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 5: Final voiceprint
    # ------------------------------------------------------------------
    voiceprint = mean_emb   # already L2-normalised above

    vp_path = VOICEPRINT_DIR / f"{agent_slug}.npy"
    np.save(str(vp_path), voiceprint)

    # ------------------------------------------------------------------
    # Step 6: Copy good clips
    # ------------------------------------------------------------------
    if not no_copy and n_used > 0:
        out_clips_dir = CLEAN_CLIPS_DIR / agent_slug
        out_clips_dir.mkdir(parents=True, exist_ok=True)
        for src_path in kept_paths:
            dst = out_clips_dir / Path(src_path).name
            if not dst.exists():
                shutil.copy2(src_path, dst)

    # ------------------------------------------------------------------
    # Print per-agent quality metrics
    # ------------------------------------------------------------------
    cohesion_flag = "" if mean_cosine >= 0.65 else "  [!low cohesion]"
    print(
        f"  Outlier removal: {n_embedded} -> {n_used} kept  "
        f"(threshold={threshold})"
    )
    print(
        f"  Cosine to mean:  mean={mean_cosine:.4f}  "
        f"min={min_cosine:.4f}  max={max_cosine:.4f}{cohesion_flag}"
    )
    print(f"  Voiceprint saved -> {vp_path}  (dim={dim})")
    if not no_copy:
        print(f"  Clean clips -> {CLEAN_CLIPS_DIR / agent_slug}/")

    return {
        "agent_name":      agent_name,
        "voiceprint_path": str(vp_path),
        "n_clips":         n_used,
        "mean_cosine":     round(mean_cosine, 6),
        "dim":             dim,
        # Extra stats for the summary table (not written to agents.json)
        "_total":          total_clips,
        "_good":           n_good,
        "_used":           n_used,
    }


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------

def discover_agents(agent_filter: str) -> List[Tuple[str, str, Path]]:
    """
    Return list of (slug, agent_name, clips_dir) for agents that have
    a populated agent_clips directory, optionally filtered by name substring.
    """
    agents = []
    if not AUDIOFY_DIR.exists():
        print(f"[Error] Audiofy data directory not found: {AUDIOFY_DIR}")
        return agents

    for agent_dir in sorted(AUDIOFY_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue

        clips_dir = agent_dir / "agent_clips"
        if not clips_dir.exists() or not any(clips_dir.glob("*.wav")):
            continue

        # Read agent name from calls.json if present
        calls_json = agent_dir / "calls.json"
        agent_name = agent_dir.name   # fallback: use slug as name
        if calls_json.exists():
            try:
                with open(calls_json, encoding="utf-8") as f:
                    data = json.load(f)
                agent_name = data.get("agent", agent_name)
            except Exception:
                pass

        if agent_filter and agent_filter.lower() not in agent_name.lower():
            continue

        agents.append((agent_dir.name, agent_name, clips_dir))

    return agents


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: List[Tuple[str, dict]]):
    if not results:
        print("\n[Summary] No agents processed.")
        return

    header = f"{'Agent':<36}  {'Clips':>5}  {'Good':>5}  {'Used':>5}  {'Mean-Cosine':>11}  {'Dim':>4}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for slug, r in results:
        name         = r["agent_name"]
        total        = r["_total"]
        good         = r["_good"]
        used         = r["_used"]
        mean_cosine  = r["mean_cosine"]
        dim          = r["dim"]
        flag = "  <" if mean_cosine < 0.65 else ""
        print(
            f"{name:<36}  {total:>5}  {good:>5}  {used:>5}  "
            f"{mean_cosine:>11.4f}  {dim:>4}{flag}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build CAM++ voiceprints from labeled Audiofy agent clips."
    )
    parser.add_argument(
        "--agent", default="",
        help="Agent name substring filter (case-insensitive).",
    )
    parser.add_argument(
        "--min-clips", type=int, default=1,
        help="Skip agents with fewer good clips than this (after quality filter).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.45,
        help="Cosine similarity threshold for outlier removal (default: 0.45).",
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Skip copying clean clips to data/agent_clean_clips/ (faster).",
    )
    args = parser.parse_args()

    print(f"[Config] threshold={args.threshold}  min-clips={args.min_clips}  "
          f"no-copy={args.no_copy}")
    print(f"[Dirs]   audiofy={AUDIOFY_DIR}")
    print(f"         voiceprints={VOICEPRINT_DIR}")
    print(f"         clean_clips={CLEAN_CLIPS_DIR}")

    # Discover agents
    agents = discover_agents(args.agent)
    if not agents:
        print(f"\n[Error] No agents found in {AUDIOFY_DIR}"
              + (f" matching '{args.agent}'" if args.agent else "") + ".")
        print("  Run scrape_audiofy.py first to download data.")
        sys.exit(1)

    print(f"\n[Found] {len(agents)} agent(s) with clips:")
    for slug_name, agent_name, clips_dir in agents:
        n = len(list(clips_dir.glob("*.wav")))
        print(f"  {agent_name:<36} ({n} WAV files)  ->  {clips_dir}")

    # Load agents.json registry (merge new results in)
    agents_json_path = VOICEPRINT_DIR / "agents.json"
    if agents_json_path.exists():
        try:
            with open(agents_json_path, encoding="utf-8") as f:
                agents_registry: dict = json.load(f)
        except Exception:
            agents_registry = {}
    else:
        agents_registry = {}

    all_results: List[Tuple[str, dict]] = []

    for agent_slug, agent_name, clips_dir in agents:
        print(f"\n{'='*60}")
        print(f"[Agent] {agent_name}  ({agent_slug})")
        print(f"{'='*60}")

        result = process_agent(
            agent_slug=agent_slug,
            agent_name=agent_name,
            clips_dir=clips_dir,
            threshold=args.threshold,
            no_copy=args.no_copy,
        )

        if result is None:
            print(f"  [skip] {agent_name}: processing failed or no clips.")
            continue

        if result["_good"] < args.min_clips:
            print(
                f"  [skip] {agent_name}: only {result['_good']} good clips "
                f"< --min-clips {args.min_clips}"
            )
            continue

        # Store in registry (strip internal underscore keys)
        registry_entry = {
            k: v for k, v in result.items() if not k.startswith("_")
        }
        agents_registry[agent_slug] = registry_entry
        all_results.append((agent_slug, result))

    # Save updated agents.json
    with open(agents_json_path, "w", encoding="utf-8") as f:
        json.dump(agents_registry, f, indent=2, ensure_ascii=False)
    print(f"\n[Registry] agents.json updated -> {agents_json_path}")
    print(f"           {len(agents_registry)} agent(s) registered total")

    # Summary table
    print_summary(all_results)


if __name__ == "__main__":
    main()
