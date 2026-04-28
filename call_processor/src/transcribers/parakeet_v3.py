"""NVIDIA Parakeet TDT v3 — fastest non-Whisper, native word timestamps."""
from __future__ import annotations
import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Dict, Any, Optional
from .base import BaseTranscriber

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
# Parakeet TDT trained on ≤60 s; use 55 s chunks to leave a small headroom
CHUNK_S = 55

# ffmpeg/ffprobe bin — same WinGet path as ui.py so subprocess calls work
_FFMPEG_BIN = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)


def _make_env() -> dict:
    env = os.environ.copy()
    if sys.platform == "win32" and os.path.isdir(_FFMPEG_BIN):
        if _FFMPEG_BIN not in env.get("PATH", ""):
            env["PATH"] = _FFMPEG_BIN + os.pathsep + env.get("PATH", "")
    return env


_ENV = _make_env()


def _silence_nemo() -> None:
    """Set all NeMo loggers to ERROR so they don't flood the console."""
    for log_name in list(logging.Logger.manager.loggerDict):
        if "nemo" in log_name.lower():
            logging.getLogger(log_name).setLevel(logging.ERROR)
    for name in ("nemo_logger", "nemo", "nemo.collections", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)


class ParakeetV3Transcriber(BaseTranscriber):
    name = "parakeet-tdt-0.6b-v3"
    supports_word_timestamps = True

    def load(self) -> None:
        if self.model is not None:
            return

        import torch
        _use_cuda = False
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.current_device()
                torch.zeros(1).cuda()
                _use_cuda = True
                torch.cuda.empty_cache()
            except Exception as _e:
                print(f"  [Parakeet] CUDA probe failed ({_e}) — CPU mode", flush=True)
        self._use_cuda = _use_cuda

        # Silence NeMo before importing — it registers loggers at import time
        _silence_nemo()

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise RuntimeError(
                "Parakeet requires nemo_toolkit[asr].\n"
                "Install: pip install nemo_toolkit[asr]"
            )

        _silence_nemo()  # silence any new loggers registered during import

        cache_dir = self.model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "nemo"
        )
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NEMO_CACHE_DIR"] = cache_dir

        import torch as _torch
        _orig_avail = _torch.cuda.is_available
        if not _use_cuda:
            _torch.cuda.is_available = lambda: False
        try:
            print(f"  [Parakeet] Downloading / loading {MODEL_ID} ...", flush=True)
            # ASRModel.from_pretrained auto-selects the right subclass
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
        except Exception as e1:
            print(f"  [Parakeet] ASRModel failed ({e1}) — trying EncDecRNNTBPEModel", flush=True)
            try:
                self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name=MODEL_ID
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Parakeet TDT v3 load failed.\n  ASRModel: {e1}\n  RNNT: {e2}"
                )
        finally:
            _torch.cuda.is_available = _orig_avail

        if _use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        _silence_nemo()
        print(f"  [Parakeet] Ready on {'CUDA' if _use_cuda else 'CPU'}.", flush=True)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _audio_duration(self, audio_path: str) -> float:
        """Return audio length in seconds. Uses soundfile (fast); falls back to ffprobe."""
        try:
            import soundfile as sf
            return sf.info(audio_path).duration
        except Exception:
            pass
        # ffprobe fallback (needs ffmpeg in PATH — _ENV has it)
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", audio_path],
                capture_output=True, text=True, env=_ENV,
            )
            return float(json.loads(r.stdout)["format"]["duration"])
        except Exception as e:
            raise RuntimeError(f"Cannot get duration of {audio_path}: {e}")

    def _wav_chunk(self, audio_path: str, start_s: float, duration_s: float) -> str:
        """Cut a 16 kHz mono WAV slice. Returns temp path (caller deletes)."""
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(start_s), "-t", str(duration_s),
             "-ac", "1", "-ar", "16000", tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_ENV,
        )
        return tmp

    def _safe_delete(self, path: str) -> None:
        try:
            os.unlink(path)
        except (PermissionError, OSError):
            pass  # Windows: NeMo may still hold the handle briefly

    def _extract_segments(self, hyp, chunk_offset: float) -> List[Dict[str, Any]]:
        """Pull segment dicts from a NeMo Hypothesis (or plain string fallback)."""
        segs = []

        ts = getattr(hyp, "timestamp", None)
        if isinstance(ts, dict):
            seg_list = ts.get("segment") or ts.get("word") or []
            for seg in seg_list:
                s = float(seg.get("start", 0)) + chunk_offset
                e = float(seg.get("end", s + 0.5)) + chunk_offset
                text = (seg.get("segment") or seg.get("word") or "").strip()
                if text:
                    segs.append({
                        "start": round(s, 2),
                        "end":   round(e, 2),
                        "text":  text,
                        "speaker": "SPEAKER_00",
                        "identified_speaker": "SPEAKER_00",
                        "confidence": 0.0,
                    })
            if segs:
                return segs

        # No timestamps — emit the whole chunk as one segment (text only)
        text = getattr(hyp, "text", None)
        if not text and isinstance(hyp, str):
            text = hyp
        text = (text or "").strip()
        # Only emit if it looks like real speech (not an empty Hypothesis repr)
        if text and not text.startswith("Hypothesis("):
            segs.append({
                "start": round(chunk_offset, 2),
                "end":   round(chunk_offset + CHUNK_S, 2),
                "text":  text,
                "speaker": "SPEAKER_00",
                "identified_speaker": "SPEAKER_00",
                "confidence": 0.0,
            })
        return segs

    # ── Main transcribe ────────────────────────────────────────────────────

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()

        total_dur = self._audio_duration(audio_path)
        total_chunks = max(1, int(total_dur / CHUNK_S) + 1)
        out: List[Dict[str, Any]] = []

        chunk_idx = 0
        start = 0.0
        while start < total_dur:
            dur = min(CHUNK_S, total_dur - start)
            if dur < 0.3:
                break
            chunk_idx += 1
            print(
                f"  [Parakeet] chunk {chunk_idx}/{total_chunks}"
                f"  {start:.0f}s–{start+dur:.0f}s"
                f"  ({total_dur - start - dur:.0f}s remaining)",
                flush=True,
            )

            tmp = self._wav_chunk(audio_path, start, dur)
            hyp = None
            try:
                # NeMo 2.x: return_hypotheses=True + timestamps=True gives
                # Hypothesis objects with .timestamp populated
                results = self.model.transcribe(
                    [tmp],
                    return_hypotheses=True,
                    timestamps=True,
                    batch_size=1,
                    verbose=False,
                )
            except TypeError:
                # Older NeMo: return_hypotheses / verbose not accepted
                try:
                    results = self.model.transcribe([tmp], timestamps=True, batch_size=1)
                except TypeError:
                    results = self.model.transcribe([tmp])
            finally:
                self._safe_delete(tmp)

            if not results:
                start += dur
                continue

            raw = results[0] if isinstance(results, (list, tuple)) else results
            # Unwrap one more level if NeMo nested the hypothesis
            if isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]

            out.extend(self._extract_segments(raw, start))
            start += dur

        before = len(out)
        out = self.filter_hallucinations(out)
        skipped = before - len(out)
        elapsed = time.time() - t0
        rtf = total_dur / elapsed if elapsed > 0 else 0
        print(
            f"  [Parakeet] {len(out)} segments in {elapsed:.1f}s"
            f"  (RTF {rtf:.1f}x, skipped {skipped} hallucinations)",
            flush=True,
        )
        return out

    # ── Cleanup ────────────────────────────────────────────────────────────

    def unload(self) -> None:
        try:
            import torch
            if self.model is not None and torch.cuda.is_available():
                try:
                    self.model = self.model.cpu()
                except Exception:
                    pass
        except ImportError:
            pass
        if self.model is not None:
            del self.model
        self.model = None
        gc.collect()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
