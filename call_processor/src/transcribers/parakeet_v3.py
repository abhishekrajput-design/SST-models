"""NVIDIA Parakeet TDT v3 — fastest non-Whisper, native word timestamps.

Fixes vs original:
- Chunks audio into 60s pieces to avoid CUDA OOM on 6 GB GPU
- Windows-safe temp file cleanup (PermissionError on unlink)
- Forces GPU CUDA memory cleanup before load
"""
from __future__ import annotations
import os
import time
import tempfile
import subprocess
from typing import List, Dict, Any
from .base import BaseTranscriber

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
CHUNK_S = 60  # seconds per chunk — keeps peak VRAM under 6 GB


class ParakeetV3Transcriber(BaseTranscriber):
    name = "parakeet-tdt-0.6b-v3"
    supports_word_timestamps = True

    def load(self) -> None:
        if self.model is not None:
            return
        # Free any leftover GPU memory from previous models
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise RuntimeError(
                "Parakeet requires 'nemo_toolkit[asr]'. Install with:\n"
                "  pip install nemo_toolkit[asr]"
            )
        cache_dir = self.model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "nemo"
        )
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NEMO_CACHE_DIR"] = cache_dir
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

    def _wav_chunk(self, audio_path: str, start_s: float, duration_s: float) -> str:
        """Extract a WAV chunk using ffmpeg. Returns temp file path (caller must delete)."""
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(start_s), "-t", str(duration_s),
             "-ac", "1", "-ar", "16000", tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return tmp

    def _safe_delete(self, path: str) -> None:
        """Delete temp file, ignoring Windows file-lock errors."""
        try:
            os.unlink(path)
        except (PermissionError, OSError):
            pass  # Windows: NeMo may still hold a handle briefly

    def _audio_duration(self, audio_path: str) -> float:
        import subprocess, json
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", audio_path],
            capture_output=True, text=True
        )
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()

        total_dur = self._audio_duration(audio_path)
        out: List[Dict[str, Any]] = []

        start = 0.0
        while start < total_dur:
            dur = min(CHUNK_S, total_dur - start)
            if dur < 0.5:
                break

            tmp = self._wav_chunk(audio_path, start, dur)
            try:
                results = self.model.transcribe([tmp], timestamps=True, batch_size=1)
            finally:
                self._safe_delete(tmp)

            hyp = results[0] if isinstance(results, list) else results

            if hasattr(hyp, "timestamp") and hyp.timestamp:
                segs = hyp.timestamp.get("segment") or hyp.timestamp.get("word") or []
                for seg in segs:
                    seg_start = float(seg.get("start", 0)) + start
                    seg_end   = float(seg.get("end", seg_start + 1)) + start
                    text      = (seg.get("segment") or seg.get("word") or "").strip()
                    if text:
                        out.append({
                            "start": round(seg_start, 2),
                            "end":   round(seg_end, 2),
                            "text":  text,
                            "speaker": "SPEAKER_00",
                            "identified_speaker": "SPEAKER_00",
                            "confidence": 0.0,
                        })
            else:
                text = (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()
                if text:
                    out.append({
                        "start": round(start, 2),
                        "end":   round(start + dur, 2),
                        "text":  text,
                        "speaker": "SPEAKER_00",
                        "identified_speaker": "SPEAKER_00",
                        "confidence": 0.0,
                    })

            start += dur

        print(f"  [Parakeet] {len(out)} segments in {time.time()-t0:.1f}s")
        return out

    def unload(self) -> None:
        import gc
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
