"""IBM Granite Speech 4.1 2B local ASR."""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List

from .base import BaseTranscriber


MODEL_ID = "ibm-granite/granite-speech-4.1-2b"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.join(
    _THIS_DIR, "..", "..", "models", "hf", "ibm-granite__granite-speech-4.1-2b"
)

CHUNK_S = 30.0

_FFMPEG_BIN = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)


def _make_env() -> dict:
    env = os.environ.copy()
    if os.name == "nt" and os.path.isdir(_FFMPEG_BIN):
        if _FFMPEG_BIN not in env.get("PATH", ""):
            env["PATH"] = _FFMPEG_BIN + os.pathsep + env.get("PATH", "")
    return env


_ENV = _make_env()


class GraniteSpeechTranscriber(BaseTranscriber):
    name = "granite-speech-4.1-2b"
    supports_word_timestamps = False
    model_id = MODEL_ID
    local_model_dir = LOCAL_MODEL_DIR
    max_new_tokens = 384

    def __init__(self, device: str = "cuda", model_dir: str | None = None):
        super().__init__(device=device, model_dir=model_dir)
        self.processor = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        preferred_local = self.model_dir or self.local_model_dir
        model_path = preferred_local if os.path.isdir(preferred_local) else self.model_id
        is_local = model_path != self.model_id
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32

        print(f"  [Granite] loading processor from {model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=is_local,
            fix_mistral_regex=True,
        )
        self.tokenizer = self.processor.tokenizer

        print(f"  [Granite] loading model on {self.device}", flush=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            local_files_only=is_local,
            device_map=self.device,
            dtype=dtype,
        )
        self.model.eval()

    def _audio_duration(self, audio_path: str) -> float:
        try:
            import soundfile as sf
            return float(sf.info(audio_path).duration)
        except Exception:
            pass
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True,
            text=True,
            env=_ENV,
            check=True,
        )
        return float(json.loads(r.stdout)["format"]["duration"])

    def _wav_chunk(self, audio_path: str, start_s: float, duration_s: float) -> str:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start_s:.3f}",
                "-i", audio_path,
                "-t", f"{duration_s:.3f}",
                "-ac", "1",
                "-ar", "16000",
                tmp,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_ENV,
        )
        return tmp

    @staticmethod
    def _safe_delete(path: str) -> None:
        try:
            os.unlink(path)
        except (PermissionError, OSError):
            pass

    def _prompt(self) -> str:
        return "<|audio|>transcribe the speech with proper punctuation and capitalization."

    def _decode_chunk(self, wav_path: str) -> str:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(wav_path, normalize=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        prompt = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": self._prompt(),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(prompt, wav, device=self.device, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        num_input_tokens = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
        text = self.tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )[0]
        return " ".join(str(text or "").split())

    @staticmethod
    def _word_text(word: dict) -> str:
        return str(word.get("word") or word.get("text") or "").strip()

    @classmethod
    def _join_words(cls, words: List[Dict[str, Any]]) -> str:
        text = " ".join(cls._word_text(w) for w in words if cls._word_text(w)).strip()
        text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _split_text_by_sentence(text: str) -> List[str]:
        text = " ".join(str(text or "").split())
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        return parts or [text]

    def _segments_from_text(self, text: str, start: float, end: float) -> List[Dict[str, Any]]:
        parts = self._split_text_by_sentence(text)
        if not parts:
            return []
        weights = [max(1, len(part.split())) for part in parts]
        total = float(sum(weights))
        cursor = start
        out: List[Dict[str, Any]] = []
        for idx, (part, weight) in enumerate(zip(parts, weights)):
            seg_end = end if idx == len(parts) - 1 else cursor + (end - start) * (weight / total)
            if seg_end <= cursor:
                seg_end = cursor + 0.2
            out.append(
                {
                    "start": round(cursor, 2),
                    "end": round(min(seg_end, end), 2),
                    "text": part,
                    "speaker": "SPEAKER_00",
                    "identified_speaker": "SPEAKER_00",
                    "confidence": 0.0,
                }
            )
            cursor = seg_end
        return out

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()
        total_dur = self._audio_duration(audio_path)
        total_chunks = max(1, int((total_dur + CHUNK_S - 0.001) // CHUNK_S))
        out: List[Dict[str, Any]] = []

        start = 0.0
        chunk_idx = 0
        while start < total_dur:
            dur = min(CHUNK_S, total_dur - start)
            if dur < 0.3:
                break
            chunk_idx += 1
            print(
                f"  [Granite] chunk {chunk_idx}/{total_chunks} "
                f"{start:.0f}s-{start + dur:.0f}s",
                flush=True,
            )
            tmp = self._wav_chunk(audio_path, start, dur)
            try:
                text = self._decode_chunk(tmp)
                out.extend(self._segments_from_text(text, start, start + dur))
            finally:
                self._safe_delete(tmp)
            start += dur

        before = len(out)
        out = self.filter_hallucinations(out)
        elapsed = time.time() - t0
        rtf = total_dur / elapsed if elapsed > 0 else 0
        print(
            f"  [Granite] {len(out)} segments in {elapsed:.1f}s "
            f"(RTF {rtf:.1f}x, skipped {before - len(out)} hallucinations)",
            flush=True,
        )
        return out

    def unload(self) -> None:
        try:
            import torch
            if self.model is not None and torch.cuda.is_available():
                try:
                    self.model = self.model.cpu()
                except Exception:
                    pass
        except Exception:
            pass
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
