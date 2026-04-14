"""Qwen3-ASR-1.7B — 52-language + dialect support, 2026.

Uses the official qwen-asr package (pip install qwen-asr).
Chunks 30-min audio into 60s pieces to fit in 6 GB VRAM.
Language arg maps ISO codes to full names expected by the API.
"""
from __future__ import annotations
import os
import time
import subprocess
import tempfile
from typing import List, Dict, Any
from .base import BaseTranscriber

# Local cache path (downloaded by download_models.py)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.join(_THIS_DIR, "..", "..", "models", "hf", "Qwen__Qwen3-ASR-1.7B")
HF_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

CHUNK_S = 60  # process in 60s chunks to avoid OOM on 6 GB GPU

# qwen-asr wants full language names
ISO_TO_LANG = {
    "en": "English", "zh": "Chinese", "fr": "French", "de": "German",
    "es": "Spanish", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
    "ru": "Russian", "it": "Italian", "ar": "Arabic", "vi": "Vietnamese",
    "nl": "Dutch", "pl": "Polish", "el": "Greek",
}


class Qwen3AsrTranscriber(BaseTranscriber):
    name = "qwen3-asr-1.7b"

    def __init__(self, device: str = "cuda", model_dir: str | None = None):
        super().__init__(device=device, model_dir=model_dir)

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_path = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HF_MODEL_ID
        device_map = "cuda:0" if self.device == "cuda" else "cpu"

        # Try qwen-asr package first; fall back to transformers Qwen2Audio
        try:
            from qwen_asr import Qwen3ASRModel
            self.model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=device_map,
                max_inference_batch_size=4,
                max_new_tokens=512,
            )
            self._backend = "qwen-asr"
        except ImportError:
            print("  [Qwen3-ASR] qwen-asr not installed — using transformers Qwen2Audio fallback")
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self.model.eval()
            self._backend = "transformers"

    def _audio_duration(self, audio_path: str) -> float:
        import json
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True,
        )
        return float(json.loads(r.stdout)["format"]["duration"])

    def _wav_chunk(self, audio_path: str, start_s: float, duration_s: float) -> str:
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
        try:
            os.unlink(path)
        except (PermissionError, OSError):
            pass

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()
        lang_full = ISO_TO_LANG.get(language, "English")
        total_dur = self._audio_duration(audio_path)
        out: List[Dict[str, Any]] = []

        start = 0.0
        while start < total_dur:
            dur = min(CHUNK_S, total_dur - start)
            if dur < 0.5:
                break

            tmp = self._wav_chunk(audio_path, start, dur)
            try:
                if self._backend == "qwen-asr":
                    results = self.model.transcribe(audio=tmp, language=lang_full)
                    text = results[0].text.strip() if results else ""
                else:
                    import librosa, torch
                    audio_arr, sr_loaded = librosa.load(tmp, sr=16000, mono=True)
                    # Build chat-template prompt for Qwen2-Audio processor
                    try:
                        conv = [{"role": "user", "content": [
                            {"type": "audio", "audio_url": "audio.wav"}
                        ]}]
                        prompt = self._processor.apply_chat_template(
                            conv, add_generation_prompt=True, tokenize=False
                        )
                    except Exception:
                        prompt = "Transcribe the speech."
                    inputs = self._processor(
                        text=[prompt], audios=[audio_arr],
                        sampling_rate=sr_loaded, return_tensors="pt", padding=True
                    ).to(self.device)
                    with torch.no_grad():
                        try:
                            ids = self.model.generate(**inputs, max_new_tokens=512)
                        except TypeError:
                            # transformers dev: check_model_inputs decorator bug
                            # Filter to known-safe keys only
                            safe = {"input_ids", "attention_mask",
                                    "audio_features", "feature_attention_mask",
                                    "inputs_embeds"}
                            ids = self.model.generate(
                                **{k: v for k, v in inputs.items() if k in safe},
                                max_new_tokens=512
                            )
                    input_len = inputs["input_ids"].shape[-1]
                    text = self._processor.decode(
                        ids[0][input_len:], skip_special_tokens=True
                    ).strip()
            except Exception as e:
                print(f"  [Qwen3-ASR] chunk {start:.0f}s failed: {e}")
                text = ""
            finally:
                self._safe_delete(tmp)

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

        print(f"  [Qwen3-ASR] {len(out)} segments in {time.time()-t0:.1f}s")
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
