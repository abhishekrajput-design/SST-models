"""Qwen3-ASR-1.7B — 52-language + dialect support, 2026.

Uses the official qwen-asr package (pip install qwen-asr) when available.
Falls back to direct transformers loading (no pipeline) so we can work around
the transformers 5.6.0.dev0 check_model_inputs() bug at generate() call time.
Chunks audio into 60s pieces to stay within VRAM / RAM budgets.
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
import tempfile
from typing import List, Dict, Any
from .base import BaseTranscriber

# Local cache path (downloaded by download_models.py)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.join(_THIS_DIR, "..", "..", "models", "hf", "Qwen__Qwen3-ASR-1.7B")
HF_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
_HF_CACHE_DIR = os.path.join(_THIS_DIR, "..", "..", "models", "hf")

CHUNK_S = 60  # process in 60s chunks to avoid OOM on 6 GB GPU

# qwen-asr wants full language names
ISO_TO_LANG = {
    "en": "English", "zh": "Chinese", "fr": "French", "de": "German",
    "es": "Spanish", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
    "ru": "Russian", "it": "Italian", "ar": "Arabic", "vi": "Vietnamese",
    "nl": "Dutch", "pl": "Polish", "el": "Greek",
}

# ---------------------------------------------------------------------------
# Transformers 5.6.0.dev0 patch — applied once at module import time.
# check_model_inputs is defined as `def check_model_inputs(func)` but somewhere
# in the code it is called as check_model_inputs() with no arguments.
# We replace it with a no-op that handles both call patterns.
# ---------------------------------------------------------------------------
def _noop_check_model_inputs(func=None, *args, **kw):
    return func if callable(func) else lambda f: f


def _patch_all_transformers_modules() -> None:
    for name, mod in list(sys.modules.items()):
        if not name or "transformers" not in name or mod is None:
            continue
        if hasattr(mod, "check_model_inputs"):
            try:
                setattr(mod, "check_model_inputs", _noop_check_model_inputs)
            except Exception:
                pass


# Run once on import (handles modules already loaded)
_patch_all_transformers_modules()


class Qwen3AsrTranscriber(BaseTranscriber):
    name = "qwen3-asr-1.7b"

    def __init__(self, device: str = "cuda", model_dir: str | None = None):
        super().__init__(device=device, model_dir=model_dir)
        self._processor = None  # only used by the transformers-manual backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_model_instance(model) -> None:
        """Patch check_model_inputs on the model instance and its class.
        Covers trust_remote_code models that embed a broken check_model_inputs
        method rather than using the module-level function."""
        noop = _noop_check_model_inputs
        # Instance attribute (overrides class method)
        try:
            object.__setattr__(model, "check_model_inputs", noop)
        except Exception:
            pass
        # Class attribute
        cls = type(model)
        if hasattr(cls, "check_model_inputs"):
            try:
                cls.check_model_inputs = noop
            except Exception:
                pass
        # Module the class was defined in
        mod = sys.modules.get(getattr(cls, "__module__", ""), None)
        if mod is not None and hasattr(mod, "check_model_inputs"):
            try:
                setattr(mod, "check_model_inputs", noop)
            except Exception:
                pass

    @staticmethod
    def _safe_generate(model, **kwargs):
        """Call model.generate(); if the check_model_inputs TypeError fires,
        bypass the wrapper via __wrapped__ and retry."""
        import torch
        try:
            with torch.no_grad():
                return model.generate(**kwargs)
        except TypeError as exc:
            if "check_model_inputs" not in str(exc):
                raise
            raw = getattr(model.generate, "__wrapped__", None)
            if raw is None:
                raise RuntimeError(
                    f"check_model_inputs bug hit and __wrapped__ unavailable: {exc}"
                ) from exc
            print("  [Qwen3-ASR] generate() wrapper bug — using __wrapped__ fallback")
            with torch.no_grad():
                return raw(model, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Patch any transformers modules already in sys.modules
        _patch_all_transformers_modules()

        model_path = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HF_MODEL_ID

        # --- Backend 1: official qwen-asr package ---
        try:
            from qwen_asr import Qwen3ASRModel
            self.model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="cuda:0" if self.device == "cuda" else "cpu",
                max_inference_batch_size=4,
                max_new_tokens=512,
            )
            self._backend = "qwen-asr"
            return
        except ImportError:
            pass

        # --- Backend 2: transformers manual load (no pipeline) ---
        print("  [Qwen3-ASR] qwen-asr not installed — loading via transformers directly")
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, cache_dir=_HF_CACHE_DIR,
        )

        dtype = torch.float32  # CPU-safe; bfloat16 on CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            dtype = torch.bfloat16

        loaded = False
        for cls_name in (
            "Qwen2AudioForConditionalGeneration",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForSeq2SeqLM",
            "AutoModel",
        ):
            try:
                model_cls = getattr(__import__("transformers"), cls_name)
                self.model = model_cls.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    cache_dir=_HF_CACHE_DIR,
                )
                self.model.eval()
                if self.device != "cpu":
                    self.model = self.model.to(self.device)
                print(f"  [Qwen3-ASR] model loaded via {cls_name}")
                loaded = True
                break
            except Exception as e:
                print(f"  [Qwen3-ASR] {cls_name} failed: {e}")

        if not loaded:
            raise RuntimeError("Could not load Qwen3-ASR model with any known class")

        # Patch newly loaded trust_remote_code modules + model instance
        _patch_all_transformers_modules()
        self._patch_model_instance(self.model)
        self._backend = "transformers-manual"

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
        import torch
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
            text = ""
            try:
                if self._backend == "qwen-asr":
                    results = self.model.transcribe(audio=tmp, language=lang_full)
                    text = results[0].text.strip() if results else ""

                else:  # transformers-manual
                    import librosa
                    feat_ext = getattr(self._processor, "feature_extractor", self._processor)
                    sr = getattr(feat_ext, "sampling_rate", 16000)
                    audio_np, _ = librosa.load(tmp, sr=sr, mono=True)

                    inputs = self._processor(
                        audios=[audio_np], return_tensors="pt", sampling_rate=sr,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    gen_ids = self._safe_generate(
                        self.model, **inputs, max_new_tokens=512,
                    )
                    # Decode only the newly generated tokens
                    prompt_len = inputs.get("input_ids", torch.empty(1, 0)).shape[-1]
                    new_ids = gen_ids[:, prompt_len:]
                    text = self._processor.batch_decode(
                        new_ids, skip_special_tokens=True,
                    )[0].strip()

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
        self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
