"""NVIDIA Canary-Qwen 2.5B — multilingual ASR with NeMo speechlm2 SALM.

Tries nemo.collections.speechlm2.models.SALM first (NeMo 2.5+), falls back
to nemo.collections.asr.models.EncDecMultiTaskModel for older NeMo builds.

Chunks audio into 25s pieces to keep peak VRAM under 6 GB on the SALM model
(~5 GB at bfloat16 leaves ~1 GB headroom for activations).
"""
from __future__ import annotations
import os
import time
import tempfile
import subprocess
from typing import List, Dict, Any
from .base import BaseTranscriber

MODEL_ID = "nvidia/canary-qwen-2.5b"
CHUNK_S = 25  # shorter than Parakeet; SALM activations are larger


class CanaryQwenTranscriber(BaseTranscriber):
    name = "canary-qwen-2.5b"
    supports_word_timestamps = False

    def load(self) -> None:
        if self.model is not None:
            return
        # Probe CUDA — driver may be incompatible even when is_available() is True
        _use_cuda = False
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.current_device()
                    torch.zeros(1).cuda()
                    _use_cuda = True
                    torch.cuda.empty_cache()
                except Exception as _e:
                    print(f"  [Canary] CUDA unavailable ({_e}), falling back to CPU")
        except ImportError:
            pass
        self._use_cuda = _use_cuda

        # Monkey-patch cuda.is_available so NeMo's internal init doesn't crash
        # on systems where the driver is present but incompatible.
        import torch as _torch
        _orig_is_avail = _torch.cuda.is_available
        if not _use_cuda:
            _torch.cuda.is_available = lambda: False

        try:
            load_exc = None
            self.model = None

            # Attempt 1: NeMo 2.5+ speechlm2 SALM
            try:
                from nemo.collections.speechlm2.models import SALM
                cache_dir = self.model_dir or os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "nemo"
                )
                os.makedirs(cache_dir, exist_ok=True)
                os.environ["NEMO_CACHE_DIR"] = cache_dir
                # from_pretrained has no dtype param; we cast after load (see below).
                self.model = SALM.from_pretrained(MODEL_ID)
                self._api = "salm"
            except (ImportError, AttributeError, TypeError) as e:
                load_exc = e

            # Attempt 2: older NeMo EncDecMultiTaskModel (Canary-1B style)
            if self.model is None:
                try:
                    import nemo.collections.asr as nemo_asr
                    cache_dir = self.model_dir or os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "nemo"
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    os.environ["NEMO_CACHE_DIR"] = cache_dir
                    for cls_name in ("EncDecMultiTaskModel", "ASRModel"):
                        try:
                            cls = getattr(nemo_asr.models, cls_name)
                            self.model = cls.from_pretrained(model_name=MODEL_ID)
                            self._api = "multitask"
                            break
                        except (TypeError, AttributeError) as e:
                            load_exc = e
                            continue
                except ImportError as e:
                    load_exc = e

            if self.model is None:
                raise RuntimeError(
                    f"Could not load Canary-Qwen via any NeMo API: {load_exc}\n"
                    "Ensure NeMo ≥2.5 is installed: pip install 'nemo_toolkit[asr]'"
                )
        finally:
            _torch.cuda.is_available = _orig_is_avail

        if _use_cuda:
            # Cast to bfloat16 before moving to GPU: float32 = ~10 GB, bfloat16 = ~5 GB.
            # bfloat16 has float32 exponent range so no overflow on audio values.
            # Convert on CPU first, then move to GPU in one shot.
            try:
                self.model = self.model.to(_torch.bfloat16)
            except Exception as _e:
                print(f"  [Canary] bfloat16 cast failed ({_e}), keeping float32")
            try:
                self.model = self.model.cuda()
            except RuntimeError as _e:
                print(f"  [Canary] CUDA OOM or error ({_e}), falling back to CPU")
                self.model = self.model.float()  # restore float32 for CPU
                self._use_cuda = False
        self.model.eval()
        print(f"  [Canary] Loaded via {self._api} API, cuda={self._use_cuda}")

    # ------------------------------------------------------------------ #

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

    def _audio_duration(self, audio_path: str) -> float:
        import json
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", audio_path],
            capture_output=True, text=True
        )
        info = json.loads(r.stdout)
        return float(info["format"]["duration"])

    def _transcribe_chunk(self, tmp_path: str) -> str:
        """Run inference on a single WAV chunk, return transcript string."""
        import torch
        with torch.no_grad():
            if self._api == "salm":
                # High-level SALM API: chat-style prompt with embedded audio path.
                # model.audio_locator_tag is the <|audio|> placeholder token string.
                locator = self.model.audio_locator_tag
                prompt = [
                    [
                        {
                            "role": "user",
                            "content": f"Transcribe the following: {locator}",
                            "audio": [tmp_path],
                        }
                    ]
                ]
                answer_ids = self.model.generate(prompt, max_new_tokens=512)
                # answer_ids shape: (1, seq_len) — includes both prompt and answer tokens.
                # Decode and strip any leading/trailing whitespace.
                tokenizer = self.model.tokenizer
                if hasattr(tokenizer, "ids_to_text"):
                    # NeMo tokenizer
                    text = tokenizer.ids_to_text(answer_ids[0].tolist())
                else:
                    # HuggingFace tokenizer
                    text = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
                return text.strip()
            else:
                # Older EncDecMultiTaskModel (Canary-1B style)
                results = self.model.transcribe([tmp_path], batch_size=1)
                if isinstance(results, list) and results:
                    hyp = results[0]
                else:
                    hyp = results
                if hasattr(hyp, "text"):
                    return hyp.text.strip()
                return str(hyp).strip()

    # ------------------------------------------------------------------ #

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()
        total_dur = self._audio_duration(audio_path)
        out: List[Dict[str, Any]] = []

        import torch
        start = 0.0
        while start < total_dur:
            dur = min(CHUNK_S, total_dur - start)
            if dur < 0.5:
                break

            tmp = self._wav_chunk(audio_path, start, dur)
            try:
                text = self._transcribe_chunk(tmp)
            finally:
                self._safe_delete(tmp)
                if self._use_cuda:
                    torch.cuda.empty_cache()

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

        print(f"  [Canary] {len(out)} segments in {time.time()-t0:.1f}s")
        return out

    def unload(self) -> None:
        import gc
        try:
            import torch
            if self.model is not None and torch.cuda.is_available():
                try:
                    self.model = self.model.cpu()
                except Exception:
                    pass
        except ImportError:
            pass
        del self.model
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
