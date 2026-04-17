"""Pluggable ASR transcriber registry. Pick a backend by string name."""
from __future__ import annotations
from .base import BaseTranscriber
from .whisper_turbo import WhisperTurboTranscriber
from .cohere       import CohereTranscriber
from .parakeet_v3  import ParakeetV3Transcriber
from .deepgram_asr import DeepgramTranscriber

# Optional / experimental models — import failures must not break the registry
try:
    from .qwen3_asr import Qwen3AsrTranscriber
except Exception as _e:
    print(f"[transcribers] qwen3_asr unavailable: {_e}")
    Qwen3AsrTranscriber = None  # type: ignore

try:
    from .vibevoice_asr import VibeVoiceAsrTranscriber
except Exception as _e:
    print(f"[transcribers] vibevoice_asr unavailable: {_e}")
    VibeVoiceAsrTranscriber = None  # type: ignore

try:
    from .canary_qwen import CanaryQwenTranscriber
except Exception as _e:
    print(f"[transcribers] canary_qwen unavailable: {_e}")
    CanaryQwenTranscriber = None  # type: ignore

try:
    from .groq_whisper import GroqWhisperTranscriber
except Exception as _e:
    print(f"[transcribers] groq_whisper unavailable: {_e}")
    GroqWhisperTranscriber = None  # type: ignore

try:
    from .assemblyai_asr import AssemblyAITranscriber
except Exception as _e:
    print(f"[transcribers] assemblyai_asr unavailable: {_e}")
    AssemblyAITranscriber = None  # type: ignore

# Aliases — the UI dropdown values map to these keys
TRANSCRIBERS = {
    # Whisper family — all routed through faster-whisper wrapper
    "whisper-large-v3-turbo":     WhisperTurboTranscriber,
    "whisper-large-v3":           WhisperTurboTranscriber,
    "large-v3-turbo":             WhisperTurboTranscriber,
    "large-v3":                   WhisperTurboTranscriber,
    "distil-large-v3":            WhisperTurboTranscriber,
    "distil-large-v3.5":          WhisperTurboTranscriber,
    "distil-whisper-large-v3.5":  WhisperTurboTranscriber,  # friendly alias
    # Non-Whisper local backends
    "cohere-transcribe-03-2026": CohereTranscriber,
    "parakeet-tdt-0.6b-v3":      ParakeetV3Transcriber,
    **({"canary-qwen-2.5b":  CanaryQwenTranscriber}  if CanaryQwenTranscriber  else {}),
    **({"qwen3-asr-1.7b":   Qwen3AsrTranscriber}   if Qwen3AsrTranscriber   else {}),
    **({"vibevoice-asr":    VibeVoiceAsrTranscriber} if VibeVoiceAsrTranscriber else {}),
    # Deepgram cloud API (no GPU needed)
    "deepgram-nova-3":           lambda **kw: DeepgramTranscriber(model="nova-3", **kw),
    "deepgram-nova-2-phonecall": lambda **kw: DeepgramTranscriber(model="nova-2-phonecall", **kw),
    "deepgram-nova-2-meeting":   lambda **kw: DeepgramTranscriber(model="nova-2-meeting", **kw),
    # Groq cloud Whisper (ultra-fast, no GPU needed)
    **({"groq-whisper-large-v3-turbo": lambda **kw: GroqWhisperTranscriber(model="whisper-large-v3-turbo", **kw),
        "groq-whisper-large-v3":       lambda **kw: GroqWhisperTranscriber(model="whisper-large-v3", **kw),
       } if GroqWhisperTranscriber else {}),
    # AssemblyAI Universal-2 (cloud, best for call-center audio)
    **({"assemblyai-universal-2": AssemblyAITranscriber} if AssemblyAITranscriber else {}),
}

DEFAULT = "whisper-large-v3-turbo"


def get_transcriber(name: str, device: str = "cuda", model_dir: str | None = None,
                    **kwargs) -> BaseTranscriber:
    """Return an instantiated transcriber. For Whisper variants the model_size
    is derived from the dropdown name."""
    factory = TRANSCRIBERS.get(name)
    if factory is None:
        raise ValueError(f"Unknown transcriber '{name}'. Options: {list(TRANSCRIBERS)}")
    if factory is WhisperTurboTranscriber:
        # Normalize friendly aliases before passing to model_size
        _size = "distil-large-v3.5" if name == "distil-whisper-large-v3.5" else name
        return factory(device=device, model_dir=model_dir, model_size=_size, **kwargs)
    # Lambda factories (e.g. Deepgram variants) and plain classes both work here
    return factory(device=device, model_dir=model_dir, **kwargs)


__all__ = [
    "BaseTranscriber",
    "TRANSCRIBERS",
    "DEFAULT",
    "get_transcriber",
    "WhisperTurboTranscriber",
    "CohereTranscriber",
    "ParakeetV3Transcriber",
    "CanaryQwenTranscriber",
    "GroqWhisperTranscriber",
    "AssemblyAITranscriber",
    "Qwen3AsrTranscriber",
    "VibeVoiceAsrTranscriber",
    "DeepgramTranscriber",
]
