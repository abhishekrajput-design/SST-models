"""Pluggable ASR transcriber registry. Pick a backend by string name."""
from __future__ import annotations
from .base import BaseTranscriber
from .whisper_turbo import WhisperTurboTranscriber
from .cohere       import CohereTranscriber
from .parakeet_v3  import ParakeetV3Transcriber
from .qwen3_asr    import Qwen3AsrTranscriber
from .vibevoice_asr import VibeVoiceAsrTranscriber
from .deepgram_asr import DeepgramTranscriber

# Aliases — the UI dropdown values map to these keys
TRANSCRIBERS = {
    # Whisper family — all routed through faster-whisper wrapper
    "whisper-large-v3-turbo": WhisperTurboTranscriber,
    "whisper-large-v3":       WhisperTurboTranscriber,
    "large-v3-turbo":         WhisperTurboTranscriber,
    "large-v3":               WhisperTurboTranscriber,
    "distil-large-v3":        WhisperTurboTranscriber,
    "distil-large-v3.5":      WhisperTurboTranscriber,
    # Non-Whisper local backends
    "cohere-transcribe-03-2026": CohereTranscriber,
    "parakeet-tdt-0.6b-v3":      ParakeetV3Transcriber,
    "qwen3-asr-1.7b":            Qwen3AsrTranscriber,
    "vibevoice-asr":             VibeVoiceAsrTranscriber,
    # Deepgram cloud API (no GPU needed)
    "deepgram-nova-3":           lambda **kw: DeepgramTranscriber(model="nova-3", **kw),
    "deepgram-nova-2-phonecall": lambda **kw: DeepgramTranscriber(model="nova-2-phonecall", **kw),
    "deepgram-nova-2-meeting":   lambda **kw: DeepgramTranscriber(model="nova-2-meeting", **kw),
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
        return factory(device=device, model_dir=model_dir, model_size=name, **kwargs)
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
    "Qwen3AsrTranscriber",
    "VibeVoiceAsrTranscriber",
    "DeepgramTranscriber",
]
