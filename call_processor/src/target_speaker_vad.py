from __future__ import annotations

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio

_TARGET_SR = 16000


def _load_audio_mono(audio_path: str) -> np.ndarray:
    data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    audio = data.mean(axis=1)
    if sr != _TARGET_SR:
        wav = torch.from_numpy(audio).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, _TARGET_SR)
        audio = wav.squeeze(0).numpy()
    return audio


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalise_stack(voiceprints: np.ndarray) -> np.ndarray:
    stack = np.asarray(voiceprints, dtype=np.float32)
    if stack.ndim == 1:
        stack = stack.reshape(1, -1)
    if stack.ndim != 2 or stack.shape[0] == 0 or stack.shape[1] == 0:
        raise ValueError(f"invalid voiceprint shape {stack.shape}")
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return stack / norms


def _merge_windows(
    windows: list[dict],
    max_gap: float,
    min_dur: float,
) -> list[dict]:
    if not windows:
        return []
    merged: list[dict] = []
    cur = dict(windows[0])
    for w in windows[1:]:
        if w["start"] - cur["end"] <= max_gap:
            cur["end"] = w["end"]
        else:
            merged.append(cur)
            cur = dict(w)
    merged.append(cur)
    return [s for s in merged if s["end"] - s["start"] >= min_dur]


class TargetSpeakerVAD:
    def __init__(
        self,
        voiceprint: np.ndarray,
        threshold: float = 0.42,
        window_s: float = 1.5,
        stride_s: float = 0.5,
        background_voiceprints: np.ndarray | None = None,
        margin: float = 0.0,
    ):
        self._voiceprints = _normalise_stack(voiceprint)
        # Kept for older callers/tests that inspect the single-target field.
        self._voiceprint = self._voiceprints[0]
        self._background_voiceprints = (
            _normalise_stack(background_voiceprints)
            if background_voiceprints is not None
            else None
        )
        if (
            self._background_voiceprints is not None
            and self._background_voiceprints.shape[1] != self._voiceprints.shape[1]
        ):
            raise ValueError(
                f"background voiceprint dim {self._background_voiceprints.shape[1]} "
                f"does not match target dim {self._voiceprints.shape[1]}"
            )
        self._threshold = threshold
        self._window_s = window_s
        self._stride_s = stride_s
        self._margin = margin

    def _score_embedding(self, emb: np.ndarray) -> dict:
        if emb.shape[0] != self._voiceprints.shape[1]:
            raise ValueError(
                f"TS-VAD embedding dim {emb.shape[0]} does not match "
                f"voiceprint dim {self._voiceprints.shape[1]}"
            )
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        target_cos = float(np.max(self._voiceprints @ emb))
        best_other = 0.0
        if self._background_voiceprints is not None and len(self._background_voiceprints):
            best_other = float(np.max(self._background_voiceprints @ emb))
        score_margin = target_cos - best_other
        is_agent = target_cos >= self._threshold
        if self._background_voiceprints is not None and len(self._background_voiceprints):
            is_agent = is_agent and score_margin >= self._margin
        return {
            "cosine": round(target_cos, 4),
            "target_cosine": round(target_cos, 4),
            "best_other_cosine": round(best_other, 4),
            "margin": round(score_margin, 4),
            "is_agent": bool(is_agent),
        }

    def detect(self, audio_path: str) -> list[dict]:
        audio = _load_audio_mono(audio_path)
        total_samples = len(audio)
        win_samples = int(self._window_s * _TARGET_SR)
        stride_samples = int(self._stride_s * _TARGET_SR)

        from src.embedding_campp import get_model
        model = get_model()

        raw_windows: list[dict] = []
        offset = 0
        while offset + win_samples <= total_samples:
            chunk = audio[offset: offset + win_samples]
            start = offset / _TARGET_SR
            end = (offset + win_samples) / _TARGET_SR

            emb = model.embed_chunk(chunk, _TARGET_SR)
            if emb is None:
                offset += stride_samples
                continue
            scored = self._score_embedding(emb)
            raw_windows.append({
                "start": round(start, 3),
                "end": round(end, 3),
                **scored,
            })
            offset += stride_samples

        if offset < total_samples:
            chunk = audio[offset:]
            if len(chunk) >= int(0.2 * _TARGET_SR):
                start = offset / _TARGET_SR
                end = total_samples / _TARGET_SR
                emb = model.embed_chunk(chunk, _TARGET_SR)
                if emb is not None:
                    scored = self._score_embedding(emb)
                    raw_windows.append({
                        "start": round(start, 3),
                        "end": round(end, 3),
                        **scored,
                    })

        smoothed: list[dict] = []
        n = len(raw_windows)
        for i, w in enumerate(raw_windows):
            if not w["is_agent"]:
                smoothed.append(w)
                continue
            prev_ok = i > 0 and raw_windows[i - 1]["is_agent"]
            next_ok = i < n - 1 and raw_windows[i + 1]["is_agent"]
            if prev_ok or next_ok:
                smoothed.append(w)
            else:
                smoothed.append({**w, "is_agent": False})

        return smoothed

    def agent_segments(self, audio_path: str) -> list[dict]:
        windows = self.detect(audio_path)
        agent_wins = [{"start": w["start"], "end": w["end"]} for w in windows if w["is_agent"]]
        return _merge_windows(agent_wins, max_gap=0.5, min_dur=1.0)


def apply_tsvad_to_diarized(
    tsvad: TargetSpeakerVAD,
    segments: list[dict],
    cosine_override: float = 0.35,
) -> tuple[list[dict], list[dict]]:
    if not segments:
        return [], []

    audio_path: str | None = None
    for seg in segments:
        if "audio_path" in seg:
            audio_path = seg["audio_path"]
            break

    window_scores: list[dict] | None = None
    if audio_path:
        window_scores = tsvad.detect(audio_path)

    def _get_cosine_for_segment(seg: dict) -> float | None:
        if "tsvad_cosine" in seg:
            return float(seg["tsvad_cosine"])
        if window_scores is None:
            return None
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        overlapping = [
            w for w in window_scores
            if w["end"] > seg_start and w["start"] < seg_end
        ]
        if not overlapping:
            return None
        weights = [
            min(w["end"], seg_end) - max(w["start"], seg_start)
            for w in overlapping
        ]
        total = sum(weights)
        if total == 0:
            return None
        return float(sum(w["cosine"] * wt for w, wt in zip(overlapping, weights)) / total)

    agent_segs: list[dict] = []
    other_segs: list[dict] = []

    for seg in segments:
        cos = _get_cosine_for_segment(seg)
        annotated = dict(seg)
        if cos is not None:
            annotated["tsvad_cosine"] = round(cos, 4)
            is_agent = cos >= cosine_override
        else:
            is_agent = seg.get("is_agent", False)

        if is_agent:
            agent_segs.append(annotated)
        else:
            other_segs.append(annotated)

    def _merge_seg_list(segs: list[dict]) -> list[dict]:
        if not segs:
            return []
        segs_sorted = sorted(segs, key=lambda s: float(s["start"]))
        merged: list[dict] = []
        cur = dict(segs_sorted[0])
        for s in segs_sorted[1:]:
            if float(s["start"]) - float(cur["end"]) <= 1.0:
                cur["end"] = max(float(cur["end"]), float(s["end"]))
                if "text" in cur and "text" in s:
                    cur["text"] = cur["text"].rstrip() + " " + s["text"].lstrip()
            else:
                merged.append(cur)
                cur = dict(s)
        merged.append(cur)
        return [s for s in merged if float(s["end"]) - float(s["start"]) >= 1.0]

    return _merge_seg_list(agent_segs), other_segs
