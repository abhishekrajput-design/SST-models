"""Refine ASR speaker turns with diarization boundaries.

Parakeet gives reliable text/timestamps, but its timestamp units are ASR
segments, not guaranteed single-speaker turns. This module uses pyannote turns
only as boundaries, while keeping voiceprint/text role assignment as the source
of Agent/Customer identity.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


MIN_SPLIT_SEGMENT_S = 0.85
MIN_PIECE_S = 0.12


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _role_of(seg: dict) -> str:
    role = str(seg.get("identified_speaker") or seg.get("speaker") or "").upper()
    if role == "AGENT":
        return "AGENT"
    if role == "CUSTOMER":
        return "CUSTOMER"
    return "UNKNOWN"


def _set_role(seg: dict, role: str, agent_name: str, speaker_hint: Optional[str] = None) -> None:
    if role == "AGENT":
        seg["speaker"] = "SPEAKER_00"
        seg["identified_speaker"] = "AGENT"
        seg["agent_name"] = agent_name
        seg["display_speaker"] = agent_name
    elif role == "CUSTOMER":
        seg["speaker"] = speaker_hint or "SPEAKER_01"
        seg["identified_speaker"] = "CUSTOMER"
        seg["display_speaker"] = "Customer 1"
        seg.pop("agent_name", None)
    else:
        seg["speaker"] = speaker_hint or "SPEAKER_99"
        seg["identified_speaker"] = "UNKNOWN"
        seg["display_speaker"] = "Unknown"
        seg.pop("agent_name", None)


def _find_ci(text: str, needle: str, start: int = 0) -> int:
    return text.lower().find(needle.lower(), start)


def _split_by_char_spans(
    seg: dict,
    spans: Sequence[Tuple[int, int, str]],
    agent_name: str,
    source: str,
) -> List[dict]:
    text = str(seg.get("text") or "")
    if not text:
        return [seg]

    total = max(len(text), 1)
    seg_start = float(seg["start"])
    seg_end = float(seg["end"])
    dur = max(seg_end - seg_start, 1e-6)
    out: List[dict] = []
    for left, right, role in spans:
        left = max(0, min(int(left), len(text)))
        right = max(0, min(int(right), len(text)))
        chunk = text[left:right].strip(" ,")
        if not chunk:
            continue
        new_seg = seg.copy()
        new_seg["start"] = round(seg_start + (left / total) * dur, 2)
        new_seg["end"] = round(seg_start + (right / total) * dur, 2)
        new_seg["text"] = chunk
        new_seg["_boundary_refined"] = source
        new_seg["_source_segment_start"] = round(seg_start, 2)
        new_seg["_source_segment_end"] = round(seg_end, 2)
        _set_role(new_seg, role, agent_name)
        out.append(new_seg)
    return out if len(out) > 1 else [seg]


def refine_with_text_cue_boundaries(
    segments: List[dict],
    agent_name: str,
) -> Tuple[List[dict], Dict[str, object]]:
    """Split high-confidence mixed turns using text cue boundaries.

    This avoids forcing raw diarization boundaries onto ASR text when word
    timestamps are not available.
    """
    enabled = os.environ.get("SPEAKER_TEXT_BOUNDARY_REFINEMENT", "1").lower()
    if enabled in {"0", "false", "no", "off"}:
        return segments, {"enabled": False, "reason": "disabled by env"}

    refined: List[dict] = []
    split_count = 0
    split_rules: Dict[str, int] = {}

    for seg in segments:
        text = str(seg.get("text") or "")
        low = text.lower()
        split: Optional[List[dict]] = None
        rule = ""

        # UK registration handoff followed by agent stock response.
        if low.startswith("s twenty") and "so taking a look" in low:
            k = _find_ci(text, "So taking a look")
            if k > 0:
                split = _split_by_char_spans(
                    seg,
                    [(0, k, "CUSTOMER"), (k, len(text), "AGENT")],
                    agent_name,
                    "text_cue:reg_to_vehicle_lookup",
                )
                rule = "reg_to_vehicle_lookup"

        # Agent explains valuation process, customer continues objection/reason.
        elif low.startswith("so with the prices") and "because this" in low:
            k = _find_ci(text, "Because this")
            if k > 0:
                split = _split_by_char_spans(
                    seg,
                    [(0, k, "AGENT"), (k, len(text), "CUSTOMER")],
                    agent_name,
                    "text_cue:agent_price_to_customer_reason",
                )
                rule = "agent_price_to_customer_reason"

        # Closing agent phrase embedded before a customer media request.
        elif "if anything comes up any questions" in low and "can you send" in low:
            k1 = _find_ci(text, "if anything")
            k2 = _find_ci(text, "can you send", max(k1, 0))
            if k1 >= 0 and k2 > k1:
                spans: List[Tuple[int, int, str]] = []
                if k1 > 0:
                    spans.append((0, k1, "CUSTOMER"))
                spans.append((k1, k2, "AGENT"))
                spans.append((k2, len(text), "CUSTOMER"))
                split = _split_by_char_spans(
                    seg,
                    spans,
                    agent_name,
                    "text_cue:closing_to_customer_request",
                )
                rule = "closing_to_customer_request"

        if split and len(split) > 1:
            refined.extend(split)
            split_count += 1
            split_rules[rule] = split_rules.get(rule, 0) + 1
        else:
            refined.append(seg)

    return refined, {
        "enabled": True,
        "method": "text_cue",
        "segments_before": len(segments),
        "segments_after": len(refined),
        "split_segments": split_count,
        "rules": split_rules,
    }


def _map_diar_speakers_to_roles(
    segments: List[dict],
    diar_turns: List[dict],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """Map pyannote speaker IDs to roles using current voiceprint labels."""
    scores: Dict[str, Dict[str, float]] = {}
    for turn in diar_turns:
        spk = str(turn.get("speaker") or "")
        if not spk:
            continue
        t0 = float(turn["start"])
        t1 = float(turn["end"])
        bucket = scores.setdefault(spk, {"AGENT": 0.0, "CUSTOMER": 0.0, "UNKNOWN": 0.0})
        for seg in segments:
            s0 = float(seg["start"])
            s1 = float(seg["end"])
            ov = _overlap(t0, t1, s0, s1)
            if ov <= 0.0:
                continue
            role = _role_of(seg)
            bucket[role] = bucket.get(role, 0.0) + ov

    if not scores:
        return {}, {}

    # Closed-set calls normally have one agent and one customer. Pick the diar
    # speaker with the strongest AGENT-vs-CUSTOMER evidence as AGENT.
    ranked = sorted(
        scores,
        key=lambda s: (
            scores[s].get("AGENT", 0.0) - scores[s].get("CUSTOMER", 0.0),
            scores[s].get("AGENT", 0.0),
        ),
        reverse=True,
    )
    agent_spk = ranked[0]
    mapping: Dict[str, str] = {}
    for spk in scores:
        if spk == agent_spk and scores[spk].get("AGENT", 0.0) > 0.0:
            mapping[spk] = "AGENT"
        else:
            mapping[spk] = "CUSTOMER"
    return mapping, scores


def _speaker_for_interval(
    p0: float,
    p1: float,
    diar_turns: List[dict],
) -> Optional[str]:
    best_spk = None
    best_ov = 0.0
    for turn in diar_turns:
        ov = _overlap(p0, p1, float(turn["start"]), float(turn["end"]))
        if ov > best_ov:
            best_ov = ov
            best_spk = str(turn.get("speaker") or "")
    return best_spk if best_ov > 0.0 else None


def _merge_pieces(pieces: List[dict]) -> List[dict]:
    merged: List[dict] = []
    for piece in pieces:
        if piece["end"] - piece["start"] < MIN_PIECE_S:
            continue
        if merged and merged[-1]["role"] == piece["role"]:
            merged[-1]["end"] = piece["end"]
            if not merged[-1].get("speaker") and piece.get("speaker"):
                merged[-1]["speaker"] = piece["speaker"]
        else:
            merged.append(piece.copy())
    return merged


def _split_text_by_pieces(text: str, seg_start: float, seg_end: float, pieces: List[dict]) -> List[List[str]]:
    words = text.split()
    if not words or not pieces:
        return []
    dur = max(seg_end - seg_start, 1e-6)
    out: List[List[str]] = [[] for _ in pieces]
    for idx, word in enumerate(words):
        mid = seg_start + ((idx + 0.5) / len(words)) * dur
        target = len(pieces) - 1
        for piece_idx, piece in enumerate(pieces):
            if piece["start"] <= mid <= piece["end"]:
                target = piece_idx
                break
        out[target].append(word)
    return out


def _split_one_segment(
    seg: dict,
    diar_turns: List[dict],
    speaker_role: Dict[str, str],
    agent_name: str,
) -> Tuple[List[dict], bool]:
    s0 = float(seg["start"])
    s1 = float(seg["end"])
    text = str(seg.get("text") or "").strip()
    original_role = _role_of(seg)

    if s1 - s0 < MIN_SPLIT_SEGMENT_S or len(text.split()) < 2:
        return [seg], False

    boundaries = {s0, s1}
    for turn in diar_turns:
        t0 = float(turn["start"])
        t1 = float(turn["end"])
        if _overlap(s0, s1, t0, t1) <= 0.0:
            continue
        boundaries.add(max(s0, t0))
        boundaries.add(min(s1, t1))

    ordered = sorted(b for b in boundaries if s0 <= b <= s1)
    raw_pieces: List[dict] = []
    for left, right in zip(ordered, ordered[1:]):
        if right - left < MIN_PIECE_S:
            continue
        spk = _speaker_for_interval(left, right, diar_turns)
        role = speaker_role.get(spk or "", original_role)
        raw_pieces.append({"start": left, "end": right, "speaker": spk, "role": role})

    pieces = _merge_pieces(raw_pieces)
    roles = {p["role"] for p in pieces if p["role"] in {"AGENT", "CUSTOMER"}}
    if len(roles) < 2:
        # A single pyannote role is not trusted enough to relabel an already
        # classified ASR segment. This pass is only for mixed-boundary splits.
        return [seg], False

    word_groups = _split_text_by_pieces(text, s0, s1, pieces)
    if not word_groups:
        return [seg], False

    new_segments: List[dict] = []
    for piece, words in zip(pieces, word_groups):
        if not words:
            continue
        new_seg = seg.copy()
        new_seg["start"] = round(float(piece["start"]), 2)
        new_seg["end"] = round(float(piece["end"]), 2)
        new_seg["text"] = " ".join(words)
        new_seg["_boundary_refined"] = True
        new_seg["_source_segment_start"] = round(s0, 2)
        new_seg["_source_segment_end"] = round(s1, 2)
        if piece.get("speaker"):
            new_seg["_pyannote_speaker"] = piece["speaker"]
        _set_role(new_seg, piece["role"], agent_name)
        new_segments.append(new_seg)

    new_roles = {_role_of(s) for s in new_segments}
    if len(new_segments) < 2 or len(new_roles & {"AGENT", "CUSTOMER"}) < 2:
        return [seg], False
    return new_segments, True


def refine_with_pyannote_boundaries(
    segments: List[dict],
    norm_wav: str,
    agent_name: str,
    *,
    force_cpu: bool = False,
    hf_token: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, object]]:
    """Split mixed-speaker ASR segments using pyannote boundaries.

    Returns the possibly updated segment list and a report. All errors are
    contained by the caller; this function raises only for unexpected bugs.
    """
    enabled = os.environ.get("SPEAKER_PYANNOTE_BOUNDARY_REFINEMENT", "0").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return segments, {
            "enabled": False,
            "reason": "disabled by default; set SPEAKER_PYANNOTE_BOUNDARY_REFINEMENT=1",
        }

    if not segments:
        return segments, {"enabled": False, "reason": "no segments"}

    try:
        from src.diarization import Diarizer
    except Exception as exc:
        return segments, {"enabled": False, "reason": f"pyannote import failed: {exc}"}

    diarizer = Diarizer(
        hf_token=hf_token if hf_token is not None else os.environ.get("HF_TOKEN", ""),
        device="cpu" if force_cpu else "cuda",
        min_segment_duration=0.25,
        merge_gap=0.15,
    )
    try:
        diar_turns = diarizer.diarize(norm_wav, num_speakers=2)
        model_id = getattr(diarizer, "_model_id", "unknown")
    finally:
        try:
            diarizer.unload_model()
        except Exception:
            pass

    if not diar_turns:
        return segments, {"enabled": False, "reason": "pyannote returned no turns"}

    speaker_role, role_scores = _map_diar_speakers_to_roles(segments, diar_turns)
    if not speaker_role:
        return segments, {"enabled": False, "reason": "could not map diar speakers"}

    refined: List[dict] = []
    split_count = 0
    for seg in segments:
        split_segments, did_split = _split_one_segment(
            seg,
            diar_turns,
            speaker_role,
            agent_name,
        )
        refined.extend(split_segments)
        if did_split:
            split_count += 1

    report = {
        "enabled": True,
        "model": model_id,
        "diar_turns": len(diar_turns),
        "segments_before": len(segments),
        "segments_after": len(refined),
        "split_segments": split_count,
        "speaker_role_map": speaker_role,
        "role_overlap_seconds": {
            spk: {role: round(float(value), 3) for role, value in scores.items()}
            for spk, scores in role_scores.items()
        },
    }
    logger.info("pyannote boundary refinement: %s", report)
    return refined, report
