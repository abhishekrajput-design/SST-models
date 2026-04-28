"""
Utilities for converting diarized segments into UI-friendly speaker roles.
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def _clean_agent_name(agent_name: str) -> str:
    return agent_name.replace("agent_", "").replace("_", " ").title()


def _segment_duration(segment: Dict) -> float:
    start = float(segment.get("start", 0) or 0)
    end = float(segment.get("end", 0) or 0)
    return max(0.0, end - start)


def _agent_identity(segment: Dict, identified: str) -> str:
    if identified.lower().startswith("agent_"):
        return identified
    if identified.upper() == "AGENT":
        return str(
            segment.get("agent_name")
            or segment.get("display_speaker")
            or "Agent"
        )
    return ""


def build_conversation_view(
    segments: List[Dict],
    agent_confidence_threshold: float = 0.25,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Resolve diarized speakers into display roles.

    - Preserve distinct agents when a diarized speaker matches one clearly.
    - If two diarized speakers collapse onto the same agent identity, keep only
      the stronger match as that agent and treat the other speaker as customer.
    - Number customer speakers only when there is more than one.
    """
    if not segments:
        return [], []

    speaker_stats: Dict[str, Dict] = {}
    for index, segment in enumerate(segments):
        diarized_label = str(segment.get("speaker") or f"SPEAKER_{index:02d}")
        identified = str(segment.get("identified_speaker") or diarized_label)
        confidence = float(segment.get("confidence", 0) or 0)
        if identified.upper() == "AGENT" and confidence <= 0:
            confidence = max(float(segment.get("_best_sim", 0) or 0), 1.0)
        duration = _segment_duration(segment)
        start = float(segment.get("start", 0) or 0)

        stats = speaker_stats.setdefault(
            diarized_label,
            {
                "first_start": start,
                "duration": 0.0,
                "segment_count": 0,
                "agent_scores": defaultdict(float),
                "best_agent_name": None,
                "best_agent_confidence": 0.0,
            },
        )
        stats["first_start"] = min(stats["first_start"], start)
        stats["duration"] += duration
        stats["segment_count"] += 1

        agent_identity = _agent_identity(segment, identified)
        if agent_identity:
            stats["agent_scores"][agent_identity] += max(confidence, 0.0) * max(duration, 1.0)
            score = stats["agent_scores"][agent_identity]
            if (
                stats["best_agent_name"] is None
                or confidence > stats["best_agent_confidence"]
                or (
                    confidence == stats["best_agent_confidence"]
                    and score > stats["agent_scores"][stats["best_agent_name"]]
                )
            ):
                stats["best_agent_name"] = agent_identity
                stats["best_agent_confidence"] = confidence

    agent_candidates: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    for diarized_label, stats in speaker_stats.items():
        if stats["best_agent_name"]:
            agent_candidates[stats["best_agent_name"]].append((diarized_label, stats))

    role_map: Dict[str, Dict] = {}
    for agent_name, candidates in agent_candidates.items():
        best_label, best_stats = max(
            candidates,
            key=lambda item: (
                item[1]["best_agent_confidence"],
                item[1]["agent_scores"][agent_name],
                item[1]["duration"],
                -item[1]["first_start"],
            ),
        )
        if best_stats["best_agent_confidence"] >= agent_confidence_threshold:
            clean_name = _clean_agent_name(agent_name)
            role_map[best_label] = {
                "role": "Agent",
                "role_class": "agent",
                "speaker_key": agent_name,
                "speaker_name": clean_name,
                "display_speaker": f"Agent ({clean_name})",
            }

    if not role_map and agent_candidates:
        best_label, best_stats = max(
            speaker_stats.items(),
            key=lambda item: (
                item[1]["best_agent_confidence"],
                item[1]["duration"],
                -item[1]["first_start"],
            ),
        )
        if best_stats["best_agent_name"] and (
            len(speaker_stats) == 1 or best_stats["best_agent_confidence"] >= agent_confidence_threshold
        ):
            agent_name = best_stats["best_agent_name"]
            clean_name = _clean_agent_name(agent_name)
            role_map[best_label] = {
                "role": "Agent",
                "role_class": "agent",
                "speaker_key": agent_name,
                "speaker_name": clean_name,
                "display_speaker": f"Agent ({clean_name})",
            }

    customer_labels = [
        label
        for label, _ in sorted(
            speaker_stats.items(),
            key=lambda item: (item[1]["first_start"], item[0]),
        )
        if label not in role_map
    ]
    many_customers = len(customer_labels) > 1
    for index, label in enumerate(customer_labels, start=1):
        display_name = f"Customer {index}" if many_customers else "Customer"
        role_map[label] = {
            "role": "Customer",
            "role_class": "customer",
            "speaker_key": f"customer_{index}",
            "speaker_name": display_name,
            "display_speaker": display_name,
        }

    annotated_segments: List[Dict] = []
    grouped_text: Dict[str, List[str]] = defaultdict(list)
    speaker_groups: Dict[str, Dict] = {}

    for segment in segments:
        diarized_label = str(segment.get("speaker") or "Unknown")
        role_info = role_map.get(
            diarized_label,
            {
                "role": "Customer",
                "role_class": "customer",
                "speaker_key": "customer_1",
                "speaker_name": "Customer",
                "display_speaker": "Customer",
            },
        )

        annotated = {
            "start": float(segment.get("start", 0) or 0),
            "end": float(segment.get("end", 0) or 0),
            "text": segment.get("text", ""),
            "confidence": float(segment.get("confidence", 0) or 0),
            "identified_speaker": segment.get("identified_speaker")
            or segment.get("speaker")
            or "Unknown",
            "speaker_label": diarized_label,
            "role": role_info["role"],
            "role_class": role_info["role_class"],
            "speaker_key": role_info["speaker_key"],
            "speaker_name": role_info["speaker_name"],
            "display_speaker": role_info["display_speaker"],
        }
        annotated_segments.append(annotated)

        group = speaker_groups.setdefault(
            role_info["speaker_key"],
            {
                "speaker_key": role_info["speaker_key"],
                "role": role_info["role"],
                "role_class": role_info["role_class"],
                "speaker_name": role_info["speaker_name"],
                "display_speaker": role_info["display_speaker"],
                "speaker_label": diarized_label,
                "segment_count": 0,
                "talk_time": 0.0,
                "first_start": annotated["start"],
            },
        )
        group["segment_count"] += 1
        group["talk_time"] += max(0.0, annotated["end"] - annotated["start"])
        group["first_start"] = min(group["first_start"], annotated["start"])

        if annotated["text"]:
            grouped_text[role_info["speaker_key"]].append(annotated["text"])

    ordered_groups: List[Dict] = []
    for speaker_key, group in sorted(
        speaker_groups.items(),
        key=lambda item: (item[1]["first_start"], item[0]),
    ):
        ordered_groups.append(
            {
                "speaker_key": speaker_key,
                "role": group["role"],
                "role_class": group["role_class"],
                "speaker_name": group["speaker_name"],
                "display_speaker": group["display_speaker"],
                "speaker_label": group["speaker_label"],
                "segment_count": group["segment_count"],
                "talk_time": round(group["talk_time"], 2),
                "text": "\n".join(grouped_text.get(speaker_key, [])),
            }
        )

    return annotated_segments, ordered_groups
