"""
Chat-style transcript formatter.

Takes pipeline output (result.json) and renders it in two clear views:
  1. CONVERSATION VIEW  -- chronological chat bubbles with timestamps
  2. SPLIT VIEW         -- Agent lines on left, Customer lines on right

Usage:
    python format_chat.py data/processed/<audio_name>/result.json
    python format_chat.py data/processed/<audio_name>/result.json --output transcript_chat.txt
"""

import json
import os
import sys
import argparse
from typing import Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def fmt_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def render_conversation(segments: List[Dict]) -> str:
    """Render segments as a chronological chat conversation."""
    lines = []
    lines.append("=" * 70)
    lines.append("  CONVERSATION TRANSCRIPT")
    lines.append("=" * 70)
    lines.append("")

    for seg in segments:
        speaker = seg.get("identified_speaker", seg.get("speaker", "Unknown"))
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = fmt_time(seg["start"])
        end = fmt_time(seg["end"])
        conf = seg.get("confidence", 0)

        # Clean up agent name for display
        display_name = speaker.replace("agent_", "").title() if speaker.startswith("agent_") else speaker

        if speaker.startswith("agent_") or speaker == "Agent":
            # Agent message -- left aligned with tag
            lines.append(f"  [{start}-{end}]  AGENT ({display_name}) [{conf:.0%}]")
            lines.append(f"  | {text}")
            lines.append("")
        else:
            # Customer message -- right aligned with tag
            lines.append(f"  [{start}-{end}]  CUSTOMER")
            lines.append(f"  | {text}")
            lines.append("")

    return "\n".join(lines)


def render_split_view(segments: List[Dict]) -> str:
    """Render two separate columns: Agent vs Customer dialogue."""
    agent_lines = []
    customer_lines = []

    for seg in segments:
        speaker = seg.get("identified_speaker", seg.get("speaker", "Unknown"))
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = fmt_time(seg["start"])
        end = fmt_time(seg["end"])
        conf = seg.get("confidence", 0)
        display_name = speaker.replace("agent_", "").title() if speaker.startswith("agent_") else speaker

        entry = f"[{start}-{end}] {text}"

        if speaker.startswith("agent_") or speaker == "Agent":
            agent_lines.append((seg["start"], display_name, entry, conf))
        else:
            customer_lines.append((seg["start"], entry))

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  AGENT DIALOGUE")
    lines.append("=" * 70)

    if agent_lines:
        agent_name = agent_lines[0][1]
        lines.append(f"  Speaker: {agent_name} (confidence avg: {sum(a[3] for a in agent_lines)/len(agent_lines):.0%})")
        lines.append("-" * 70)
        for _, _, text, _ in agent_lines:
            lines.append(f"  {text}")
    else:
        lines.append("  (no agent segments detected)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  CUSTOMER DIALOGUE")
    lines.append("=" * 70)
    lines.append("-" * 70)

    if customer_lines:
        for _, text in customer_lines:
            lines.append(f"  {text}")
    else:
        lines.append("  (no customer segments detected)")

    return "\n".join(lines)


def render_stats(result: Dict) -> str:
    """Render summary statistics."""
    segments = result.get("segments", [])
    agent_segs = [s for s in segments if s.get("identified_speaker", "").startswith("agent_")]
    customer_segs = [s for s in segments if not s.get("identified_speaker", "").startswith("agent_")]

    agent_time = sum(s["end"] - s["start"] for s in agent_segs)
    customer_time = sum(s["end"] - s["start"] for s in customer_segs)
    total_time = agent_time + customer_time

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  CALL STATISTICS")
    lines.append("=" * 70)
    lines.append(f"  Audio file    : {result.get('audio_file', 'N/A')}")
    lines.append(f"  Processed     : {result.get('processed_at', 'N/A')}")
    lines.append(f"  Processing    : {result.get('processing_time_seconds', 0):.1f}s")
    lines.append(f"  Total segments: {len(segments)}")
    lines.append(f"  Agent segments: {len(agent_segs)} ({fmt_time(agent_time)} total)")
    lines.append(f"  Customer segs : {len(customer_segs)} ({fmt_time(customer_time)} total)")
    if total_time > 0:
        lines.append(f"  Talk ratio    : Agent {agent_time/total_time:.0%} / Customer {customer_time/total_time:.0%}")

    # Agent identification
    agent_names = set(s.get("identified_speaker", "") for s in agent_segs)
    if agent_names:
        names = ", ".join(n.replace("agent_", "").title() for n in agent_names)
        lines.append(f"  Identified as : {names}")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_result(result: Dict) -> str:
    """Generate the full formatted output."""
    parts = []
    parts.append(render_stats(result))
    parts.append(render_conversation(result.get("segments", [])))
    parts.append(render_split_view(result.get("segments", [])))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Format pipeline output as chat transcript")
    parser.add_argument("input", help="Path to result.json from pipeline output")
    parser.add_argument("--output", "-o", default=None, help="Save formatted output to file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        result = json.load(f)

    formatted = format_result(result)
    print(formatted)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
