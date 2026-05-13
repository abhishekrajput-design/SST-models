"""Reverse "UTF-8 -> mis-read-as-Latin-1 -> re-encoded as UTF-8" mojibake.

When a tool reads a UTF-8 file as Latin-1 (or Windows-1252) and writes it back
out as UTF-8, every non-ASCII character ends up as 2-3 codepoints of garbage:

    "🔄" (U+1F504, bytes F0 9F 94 84)
      -> read as Latin-1: chars ð Ÿ ” „ (U+00F0 U+0178 U+201D U+201E)
      -> re-encoded as UTF-8: bytes C3 B0 C5 B8 E2 80 9D E2 80 9E
      -> shown by a UTF-8-aware viewer as: "ðŸ"„"

Reversing it: decode UTF-8 -> encode as Latin-1 (recovers the original bytes)
-> decode as UTF-8 (recovers the original character). This is round-trippable
provided the file contains no genuine Latin-1 codepoints, which is the case
for our source files — every non-ASCII char in them is currently mojibake.

Usage:
  python fix_mojibake.py <file>...           # dry-run, prints counts
  python fix_mojibake.py <file>... --write   # actually rewrite the file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Second-pass replacements for emoji whose original bytes contain values that
# cp1252 leaves undefined (0x81, 0x8D, 0x8F, 0x90, 0x9D). The round-trip in
# repair_text() cannot recover these; the table below covers what we've seen
# in this codebase.
FALLBACK_REPLACEMENTS = [
    ("â˜ï¸", "☁️"),
    ("ðŸ–¥ï¸", "🖥️"),
    ("ðŸŽ™ï¸", "🎙️"),
    ("ðŸ“", "📁"),
    ("ðŸ", "📁"),
]


def repair_text(text: str) -> tuple[str, int]:
    """Round-trip every non-ASCII run via latin-1 -> utf-8. Returns (new_text,
    n_chars_changed). Falls back to keeping the original if a run cannot be
    cleanly recovered."""
    out: list[str] = []
    i = 0
    changed = 0
    while i < len(text):
        ch = text[i]
        if ord(ch) < 0x80:
            out.append(ch)
            i += 1
            continue
        # Collect a contiguous run of non-ASCII characters
        j = i
        while j < len(text) and ord(text[j]) >= 0x80:
            j += 1
        run = text[i:j]
        # Try cp1252 first (covers Windows tools that misread as Win-1252,
        # including chars in 0x80-0x9F like smart quotes), then plain latin-1.
        fixed = run
        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = run.encode(encoding).decode("utf-8")
                # Heuristic: accept the round-trip only if the result is
                # SHORTER (mojibake expands chars) AND contains fewer
                # non-ASCII chars (real chars are usually intentional).
                if len(candidate) < len(run):
                    fixed = candidate
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        out.append(fixed)
        if fixed != run:
            changed += len(run)
        i = j
    result = "".join(out)
    for bad, good in FALLBACK_REPLACEMENTS:
        if bad in result:
            changed += result.count(bad) * len(bad)
            result = result.replace(bad, good)
    return result, changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--write", action="store_true",
                   help="actually rewrite the file (default: dry-run)")
    args = p.parse_args()
    for path in args.files:
        fp = Path(path)
        raw = fp.read_bytes()
        had_bom = raw.startswith(b"\xef\xbb\xbf")
        if had_bom:
            raw = raw[3:]
        text = raw.decode("utf-8", errors="replace")
        new_text, n = repair_text(text)
        suffix = "  (was BOM-prefixed)" if had_bom else ""
        print(f"{path}: {n} chars repaired{suffix}")
        if not n:
            continue
        if args.write:
            # Write without BOM — adds friction for Linux Python which then
            # rejects the file at parse time. The Windows tools that produced
            # the BOM in the first place no longer need it.
            fp.write_bytes(new_text.encode("utf-8"))
            print("  -> rewritten")
        else:
            # Show a short preview of the first repaired location for sanity
            for idx in range(len(text)):
                if idx < len(new_text) and text[idx] != new_text[idx]:
                    s = max(0, idx - 10)
                    e = min(len(text), idx + 20)
                    print(f"  preview before: {text[s:e]!r}")
                    print(f"  preview after:  {new_text[s:e]!r}")
                    break


if __name__ == "__main__":
    main()
