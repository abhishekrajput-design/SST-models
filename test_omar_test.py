import sys
import json
from pathlib import Path

# Add project root to sys.path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from call_processor.src.diar_multi import diarize_multi

# The test audio file
wav_path = "C:/Users/abhis/Desktop/SST-models/testing-audio/omar_test/20260505T073055769_385036.mp3"

# Let's generate a dummy transcript with roughly the ground truth timestamps
segments = [
    {"start": 0.0, "end": 1.32, "text": "Hi, I'm Arcester. Is this a good time to chat?"},
    {"start": 1.32, "end": 1.64, "text": "Yeah speaking."},
    {"start": 1.64, "end": 4.16, "text": "Awesome, I wanted to discuss the Mini Hatch you have available."},
    {"start": 4.16, "end": 6.24, "text": "Great, we have a few on the lot. When would you like to view it?"},
    {"start": 6.24, "end": 8.64, "text": "How about tomorrow morning around ten?"},
    {"start": 8.64, "end": 9.28, "text": "Five o'clock."},
    {"start": 9.28, "end": 12.48, "text": "That's correct. I mean, that works for me if five is better."}
]

res = diarize_multi(segments, wav_path, force_cpu=False)

for s in res.get("segments", []):
    print(f"{s['start']:.2f}-{s['end']:.2f}: {s.get('identified_speaker', 'UNKNOWN')} ({s.get('_best_sim', 0):.3f}) - {s['text']}")
