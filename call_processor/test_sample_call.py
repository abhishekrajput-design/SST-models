import json
import subprocess
import os
from pathlib import Path
from src.process_audio import process_call

# Test one of the API ground truth calls
test_file = "data/raw_calls/api_69c842366a2041f487a6b158.mp3"

if not os.path.exists(test_file):
    print(f"Test file not found: {test_file}")
    print("Available files:")
    for f in sorted(Path("data/raw_calls").glob("*.mp3"))[:10]:
        print(f"  {f}")
    exit(1)

# Process with our system
output_dir = "data/processed/test_sample_call"
os.makedirs(output_dir, exist_ok=True)

print(f"Testing file: {test_file}")
print(f"Output dir: {output_dir}\n")

# Run the full pipeline: Parakeet transcription + diarization
result = process_call(test_file, output_dir=output_dir, transcriber='parakeet')
if result:
    print('✓ Processing complete')
    print(f'  Segments: {len(result.get("segments", []))}')
    print(f'  Agent identified: {result.get("agent_name", "Unknown")}')
    print(f'  Agent similarity: {result.get("agent_similarity", 0.0)}')
    
    # Count agent vs customer segments
    agent_count = sum(1 for s in result.get('segments', []) if s.get('identified_speaker') == 'AGENT')
    customer_count = sum(1 for s in result.get('segments', []) if s.get('identified_speaker') == 'CUSTOMER')
    print(f'  Agent segments: {agent_count}')
    print(f'  Customer segments: {customer_count}')
    
    # Show first 5 segments
    print('\nFirst 5 segments:')
    for i, seg in enumerate(result.get('segments', [])[:5]):
        role = seg.get('identified_speaker', '?')
        text = seg.get('text', '')[:40]
        print(f'  [{i}] {role:8s} | {text}')
else:
    print('✗ Processing failed')
