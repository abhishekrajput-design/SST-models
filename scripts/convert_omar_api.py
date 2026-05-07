import json
import os
import datetime

INDEX_PATH = "call_processor/data/audiofy/_dataset/index.json"
CALLS = [
    ("69efc071f91ac02559f83167", "traning_data/omar_el_harchaoui/call_01/data.json"),
    ("69efbb58f91ac02559f827f6", "traning_data/omar_el_harchaoui/call_02/data.json"),
    ("69efb362f91ac02559f813fe", "traning_data/omar_el_harchaoui/call_03/data.json"),
]

def parse_time(ts):
    # '00:00:02.400' -> seconds float
    if '.' in ts:
        time_part, ms_part = ts.split('.')
        ms = float('.' + ms_part)
    else:
        time_part = ts
        ms = 0.0
    h, m, s = map(int, time_part.split(':'))
    return h * 3600 + m * 60 + s + ms

def main():
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    call_map = {x["_id"]: x for x in index}
    
    for call_id, out_path in CALLS:
        api_data = call_map.get(call_id)
        if not api_data:
            print(f"Missing {call_id}")
            continue
            
        segments = []
        for s in api_data.get("speaker_json", []):
            spk = "customer" if s["speaker"] == "Customer" else "agent"
            segments.append({
                "start": parse_time(s["start"]),
                "end": parse_time(s["end"]),
                "speaker": spk,
                "text": s.get("phrase", "")
            })
            
        out_data = {
            "call_id": call_id,
            "agent_name": "Omar El Harchaoui",
            "source": "api",
            "segments": segments
        }
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
            
        print(f"Wrote {out_path} with {len(segments)} segments.")

if __name__ == '__main__':
    main()
