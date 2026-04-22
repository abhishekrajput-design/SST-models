import time
import os

def transcribe_audio():
    audio_path = r"call_processor\data\processed\audio_04_16_2026_15_34_29_kixm5p\clean_audio_04_16_2026_15_34_29_kixm5p.wav"
    out_txt = r"call_processor\data\processed\audio_04_16_2026_15_34_29_kixm5p\transcript_fast.txt"
    
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    print("Loading Faster-Whisper large-v3...")
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "float32"
    
    model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "call_processor", "models", "faster-whisper")
    whisper_mod = WhisperModel("large-v3", device=device, compute_type=compute, download_root=model_dir)

    print("Transcribing...")
    t0 = time.time()
    
    # beam_size 5, condition_on_previous_text True, vad_filter True
    segs_iter, info = whisper_mod.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
        temperature=0
    )
    
    segments_texts = []
    for seg in segs_iter:
        line = f"[{seg.start:05.2f}s -> {seg.end:05.2f}s] {seg.text.strip()}"
        print(line)
        segments_texts.append(line)
        
    print(f"Transcription finished in {time.time()-t0:.2f}s")
    
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(segments_texts))
        
    print(f"Saved to {out_txt}")

if __name__ == "__main__":
    transcribe_audio()
