import time
import os
import sys
from pathlib import Path

def main():
    audio_path = "audio_04_16_2026_15_34_29_kixm5p.mp3"
    clean_path = "enhanced_dfnet_audio.wav"

    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    # 1. Provide Clear Voice using DeepFilterNet
    print(f"\n[Stage 1] Loading DeepFilterNet (State-of-the-art Denoising)...")
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError:
        print("deepfilternet not installed. Please run: pip install deepfilternet")
        return

    t0 = time.time()
    # init_df takes care of loading defaults
    model, df_state, _ = init_df()
    
    print(f"Enhancing {audio_path}...")
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    
    # Process the audio array
    enhanced = enhance(model, df_state, audio)
    
    # Save clear audio output
    save_audio(clean_path, enhanced, df_state.sr())
    print(f"-> Audio enhancement complete in {time.time()-t0:.2f}s.")
    print(f"-> Output saved to: {clean_path}\n")

    # 2. Transcribe clear audio using large-v3
    print(f"[Stage 2] Loading Whisper large-v3 transcriber...")
    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper or torch not installed.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "float32"
    
    # Pointing to the models dir of the project
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "call_processor", "models", "faster-whisper")
    os.makedirs(model_dir, exist_ok=True)
    
    whisper_mod = WhisperModel("large-v3", device=device, compute_type=compute, download_root=model_dir)

    print("Transcribing...")
    t1 = time.time()
    segs_iter, info = whisper_mod.transcribe(
        clean_path,
        language="en",
        beam_size=5,
        vad_filter=True, # Skips silence chunks
        vad_parameters={"min_silence_duration_ms": 500}
    )
    
    print("\n" + "="*50)
    print(" 🎙️ FINAL CLEAR AUDIO TRANSCRIPT")
    print("="*50)
    
    for seg in segs_iter:
        print(f"[{seg.start:05.2f}s -> {seg.end:05.2f}s] {seg.text.strip()}")
        
    print("="*50)
    print(f"-> Transcription finished in {time.time()-t1:.2f}s\n")


if __name__ == "__main__":
    main()
