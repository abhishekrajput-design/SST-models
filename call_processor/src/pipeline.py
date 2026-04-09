"""
Full call processing pipeline orchestrator.
Coordinates diarization → embedding → matching → transcription
with sequential model loading for 4GB VRAM optimization.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CallProcessor:
    """
    End-to-end call processing pipeline.

    Processes a call recording through 3 sequential stages:
    1. Diarization (pyannote) — detect speaker segments
    2. Speaker ID (SpeechBrain) — match voices to known agents
    3. Transcription (Whisper) — convert speech to text

    Each stage loads its model, processes, then unloads to stay within 4GB VRAM.
    """

    def __init__(
        self,
        hf_token: str,
        embeddings_path: str = "embeddings/agent_embeddings.pkl",
        model_cache_dir: str = "models",
        whisper_model: str = "large-v3",
        whisper_compute_type: str = "int8",
        language: str = "en",
        device: str = "cuda",
        similarity_threshold: float = 0.25,
        min_segment_duration: float = 1.0,
        output_dir: str = "data/processed",
    ):
        self.hf_token = hf_token
        self.embeddings_path = embeddings_path
        self.model_cache_dir = model_cache_dir
        self.whisper_model = whisper_model
        self.whisper_compute_type = whisper_compute_type
        self.language = language
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.min_segment_duration = min_segment_duration
        self.output_dir = output_dir

    def process(self, audio_path: str) -> Dict:
        """
        Run the full pipeline on a call recording.

        Args:
            audio_path: Path to the call audio file.

        Returns:
            Dict with 'segments' list and metadata.
        """
        start_time = time.time()
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        segments_dir = os.path.join(self.output_dir, audio_name, "segments")

        logger.info(f"{'='*60}")
        logger.info(f"Processing: {audio_path}")
        logger.info(f"{'='*60}")

        # ── Stage 1: Diarization ──────────────────────────────────
        logger.info("\n[Stage 1/3] Speaker Diarization")
        segments, segment_paths = self._stage_diarization(audio_path, segments_dir)

        if not segments:
            logger.warning("No speech segments found!")
            return {"segments": [], "metadata": {"error": "no segments found"}}

        # ── Stage 2: Speaker Identification ───────────────────────
        logger.info("\n[Stage 2/3] Speaker Identification")
        segments = self._stage_speaker_id(segments, segment_paths)

        # ── Stage 3: Transcription ────────────────────────────────
        logger.info("\n[Stage 3/3] Transcription")
        segments = self._stage_transcription(segments, segment_paths)

        # ── Build result ──────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)

        result = {
            "audio_file": audio_path,
            "processed_at": datetime.now().isoformat(),
            "processing_time_seconds": elapsed,
            "total_segments": len(segments),
            "segments": segments,
        }

        # Save outputs
        output_subdir = os.path.join(self.output_dir, audio_name)
        self.save_json(result, os.path.join(output_subdir, "result.json"))
        self.save_transcript(result, os.path.join(output_subdir, "transcript.txt"))

        logger.info(f"\nPipeline complete in {elapsed}s")
        logger.info(f"Output: {output_subdir}")

        return result

    def _stage_diarization(self, audio_path: str, segments_dir: str):
        """Stage 1: Run diarization and extract segment WAVs."""
        from src.diarization import Diarizer

        diarizer = Diarizer(
            hf_token=self.hf_token,
            device=self.device,
            min_segment_duration=self.min_segment_duration,
        )

        try:
            segments = diarizer.diarize(audio_path)
            segment_paths = diarizer.save_segments(audio_path, segments, segments_dir)
        finally:
            diarizer.unload_model()

        logger.info(f"Found {len(segments)} segments from diarization")
        return segments, segment_paths

    def _stage_speaker_id(self, segments: List[Dict], segment_paths: List[str]):
        """Stage 2: Extract embeddings and match to known agents."""
        from src.embedding import EmbeddingExtractor
        from src.speaker_matcher import SpeakerMatcher

        extractor = EmbeddingExtractor(
            device=self.device,
            model_dir=os.path.join(self.model_cache_dir, "spkrec-ecapa"),
        )

        try:
            embeddings = extractor.extract_embeddings_batch(segment_paths)
        finally:
            extractor.unload_model()

        # Match embeddings to agents (CPU-only, no model needed)
        if os.path.exists(self.embeddings_path):
            matcher = SpeakerMatcher(
                embeddings_path=self.embeddings_path,
                threshold=self.similarity_threshold,
            )
            segments = matcher.match_segments(segments, embeddings)
        else:
            logger.warning(
                f"No agent embeddings found at {self.embeddings_path}. "
                "Run enroll_agents.py first. Labeling all speakers as 'Unknown'."
            )
            for seg in segments:
                seg["identified_speaker"] = seg["speaker"]
                seg["confidence"] = 0.0

        return segments

    def _stage_transcription(self, segments: List[Dict], segment_paths: List[str]):
        """Stage 3: Transcribe each segment."""
        from src.transcription import Transcriber

        transcriber = Transcriber(
            model_size=self.whisper_model,
            device=self.device,
            compute_type=self.whisper_compute_type,
            language=self.language,
        )

        try:
            segments = transcriber.transcribe_segments(segment_paths, segments)
        finally:
            transcriber.unload_model()

        return segments

    @staticmethod
    def save_json(result: Dict, output_path: str):
        """Save pipeline result as formatted JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON: {output_path}")

    @staticmethod
    def save_transcript(result: Dict, output_path: str):
        """Save human-readable transcript."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines = []
        lines.append(f"Call Transcript")
        lines.append(f"Audio: {result['audio_file']}")
        lines.append(f"Processed: {result['processed_at']}")
        lines.append(f"Processing Time: {result['processing_time_seconds']}s")
        lines.append("=" * 60)
        lines.append("")

        for seg in result["segments"]:
            speaker = seg.get("identified_speaker", seg.get("speaker", "Unknown"))
            start = seg["start"]
            end = seg["end"]
            text = seg.get("text", "")
            confidence = seg.get("confidence", 0)

            timestamp = f"[{_format_time(start)} → {_format_time(end)}]"
            lines.append(f"{timestamp} {speaker} (conf: {confidence:.2f})")
            lines.append(f"  {text}")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved transcript: {output_path}")


def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
