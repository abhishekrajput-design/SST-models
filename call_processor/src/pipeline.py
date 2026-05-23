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

from src.conversation_roles import build_conversation_view

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
        whisper_compute_type: str = "float16",
        whisper_device: str = "auto",
        language: str = "en",
        device: str = "cuda",
        similarity_threshold: float = 0.62,
        min_segment_duration: float = 1.0,
        output_dir: str = "data/processed",
        initial_prompt: Optional[str] = "CarPlanet car dealership. Agent speaking with customer.",
    ):
        self.hf_token = hf_token
        self.embeddings_path = embeddings_path
        self.model_cache_dir = model_cache_dir
        self.whisper_model = whisper_model
        self.whisper_compute_type = whisper_compute_type
        self.whisper_device = whisper_device
        self.language = language
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.min_segment_duration = min_segment_duration
        self.output_dir = output_dir
        self.initial_prompt = initial_prompt

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
            return {
                "audio_file": audio_path,
                "processed_at": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "total_segments": 0,
                "segments": [],
                "error": "no segments found",
            }

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
            cache_dir=os.path.join(self.model_cache_dir, "pyannote"),
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
        from src.config import PipelineConfig
        from src.speaker_matcher import SpeakerMatcher

        cfg = PipelineConfig()

        if cfg.embedding_model == "cam++":
            from src.embedding_campp import EmbeddingModel
            model = EmbeddingModel()
            model.load()
            try:
                embeddings = model.embed_batch(segment_paths)
            finally:
                pass  # CAM++ uses module-level singleton; don't unload
        else:
            from src.embedding import EmbeddingExtractor
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
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
            language=self.language,
            download_root=os.path.join(self.model_cache_dir, "faster-whisper"),
            initial_prompt=self.initial_prompt,
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
        """Save chat-style transcript with Agent vs Customer split."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        segments, speaker_groups = build_conversation_view(result.get("segments", []))

        def fmt(sec):
            return f"{int(sec//60):02d}:{int(sec%60):02d}"

        agent_segs = [s for s in segments if s["role"] == "Agent"]
        cust_segs = [s for s in segments if s["role"] == "Customer"]
        agent_speakers = [g for g in speaker_groups if g["role"] == "Agent"]
        customer_speakers = [g for g in speaker_groups if g["role"] == "Customer"]
        agent_time = sum(s["end"] - s["start"] for s in agent_segs)
        cust_time = sum(s["end"] - s["start"] for s in cust_segs)
        total = agent_time + cust_time

        lines = []
        # Header
        lines.append("=" * 70)
        lines.append(f"  CALL TRANSCRIPT")
        lines.append(f"  Audio: {result.get('audio_file', 'N/A')}")
        lines.append(f"  Processed: {result.get('processed_at', '')}")
        lines.append(
            f"  Segments: {len(segments)} | "
            f"Agents: {len(agent_speakers)} ({fmt(agent_time)}) | "
            f"Customer Speakers: {len(customer_speakers)} ({fmt(cust_time)})"
        )
        if total > 0:
            lines.append(f"  Talk ratio: Agent {agent_time/total:.0%} / Customer {cust_time/total:.0%}")
        lines.append("=" * 70)
        lines.append("")

        # Chronological conversation
        lines.append("-" * 70)
        lines.append("  CONVERSATION")
        lines.append("-" * 70)
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            conf = seg.get("confidence", 0)
            lines.append(
                f"  [{fmt(seg['start'])}-{fmt(seg['end'])}] "
                f"{seg['role'].upper()} ({seg['display_speaker']}, {seg['speaker_label']}) "
                f"[{conf:.0%}]"
            )
            lines.append(f"    {text}")
            lines.append("")

        # Speaker-level dialogue
        lines.append("-" * 70)
        lines.append("  SPEAKER TRANSCRIPTS")
        lines.append("-" * 70)
        for group in speaker_groups:
            lines.append(
                f"  {group['display_speaker']} [{group['speaker_label']}] "
                f"{group['segment_count']} segments / {fmt(group['talk_time'])}"
            )
            if group["text"]:
                lines.append(f"    {group['text']}")
            lines.append("")

        lines.append("")
        lines.append("=" * 70)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved transcript: {output_path}")


def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
