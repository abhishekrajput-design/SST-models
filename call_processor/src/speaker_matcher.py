"""
Speaker identification via cosine similarity matching.
Compares segment embeddings against enrolled agent voiceprints.
"""

import pickle
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerMatcher:
    """Match unknown speaker embeddings to enrolled agent identities."""

    def __init__(
        self,
        embeddings_path: str = "embeddings/agent_embeddings.pkl",
        threshold: float = 0.25,
    ):
        """
        Args:
            embeddings_path: Path to the agent embeddings pickle file.
            threshold: Minimum cosine similarity to consider a match.
                       Below this → labeled as "Customer".
        """
        self.embeddings_path = embeddings_path
        self.threshold = threshold
        self.agent_embeddings: Dict[str, np.ndarray] = {}

    def load_embeddings(self):
        """Load agent embeddings from pickle file."""
        with open(self.embeddings_path, "rb") as f:
            self.agent_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(self.agent_embeddings)} agent embeddings")
        for name in self.agent_embeddings:
            logger.info(f"  - {name}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Match a single embedding against all enrolled agents.

        Args:
            embedding: 192-dim speaker embedding.

        Returns:
            Tuple of (speaker_name, confidence_score).
            Returns ("Customer", score) if no agent matches above threshold.
        """
        if not self.agent_embeddings:
            self.load_embeddings()

        best_name = "Customer"
        best_score = -1.0

        for agent_name, agent_emb in self.agent_embeddings.items():
            score = self._cosine_similarity(embedding, agent_emb)
            if score > best_score:
                best_score = score
                best_name = agent_name

        if best_score < self.threshold:
            return "Customer", round(best_score, 4)

        return best_name, round(best_score, 4)

    def match_segments(
        self,
        segments: List[Dict],
        embeddings: List[Optional[np.ndarray]],
    ) -> List[Dict]:
        """
        Assign speaker identities to all diarized segments.

        Uses a two-pass approach:
        1. Match each segment embedding to agents
        2. For segments with the same diarization label, use majority vote
           to assign consistent identity

        Args:
            segments: List of diarization segments with 'speaker' labels.
            embeddings: Corresponding embeddings for each segment.

        Returns:
            Segments with 'identified_speaker' and 'confidence' fields added.
        """
        if not self.agent_embeddings:
            self.load_embeddings()

        # Pass 1: Individual matching
        for seg, emb in zip(segments, embeddings):
            if emb is not None:
                name, confidence = self.match(emb)
                seg["identified_speaker"] = name
                seg["confidence"] = confidence
            else:
                seg["identified_speaker"] = "Customer"
                seg["confidence"] = 0.0

        # Pass 2: Majority vote per diarization speaker label
        # Group by original diarization label
        speaker_votes: Dict[str, List[Tuple[str, float]]] = {}
        for seg in segments:
            diar_label = seg["speaker"]
            if diar_label not in speaker_votes:
                speaker_votes[diar_label] = []
            speaker_votes[diar_label].append(
                (seg["identified_speaker"], seg["confidence"])
            )

        # Determine best identity for each diarization label
        speaker_map: Dict[str, Tuple[str, float]] = {}
        for diar_label, votes in speaker_votes.items():
            # Weight by confidence: pick the identity with highest total confidence
            identity_scores: Dict[str, float] = {}
            for name, conf in votes:
                identity_scores[name] = identity_scores.get(name, 0.0) + conf

            best_identity = max(identity_scores, key=identity_scores.get)
            avg_confidence = identity_scores[best_identity] / sum(
                1 for n, _ in votes if n == best_identity
            )
            speaker_map[diar_label] = (best_identity, round(avg_confidence, 4))

        # Apply consistent labels
        for seg in segments:
            mapped_name, mapped_conf = speaker_map[seg["speaker"]]
            seg["identified_speaker"] = mapped_name
            seg["confidence"] = mapped_conf

        logger.info("Speaker mapping:")
        for diar_label, (name, conf) in speaker_map.items():
            logger.info(f"  {diar_label} → {name} (confidence: {conf:.4f})")

        return segments
