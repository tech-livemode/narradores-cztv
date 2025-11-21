"""Typed helpers used across scoring modules."""

from dataclasses import dataclass

@dataclass(frozen=True)
class ScoringContext:
    """Encapsulates weighting rules for full streams and highlight reels."""

    audio_weight: float = 0.60
    text_weight: float = 0.40
    highlights_audio_weight: float = 0.70
    highlights_text_weight: float = 0.30


@dataclass
class Window:
    """Represents a textual/audio slice scored by the LLM."""

    start: float
    end: float
    text: str
    kind: str  # 'peak' | 'uniform' | 'analytic' | 'fallback'
    weight: float
    narrator_share: float = 0.0
    leader_voice_ratio: float = 0.0


__all__ = ["ScoringContext", "Window"]
