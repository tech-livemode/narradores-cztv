"""Scoring package exposing the public evaluation entrypoints."""

from .entrypoint import graduate_audio_and_text
from .report import print_score_report

__all__ = [
    "graduate_audio_and_text",
    "print_score_report",
]
