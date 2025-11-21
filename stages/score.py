# Adaptador que expõe o pipeline de pontuação (55% áudio + 45% texto; 65/35 para highlights).

from .scoring import graduate_audio_and_text, print_score_report

__all__ = [
    "graduate_audio_and_text",
    "print_score_report",
]
