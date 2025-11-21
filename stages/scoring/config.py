# Centraliza configurações e constantes para o pipeline de avaliação de audio

import re
from config import settings

from .types import ScoringContext


SCORING_CONTEXT = ScoringContext()

ANALYTIC_HINTS = re.compile(
    r"(posse|estat(í|i)stic|t(á|a)tic|linha de 4|linha de 5|xg|expected goals|na minha opini|sistema defensivo|compacta|amplitude|profundidade)",
    re.IGNORECASE,
)

ACTION_TRIGGERS = re.compile(
    r"(gol|defendeu|trave|na trave|pra fora|é agora|contra-ataque|contra ataque|chute|bateu|cabeceou|cruzou|finalizou)",
    re.IGNORECASE,
)

DEFAULT_LLM_WINDOWS = 8
DEFAULT_LLM_WINDOW_SECONDS = 25.0

LUISINHO_PROFILE = {
    "speech_rate_mean": 3.06,
    "speech_rate_std": 0.13,
    "emotion_corr_mean": 0.30,
    "emotion_corr_std": 0.05,
    "vocal_dyn_mean": 9.8,
    "vocal_dyn_std": 0.6,
}

BASELINE_STATS = {
    "pitch": (150, 40),  # média, desvio
    "rms": (0.12, 0.05),
    "speech_rate": (1.4, 0.5),
    "emotion_correlation": (0.45, 0.20),
    "appropriate_fun": (6.0, 2.0),
}

SEMANTIC_CLUSTERS = {
    "emotion": [
        "gol incrível",
        "vitória épica",
        "explosão de alegria",
        "grande jogada",
        "lindo lance",
        "emoção pura",
        "torcida vibra",
        "grito de gol",
        "momento histórico",
    ],
    "storytelling": [
        "história do jogo",
        "momento decisivo",
        "contexto da partida",
        "narração envolvente",
        "explicando a jogada",
        "lembrando o início do jogo",
    ],
    "game_rhythm": [
        "ataque rápido",
        "pressão do time",
        "defesa organizada",
        "transição veloz",
        "posse de bola",
        "ritmo intenso",
    ],
}


def ensure_settings_defaults() -> None:
    # Popula valores opicionais de configurações esperado pela pipeline de avaliação.
    if not hasattr(settings, "llm_windowed_scoring"):
        settings.llm_windowed_scoring = True
    if not hasattr(settings, "llm_windows"):
        settings.llm_windows = DEFAULT_LLM_WINDOWS
    if not hasattr(settings, "llm_window_seconds"):
        settings.llm_window_seconds = DEFAULT_LLM_WINDOW_SECONDS
    if not hasattr(settings, "enable_light_diarization"):
        settings.enable_light_diarization = True
    if not hasattr(settings, "light_diarization_k"):
        settings.light_diarization_k = 2
    if not hasattr(settings, "light_diarization_min_conf"):
        settings.light_diarization_min_conf = 0.55
    if not hasattr(settings, "light_diarization_focus_audio"):
        settings.light_diarization_focus_audio = True
    if not hasattr(settings, "light_diarization_window_boost"):
        settings.light_diarization_window_boost = 0.35


ensure_settings_defaults()

__all__ = [
    "SCORING_CONTEXT",
    "ANALYTIC_HINTS",
    "ACTION_TRIGGERS",
    "DEFAULT_LLM_WINDOWS",
    "DEFAULT_LLM_WINDOW_SECONDS",
    "LUISINHO_PROFILE",
    "BASELINE_STATS",
    "SEMANTIC_CLUSTERS",
    "settings",
]
