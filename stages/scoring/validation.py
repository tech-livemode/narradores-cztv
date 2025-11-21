"""Validation utilities: highlight detection and sports-content gating."""

import re
from typing import Dict, List, Tuple
import numpy as np  # type: ignore
from model.scoreresult import AudioScore, FinalScore, TextScore
from .config import ACTION_TRIGGERS


def _is_highlights_mode(stream_id: str, transcripts: List[Dict], total_duration: float) -> bool:
    # Detecta se o conteúdo é "melhores momentos"/highlights."""
    sid = (stream_id or "").lower()
    if any(x in sid for x in ["melhores_momentos", "melhores-momentos", "melhores", "highlights", "resumo"]):
        return True
    if total_duration and total_duration <= 900:
        full_text = " ".join([t.get("text", "") for t in transcripts]).lower()
        hits = len(ACTION_TRIGGERS.findall(full_text))
        return hits >= 10
    return False


def _validate_sports_content(text: str, duration: float) -> Tuple[bool, float]:
    # Valida se o conteúdo é narração esportiva usando densidade normalizada."""
    text_lower = text.lower()
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return False, 0.0

    confidence_score = 0.0

    log_words = np.log(total_words) if total_words > 1 else 1

    sports_groups = {
        'core': re.compile(r'(gol|bola|jogo|partida|time|equipe|campo)', re.IGNORECASE),
        'actors': re.compile(r'(jogador|atleta|árbitro|técnico|goleiro)', re.IGNORECASE),
        'actions': re.compile(r'(chut|pass|drible|marc|defend|atac)', re.IGNORECASE),
        'competition': re.compile(r'(campeonato|taça|copa|título|placar|resultado)', re.IGNORECASE),
    }

    groups_present = sum(1 for pattern in sports_groups.values() if pattern.search(text))
    keyword_density = groups_present / len(sports_groups)

    if keyword_density >= 0.75:
        confidence_score += 0.40
    elif keyword_density >= 0.50:
        confidence_score += 0.25
    elif keyword_density >= 0.25:
        confidence_score += 0.10

    if duration >= 60:
        confidence_score += 0.20
    elif duration >= 30:
        confidence_score += 0.10
    elif duration >= 15:
        confidence_score += 0.05

    action_pattern = re.compile(r'(pass|chut|cruz|marc|defend|atac|corr|disput|arremat|finaliz|salv|intercept)', re.IGNORECASE)
    action_matches = len(action_pattern.findall(text))
    action_density = action_matches / log_words

    if action_density >= 3.0:
        confidence_score += 0.20
    elif action_density >= 1.5:
        confidence_score += 0.12
    elif action_density >= 0.5:
        confidence_score += 0.05

    non_sports_pattern = re.compile(r'(motiva[cç]ão|reflexão|podcast|inscre|curtir|compartilh|like|status)', re.IGNORECASE)
    non_sports_matches = len(non_sports_pattern.findall(text))

    if non_sports_matches == 0:
        confidence_score += 0.20
    elif non_sports_matches <= 2:
        confidence_score += 0.10
    else:
        confidence_score -= 0.10

    confidence_score = max(0.0, min(1.0, confidence_score))
    is_sports = confidence_score >= 0.50

    return is_sports, confidence_score


def _create_non_sports_score(stream_id: str, confidence: float, segments: int, duration: float, text: str) -> FinalScore:
    """Cria um score para conteúdo que não é narração esportiva."""
    audio_score = AudioScore(2, 2, 2, 20, {"reason": "not_sports_content"})
    text_score = TextScore(1, 1, 1, 1, 10, {"reason": "not_sports_content"})

    final_score_val = 15.0
    content_type = _detect_content_type(text.lower())

    return FinalScore(
        stream_id=stream_id,
        audio_score=audio_score,
        text_score=text_score,
        final_score=final_score_val,
        classification="Insuficiente",
        strengths=[],
        improvements=[
            f"❌ CONTEÚDO NÃO É NARRAÇÃO ESPORTIVA (detectado como: {content_type})",
            f"Confiança de ser conteúdo esportivo: {confidence:.1%}",
            "Este áudio parece ser de outro tipo de conteúdo",
            "Para ser avaliado, envie uma narração de evento esportivo real",
        ],
        total_segments=segments,
        total_duration=duration,
    )


def _detect_content_type(text: str) -> str:
    # Tenta detectar o tipo de conteúdo baseado em palavras-chave.
    if any(word in text for word in ['motivação', 'sucesso', 'frase', 'reflexão']):
        return "conteúdo motivacional"
    elif any(word in text for word in ['podcast', 'episódio', 'programa']):
        return "podcast/programa"
    elif any(word in text for word in ['notícia', 'aconteceu', 'hoje']):
        return "notícia/informativo"
    elif any(word in text for word in ['música', 'canção', 'letra']):
        return "música/áudio musical"
    else:
        return "conteúdo genérico"


__all__ = [
    "_is_highlights_mode",
    "_validate_sports_content",
    "_create_non_sports_score",
    "_detect_content_type",
]
