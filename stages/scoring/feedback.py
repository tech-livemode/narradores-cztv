# Classification helpers and structured feedback generation.

from typing import Tuple

from model.scoreresult import AudioScore, TextScore


def _get_classification(score: float) -> str:
    """
    Classifica o narrador baseado no score final.
    """
    if score >= 85:
        return "Excelente"
    elif score >= 70:
        return "Bom"
    elif score >= 55:
        return "Regular"
    elif score >= 40:
        return "Abaixo da média"
    else:
        return "Insuficiente"


def _generate_feedback(audio_score: AudioScore, text_score: TextScore, sports_confidence: float = 1.0) -> Tuple[list, list]:
    """
    Gera feedback estruturado com pontos fortes e áreas de melhoria.
    """
    strengths = []
    improvements = []

    if audio_score.vocal_dynamics >= 7:
        strengths.append("Boa dinâmica vocal e expressividade")
    elif audio_score.vocal_dynamics < 5:
        improvements.append("Trabalhar variação de pitch e energia vocal")

    if audio_score.speech_pacing >= 7:
        strengths.append("Ritmo de fala adequado com boa variação")
    elif audio_score.speech_pacing < 5:
        improvements.append("Ajustar ritmo da fala (muito rápido ou muito lento)")

    if audio_score.emotion_audio >= 7:
        strengths.append("Transmite emoção pela voz nos momentos importantes")
    elif audio_score.emotion_audio < 5:
        improvements.append("Aumentar emoção e energia nos momentos-chave")

    if text_score.emotion_content >= 7:
        strengths.append("Captura bem a emoção e paixão do jogo")
    elif text_score.emotion_content < 5:
        improvements.append("Transmitir mais paixão e emoção no conteúdo")

    if text_score.storytelling >= 7:
        strengths.append("Excelente storytelling e contextualização")
    elif text_score.storytelling < 5:
        improvements.append("Contar mais a história e contexto do jogo")

    if text_score.game_rhythm >= 7:
        strengths.append("Dá bom ritmo ao jogo e menciona bem o elenco")
    elif text_score.game_rhythm < 5:
        improvements.append("Ser mais generoso com menções ao elenco completo")

    if text_score.appropriate_fun >= 7:
        strengths.append("Boa resenha, divertido sem excessos")
    elif text_score.appropriate_fun < 5:
        improvements.append("Equilibrar diversão sem ser comediante")

    if sports_confidence < 0.5:
        improvements.insert(0, "⚠️ CRÍTICO: Conteúdo não parece ser narração esportiva")
    elif sports_confidence < 0.7:
        improvements.append("Adicionar mais contexto e vocabulário esportivo")

    if not strengths:
        strengths.append("Narração com pontos positivos a desenvolver")
    if not improvements:
        improvements.append("Continuar mantendo o bom nível")

    return strengths, improvements


__all__ = ["_get_classification", "_generate_feedback"]
