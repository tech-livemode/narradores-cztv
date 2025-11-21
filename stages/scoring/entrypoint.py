# Entrypoint orchestrating the full scoring pipeline.

import logging
import re
from typing import Dict, List, Optional
import numpy as np  # type: ignore

from model.scoreaccumulator import ScoreAccumulator
from model.processedsegment import ProcessedSegment
from model.scoreresult import AudioScore, FinalScore, TextScore

from .audio import _evaluate_audio
from .config import LUISINHO_PROFILE, SCORING_CONTEXT, settings
from .diarization_light import _assign_light_speaker_labels
from .feedback import _generate_feedback, _get_classification
from .llm_aggregate import _aggregate_llm_windows
from .text import _evaluate_text, _score_appropriate_fun
from .utils import _to_similarity, _z
from .validation import _create_non_sports_score, _is_highlights_mode, _validate_sports_content
from .windows import _select_windows

logger = logging.getLogger(__name__)

PASSION_KEYWORDS = re.compile(
    r"(gol+|explode|torcida|apaixonad|sensacional|inacredit|absurdo|"
    r"emocionante|que lance|vibra|histórico|épico)",
    re.IGNORECASE,
)


def _text_gate_for_narrativity(narrativity: float) -> float:
    if narrativity >= 0.65:
        return 1.0
    if narrativity >= 0.50:
        return 0.97
    if narrativity >= 0.45:
        return 0.94
    if narrativity >= 0.35:
        return 0.88
    return 0.80


def _highlight_heuristic_weight(narrativity: float) -> float:
    if narrativity >= 0.80:
        return 0.80
    if narrativity <= 0.55:
        return 0.60
    return 0.70


def _full_stream_blend_weight(narrativity: float) -> Optional[float]:
    if narrativity >= 0.90:
        return 0.85
    if narrativity >= 0.80:
        return 0.75
    if narrativity >= 0.70:
        return 0.65
    if narrativity >= 0.60:
        return 0.60
    return None


def _passion_bonus(full_text: str, narrativity: float) -> float:
    if narrativity < 0.85:
        return 0.0
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", full_text)
        if s and s.strip()
    ]
    passionate = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) < 18:
            continue
        if PASSION_KEYWORDS.search(sentence):
            passionate += 1
        if passionate >= 3:
            break
    if passionate >= 3:
        return 5.0
    if passionate >= 1:
        return 3.0
    return 0.0


def graduate_audio_and_text(
    segments: List[ProcessedSegment],
    transcripts: List[Dict],
    score_accumulator: Optional[ScoreAccumulator] = None,
) -> FinalScore:
    # Executa a pipeline completa de graduação de áudio/texto.
    logger.info("Iniciando avaliação do narrador...")

    if not segments or not transcripts:
        logger.warning("Dados insuficientes para avaliação")
        return _create_empty_score()

    stream_id = segments[0].stream_id if segments else "unknown"
    total_duration = sum(seg.end_time - seg.start_time for seg in segments)
    is_highlights = _is_highlights_mode(stream_id, transcripts, total_duration)

    logger.info(
        "LLM flags → enable_llm_scoring=%s | llm_validation_only=%s | llm_full_scoring=%s",
        getattr(settings, "enable_llm_scoring", False),
        getattr(settings, "llm_validation_only", False),
        getattr(settings, "llm_full_scoring", False),
    )
    full_text = " ".join([t.get("text", "") for t in transcripts])

    is_sports_content = None
    sports_confidence = 0.0
    if getattr(settings, "enable_llm_scoring", False):
        try:
            from stages.llm_scorer import hybrid_scoring_mode

            logger.info("🤖 LLM: validando conteúdo esportivo (modo validação)")
            hybrid = hybrid_scoring_mode(full_text, total_duration, use_llm=True)
            is_sports_content = bool(hybrid.get("is_sports", False))
            sports_confidence = float(hybrid.get("confidence", 0.0))
            logger.info(
                f"Validação de conteúdo via {hybrid.get('validation_method')}: is_sports={is_sports_content}, conf={sports_confidence:.2f}"
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Falha na validação com LLM, usando regras: {e}")

    if is_sports_content is None:
        is_sports_content, sports_confidence = _validate_sports_content(full_text, total_duration)

    if not is_sports_content:
        logger.warning(
            f"⚠️ Conteúdo não parece ser narração esportiva (confiança: {sports_confidence:.1%})"
        )
        return _create_non_sports_score(stream_id, sports_confidence, len(segments), total_duration, full_text)

    logger.info(f"✅ Conteúdo esportivo detectado (confiança: {sports_confidence:.1%})")

    if bool(getattr(settings, "enable_light_diarization", True)):
        try:
            _assign_light_speaker_labels(segments)
            logger.info("🔊 Light diarization habilitada: labels narrator/other atribuídos")
        except Exception as _e:  # pragma: no cover
            logger.warning(f"Light diarization falhou: {_e}")

    audio_score = _evaluate_audio(segments)
    logger.info(f"Score de áudio: {audio_score.total_score:.2f}/100")

    speech_rate_cur = float(audio_score.details.get("avg_speech_rate") or 0.0)
    emotion_corr_raw = float(audio_score.details.get("emotion_corr_raw") or 0.0)
    vocal_dyn_cur = float(audio_score.vocal_dynamics or 0.0)

    z_sr = _z(
        speech_rate_cur, LUISINHO_PROFILE["speech_rate_mean"], LUISINHO_PROFILE["speech_rate_std"]
    )
    z_em = _z(
        emotion_corr_raw, LUISINHO_PROFILE["emotion_corr_mean"], LUISINHO_PROFILE["emotion_corr_std"]
    )
    z_vd = _z(vocal_dyn_cur, LUISINHO_PROFILE["vocal_dyn_mean"], LUISINHO_PROFILE["vocal_dyn_std"])

    zbar = (0.5 * z_sr) + (0.3 * z_em) + (0.2 * z_vd)
    narrativity = _to_similarity(zbar, span=2.5)

    total_words = len(full_text.split())

    if score_accumulator is not None:
        logger.info("Usando score incremental do acumulador para avaliação textual.")
        summary = score_accumulator.summarize()
        emotion_content = summary.get("emotion", 0.0)
        storytelling = summary.get("storytelling", 0.0)
        game_rhythm = summary.get("game_rhythm", 0.0)
        appropriate_fun = _score_appropriate_fun(full_text)
        total = (
            emotion_content * 0.25 +
            storytelling * 0.35 +
            game_rhythm * 0.25 +
            appropriate_fun * 0.15
        ) * 10
        words = full_text.split()
        unique_words = set(w.lower() for w in words)
        lexical_diversity = len(unique_words) / len(words) if len(words) > 0 else 0
        details = {
            "total_words": len(words),
            "unique_words": len(unique_words),
            "lexical_diversity": lexical_diversity,
            "avg_words_per_segment": len(words) / len(transcripts) if transcripts else 0,
            "source": "score_accumulator",
        }
        text_score = TextScore(
            emotion_content=emotion_content,
            storytelling=storytelling,
            game_rhythm=game_rhythm,
            appropriate_fun=appropriate_fun,
            total_score=total,
            details=details,
        )
        heur_text_total_baseline = float(text_score.total_score)
    else:
        text_score = _evaluate_text(transcripts, segments)
        heur_text_total_baseline = float(text_score.total_score)

    if (
        getattr(settings, "enable_llm_scoring", False)
        and getattr(settings, "llm_full_scoring", False)
        and not getattr(settings, "llm_validation_only", False)
    ):
        try:
            from stages.llm_scorer import configure_llm

            model = configure_llm()
            if model:
                use_windowed = bool(getattr(settings, "llm_windowed_scoring", True))
                if use_windowed:
                    wins = _select_windows(segments, transcripts, stream_id, total_duration, is_highlights)
                    agg = _aggregate_llm_windows(
                        wins,
                        model,
                        float(audio_score.total_score),
                        float(narrativity),
                        bool(is_highlights),
                    )
                else:
                    agg = None

                if agg is not None:
                    emo, sto, pace, fun, text_total_llm, details = agg
                    text_score = TextScore(
                        emotion_content=float(emo),
                        storytelling=float(sto),
                        game_rhythm=float(pace),
                        appropriate_fun=float(fun),
                        total_score=float(text_total_llm),
                        details=details,
                    )
                    blend_applied = False
                    if is_highlights:
                        heur_weight = _highlight_heuristic_weight(float(narrativity))
                        llm_weight = 1.0 - heur_weight
                        blended = heur_weight * heur_text_total_baseline + llm_weight * float(text_score.total_score)
                        det = text_score.details or {}
                        det["blended"] = {
                            "mode": "highlights_dynamic_blend",
                            "heur": heur_text_total_baseline,
                            "llm": float(text_score.total_score),
                            "weights": {"heur": heur_weight, "llm": llm_weight},
                        }
                        text_score = TextScore(
                            emotion_content=text_score.emotion_content,
                            storytelling=text_score.storytelling,
                            game_rhythm=text_score.game_rhythm,
                            appropriate_fun=text_score.appropriate_fun,
                            total_score=float(blended),
                            details=det,
                        )
                        blend_applied = True
                    else:
                        heur_weight = _full_stream_blend_weight(float(narrativity))
                        if heur_weight is not None:
                            llm_weight = 1.0 - heur_weight
                            blended = heur_weight * heur_text_total_baseline + llm_weight * float(text_score.total_score)
                            det = text_score.details or {}
                            det["blended"] = {
                                "mode": "full_dynamic_blend",
                                "heur": heur_text_total_baseline,
                                "llm": float(text_score.total_score),
                                "weights": {"heur": heur_weight, "llm": llm_weight},
                            }
                            text_score = TextScore(
                                emotion_content=text_score.emotion_content,
                                storytelling=text_score.storytelling,
                                game_rhythm=text_score.game_rhythm,
                                appropriate_fun=text_score.appropriate_fun,
                                total_score=float(blended),
                                details=det,
                            )
                            blend_applied = True
                    logger.info(
                        "LLM_WIN: emo=%.1f sto=%.1f pace=%.1f fun=%.1f narr=%.2f total=%.1f wins=%d",
                        emo,
                        sto,
                        pace,
                        fun,
                        narrativity,
                        text_total_llm,
                        len(wins),
                    )
                else:
                    from stages.llm_scorer import evaluate_narration_criteria_llm

                    llm_result = evaluate_narration_criteria_llm(full_text, model)
                    if llm_result:
                        style_block = llm_result.get("style_alignment_luisinho", {}) or {}
                        style_score = float(style_block.get("score", 0.0))
                        signals = style_block.get("signals", {}) or {}
                        analysis_ratio = float(signals.get("analysis_ratio", 0.0))
                        offtopic_ratio = float(signals.get("offtopic_ratio", 0.0))
                        emo = float(llm_result.get("emotion", {}).get("score", 0.0))
                        sto = float(llm_result.get("storytelling", {}).get("score", 0.0))
                        pace = float(llm_result.get("game_pace", {}).get("score", 0.0))
                        fun = float(llm_result.get("appropriate_commentary", {}).get("score", 0.0))
                        style_gate = 0.7 + 0.04 * style_score
                        emo *= style_gate
                        pace *= style_gate
                        if analysis_ratio >= 0.45:
                            sto -= 4.0
                            pace -= 2.5
                        elif analysis_ratio >= 0.35:
                            sto -= 2.0
                            pace -= 1.5
                        if offtopic_ratio >= 0.18:
                            fun -= max(0.5, 1.0 * (1.0 - (style_score / 10.0)))
                        emo = float(np.clip(emo, 0, 10))
                        sto = float(np.clip(sto, 0, 10))
                        pace = float(np.clip(pace, 0, 10))
                        fun = float(np.clip(fun, 0, 10))
                        text_total_llm = (0.30 * emo + 0.30 * sto + 0.25 * pace + 0.15 * fun) * 10
                        text_total_llm *= (0.75 + 0.25 * narrativity)
                        audio_total = float(audio_score.total_score)
                        if text_total_llm > audio_total + 8:
                            text_total_llm = audio_total + 8
                        if narrativity < 0.5 and text_total_llm > 78:
                            text_total_llm = 78
                        text_score = TextScore(
                            emotion_content=emo,
                            storytelling=sto,
                            game_rhythm=pace,
                            appropriate_fun=fun,
                            total_score=float(text_total_llm),
                            details={"source": "llm", "signals": signals, "style_alignment": style_score},
                        )
                        if is_highlights:
                            heur_weight = _highlight_heuristic_weight(float(narrativity))
                            llm_weight = 1.0 - heur_weight
                            blended = heur_weight * heur_text_total_baseline + llm_weight * float(text_score.total_score)
                            det = text_score.details or {}
                            det["blended"] = {
                                "mode": "highlights_dynamic_blend",
                                "heur": heur_text_total_baseline,
                                "llm": float(text_score.total_score),
                                "weights": {"heur": heur_weight, "llm": llm_weight},
                            }
                            text_score = TextScore(
                                emotion_content=text_score.emotion_content,
                                storytelling=text_score.storytelling,
                                game_rhythm=text_score.game_rhythm,
                                appropriate_fun=text_score.appropriate_fun,
                                total_score=float(blended),
                                details=det,
                            )
                        else:
                            heur_weight = _full_stream_blend_weight(float(narrativity))
                            if heur_weight is not None:
                                llm_weight = 1.0 - heur_weight
                                blended = heur_weight * heur_text_total_baseline + llm_weight * float(text_score.total_score)
                                det = text_score.details or {}
                                det["blended"] = {
                                    "mode": "full_dynamic_blend",
                                    "heur": heur_text_total_baseline,
                                    "llm": float(text_score.total_score),
                                    "weights": {"heur": heur_weight, "llm": llm_weight},
                                }
                                text_score = TextScore(
                                    emotion_content=text_score.emotion_content,
                                    storytelling=text_score.storytelling,
                                    game_rhythm=text_score.game_rhythm,
                                    appropriate_fun=text_score.appropriate_fun,
                                    total_score=float(blended),
                                    details=det,
                                )
                        logger.info(
                            "LLM_TEXT: emo=%.1f sto=%.1f pace=%.1f fun=%.1f narr=%.2f total=%.1f",
                            emo,
                            sto,
                            pace,
                            fun,
                            narrativity,
                            text_total_llm,
                        )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Falha ao usar LLM para scoring de texto: {e}")

    passion_bonus = _passion_bonus(full_text, float(narrativity))
    if passion_bonus > 0:
        det = dict(text_score.details or {})
        adjustments = det.setdefault("adjustments", {})
        adjustments["passion_bonus"] = passion_bonus
        text_score = TextScore(
            emotion_content=text_score.emotion_content,
            storytelling=text_score.storytelling,
            game_rhythm=text_score.game_rhythm,
            appropriate_fun=text_score.appropriate_fun,
            total_score=float(text_score.total_score + passion_bonus),
            details=det,
        )

    logger.info(f"Score de texto: {text_score.total_score:.2f}/100")

    text_gate = _text_gate_for_narrativity(float(narrativity))
    text_total_adj = float(text_score.total_score) * text_gate
    if narrativity >= 0.90:
        text_total_adj = max(text_total_adj, 52.0)
    elif narrativity >= 0.85:
        text_total_adj = max(text_total_adj, 50.0)

    details = text_score.details or {}
    if (audio_score.total_score >= 80) and (total_words > 5000):
        confidence_factor = 1.0
    else:
        confidence_factor = 0.5 + (sports_confidence / 2)

    if is_highlights:
        base_score = (
            audio_score.total_score * SCORING_CONTEXT.highlights_audio_weight
            + text_total_adj * SCORING_CONTEXT.highlights_text_weight
        )
    else:
        base_score = (
            audio_score.total_score * SCORING_CONTEXT.audio_weight
            + text_total_adj * SCORING_CONTEXT.text_weight
        )
    final_score_value = base_score * confidence_factor

    style_bonus = 0
    if narrativity >= 0.80 and text_total_adj >= 60:
        style_bonus = 4
    if narrativity >= 0.90 and text_total_adj >= 70:
        style_bonus = 8
    if narrativity >= 0.92 and text_total_adj >= 65:
        style_bonus = 10

    style_penalty = 0
    if narrativity <= 0.45 and text_total_adj >= 78:
        style_penalty = 2
    if narrativity <= 0.35 and text_total_adj >= 80:
        style_penalty = 3

    final_score_value = final_score_value + style_bonus - style_penalty

    # Style boost: narrador muito próximo do perfil + áudio forte + excitação
    try:
        sig = (text_score.details or {}).get("signals_avg", {})
        global_interj = float(sig.get("interjection_rate_per_100w", 0.0))
        style_avg = float(sig.get("style_alignment", 0.0))
    except Exception:
        global_interj = 0.0
        style_avg = 0.0
    if narrativity >= 0.90 and audio_score.total_score >= 84 and global_interj >= 2.0:
        final_score_value += 3
        if style_avg >= 4.0:
            final_score_value += 2

    if (not is_highlights) and (audio_score.total_score >= 84) and (float(text_score.total_score) >= 40):
        final_score_value += 2

    bonus = 0
    if audio_score.total_score >= 85 and text_score.total_score >= 60:
        bonus = 3
    if audio_score.total_score >= 87 and text_score.total_score >= 70:
        bonus = 5

    final_score_value = min(100, final_score_value + bonus)

    classification = _get_classification(final_score_value)

    strengths, improvements = _generate_feedback(audio_score, text_score, sports_confidence)

    if getattr(settings, "enable_llm_scoring", False):
        try:
            from stages.llm_scorer import configure_llm, generate_personalized_feedback_llm

            model = configure_llm()
            if model:
                audio_dict = {
                    "vocal_dynamics": audio_score.vocal_dynamics,
                    "speech_pacing": audio_score.speech_pacing,
                    "emotion_audio": audio_score.emotion_audio,
                }
                text_dict = {
                    "emotion_content": text_score.emotion_content,
                    "storytelling": text_score.storytelling,
                    "game_pace": text_score.game_rhythm,
                    "appropriate_commentary": text_score.appropriate_fun,
                }
                llm_strengths, llm_improvements = generate_personalized_feedback_llm(
                    audio_dict, text_dict, final_score_value, classification, model
                )
                if llm_strengths or llm_improvements:
                    strengths = llm_strengths if llm_strengths else strengths
                    improvements = llm_improvements if llm_improvements else improvements
        except Exception as e:  # pragma: no cover
            logger.warning(f"Falha ao gerar feedback com LLM: {e}")

    final_result = FinalScore(
        stream_id=stream_id,
        audio_score=audio_score,
        text_score=text_score,
        final_score=final_score_value,
        classification=classification,
        strengths=strengths,
        improvements=improvements,
        total_segments=len(segments),
        total_duration=total_duration,
    )

    try:
        if final_score_value >= 70.0:
            from smtp_sender import Send_Mail

            assunto = f"Narrador destaque: {stream_id}"
            mensagem = (
                f"O narrador {stream_id} atingiu {final_score_value:.2f} pontos (classificação: {classification}).\n"
                f"Áudio: {audio_score.total_score:.2f} | Texto: {text_score.total_score:.2f}\n"
                f"Modalidade: {'highlights' if is_highlights else 'stream completo'}\n"
                f"Total de segmentos: {len(segments)} | Duração: {total_duration:.1f}s."
            )
            Send_Mail(assunto, mensagem)
    except Exception as notify_err:  # pragma: no cover
        logger.warning(f"Falha ao enviar notificação por e-mail: {notify_err}")

    logger.info(f"✅ Avaliação completa: {final_score_value:.2f}/100 - {classification}")
    return final_result


def _create_empty_score() -> FinalScore:
    empty_audio = AudioScore(0, 0, 0, 0, None)
    empty_text = TextScore(0, 0, 0, 0, 0, None)

    return FinalScore(
        stream_id="unknown",
        audio_score=empty_audio,
        text_score=empty_text,
        final_score=0,
        classification="Insuficiente",
        strengths=["Dados insuficientes para avaliação"],
        improvements=["Necessário mais conteúdo para análise"],
        total_segments=0,
        total_duration=0,
    )


__all__ = ["graduate_audio_and_text"]
