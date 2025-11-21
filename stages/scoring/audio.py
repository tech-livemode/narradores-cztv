# Audio scoring logic (dinâmica vocal, ritmo, emoção).

import logging
from typing import List
import numpy as np  # type: ignore
try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:  # pragma: no cover
    spearmanr = None  # type: ignore

from model.processedsegment import ProcessedSegment
from model.scoreresult import AudioScore

from .config import settings
from .utils import _normalize_feature

logger = logging.getLogger(__name__)


def _evaluate_audio(segments: List[ProcessedSegment]) -> AudioScore:
    # Calcula o AudioScore (0–100) normalizando dinâmica vocal, ritmo e emoção.

    # Se habilitado, foca as métricas de áudio nos segmentos do narrador
    use_narr = bool(getattr(settings, "light_diarization_focus_audio", True))
    if use_narr:
        narr_feats = [
            s.features
            for s in segments
            if getattr(s, "meta", {})
            and getattr(s, "meta", {}).get("light_speaker") == "narrator"
            and s.features
        ]
        all_features = narr_feats if narr_feats else [seg.features for seg in segments if seg.features]
    else:
        all_features = [seg.features for seg in segments if seg.features]

    if not all_features:
        logger.warning("Nenhuma feature de áudio disponível")
        return AudioScore(0, 0, 0, 0, {})

    vocal_dynamics = _score_vocal_dynamics(all_features)
    speech_pacing = _score_speech_pacing(all_features)
    emotion_audio = _score_audio_emotion(all_features)

    emotion_corr_raw = _score_audio_emotion.last_corr
    if np.isnan(emotion_corr_raw):
        emotion_corr_raw = 0.0
    else:
        emotion_corr_raw = float(emotion_corr_raw)

    # Score base e bônus contextual
    total = (vocal_dynamics + speech_pacing + emotion_audio) / 3
    if vocal_dynamics > 7 and emotion_audio > 7 and 5 <= speech_pacing <= 8:
        total = min(total + 1.5, 10)

    details = {
        "avg_pitch": np.mean([f.get('pitch') for f in all_features if f.get('pitch')]),
        "avg_rms": np.mean([f.get('rms') for f in all_features if f.get('rms')]),
        "avg_speech_rate": np.mean([f.get('speech_rate') for f in all_features if f.get('speech_rate')]),
        "avg_snr": np.mean([f.get('snr') for f in all_features if f.get('snr')]),
        "emotion_corr_raw": emotion_corr_raw,
        "segments_used_for_emotion": len(all_features),
    }

    return AudioScore(
        vocal_dynamics=vocal_dynamics,
        speech_pacing=speech_pacing,
        emotion_audio=emotion_audio,
        total_score=total * 10,
        details=details,
    )


def _score_vocal_dynamics(features: List) -> float:
    # Mede a expressividade da voz pela variação e alcance do pitch e energia.
    try:
        pitch_values = [f.get('pitch') for f in features if f.get('pitch') and not np.isnan(f.get('pitch'))]
        rms_values = [f.get('rms') for f in features if f.get('rms')]

        if not pitch_values or not rms_values:
            return 5.0

        pitch_range = max(pitch_values) - min(pitch_values)
        pitch_var_coeff = np.std(pitch_values) / (np.mean(pitch_values) + 1e-5)

        rms_range = max(rms_values) - min(rms_values)
        rms_var_coeff = np.std(rms_values) / (np.mean(rms_values) + 1e-5)

        # Combina variação e consistência
        expressiveness = (pitch_range / 150) * 5 + (rms_range / 0.1) * 2 + (pitch_var_coeff + rms_var_coeff) * 1.5
        score = np.clip(expressiveness, 0, 10)
        return float(score)
    except Exception as e:
        logger.warning(f"Erro ao calcular dinâmica vocal: {e}")
        return 5.0


def _score_speech_pacing(features: List) -> float:
    # Avalia o ritmo da fala considerando velocidade média e estabilidade.
    try:
        speech_rates = [f.get('speech_rate') for f in features if f.get('speech_rate')]
        if not speech_rates:
            return 5.0

        avg_rate = np.mean(speech_rates)
        rate_var = np.std(speech_rates)

        avg_rate_score = _normalize_feature(avg_rate, "speech_rate")

        target_var = 0.8
        stability_score = np.clip(10 - abs(rate_var - target_var) * 8, 0, 10)

        score = avg_rate_score * 0.7 + stability_score * 0.3
        return float(np.clip(score, 0, 10))
    except Exception as e:
        logger.warning(f"Erro ao calcular ritmo da fala: {e}")
        return 5.0


def _score_audio_emotion(features: List) -> float:
    # Mede emoção considerando correlação entre pitch e energia (RMS) e suas dinâmicas.
    try:
        rms = np.array([f.get('rms', np.nan) for f in features], dtype=float)
        pitch = np.array([f.get('pitch', np.nan) for f in features], dtype=float)

        voiced_mask = (
            ~np.isnan(rms)
            & ~np.isnan(pitch)
            & (pitch > 60.0)
            & (rms > 1e-6)
        )

        rms = rms[voiced_mask]
        pitch = pitch[voiced_mask]

        if len(rms) < 3:
            _score_audio_emotion.last_corr = 0.0
            return 5.0

        pearson = np.corrcoef(rms, pitch)[0, 1]

        dyn = np.nan
        if len(rms) > 3:
            drms = np.diff(rms)
            dpitch = np.diff(pitch)
            if len(drms) > 1 and len(dpitch) > 1:
                dyn = np.corrcoef(drms, dpitch)[0, 1]

        spearman_val = np.nan
        if spearmanr is not None:
            try:
                spearman_result = spearmanr(rms, pitch)
                if hasattr(spearman_result, "correlation"):
                    spearman_val = spearman_result.correlation
                elif isinstance(spearman_result, tuple):
                    spearman_val = spearman_result[0]
            except Exception:
                spearman_val = np.nan

        measured = []
        weights = []
        for value, weight in ((pearson, 0.4), (dyn, 0.4), (spearman_val, 0.2)):
            if value is not None and not np.isnan(value):
                measured.append(value)
                weights.append(weight)

        if weights:
            normalized_weights = np.array(weights) / np.sum(weights)
            corr_raw = float(np.dot(normalized_weights, measured))
        else:
            corr_raw = 0.0

        corr = float(np.clip(corr_raw, -1.0, 1.0))
        corr_norm = (corr + 1.0) / 2.0

        logger.debug(
            f"[audio_emotion] n={len(rms)}, corr={corr:.3f}, "
            f"pearson={pearson:.3f}, dyn={dyn:.3f}, spearman={spearman_val:.3f}, corr_norm={corr_norm:.3f}"
        )

        normalized = _normalize_feature(corr_norm, "emotion_correlation")
        _score_audio_emotion.last_corr = corr
        return normalized
    except Exception as e:
        logger.warning(f"Erro ao calcular correlação pitch×energia: {e}")
        _score_audio_emotion.last_corr = 0.0
        return 5.0


_score_audio_emotion.last_corr = 0.0

__all__ = ["_evaluate_audio"]
