"""LLM scoring aggregation across selected windows."""

from typing import List, Optional
import numpy as np  # type: ignore

from .types import Window


def _aggregate_llm_windows(
    windows: List[Window],
    model,
    audio_total: float,
    narrativity: float,
    is_highlights: bool,
):
    # Pontua cada janela via LLM e aplica agregação trimmed ponderada.
    from stages.llm_scorer import evaluate_narration_criteria_llm

    if not windows:
        return None

    num = {"emo": 0.0, "sto": 0.0, "pace": 0.0, "fun": 0.0, "style": 0.0}
    den = 0.0
    signals_acc = {
        "analysis_ratio": 0.0,
        "offtopic_ratio": 0.0,
        "interjection_rate_per_100w": 0.0,
        "imperative_rate_per_100w": 0.0,
    }

    per_text_vals = []

    for w in windows:
        res = evaluate_narration_criteria_llm(w.text, model)
        if not res:
            continue
        style_block = res.get("style_alignment_luisinho", {}) or {}
        style_score = float(style_block.get("score", 0.0))
        signals = style_block.get("signals", {}) or {}
        analysis_ratio = float(signals.get("analysis_ratio", 0.0))
        offtopic_ratio = float(signals.get("offtopic_ratio", 0.0))
        interj = float(signals.get("interjection_rate_per_100w", 0.0))
        imper = float(signals.get("imperative_rate_per_100w", 0.0))

        emo = float(res.get("emotion", {}).get("score", 0.0))
        sto = float(res.get("storytelling", {}).get("score", 0.0))
        pace = float(res.get("game_pace", {}).get("score", 0.0))
        fun = float(res.get("appropriate_commentary", {}).get("score", 0.0))

        if is_highlights:
            style_gate = 0.92 + 0.03 * style_score
            off_penalty_thr = 0.22
        else:
            style_gate = 0.90 + 0.02 * style_score
            off_penalty_thr = 0.18
        style_gate = float(np.clip(style_gate, 0.90, 1.12))
        emo *= style_gate
        pace *= style_gate

        analysis_factor = 0.5 if interj >= 2.0 else 1.0
        if analysis_ratio >= 0.50:
            sto -= 2.5 * analysis_factor
            pace -= 1.5 * analysis_factor
        elif analysis_ratio >= 0.38:
            sto -= 1.0 * analysis_factor
            pace -= 0.5 * analysis_factor

        if offtopic_ratio >= 0.28:
            penalty = max(0.5, 0.7 * (1.0 - (style_score / 10.0)))
            if narrativity < 0.80 and offtopic_ratio >= 0.35:
                penalty *= 1.25
            fun -= penalty

        emo = float(np.clip(emo, 0, 10))
        sto = float(np.clip(sto, 0, 10))
        pace = float(np.clip(pace, 0, 10))
        fun = float(np.clip(fun, 0, 10))

        text_win = (0.30 * emo + 0.30 * sto + 0.25 * pace + 0.15 * fun) * 10
        if narrativity >= 0.85:
            text_win *= (0.7 + 0.5 * narrativity)
        else:
            text_win *= (0.6 + 0.4 * narrativity)

        if text_win > audio_total + 8:
            text_win = audio_total + 8
        if narrativity < 0.5 and text_win > 78:
            text_win = 78

        floor_min = 48.0 if narrativity >= 0.90 else (45.0 if narrativity >= 0.85 else (38.0 if narrativity >= 0.55 else 35.0))
        if interj >= 6 or imper >= 3:
            text_win = max(text_win, floor_min)

        per_text_vals.append((float(text_win), float(w.weight)))
        num["emo"] += emo * w.weight
        num["sto"] += sto * w.weight
        num["pace"] += pace * w.weight
        num["fun"] += fun * w.weight
        num["style"] += style_score * w.weight
        signals_acc["analysis_ratio"] += analysis_ratio * w.weight
        signals_acc["offtopic_ratio"] += offtopic_ratio * w.weight
        signals_acc["interjection_rate_per_100w"] += interj * w.weight
        signals_acc["imperative_rate_per_100w"] += imper * w.weight
        den += w.weight

    if den <= 0 or not per_text_vals:
        return None

    emo = num["emo"] / den
    sto = num["sto"] / den
    pace = num["pace"] / den
    fun = num["fun"] / den
    style = num["style"] / den
    analysis_ratio = signals_acc["analysis_ratio"] / den
    offtopic_ratio = signals_acc["offtopic_ratio"] / den
    interj = signals_acc["interjection_rate_per_100w"] / den
    imper = signals_acc["imperative_rate_per_100w"] / den

    vals = sorted(per_text_vals, key=lambda x: x[0])
    if narrativity >= 0.90 and len(vals) >= 5:
        vals = vals[2:]
    elif narrativity >= 0.80 and len(vals) >= 7:
        vals = vals[2:-1]
    elif len(vals) >= 6:
        vals = vals[1:-1]
    num_t = sum(v * w for v, w in vals)
    den_t = sum(w for _v, w in vals)
    text_total = num_t / max(den_t, 1e-6)

    if float(audio_total) >= 88 and (interj >= 5.0 or imper >= 2.5):
        text_total = max(text_total, 42.0)

    details = {
        "source": "llm_windowed",
        "windows": [
            {
                "start": w.start,
                "end": w.end,
                "kind": w.kind,
                "weight": w.weight,
                "narrator_share": getattr(w, "narrator_share", 0.0),
                "leader_voice_ratio": getattr(w, "leader_voice_ratio", 0.0),
            }
            for w in windows
        ],
        "signals_avg": {
            "analysis_ratio": analysis_ratio,
            "offtopic_ratio": offtopic_ratio,
            "interjection_rate_per_100w": interj,
            "imperative_rate_per_100w": imper,
            "style_alignment": style,
        },
        "aggregation": "trimmed_weighted",
    }

    return emo, sto, pace, fun, float(text_total), details


__all__ = ["_aggregate_llm_windows"]
