"""Window selection and materialization for LLM evaluation."""

import re
from typing import Dict, List, Optional
import numpy as np # type: ignore
from model.processedsegment import ProcessedSegment
from .config import (
    ACTION_TRIGGERS,
    ANALYTIC_HINTS,
    DEFAULT_LLM_WINDOW_SECONDS,
    DEFAULT_LLM_WINDOWS,
    settings,
)
from .diarization_light import _leader_voice_ratio, _narrator_share_for_window
from .types import Window
from .utils import _zscore

INTERJECTION_PATTERN = re.compile(
    r"(gol+|que lance|é agora|segura|vamos|bora|explod(e|iu)|sensacional|absurdo|\buau\b|inacreditável)",
    re.IGNORECASE,
)
IMPERATIVE_PATTERN = re.compile(
    r"(bate|cruza|marca|vem|olha|vai|segura|chuta|toca|corre)!?",
    re.IGNORECASE,
)


def _segments_to_blocks(segments: List[ProcessedSegment]) -> List[tuple]:
    # Converte segments em blocos (start, end, speech_rate, rms, pitch).
    blocks = []
    for s in segments:
        st = float(getattr(s, "start_time", 0.0) or 0.0)
        et = float(getattr(s, "end_time", st) or st)
        f = getattr(s, "features", {}) or {}
        blocks.append(
            (
                st,
                et,
                float(f.get("speech_rate") or 0.0),
                float(f.get("rms") or 0.0),
                float(f.get("pitch") or 0.0),
            )
        )
    return blocks


def _build_text_index(transcripts: List[Dict]) -> List[tuple]:
    # Cria índice [(start, end, text)] a partir de transcripts quando possível.
    with_ts = []
    has_ts = False
    for t in transcripts:
        st = t.get("start") if t.get("start") is not None else t.get("start_time")
        et = t.get("end") if t.get("end") is not None else t.get("end_time")
        txt = t.get("text", "")
        if isinstance(st, (int, float)) and isinstance(et, (int, float)):
            has_ts = True
            with_ts.append((float(st), float(et), txt))
    if has_ts and with_ts:
        return with_ts
    full_text = " ".join([t.get("text", "") for t in transcripts])
    return [(0.0, float("inf"), full_text)]


def _collect_text(text_index: List[tuple], start: float, end: float) -> str:
    # Concatena texto cuja janela [start,end] cobre parcialmente.
    if len(text_index) == 1 and text_index[0][1] == float("inf"):
        return text_index[0][2]
    parts = []
    for st, et, tx in text_index:
        if et >= start and st <= end:
            parts.append(tx)
    return " ".join(parts)


def _pick_peak_windows(blocks: List[tuple], k: int, win_seconds: float) -> List[Window]:
    # Seleciona janelas com picos usando z-score de speech rate/rms.
    if not blocks:
        return []
    starts = np.array([b[0] for b in blocks], dtype=float)
    ends = np.array([b[1] for b in blocks], dtype=float)
    sr = np.array([b[2] for b in blocks], dtype=float)
    rms = np.array([b[3] for b in blocks], dtype=float)

    z = np.maximum(_zscore(sr), _zscore(rms))
    order = np.argsort(-z)
    chosen = []
    half = win_seconds / 2.0

    for idx in order:
        if len(chosen) >= k:
            break
        center = float((starts[idx] + ends[idx]) / 2.0)
        wst = max(0.0, center - half)
        wed = center + half
        overlap = any(not (wed <= c.start or wst >= c.end) for c in chosen)
        if overlap:
            continue
        chosen.append(Window(wst, wed, "", "peak", 1.5))
    return chosen


def _pick_uniform_windows(blocks: List[tuple], k: int, win_seconds: float) -> List[Window]:
    if not blocks:
        return []
    total_start = float(min(b[0] for b in blocks))
    total_end = float(max(b[1] for b in blocks))
    total_dur = max(0.0, total_end - total_start)
    if total_dur <= 0:
        return []
    step = total_dur / (k + 1)
    half = win_seconds / 2.0
    wins = []
    for i in range(1, k + 1):
        center = total_start + i * step
        wst = max(0.0, center - half)
        wed = center + half
        wins.append(Window(wst, wed, "", "uniform", 1.0))
    return wins


def _pick_analytic_window(text_index: List[tuple], blocks: List[tuple], win_seconds: float) -> List[Window]:
    # Seleciona 1 janela com maior densidade de termos analíticos.
    if not blocks:
        return []
    total_start = float(min(b[0] for b in blocks))
    total_end = float(max(b[1] for b in blocks))
    total_dur = max(0.0, total_end - total_start)
    if total_dur <= 0:
        return []
    half = win_seconds / 2.0
    best = None
    best_score = -1
    probes = max(6, int(total_dur // (win_seconds * 1.2)))
    for i in range(probes):
        center = total_start + (i + 0.5) * (total_dur / probes)
        wst = max(0.0, center - half)
        wed = center + half
        tx = _collect_text(text_index, wst, wed)
        score = len(ANALYTIC_HINTS.findall(tx or ""))
        if score > best_score:
            best_score = score
            best = (wst, wed)
    if best is None:
        return []
    return [Window(best[0], best[1], "", "analytic", 0.7)]


def _materialize_text(wins: List[Window], text_index: List[tuple]) -> None:
    for w in wins:
        w.text = _collect_text(text_index, w.start, w.end)
        if not w.text or len(w.text.strip()) < 50:
            w.kind = "fallback"
            w.weight = 0.8 if w.weight < 0.8 else w.weight
        else:
            if w.kind == "longform":
                word_count = len(w.text.split())
                if word_count >= 120:
                    w.weight *= 1.15
                elif word_count >= 80:
                    w.weight *= 1.05


def _pick_longform_windows(
    text_index: List[tuple],
    total_start: float,
    total_end: float,
    k: int,
    win_seconds: float,
) -> List[Window]:
    if k <= 0 or total_end <= total_start:
        return []
    half = win_seconds / 2.0
    span = total_end - total_start
    probes = max(6, k * 3)
    candidates = []
    for i in range(probes):
        center = total_start + (i + 0.5) * (span / probes)
        wst = max(0.0, center - half)
        wed = center + half
        tx = _collect_text(text_index, wst, wed)
        text = tx or ""
        word_count = len(text.split())
        emotion_hits = len(INTERJECTION_PATTERN.findall(text)) + len(IMPERATIVE_PATTERN.findall(text))
        score = word_count + (25 * emotion_hits)
        candidates.append((score, word_count, wst, wed))
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: List[Window] = []
    for _score, word_count, wst, wed in candidates:
        if len(selected) >= k:
            break
        overlap = any(not (wed <= s.start or wst >= s.end) for s in selected)
        if overlap:
            continue
        weight = 1.30 if word_count >= 80 else 1.15
        selected.append(Window(wst, wed, "", "longform", weight))
    return selected


def _select_windows(
    segments: List[ProcessedSegment],
    transcripts: List[Dict],
    stream_id: str,
    total_duration: float,
    is_highlights: Optional[bool] = None,
) -> List[Window]:
    # Seleciona janelas (picos + uniforme + analítica) e materializa textos.
    from .validation import _is_highlights_mode

    if is_highlights is None:
        is_highlights = _is_highlights_mode(stream_id, transcripts, total_duration)

    n_windows = int(getattr(settings, "llm_windows", DEFAULT_LLM_WINDOWS))
    win_seconds = float(getattr(settings, "llm_window_seconds", DEFAULT_LLM_WINDOW_SECONDS))
    if is_highlights:
        win_seconds = max(18.0, win_seconds * 0.8)
        n_windows = max(6, int(round(n_windows * 1.0)))

    blocks = _segments_to_blocks(segments)
    text_index = _build_text_index(transcripts)

    total_start = float(min(b[0] for b in blocks)) if blocks else 0.0
    total_end = float(max(b[1] for b in blocks)) if blocks else win_seconds

    n_peak = max(3, int(round(n_windows * 0.50)))
    n_uniform = max(2, int(round(n_windows * 0.20)))
    n_analytic = max(1, int(round(n_windows * 0.10)))
    n_long = max(2, int(round(n_windows * 0.25)))

    allocations = [
        ["peak", n_peak, 3],
        ["long", n_long, 2],
        ["uniform", n_uniform, 2],
        ["analytic", n_analytic, 1],
    ]
    total_alloc = sum(a[1] for a in allocations)
    while total_alloc > n_windows:
        for entry in allocations:
            if total_alloc <= n_windows:
                break
            if entry[1] > entry[2]:
                entry[1] -= 1
                total_alloc -= 1
    n_peak = allocations[0][1]
    n_long = allocations[1][1]
    n_uniform = allocations[2][1]
    n_analytic = allocations[3][1]

    wins = []
    wins.extend(_pick_peak_windows(blocks, n_peak, win_seconds))
    wins.extend(_pick_uniform_windows(blocks, n_uniform, win_seconds))
    wins.extend(_pick_analytic_window(text_index, blocks, win_seconds))
    wins.extend(_pick_longform_windows(text_index, total_start, total_end, n_long, win_seconds))

    # Add up to 2 pre-peak windows (lead-in) for the strongest peaks
    try:
        peaks = [w for w in wins if w.kind == "peak"]
        peaks_sorted = sorted(peaks, key=lambda w: w.weight, reverse=True)[:2]
        half = win_seconds / 2.0
        added = []
        for p in peaks_sorted:
            center = (p.start + p.end) / 2.0
            pre_center = max(0.0, center - 10.0)
            wst = max(0.0, pre_center - half)
            wed = wst + win_seconds
            overlap = any(not (wed <= e.start or wst >= e.end) for e in wins + added)
            if overlap:
                continue
            added.append(Window(wst, wed, "", "prepeak", 1.20))
        if added:
            wins.extend(added)
            # keep total count under budget by removing weakest uniforms/others
            if len(wins) > n_windows:
                to_remove = len(wins) - n_windows
                rank = {"uniform": 0, "analytic": 1, "longform": 2, "fallback": 2, "prepeak": 3, "peak": 4}
                removable = sorted([w for w in wins], key=lambda w: (rank.get(w.kind, 2), w.weight))
                # never remove peaks or prepeaks first unless necessary
                filtered = []
                removed = 0
                for w in removable:
                    if removed < to_remove and rank.get(w.kind, 2) <= 2:
                        removed += 1
                        continue
                    filtered.append(w)
                # if still over, remove globally lowest weights
                while len(filtered) > n_windows:
                    idx = min(range(len(filtered)), key=lambda i: filtered[i].weight)
                    filtered.pop(idx)
                wins = filtered
    except Exception:
        pass

    if not wins:
        only_text = text_index[0][2] if text_index else ""
        return [Window(0.0, win_seconds, only_text, "fallback", 1.0)]

    _materialize_text(wins, text_index)

    if bool(getattr(settings, "enable_light_diarization", True)):
        boost = float(getattr(settings, "light_diarization_window_boost", 0.35))
        min_share = 0.60
        kept = []
        for w in wins:
            w.narrator_share = _narrator_share_for_window(segments, w.start, w.end)
            w.leader_voice_ratio = _leader_voice_ratio(segments, w.start, w.end)
            if w.narrator_share < min_share:
                continue

            frac = max(0.0, min(1.0, (w.narrator_share - 0.50) / 0.40))
            w.weight *= (1.0 + max(boost, 0.20) * frac)

            if w.narrator_share >= 0.80:
                w.weight *= 1.12
            elif w.narrator_share < 0.55:
                w.weight *= 0.85
            elif w.narrator_share < 0.60:
                w.weight *= 0.90

            if w.leader_voice_ratio >= 0.65:
                w.weight *= 1.06

            if ACTION_TRIGGERS.search(w.text or ""):
                w.weight *= 1.10

            kept.append(w)
        if kept:
            wins = kept

    dense = []
    for w in wins:
        wl = len((w.text or "").split())
        if wl >= 60:
            dense.append(w)
        elif w.kind == "longform" and wl >= 80:
            dense.append(w)
        elif is_highlights and w.kind == "peak" and wl >= 35:
            dense.append(w)
    return dense if dense else wins


__all__ = ["_select_windows"]
