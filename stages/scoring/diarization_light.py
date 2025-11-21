# Light diarization helpers for narrator clustering and window metrics.

from typing import List, TYPE_CHECKING
import numpy as np  # type: ignore

if TYPE_CHECKING:
    from model.processedsegment import ProcessedSegment


def _ensure_meta(seg: "ProcessedSegment") -> dict:
    m = getattr(seg, "meta", None)
    if m is None:
        m = {}
        setattr(seg, "meta", m)
    return m


def _k2_cluster(vals: np.ndarray) -> np.ndarray:
    # k=2 simples em (speech_rate, rms). Retorna labels 0/1.
    if vals.shape[0] < 4:
        return np.zeros((vals.shape[0],), dtype=int)
    # inicia por quantis para robustez
    q_sr_lo, q_sr_hi = np.quantile(vals[:, 0], [0.35, 0.65])
    q_rm_lo, q_rm_hi = np.quantile(vals[:, 1], [0.35, 0.65])
    c0 = np.array([q_sr_lo, q_rm_lo])
    c1 = np.array([q_sr_hi, q_rm_hi])
    for _ in range(8):
        d0 = np.linalg.norm(vals - c0, axis=1)
        d1 = np.linalg.norm(vals - c1, axis=1)
        lab = (d1 < d0).astype(int)
        if np.any(lab == 0):
            c0 = vals[lab == 0].mean(axis=0)
        if np.any(lab == 1):
            c1 = vals[lab == 1].mean(axis=0)
    return lab


def _assign_light_speaker_labels(segments: List["ProcessedSegment"]) -> None:
    # Atribui meta['light_speaker'] ∈ {'narrator','other'} via k=2 em (speech_rate, rms).
    if not segments:
        return
    feats = []
    for s in segments:
        f = getattr(s, "features", {}) or {}
        sr = float(f.get("speech_rate") or 0.0)
        rm = float(f.get("rms") or 0.0)
        feats.append([sr, rm])
    X = np.array(feats, dtype=float)
    labs = _k2_cluster(X)
    # escolhe 'narrator' como cluster com maior média de (speech_rate + rms)
    m0 = X[labs == 0].mean(axis=0) if np.any(labs == 0) else np.array([0.0, 0.0])
    m1 = X[labs == 1].mean(axis=0) if np.any(labs == 1) else np.array([0.0, 0.0])
    narrator_label = 1 if (m1[0] + m1[1]) >= (m0[0] + m0[1]) else 0
    for s, lb in zip(segments, labs):
        m = _ensure_meta(s)
        m["light_speaker"] = "narrator" if lb == narrator_label else "other"


def _narrator_share_for_window(segments: List["ProcessedSegment"], start: float, end: float) -> float:
    dur_narr = 0.0
    dur_tot = 0.0
    for s in segments:
        st = float(getattr(s, "start_time", 0.0) or 0.0)
        et = float(getattr(s, "end_time", st) or st)
        ov = max(0.0, min(end, et) - max(start, st))
        if ov <= 0:
            continue
        dur_tot += ov
        role = (getattr(s, "meta", {}) or {}).get("light_speaker")
        if role == "narrator":
            dur_narr += ov
    return 0.0 if dur_tot <= 0 else float(dur_narr / dur_tot)


def _leader_voice_ratio(segments: List["ProcessedSegment"], start: float, end: float) -> float:
    # Proxy de dominância: fração de frames rms no top 20% dentro da janela.
    rms_list = []
    for s in segments:
        st = float(getattr(s, "start_time", 0.0) or 0.0)
        et = float(getattr(s, "end_time", st) or st)
        if et < start or st > end:
            continue
        f = getattr(s, "features", {}) or {}
        rv = f.get("rms")
        if rv is not None:
            rms_list.append(float(rv))
    if not rms_list:
        return 0.0
    arr = np.array(rms_list, dtype=float)
    thr = np.quantile(arr, 0.80)
    return float((arr >= thr).mean())


__all__ = [
    "_assign_light_speaker_labels",
    "_narrator_share_for_window",
    "_leader_voice_ratio",
]
