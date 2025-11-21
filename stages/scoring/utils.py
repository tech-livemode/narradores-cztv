"""Utility helpers shared across scoring modules."""

import numpy as np  # type: ignore
from .config import BASELINE_STATS


def _normalize_feature(value: float, feature: str) -> float:
    # Normaliza o valor de uma feature para a escala 0–10 com base no baseline."""
    mean, std = BASELINE_STATS.get(feature, (0, 1))
    z_score = (value - mean) / std
    normalized = 5 + (z_score * 2)  # converte z-score em escala 0–10 centrada
    return float(np.clip(normalized, 0, 10))


def _zscore(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    mu = float(np.mean(arr))
    sd = float(np.std(arr) + 1e-6)
    return (arr - mu) / sd


def _z(value: float, mean: float, std: float, eps: float = 1e-6) -> float:
    # Convenience z-score helper used to compute narrativity similarity."""
    return abs((float(value) - float(mean)) / (float(std) + eps))


def _to_similarity(zbar: float, span: float = 3.5) -> float:
    # Maps an aggregated z-score to [0,1], protecting spontaneous styles."""
    s = 1.0 - (zbar / float(span))
    s = float(np.clip(s, 0.0, 1.0))
    return max(0.55, s)


__all__ = ["_normalize_feature", "_zscore", "_z", "_to_similarity"]
