from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioFeatures:
    rms_mean: float
    rms_std: float
    pitch_median: float
    pitch_std: float
    f1: Optional[float]
    f2: Optional[float]
    speech_rate: float
    snr: float