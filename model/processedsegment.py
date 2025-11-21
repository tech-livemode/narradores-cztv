from dataclasses import dataclass
from .audiofeatures import AudioFeatures

@dataclass
class ProcessedSegment:
    segment_id: int
    stream_id: str
    chunk_seq: int
    start_time: float
    end_time: float
    audio_data: bytes
    features: AudioFeatures
    local_path: str
    vad_confidence: float