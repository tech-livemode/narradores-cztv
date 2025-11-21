from dataclasses import dataclass
from typing import Optional

@dataclass
class VADSegment:
    stream_id: str
    chunk_seq: int
    start_time: float
    end_time: float
    audio_data: bytes
    confidence: float
    segment_id: Optional[int] = None