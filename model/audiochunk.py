from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class AudioChunk:
    stream_id: str
    chunk_seq: int
    audio_data: bytes
    sample_rate: int
    channels: int
    timestamp: datetime
    metadata: Optional[dict] = None