from dataclasses import dataclass
from typing import List

@dataclass
class SpeakerEmbedding:
    segment_id: str
    speaker_label: str
    embedding: List[float]