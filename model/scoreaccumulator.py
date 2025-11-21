from collections import defaultdict
import threading
import numpy as np  # type: ignore

class ScoreAccumulator:
    def __init__(self):
        self.scores = defaultdict(list)
        # Guard da atualização: SentenceTransformer não é thread-safe
        self._lock = threading.Lock()

    def update(self, text_chunk: str):
        # Import aqui evita o ciclo no carregamento inicial
        from stages.score import _semantic_similarity_score

        with self._lock:
            for cat in ["emotion", "storytelling", "game_rhythm"]:
                score = _semantic_similarity_score(text_chunk, cat)
                self.scores[cat].append(score)

    def summarize(self):
        with self._lock:
            return {k: float(np.mean(v)) if v else 0.0 for k, v in self.scores.items()}
