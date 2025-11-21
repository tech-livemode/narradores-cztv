"""Text scoring logic combining heuristics and semantic embeddings."""

import logging
import re
from typing import Dict, List
import nltk # type: ignore
import numpy as np  # type: ignore
from nltk.tokenize import sent_tokenize # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore

from model.processedsegment import ProcessedSegment
from model.scoreresult import TextScore

from .config import SEMANTIC_CLUSTERS
from .utils import _normalize_feature

logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

_semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
try:
    _semantic_model.encode(["warmup"], convert_to_tensor=True)
except Exception as warm_exc:  # pragma: no cover
    logger.warning(f"Falha ao aquecer modelo semântico: {warm_exc}")


def _semantic_similarity_score(text: str, category: str) -> float:
    """
    Calcula a similaridade média por sentença com o cluster semântico.
    """
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return 5.0

        MAX_SENTENCES = 200
        if len(sentences) > MAX_SENTENCES:
            step = len(sentences) // MAX_SENTENCES
            sentences = [sentences[i] for i in range(0, len(sentences), step)][:MAX_SENTENCES]

        text_embs = _semantic_model.encode(sentences, convert_to_tensor=True)
        cluster_embs = _semantic_model.encode(SEMANTIC_CLUSTERS[category], convert_to_tensor=True)

        sims = util.cos_sim(text_embs, cluster_embs)
        mean_sim = sims.mean().item()

        if len(sentences) < 3:
            softened = 5.0 + (mean_sim - 0.5) * 5
            return float(np.clip(softened, 0, 10))

        mapped = (np.tanh((mean_sim - 0.3) * 3) + 1) * 5
        return float(np.clip(mapped, 0, 10))

    except Exception as e:
        logger.warning(f"Erro na similaridade semântica ({category}): {e}")
        return 5.0


def _score_appropriate_fun(text: str) -> float:
    """Avalia equilíbrio entre seriedade e leveza usando grupos semânticos."""
    try:
        text_lower = text.lower()
        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return 0.0

        fun_groups = {
            'enthusiasm': re.compile(r'(vamos|bora|anim|entusiasm)', re.IGNORECASE),
            'positivity': ['alegria', 'festa', 'celebração', 'feliz'],
            'engagement': ['olha', 'veja', 'repare', 'olhem', 'vejam'],
        }

        comedy_patterns = re.compile(
            r'(piada|engraçado|com[eé]dia|palhaç|zueira|zoeira|meme|hil[aá]rio)',
            re.IGNORECASE
        )

        profanity_patterns = re.compile(
            r'(porra|caralho|merda|cacete|pqp|c\*?aralh|p\*?rra|m\*?rda)',
            re.IGNORECASE
        )
        interjection_patterns = re.compile(
            r'(gol+|que lance|incr[ií]vel|olha|vai|agora|pra fora|defendeu|uu+|ah+|uau)',
            re.IGNORECASE
        )

        groups_present = 0
        for _, patterns in fun_groups.items():
            if isinstance(patterns, re.Pattern):
                if patterns.search(text):
                    groups_present += 1
            else:
                if any(word in text_lower for word in patterns):
                    groups_present += 1

        group_score = (groups_present / len(fun_groups)) * 10

        raw_comedy = len(comedy_patterns.findall(text))
        profanity_hits = len(profanity_patterns.findall(text))
        comedy_matches = max(0, raw_comedy - profanity_hits)
        comedy_matches = min(comedy_matches, 10)

        allowance = 5
        effective_comedy = max(0, comedy_matches - allowance)

        comedy_penalty = 3.0 * (1 - np.exp(-0.45 * effective_comedy))

        raw_score = max(0, group_score - comedy_penalty)
        interj = len(interjection_patterns.findall(text))
        if interj >= 6:
            raw_score = min(10, raw_score + 1.0)
        elif interj >= 3:
            raw_score = min(10, raw_score + 0.5)
        smoothed = 10 / (1 + np.exp(-0.6 * (raw_score - 4)))

        normalized = _normalize_feature(smoothed, "appropriate_fun")
        return float(np.clip(normalized, 0, 10))

    except Exception as e:
        logger.warning(f"Erro ao calcular diversão apropriada: {e}")
        return 5.0


def _evaluate_text(transcripts: List[Dict], segments: List[ProcessedSegment]) -> TextScore:
    """Consolida o ASR, mede emoção, storytelling, ritmo e resenha apropriada."""
    full_text = " ".join([t.get("text", "") for t in transcripts])
    all_words = [t.get("words", []) for t in transcripts]

    if not full_text.strip():
        logger.warning("Nenhum texto para avaliar")
        return TextScore(0, 0, 0, 0, 0, {})

    emotion_content = _semantic_similarity_score(full_text, "emotion")
    storytelling = _semantic_similarity_score(full_text, "storytelling")
    game_rhythm = _semantic_similarity_score(full_text, "game_rhythm")

    appropriate_fun = _score_appropriate_fun(full_text)

    total = (
        emotion_content * 0.25 +
        storytelling * 0.35 +
        game_rhythm * 0.25 +
        appropriate_fun * 0.15
    ) * 10

    words = full_text.split()
    unique_words = set(full_text.lower().split())
    lexical_diversity = len(unique_words) / len(words) if len(words) > 0 else 0

    details = {
        "total_words": len(words),
        "unique_words": len(unique_words),
        "lexical_diversity": lexical_diversity,
        "avg_words_per_segment": len(words) / len(transcripts) if transcripts else 0,
    }

    return TextScore(
        emotion_content=emotion_content,
        storytelling=storytelling,
        game_rhythm=game_rhythm,
        appropriate_fun=appropriate_fun,
        total_score=total,
        details=details,
    )


__all__ = ["_evaluate_text", "_score_appropriate_fun"]
