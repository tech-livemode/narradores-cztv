import logging
from typing import List, Dict, Optional
import os
import platform

import numpy as np  # type: ignore

from faster_whisper import WhisperModel  # type: ignore


logger = logging.getLogger(__name__)

_MODEL: Optional[WhisperModel] = None
_BEAM_SIZE: int = 1
_FORCE_LANG: bool = False
_WHISPER_LANG: Optional[str] = None
_USE_VAD: bool = False


def init_asr_worker(settings: Dict):
    """
    Inicializa o modelo Whisper no processo worker.
    Espera um dicionário com chaves: whisper_model, whisper_cpu_threads, whisper_num_workers,
    whisper_compute_type, whisper_beam_size, force_language_pt, whisper_language, use_vad_filter.
    """
    global _MODEL, _BEAM_SIZE, _FORCE_LANG, _WHISPER_LANG, _USE_VAD

    prefer_gpu = bool(settings.get("prefer_gpu", False))
    # 'auto' deixa o CTranslate2 escolher CUDA/MPS quando disponíveis
    device = "auto" if prefer_gpu else "cpu"
    compute_type = settings.get("whisper_compute_type", "int8_float32")

    # Sinaliza MPS em macOS se preferir GPU
    if prefer_gpu and platform.system() == "Darwin":
        os.environ.setdefault("CT2_USE_MPS", "1")
    cpu_threads = int(settings.get("whisper_cpu_threads", 2))
    num_workers = int(settings.get("whisper_num_workers", 1))
    model_name = settings.get("whisper_model", "medium")

    _BEAM_SIZE = int(settings.get("whisper_beam_size", 1))
    _FORCE_LANG = bool(settings.get("force_language_pt", False))
    _WHISPER_LANG = settings.get("whisper_language")
    _USE_VAD = bool(settings.get("use_vad_filter", False))

    logger.info(
        "[ASR Worker] Inicializando Whisper: %s (device=%s, compute=%s), cpu_threads=%s, workers=%s",
        model_name,
        device,
        compute_type,
        cpu_threads,
        num_workers,
    )

    # Tenta inicializar com compute_type preferido e cai para alternativas se não suportado
    fallbacks_gpu = [compute_type, "int8_float16", "float32", "int8_float32", "int8"]
    fallbacks_cpu = [compute_type, "int8_float32", "float32", "int8"]
    candidates = fallbacks_gpu if prefer_gpu else fallbacks_cpu
    last_err: Optional[Exception] = None
    for ctype in candidates:
        try:
            _model = WhisperModel(
                model_name,
                device=device,
                compute_type=ctype,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                download_root="./models",
            )
            _MODEL = _model
            logger.info("[ASR Worker] Whisper pronto (compute_type=%s)", ctype)
            break
        except Exception as e:
            logger.warning("[ASR Worker] Falha com compute_type=%s: %s", ctype, e)
            last_err = e
            _MODEL = None
            continue
    if _MODEL is None:
        raise last_err if last_err else RuntimeError("Falha ao inicializar WhisperModel")


def _transcribe_path(path: str) -> Dict:
    assert _MODEL is not None, "Modelo Whisper não inicializado no worker"
    segments_iter, info = _MODEL.transcribe(
        path,
        language=_WHISPER_LANG if _FORCE_LANG else None,
        beam_size=_BEAM_SIZE,
        vad_filter=_USE_VAD,
        word_timestamps=False,
        condition_on_previous_text=False,
    )

    text_parts = []
    avg_logprob_sum = 0.0
    n = 0
    for s in segments_iter:
        if s.text:
            text_parts.append(s.text.strip())
        if hasattr(s, "avg_logprob") and s.avg_logprob is not None:
            avg_logprob_sum += float(s.avg_logprob)
            n += 1

    full_text = " ".join(text_parts).strip()
    avg_logprob = (avg_logprob_sum / n) if n > 0 else 0.0
    lang_code = getattr(info, "language", None)
    lang_prob = float(getattr(info, "language_probability", 0.0)) if hasattr(info, "language_probability") else 0.0

    return full_text, avg_logprob, lang_code, lang_prob


def transcribe_paths(payload: List[Dict]) -> List[Dict]:
    """
    Transcreve uma lista de segmentos (por caminho de arquivo). Retorna lista de transcripts.
    payload: [{segment_id, speaker, path, start_time, end_time, stream_id, chunk_seq}]
    """
    results: List[Dict] = []
    for item in payload:
        try:
            text, avg_logprob, lang_code, lang_prob = _transcribe_path(item["path"])
            results.append({
                "segment_id": item.get("segment_id"),
                "stream_id": item.get("stream_id"),
                "chunk_seq": item.get("chunk_seq"),
                "speaker": item.get("speaker", "SPEAKER_00"),
                "text": text,
                "language": lang_code,
                "avg_logprob": float(avg_logprob),
                "no_speech_prob": float(1.0 - lang_prob) if lang_prob else None,
                "words": [],
                "path": item.get("path"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
            })
        except Exception as e:
            results.append({
                "segment_id": item.get("segment_id"),
                "error": str(e),
            })
    return results
