# Estágio 4: ASR (Reconhecimento de fala)
import logging
from typing import List, Dict, Optional
from collections import defaultdict
import threading
import queue
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.queues import Queue as MPQueue
import numpy as np  # type: ignore
import torch  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import re
import os

from faster_whisper import WhisperModel # type: ignore

from config import settings
from model.processedsegment import ProcessedSegment

logger = logging.getLogger(__name__)

_model_pool_lock = threading.Lock()
_model_pool = None
_whisper_device = None
_whisper_compute_type = None

_progress_lock = threading.Lock()
_total_segments_seen = 0
_total_segments_completed = 0

# Controla quantos chunks podem executar ASR simultaneamente
_asr_gate = threading.Semaphore(getattr(settings, "asr_concurrency", 1))

# Subprocesso persistente para ASR
_worker_lock = threading.Lock()
_worker_proc: Optional[mp.Process] = None
_worker_task_q: Optional[MPQueue] = None
_worker_result_q: Optional[MPQueue] = None


def _ensure_model_pool() -> queue.Queue:
    """
    Garante que exista um pool de modelos Whisper prontos para uso.
    """
    global _model_pool, _whisper_device, _whisper_compute_type

    if _model_pool is None:
        with _model_pool_lock:
            if _model_pool is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                default_compute = "float16" if device == "cuda" else "int8"
                compute_type = getattr(settings, "whisper_compute_type", default_compute)
                pool_size = max(1, getattr(settings, "asr_model_pool_size", 1))
                logger.info(
                    "Carregando pool Whisper: %s (%s, %s) x%d",
                    settings.whisper_model,
                    device,
                    compute_type,
                    pool_size,
                )
                _whisper_device = device
                _whisper_compute_type = compute_type
                new_pool = queue.Queue(pool_size)
                cpu_threads = int(getattr(settings, "whisper_cpu_threads", 0))
                num_workers = int(getattr(settings, "whisper_num_workers", 1))
                for _ in range(pool_size):
                    model = WhisperModel(
                        settings.whisper_model,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        num_workers=num_workers,
                        download_root="./models"
                    )
                    new_pool.put(model)
                _model_pool = new_pool
    return _model_pool


@contextmanager
def _borrow_model():
    pool = _ensure_model_pool()
    model = pool.get()
    try:
        yield model
    finally:
        pool.put(model)


def _asr_worker_main(task_q: mp.Queue, result_q: mp.Queue, cfg: dict):
    try:
        device = cfg.get("device", "cpu")
        compute_type = cfg.get("compute_type", "int8")
        model_name = cfg.get("model", "medium")
        cpu_threads = int(cfg.get("cpu_threads", 1))
        num_workers = int(cfg.get("num_workers", 1))
        beam_size = int(cfg.get("beam_size", 1))
        force_lang = bool(cfg.get("force_lang", False))
        whisper_lang = cfg.get("whisper_lang")
        use_vad = bool(cfg.get("use_vad", False))

        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root="./models",
        )

        while True:
            payload = task_q.get()
            if payload is None:
                break

            segments = payload  # list of dicts
            results = []
            for seg in segments:
                try:
                    segments_iter, info = model.transcribe(
                        seg["path"],
                        language=whisper_lang if force_lang else None,
                        beam_size=beam_size,
                        vad_filter=use_vad,
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

                    results.append({
                        "segment_id": seg["segment_id"],
                        "stream_id": seg.get("stream_id"),
                        "chunk_seq": seg.get("chunk_seq"),
                        "speaker": seg.get("speaker"),
                        "text": full_text,
                        "language": lang_code,
                        "avg_logprob": float(avg_logprob),
                        "no_speech_prob": float(1.0 - lang_prob) if lang_prob else None,
                        "words": [],
                        "path": seg["path"],
                        "start_time": seg.get("start_time"),
                        "end_time": seg.get("end_time"),
                    })
                except Exception as e:
                    results.append({
                        "segment_id": seg.get("segment_id"),
                        "error": str(e),
                    })

            result_q.put(results)
    except Exception as e:
        try:
            result_q.put({"fatal_error": str(e)})
        except Exception:
            pass


def _ensure_asr_worker_started():
    if not getattr(settings, "use_subprocess_asr", False):
        return None
    global _worker_proc, _worker_task_q, _worker_result_q
    with _worker_lock:
        if _worker_proc is not None and _worker_proc.is_alive():
            return _worker_proc
        # Build config snapshot for worker
        device = "cuda" if torch.cuda.is_available() else "cpu"
        default_compute = "float16" if device == "cuda" else "int8"
        cfg = {
            "device": device,
            "compute_type": getattr(settings, "whisper_compute_type", default_compute),
            "model": settings.whisper_model,
            "cpu_threads": getattr(settings, "whisper_cpu_threads", 1),
            "num_workers": getattr(settings, "whisper_num_workers", 1),
            "beam_size": getattr(settings, "whisper_beam_size", 1),
            "force_lang": getattr(settings, "force_language_pt", False),
            "whisper_lang": getattr(settings, "whisper_language", None),
            "use_vad": getattr(settings, "use_vad_filter", False),
        }
        _worker_task_q = mp.Queue()
        _worker_result_q = mp.Queue()
        _worker_proc = mp.Process(target=_asr_worker_main, args=(_worker_task_q, _worker_result_q, cfg), daemon=True)
        _worker_proc.start()
        logger.info("Subprocesso de ASR iniciado (PID %s)", _worker_proc.pid)
        return _worker_proc


def _segments_to_payload(segments: List[ProcessedSegment]) -> list[dict]:
    payload = []
    min_dur = float(getattr(settings, "min_asr_segment_duration", 0.0))
    for seg in segments:
        start_t = getattr(seg, "start_time", 0.0)
        end_t = getattr(seg, "end_time", start_t)
        if min_dur > 0 and (end_t - start_t) < min_dur:
            continue
        payload.append({
            "segment_id": getattr(seg, "segment_id", None),
            "stream_id": getattr(seg, "stream_id", None),
            "chunk_seq": getattr(seg, "chunk_seq", None),
            "speaker": getattr(seg, "speaker_label", "SPEAKER_00"),
            "path": getattr(seg, "local_path", None),
            "start_time": start_t,
            "end_time": end_t,
        })
    return payload


def _log_progress(completed: int, seen: int) -> None:
    if seen <= 0:
        return
    progress = (completed / seen) * 100
    logger.info(f"ASR progresso total: {completed}/{seen} ({progress:.1f}%)")


def _register_new_segments(count: int) -> None:
    if count <= 0:
        return
    global _total_segments_seen
    with _progress_lock:
        _total_segments_seen += count
        completed = _total_segments_completed
        seen = _total_segments_seen
    _log_progress(completed, seen)


def _mark_segment_completed() -> None:
    global _total_segments_completed
    with _progress_lock:
        _total_segments_completed += 1
        completed = _total_segments_completed
        seen = _total_segments_seen
    _log_progress(completed, seen)


def _update_speech_rate_with_transcripts(
    segments: List[ProcessedSegment],
    transcripts: List[Dict]
) -> None:
    """
    Ajusta a taxa de fala dos segmentos com base nas transcrições concluídas.
    """
    if not segments or not transcripts:
        return

    seg_by_id = {str(getattr(seg, "segment_id")): seg for seg in segments}

    updated_segments = set()
    for tr in transcripts:
        seg_id = tr.get("segment_id")
        if seg_id is None:
            continue
        seg = seg_by_id.get(str(seg_id))
        if not seg:
            continue

        text = tr.get("text", "")
        words = [w for w in text.split() if w.strip()]
        if not words:
            continue

        start_time = tr.get("start_time")
        end_time = tr.get("end_time")
        if start_time is None:
            start_time = getattr(seg, "start_time", 0.0)
        if end_time is None:
            end_time = getattr(seg, "end_time", start_time)

        duration = float(end_time) - float(start_time)
        if duration <= 0:
            duration = float(getattr(seg, "end_time", 0.0)) - float(getattr(seg, "start_time", 0.0))
        if duration <= 0:
            continue

        rate = len(words) / duration

        features = getattr(seg, "features", None)
        if isinstance(features, dict):
            features["speech_rate"] = float(rate)
        elif features is not None:
            try:
                setattr(features, "speech_rate", float(rate))
            except Exception:
                features = {"speech_rate": float(rate)}
        else:
            features = {"speech_rate": float(rate)}

        seg.features = features
        updated_segments.add(str(seg_id))

    # Remove valores herdados de segmentos que não foram atualizados (por exemplo, ignorados no ASR)
    for seg in segments:
        if str(getattr(seg, "segment_id", "")) in updated_segments:
            continue
        features = getattr(seg, "features", None)
        if isinstance(features, dict):
            features.pop("speech_rate", None)
        elif features is not None:
            try:
                setattr(features, "speech_rate", None)
            except Exception:
                pass


def asr_audio(segments: List[ProcessedSegment], score_accumulator=None):
    """
    Transcreve o audio e verifica usando o outra IA (ex. Gemini)
    """
    with _asr_gate:
        # Caminho via subprocesso dedicado (estabilidade)
        if getattr(settings, "use_subprocess_asr", False):
            _ensure_asr_worker_started()
            if _worker_task_q is None or _worker_result_q is None:
                logger.error("Fila do ASR worker não inicializada.")
                transcripts = []
            else:
                payload = _segments_to_payload(segments)
                _register_new_segments(len(payload))
                _worker_task_q.put(payload)
                logger.info("ASR worker: processando %d segmentos...", len(payload))
                # Espera com timeout para detectar crash do worker
                results = None
                while results is None:
                    try:
                        results = _worker_result_q.get(timeout=1.0)
                    except queue.Empty:
                        if _worker_proc is None or not _worker_proc.is_alive():
                            logger.error("ASR worker morreu durante o processamento.")
                            transcripts = []
                            break
                if isinstance(results, dict) and results and results.get("fatal_error"):
                    logger.error("ASR worker falhou: %s", results["fatal_error"])
                    transcripts = []
                elif results is not None:
                    for _ in results:
                        _mark_segment_completed()
                    transcripts = [r for r in results if not r.get("error")]
                    _update_speech_rate_with_transcripts(segments, transcripts)
        else:
            pool = _ensure_model_pool()
            if _whisper_device and _whisper_compute_type:
                logger.debug(
                    "Pool Whisper pronto (%s, %s) - capacidade atual: %d",
                    _whisper_device,
                    _whisper_compute_type,
                    pool.maxsize if hasattr(pool, "maxsize") else -1,
                )
            transcripts = _transcribe_audio(segments)
            _update_speech_rate_with_transcripts(segments, transcripts)

        # Correção de transcrição com LLM (se habilitado)
        if getattr(settings, 'enable_transcription_correction', False) and transcripts:
            logger.info("🤖 Iniciando correção de transcrição com LLM...")
            transcripts = _correct_transcriptions_with_llm(transcripts)
            _update_speech_rate_with_transcripts(segments, transcripts)

    for t in transcripts:
        try:
            score_accumulator.update(t["text"])
        except Exception as e:
            logger.warning(f"Falha ao acumular score de {t.get('segment_id', '?')}: {e}")
    logger.info(f"🧩 Análise semântica aplicada a {len(transcripts)} segmentos pós-correção.")

    #for el in transcripts:
    #    print(f"[{el['segment_id']} - {el['speaker']}] {el['text']}")

    logger.info(f"✅ ASR concluído: {len(transcripts)} segmentos transcritos")

    if transcripts:
        grouped = defaultdict(list)
        for t in transcripts:
            grouped[str(t.get("chunk_seq", "?"))].append(t)

        for chunk_id, items in sorted(grouped.items(), key=lambda kv: kv[0]):
            logger.info(f"📦 Chunk {chunk_id}: {len(items)} transcrição(ões)")
            for t in items:
                text = t.get("text", "").strip()
                if not text:
                    text = "<vazio>"
                snippet = text if len(text) <= 140 else text[:137] + "..."
                seg_id = t.get("segment_id", "?")
                speaker = t.get("speaker", "SPEAKER_00")
                logger.info(f"   • [{seg_id}] {speaker}: {snippet}")

    return transcripts


def _transcribe_audio(segments: List[ProcessedSegment]):
    """
    Transcreve os segmentos de áudio usando o Whisper (faster-whisper)
    """
    results = []
    if not segments:
        logger.warning("Nenhum segmento recebido para transcrição.")
        return results

    # Filtra segmentos muito curtos
    def _seg_duration(s: ProcessedSegment) -> float:
        try:
            st = float(getattr(s, "start_time", 0.0) or 0.0)
            et = float(getattr(s, "end_time", 0.0) or 0.0)
            dur = et - st
            if dur <= 0:
                audio_bytes = getattr(s, "audio_data", b"")
                if audio_bytes:
                    # int16 mono
                    dur = len(audio_bytes) / 2 / float(getattr(settings, "sample_rate", 16000))
            return max(0.0, dur)
        except Exception:
            return 0.0

    min_dur = float(getattr(settings, "min_asr_segment_duration", 0.0) or 0.0)
    if min_dur > 0:
        original_len = len(segments)
        segments = [s for s in segments if _seg_duration(s) >= min_dur]
        dropped = original_len - len(segments)
        if dropped > 0:
            logger.info(f"ASR: ignorando {dropped} segmento(s) < {min_dur:.2f}s")

    # Parâmetros opcionais vindos de settings (com fallback)
    beam_size = getattr(settings, "whisper_beam_size", 1)
    force_lang = getattr(settings, "force_language_pt", False)
    whisper_lang = getattr(settings, "whisper_language", None)
    use_vad = getattr(settings, "use_vad_filter", True)
    enable_parallel = (
        getattr(settings, "enable_parallel_asr", False)
        and getattr(settings, "allow_threaded_asr", False)
    )
    max_workers = getattr(settings, "max_workers_asr", 4)

    total_segments = len(segments)
    _register_new_segments(total_segments)

    # Processamento paralelo se habilitado e há múltiplos segmentos
    if enable_parallel and len(segments) > 1:
        logger.info(f"🚀 Processando {len(segments)} segmentos em paralelo (max_workers={max_workers})...")
        
        # Função para processar um único segmento
        process_func = partial(
            _transcribe_single_segment,
            beam_size=beam_size,
            force_lang=force_lang,
            whisper_lang=whisper_lang,
            use_vad=use_vad
        )
        
        max_workers = min(max_workers, getattr(settings, "asr_model_pool_size", 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submete todas as tarefas
            future_to_seg = {executor.submit(process_func, seg): seg for seg in segments}
            
            # Coleta resultados conforme completam
            for future in as_completed(future_to_seg):
                seg = future_to_seg[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Falha ao transcrever {getattr(seg, 'segment_id', '?')}: {e}")
                finally:
                    _mark_segment_completed()
        
        logger.info(f"✅ ASR paralelo concluído: {len(results)} segmentos transcritos")
        return results
    
    # Processamento sequencial (fallback ou quando paralelo desabilitado)
    logger.info(f"Processando {len(segments)} segmentos sequencialmente...")
    for seg in segments:
        try:
            with _borrow_model() as model:
                # Usa o áudio em memória para evitar I/O de disco
                audio = np.frombuffer(getattr(seg, "audio_data", b""), dtype=np.int16).astype(np.float32) / 32768.0
                if audio.size == 0:
                    # fallback para arquivo, se necessário
                    source = getattr(seg, "local_path", None)
                else:
                    source = audio
                segments_iter, info = model.transcribe(
                    source,
                    language=whisper_lang if force_lang else None,
                    beam_size=beam_size,
                    vad_filter=use_vad,
                    word_timestamps=False,  # Desabilita timestamps de palavras para acelerar
                    condition_on_previous_text=False  # Acelera processamento
                )
                # Consolida texto e palavras
                text_parts = []
                word_timestamps = []
                avg_logprob_sum = 0.0
                n = 0

                for s in segments_iter:
                    if s.text:
                        text_parts.append(s.text.strip())
                    if hasattr(s, "avg_logprob") and s.avg_logprob is not None:
                        avg_logprob_sum += float(s.avg_logprob)
                        n += 1
                    if getattr(s, "words", None):
                        for w in s.words:
                            word_timestamps.append({
                                "word": w.word,
                                "start": float(w.start) if w.start is not None else None,
                                "end": float(w.end) if w.end is not None else None,
                                "probability": float(getattr(w, "probability", 0.0))
                            })

                full_text = " ".join(text_parts).strip()
                avg_logprob = (avg_logprob_sum / n) if n > 0 else 0.0
                lang_code = getattr(info, "language", None)
                lang_prob = float(getattr(info, "language_probability", 0.0)) if hasattr(info, "language_probability") else 0.0

                result = {
                    "segment_id": seg.segment_id,
                    "stream_id": getattr(seg, "stream_id", None),
                    "chunk_seq": getattr(seg, "chunk_seq", None),
                    "speaker": getattr(seg, "speaker_label", "SPEAKER_00"),
                    "text": full_text,
                    "language": lang_code,
                    "avg_logprob": float(avg_logprob),
                    "no_speech_prob": float(1.0 - lang_prob) if lang_prob else None,
                    "words": word_timestamps,
                    "path": seg.local_path,
                    "start_time": getattr(seg, "start_time", None),
                    "end_time": getattr(seg, "end_time", None),
                }

                logger.info(f"🗣️ {result['segment_id']} ({result['speaker']}): {result['text'][:80]}...")
                results.append(result)

        except Exception as e:
            logger.warning(f"Falha ao transcrever {getattr(seg, 'segment_id', '?')}: {e}")
        finally:
            _mark_segment_completed()

    return results


def _transcribe_single_segment(
    seg: ProcessedSegment,
    beam_size: int,
    force_lang: bool,
    whisper_lang: str,
    use_vad: bool
) -> dict:
    """
    Transcreve um único segmento de áudio.
    Função auxiliar para processamento paralelo.
    """
    try:
        with _borrow_model() as model:
            # Usa o áudio em memória para evitar I/O de disco
            audio = np.frombuffer(getattr(seg, "audio_data", b""), dtype=np.int16).astype(np.float32) / 32768.0
            if audio.size == 0:
                source = getattr(seg, "local_path", None)
            else:
                source = audio
            segments_iter, info = model.transcribe(
                source,
                language=whisper_lang if force_lang else None,
                beam_size=beam_size,
                vad_filter=use_vad,
                word_timestamps=False,
                condition_on_previous_text=False
            )

            # Consolida texto
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

            result = {
                "segment_id": seg.segment_id,
                "stream_id": getattr(seg, "stream_id", None),
                "chunk_seq": getattr(seg, "chunk_seq", None),
                "speaker": getattr(seg, "speaker_label", "SPEAKER_00"),
                "text": full_text,
                "language": lang_code,
                "avg_logprob": float(avg_logprob),
                "no_speech_prob": float(1.0 - lang_prob) if lang_prob else None,
                "words": [],
                "path": seg.local_path,
                "start_time": getattr(seg, "start_time", None),
                "end_time": getattr(seg, "end_time", None),
            }

        logger.info(f"🗣️ {result['segment_id']} ({result['speaker']}): {result['text'][:80]}...")
        return result

    except Exception as e:
        logger.warning(f"Erro ao transcrever {getattr(seg, 'segment_id', '?')}: {e}")
        return None


import google.generativeai as genai  # type: ignore

def configure_llm():
    """
    Configura a API do Gemini.
    Adicione sua API key em config.py ou variável de ambiente.
    """
    api_key = (
        getattr(settings, 'gemini_api_key_1', None)
        or getattr(settings, 'gemini_api_key', None)
        or os.getenv('GEMINI_API_KEY')
    )
    if not api_key:
        logger.warning("⚠️ GEMINI_API_KEY não configurada. LLM scorer desabilitado.")
        return None
    
    genai.configure(api_key=api_key)
    #for m in genai.list_models():
    #    print(m.name)

    model = genai.GenerativeModel("gemini-2.5-flash")
    logger.info("✅ Gemini 1.5 Flash configurado")
    return model

def _correct_transcriptions_with_llm(transcripts: List[Dict]) -> List[Dict]:
    """
    Corrige erros de transcrição usando LLM (Gemini).
    Foca em nomes de jogadores, times, campeonatos e termos técnicos de futebol.
    Agrupa transcrições em batches para economizar tokens.
    """
    if not transcripts:
        return transcripts
    
    try:
        model = configure_llm()
        if not model:
            logger.warning("⚠️ LLM não configurado, pulando correção de transcrição")
            return transcripts
        logger.info("🤖 Iniciando correção LLM das transcrições detectadas")
        
        batch_size = getattr(settings, 'correction_batch_size', 5)
        corrected_transcripts = []
        
        # Processa em batches
        for i in range(0, len(transcripts), batch_size):
            batch = transcripts[i:i + batch_size]
            
            # Prepara o texto para correção
            texts_to_correct = []
            for idx, t in enumerate(batch):
                texts_to_correct.append(f"{idx+1}. {t['text']}")
            
            batch_text = "\n".join(texts_to_correct)
            
            # Prompt especializado em futebol
            prompt = f"""Você é um especialista em futebol brasileiro e internacional. Sua tarefa é corrigir erros de transcrição automática (ASR) em narrações esportivas.

FOCO DE CORREÇÃO:
- Nomes de jogadores (ex: "ara sca eta" → "Arrascaeta", "gabi gol" → "Gabigol")
- Nomes de times (ex: "flamengo", "palmeiras", "corinthians", "são paulo")
- Nomes de campeonatos (ex: "brasileirão", "libertadores", "copa do brasil", "champions league")
- Termos técnicos (ex: "impedimento", "escanteio", "pênalti")
- Estádios (ex: "maracanã", "morumbi", "neo química arena")

REGRAS:
1. Corrija APENAS erros evidentes de transcrição
2. Mantenha o resto do texto EXATAMENTE como está
3. Não adicione ou remova palavras
4. Não corrija gramática ou estilo
5. Preserve pontuação e capitalização quando apropriado
6. Se não houver erros, mantenha o texto original

TEXTOS PARA CORREÇÃO:
{batch_text}

RETORNE APENAS UM JSON no formato:
{{
  "corrections": [
    {{"index": 1, "original": "texto original", "corrected": "texto corrigido"}},
    {{"index": 2, "original": "texto original", "corrected": "texto corrigido"}}
  ]
}}

Se um texto não precisa de correção, use o mesmo texto em "corrected"."""

            try:
                response = model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Remove markdown code blocks se existirem
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                    result_text = result_text.strip()
                    if result_text.endswith("```"):
                        result_text = result_text[:-3].strip()
                
                corrections = json.loads(result_text)
                
                # Aplica correções
                for correction in corrections.get('corrections', []):
                    idx = correction['index'] - 1  # Ajusta para 0-indexed
                    if 0 <= idx < len(batch):
                        original_text = batch[idx]['text']
                        corrected_text = correction.get('corrected', original_text)
                        
                        # Log se houve mudança
                        if original_text != corrected_text:
                            logger.info(f"✏️ Correção LLM [{batch[idx]['segment_id']}] index {correction['index']}:")
                            logger.info(f"   Antes: {original_text}")
                            logger.info(f"   Depois: {corrected_text}")
                        
                        # Atualiza o texto
                        batch[idx]['text'] = corrected_text
                        batch[idx]['original_text'] = original_text  # Guarda o original
                        batch[idx]['was_corrected'] = (original_text != corrected_text)
                
                corrected_transcripts.extend(batch)
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Erro ao parsear resposta do LLM (batch {i//batch_size + 1}): {e}")
                logger.warning(f"Resposta: {result_text[:200]}...")
                # Em caso de erro, mantém os textos originais
                corrected_transcripts.extend(batch)
            
            except Exception as e:
                logger.warning(f"⚠️ Erro na correção LLM (batch {i//batch_size + 1}): {e}")
                corrected_transcripts.extend(batch)
        
        # Conta correções realizadas
        num_corrected = sum(1 for t in corrected_transcripts if t.get('was_corrected', False))
        logger.info(f"✅ Correção LLM concluída: {num_corrected}/{len(transcripts)} segmentos corrigidos")
        
        return corrected_transcripts
        
    except ImportError:
        logger.warning("⚠️ Módulo llm_scorer não disponível, pulando correção")
        return transcripts
    except Exception as e:
        logger.error(f"❌ Erro na correção de transcrição: {e}")
        return transcripts
        
