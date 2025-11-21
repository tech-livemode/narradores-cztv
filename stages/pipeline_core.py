# Pipeline core: capture → preprocess/diarize → ASR → scoring.

import json
import logging
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from config import settings
from model.audiochunk import AudioChunk
from model.scoreaccumulator import ScoreAccumulator
from stages.asr import (
    _correct_transcriptions_with_llm,
    _update_speech_rate_with_transcripts,
    asr_audio,
)
from stages.asr_pool import init_asr_worker, transcribe_paths
from stages.capture import capture_audio
from stages.preprocessing import preprocess_chunk

from .cache_io import _save_intermediates

logger = logging.getLogger(__name__)

DEFAULT_URL: Optional[str] = None
DEFAULT_STREAM_ID: Optional[str] = None
DEFAULT_GEMINI_KEY: Optional[str] = None

score_accumulator = ScoreAccumulator()


def process_single_chunk(chunk: AudioChunk, output_dir: Path) -> List:
    """
    Pré-processa e (opcionalmente) diariza um chunk antes do ASR.
    """
    try:
        logger.info(f"🔹 Processando chunk {chunk.chunk_seq} ({len(chunk.audio_data)} bytes)")

        processed_segments = preprocess_chunk(chunk, output_dir)

        if settings.enable_diarization:
            logger.info(f"⏳ Diarizando chunk {chunk.chunk_seq}...")
            from stages.diarization import diarizate_audio  # lazy import

            diarizated_segments = diarizate_audio(processed_segments)
        else:
            for seg in processed_segments:
                seg.speaker_label = "SPEAKER_00"
                seg.embedding = None
            diarizated_segments = processed_segments

        logger.info(
            f"✅ Chunk {chunk.chunk_seq} pré-processado: {len(diarizated_segments)} segmentos prontos para ASR"
        )
        return diarizated_segments

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"❌ Erro ao processar chunk {chunk.chunk_seq}: {e}")
        import traceback

        traceback.print_exc()
        return []


def main_parallel(
    url: Optional[str] = None,
    stream_id: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
):
    """
    Executa o pipeline completo (captura → preprocessamento → ASR → score).
    """

    url = url or DEFAULT_URL
    stream_id = stream_id or DEFAULT_STREAM_ID

    start_time = time.time()

    if gemini_api_key is None:
        gemini_api_key = DEFAULT_GEMINI_KEY

    global score_accumulator
    score_accumulator = ScoreAccumulator()

    logger.info("Iniciando processamento com PRÉ-PROCESSAMENTO paralelo e ASR sequencial...")
    logger.info(f"🎬 Job atual: stream_id='{stream_id}', url='{url}'")
    if gemini_api_key:
        try:
            settings.gemini_api_key_1 = gemini_api_key
            logger.info("🔑 Gemini API key (gemini_api_key_1) atualizada para este job")
        except Exception as e:
            logger.warning(f"Falha ao atualizar gemini_api_key_1: {e}")

    try:
        from config import PROCESSED_DIR
        output_dir = PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        enable_parallel = getattr(settings, "enable_parallel_chunks", False)
        max_workers = getattr(settings, "max_workers_chunks", 1)
        asr_workers = max(1, int(getattr(settings, "asr_processes", 1)))

        all_segments = []
        all_transcripts = []

        if not enable_parallel or max_workers <= 1:
            for chunk in capture_audio(url, stream_id):
                try:
                    segments = process_single_chunk(chunk, output_dir)
                    transcripts = asr_audio(segments, score_accumulator)
                    all_segments.extend(segments)
                    all_transcripts.extend(transcripts)
                    logger.info(
                        f"✅ Chunk {chunk.chunk_seq} concluído e acumulado: {len(segments)} segmentos, {len(transcripts)} transcrições"
                    )
                    logger.info(f"Resumo parcial dos scores: {score_accumulator.summarize()}")
                except Exception as e:
                    logger.error(f"❌ Erro no chunk {chunk.chunk_seq}: {e}")
        else:
            logger.info(
                f"🚀 Pré-processamento paralelo ({max_workers}) + ASR em pool de processos ({asr_workers})"
            )
            buffer_size = max_workers * 2
            settings_dict: Dict[str, Optional[str]] = {
                "whisper_model": settings.whisper_model,
                "whisper_cpu_threads": settings.whisper_cpu_threads,
                "whisper_num_workers": settings.whisper_num_workers,
                "whisper_compute_type": settings.whisper_compute_type,
                "whisper_beam_size": settings.whisper_beam_size,
                "force_language_pt": settings.force_language_pt,
                "whisper_language": settings.whisper_language,
                "use_vad_filter": settings.use_vad_filter,
                "prefer_gpu": getattr(settings, "prefer_gpu", False),
            }
            with ThreadPoolExecutor(max_workers=max_workers) as pre_executor, ProcessPoolExecutor(
                max_workers=asr_workers, initializer=init_asr_worker, initargs=(settings_dict,)
            ) as asr_pool:
                pre_futures = {}
                asr_futures = {}
                for chunk in capture_audio(url, stream_id):
                    pf = pre_executor.submit(process_single_chunk, chunk, output_dir)
                    pre_futures[pf] = (chunk.chunk_seq, chunk.stream_id)
                    logger.info(f"📤 Chunk {chunk.chunk_seq} submetido para pré-processamento paralelo")

                    if len(pre_futures) >= buffer_size:
                        done = []
                        for pf in as_completed(list(pre_futures.keys())):
                            seq, _sid = pre_futures[pf]
                            try:
                                segments = pf.result()
                                min_dur = float(getattr(settings, "min_asr_segment_duration", 0.0) or 0.0)
                                payload = []
                                for s in segments:
                                    st = float(getattr(s, "start_time", 0.0) or 0.0)
                                    et = float(getattr(s, "end_time", 0.0) or 0.0)
                                    if min_dur > 0 and (et - st) < min_dur:
                                        continue
                                    payload.append(
                                        {
                                            "segment_id": getattr(s, "segment_id", None),
                                            "stream_id": getattr(s, "stream_id", None),
                                            "chunk_seq": getattr(s, "chunk_seq", None),
                                            "speaker": getattr(s, "speaker_label", "SPEAKER_00"),
                                            "path": getattr(s, "local_path", None),
                                            "start_time": getattr(s, "start_time", None),
                                            "end_time": getattr(s, "end_time", None),
                                        }
                                    )
                                if not payload:
                                    logger.info(
                                        f"ASR: nenhum segmento elegível após filtro de {min_dur:.2f}s (chunk {seq})"
                                    )
                                    continue
                                af = asr_pool.submit(transcribe_paths, payload)
                                asr_futures[af] = (seq, segments)
                                logger.info(
                                    f"🧠 ASR submetido (chunk {seq}) com {len(segments)} segmentos"
                                )
                            except Exception as e:
                                logger.error(f"❌ Erro no chunk {seq}: {e}")
                            done.append(pf)
                            break
                        for pf in done:
                            del pre_futures[pf]

                    if len(asr_futures) >= asr_workers * 2:
                        for af in as_completed(list(asr_futures.keys())):
                            seq, segs = asr_futures[af]
                            try:
                                transcripts = af.result()
                                try:
                                    _update_speech_rate_with_transcripts(segs, transcripts)
                                except Exception:
                                    pass
                                if getattr(settings, "enable_transcription_correction", False):
                                    try:
                                        transcripts = _correct_transcriptions_with_llm(transcripts)
                                        _update_speech_rate_with_transcripts(segs, transcripts)
                                    except Exception as e:
                                        logger.warning(f"⚠️ Falha na correção LLM do chunk {seq}: {e}")
                                all_segments.extend(segs)
                                all_transcripts.extend(transcripts)
                                for t in transcripts:
                                    try:
                                        score_accumulator.update(t.get("text", ""))
                                    except Exception:
                                        pass
                                logger.info(
                                    f"✅ Chunk {seq} ASR concluído e acumulado: {len(segs)} segmentos, {len(transcripts)} transcrições"
                                )
                                logger.info(f"Resumo parcial dos scores: {score_accumulator.summarize()}")
                            except Exception as e:
                                logger.error(f"❌ Erro no ASR do chunk {seq}: {e}")
                            del asr_futures[af]
                            break

                logger.info("⏳ Aguardando chunks finais de pré-processamento/ASR…")
                for pf in as_completed(list(pre_futures.keys())):
                    seq, _sid = pre_futures[pf]
                    try:
                        segments = pf.result()
                        min_dur = float(getattr(settings, "min_asr_segment_duration", 0.0) or 0.0)
                        payload = []
                        for s in segments:
                            st = float(getattr(s, "start_time", 0.0) or 0.0)
                            et = float(getattr(s, "end_time", 0.0) or 0.0)
                            if min_dur > 0 and (et - st) < min_dur:
                                continue
                            payload.append(
                                {
                                    "segment_id": getattr(s, "segment_id", None),
                                    "stream_id": getattr(s, "stream_id", None),
                                    "chunk_seq": getattr(s, "chunk_seq", None),
                                    "speaker": getattr(s, "speaker_label", "SPEAKER_00"),
                                    "path": getattr(s, "local_path", None),
                                    "start_time": getattr(s, "start_time", None),
                                    "end_time": getattr(s, "end_time", None),
                                }
                            )
                        if payload:
                            af = asr_pool.submit(transcribe_paths, payload)
                            asr_futures[af] = (seq, segments)
                    except Exception as e:
                        logger.error(f"❌ Erro no chunk {seq}: {e}")

                for af in as_completed(list(asr_futures.keys())):
                    seq, segs = asr_futures[af]
                    try:
                        transcripts = af.result()
                        try:
                            _update_speech_rate_with_transcripts(segs, transcripts)
                        except Exception:
                            pass
                        if getattr(settings, "enable_transcription_correction", False):
                            try:
                                transcripts = _correct_transcriptions_with_llm(transcripts)
                                _update_speech_rate_with_transcripts(segs, transcripts)
                            except Exception as e:
                                logger.warning(f"⚠️ Falha na correção LLM do chunk {seq}: {e}")
                        all_segments.extend(segs)
                        all_transcripts.extend(transcripts)
                        for t in transcripts:
                            try:
                                score_accumulator.update(t.get("text", ""))
                            except Exception:
                                pass
                        logger.info(f"✅ Chunk {seq} ASR concluído e acumulado")
                        logger.info(f"Resumo parcial dos scores: {score_accumulator.summarize()}")
                    except Exception as e:
                        logger.error(f"❌ Erro no ASR do chunk {seq}: {e}")

        if all_segments and all_transcripts:
            logger.info("\n🎯 Iniciando avaliação final do narrador…")
            logger.info(f"   Total: {len(all_segments)} segmentos, {len(all_transcripts)} transcrições")

            from stages.score import graduate_audio_and_text, print_score_report

            final_score = graduate_audio_and_text(
                segments=all_segments,
                transcripts=all_transcripts,
                score_accumulator=score_accumulator,
            )

            try:
                _save_intermediates(stream_id, all_segments, all_transcripts, output_dir)
                logger.info(f"💾 Intermediários salvos para re-score: {output_dir / stream_id}")
            except Exception as e:
                logger.warning(f"Não foi possível salvar intermediários: {e}")

            print_score_report(final_score)

            score_file = output_dir / stream_id / "score_result.json"
            score_file.parent.mkdir(parents=True, exist_ok=True)
            with open(score_file, "w", encoding="utf-8") as f:
                json.dump(asdict(final_score), f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Score salvo em: {score_file}")
        else:
            logger.warning("⚠️ Não há dados suficientes para avaliação")
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"❌ Erro no processamento paralelo: {e}")
        import traceback

        traceback.print_exc()
    elapsed_time = time.time() - start_time
    logger.info(f"⏰ Tempo total de execução: {elapsed_time:.2f}s")


__all__ = [
    "process_single_chunk",
    "main_parallel",
    "DEFAULT_URL",
    "DEFAULT_STREAM_ID",
    "DEFAULT_GEMINI_KEY",
]
