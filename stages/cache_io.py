"""Cache utilities for reusing intermediates and running lightweight modes."""

import json
import logging
import time
from dataclasses import asdict
from glob import glob
from pathlib import Path
from typing import List, Optional

from model.scoreaccumulator import ScoreAccumulator
from stages.asr import asr_audio, _update_speech_rate_with_transcripts

logger = logging.getLogger(__name__)


def _segments_to_minimal_dicts(segs):
    out = []
    for s in segs:
        out.append(
            {
                "stream_id": getattr(s, "stream_id", None),
                "chunk_seq": getattr(s, "chunk_seq", None),
                "start_time": float(getattr(s, "start_time", 0.0) or 0.0),
                "end_time": float(getattr(s, "end_time", 0.0) or 0.0),
                "speaker_label": getattr(s, "speaker_label", None),
                "features": getattr(s, "features", None),
                "text": getattr(s, "text", None),
            }
        )
    return out


def _load_cached_intermediates(stream_id: str, base_dir: str = None):
    """Carrega segments.json e transcripts.json do cache de processed/<stream_id>."""
    from config import PROCESSED_DIR
    if base_dir is None:
        base_dir = str(PROCESSED_DIR)
    seg_path = Path(base_dir) / stream_id / "segments.json"
    tr_path = Path(base_dir) / stream_id / "transcripts.json"
    if not seg_path.exists() or not tr_path.exists():
        raise FileNotFoundError(
            f"Cache não encontrado para '{stream_id}'. Esperado {seg_path} e {tr_path}"
        )
    with open(seg_path, "r", encoding="utf-8") as f:
        seg_dicts = json.load(f)
    with open(tr_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    return seg_dicts, transcripts


def _save_intermediates(stream_id: str, segments, transcripts, output_dir: Path):
    """Persiste segmentos mínimos e transcrições para reuso posterior."""
    out_dir = output_dir / stream_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "segments.json", "w", encoding="utf-8") as f:
        json.dump(_segments_to_minimal_dicts(segments), f, ensure_ascii=False)
    with open(out_dir / "transcripts.json", "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False)


def _list_chunk_wavs(stream_id: str, base_dir: str = None) -> List[Path]:
    """Lista segment_*_*.wav ordenados por chunk e índice."""
    from config import PROCESSED_DIR
    if base_dir is None:
        base_dir = str(PROCESSED_DIR)
    p = Path(base_dir) / stream_id
    if not p.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {p}")
    wavs = [Path(x) for x in glob(str(p / "segment_*_*.wav"))]

    def _key(pth: Path):
        name = pth.stem
        parts = name.split("_")
        try:
            ch = int(parts[1])
        except Exception:
            ch = 0
        try:
            ix = int(parts[2])
        except Exception:
            ix = 0
        return (ch, ix)

    wavs.sort(key=_key)
    return wavs


def _segments_from_wavs(stream_id: str, base_dir: str = "processed"):
    """Reconstrói objetos compatíveis com ProcessedSegment a partir dos WAVs."""
    from types import SimpleNamespace
    import wave

    segs = []
    wavs = _list_chunk_wavs(stream_id, base_dir)
    if not wavs:
        raise FileNotFoundError(
            f"Nenhum WAV segment_*_*.wav encontrado em processed/{stream_id}"
        )

    for wav_path in wavs:
        stem = wav_path.stem
        parts = stem.split("_")
        try:
            chunk_seq = int(parts[1])
        except Exception:
            chunk_seq = 0

        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = max(1, wf.getframerate())
                duration = float(frames) / float(rate)
        except Exception:
            duration = 0.0

        segs.append(
            SimpleNamespace(
                stream_id=stream_id,
                chunk_seq=chunk_seq,
                start_time=0.0,
                end_time=duration,
                speaker_label="SPEAKER_00",
                features={},
                text=None,
                local_path=str(wav_path),
            )
        )

    return segs


def main_from_chunks(stream_id: str):
    """Executa ASR + score usando processed/<id>/segment_*_*.wav como entrada."""

    start_time = time.time()
    from config import PROCESSED_DIR
    output_dir = PROCESSED_DIR
    output_dir.mkdir(exist_ok=True)

    seg_dicts = None
    transcripts = None
    try:
        seg_dicts, transcripts = _load_cached_intermediates(stream_id)
        logger.info(
            f"📦 Cache existente encontrado para {stream_id} (segments.json + transcripts.json)"
        )
    except Exception:
        logger.info(
            f"ℹ️ Nenhum cache completo encontrado para {stream_id}; reconstruindo de WAVs…"
        )

    if seg_dicts is None:
        segments = _segments_from_wavs(stream_id)
    else:
        from types import SimpleNamespace

        segments = [
            SimpleNamespace(
                stream_id=d.get("stream_id") or stream_id,
                chunk_seq=d.get("chunk_seq", 0),
                start_time=float(d.get("start_time", 0.0) or 0.0),
                end_time=float(d.get("end_time", 0.0) or 0.0),
                speaker_label=d.get("speaker_label") or "SPEAKER_00",
                features=d.get("features", {}),
                text=d.get("text"),
                local_path=None,
            )
            for d in seg_dicts
        ]
        if any(getattr(s, "local_path", None) is None for s in segments):
            wavs = _list_chunk_wavs(stream_id)
            for i, s in enumerate(segments):
                if i < len(wavs) and getattr(s, "local_path", None) is None:
                    s.local_path = str(wavs[i])

    all_transcripts = transcripts
    if all_transcripts is None:
        logger.info(f"🧠 Rodando ASR sobre {len(segments)} segmentos carregados de disco…")
        local_acc = ScoreAccumulator()
        all_transcripts = asr_audio(segments, local_acc)
        try:
            _update_speech_rate_with_transcripts(segments, all_transcripts)
        except Exception:
            pass
        try:
            _save_intermediates(stream_id, segments, all_transcripts, output_dir)
            logger.info(f"💾 Intermediários reconstruídos e salvos em processed/{stream_id}")
        except Exception as e:
            logger.warning(f"Não foi possível salvar intermediários: {e}")

    from stages.score import graduate_audio_and_text, print_score_report

    final_score = graduate_audio_and_text(
        segments=segments,
        transcripts=all_transcripts,
        score_accumulator=None,
    )
    print_score_report(final_score)

    out_dir = output_dir / stream_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "score_result_rescored.json", "w", encoding="utf-8") as f:
        json.dump(asdict(final_score), f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    logger.info(
        f"✅ from-chunks concluído em {elapsed:.2f}s → {out_dir / 'score_result_rescored.json'}"
    )


def rescored_run(stream_id: str) -> bool:
    """Reexecuta somente o score usando caches existentes para o stream informado."""

    try:
        seg_dicts, transcripts = _load_cached_intermediates(stream_id)
    except Exception as e:
        logger.error(f"❌ Cache ausente para {stream_id}: {e}")
        return False

    from types import SimpleNamespace

    segments = [
        SimpleNamespace(
            stream_id=d.get("stream_id"),
            chunk_seq=d.get("chunk_seq"),
            start_time=d.get("start_time", 0.0),
            end_time=d.get("end_time", 0.0),
            speaker_label=d.get("speaker_label"),
            features=d.get("features", {}),
            text=d.get("text"),
        )
        for d in seg_dicts
    ]

    from stages.score import graduate_audio_and_text, print_score_report

    final_score = graduate_audio_and_text(
        segments=segments,
        transcripts=transcripts,
        score_accumulator=None,
    )
    print_score_report(final_score)

    from config import PROCESSED_DIR
    out_dir = PROCESSED_DIR / stream_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "score_result_rescored.json", "w", encoding="utf-8") as f:
        json.dump(asdict(final_score), f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Score-only salvo em: {out_dir / 'score_result_rescored.json'}")
    return True


def run_mvp(job_queue: Optional[List[dict]]) -> bool:
    """Percorre a JOB_QUEUE e tenta reavaliar cada stream_id usando caches."""

    logger.info("🟦 MVP: reavaliando somente a partir de caches em processed/<stream_id>")
    pending = []
    done = []
    jobs = job_queue if job_queue else []
    for job in jobs:
        sid = job.get("stream_id") if isinstance(job, dict) else None
        if not sid:
            continue
        ok = rescored_run(sid)
        if ok:
            done.append(sid)
        else:
            pending.append(sid)
    if pending:
        logger.warning(
            "⚠️ Sem cache para: "
            + ", ".join(pending)
            + ". Rode o pipeline normal para gerar segments.json e transcripts.json."
        )
    return bool(done)


__all__ = [
    "_load_cached_intermediates",
    "_save_intermediates",
    "_list_chunk_wavs",
    "_segments_from_wavs",
    "main_from_chunks",
    "rescored_run",
    "run_mvp",
]
