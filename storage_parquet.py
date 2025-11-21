from __future__ import annotations
import os, json, hashlib, tempfile, datetime
from typing import Optional, Dict, Any
import polars as pl # type: ignore
import portalocker  # type: ignore
from slugify import slugify # type: ignore
from config import DATA_LAKE_DIR

DATA_ROOT = str(DATA_LAKE_DIR)
ANALYSES_DIR = os.path.join(DATA_ROOT, "analyses")
os.makedirs(ANALYSES_DIR, exist_ok=True)

def _now_utc_iso_filename() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

def make_analysis_key(
    source_platform: str,
    source_external_id: str,
    pipeline_version: str,
    stream_id: Optional[str],
) -> str:
    """
    chave determinística: evita duplicar a mesma análise (mesmo run/stream config).
    ajuste os campos se quiser que 'params' também entrem na idempotência.
    """
    raw = f"{source_platform}|{source_external_id}|{pipeline_version}|{stream_id or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def write_analysis_parquet(
    *,
    source_platform: str,
    source_external_id: str,
    source_title: Optional[str],
    run_id: str,
    pipeline_version: str,
    asr_model: Optional[str],
    scoring_model: Optional[str],
    payload: Dict[str, Any],
    narrator_name: Optional[str] = None,
    narrator_id: Optional[str] = None,
) -> str:
    """
    Cria (ou sobrescreve) 1 arquivo Parquet por análise.
    Retorna o path absoluto gravado.
    """
    assert "final_score" in payload, "payload precisa ter final_score"
    stream = slugify(payload.get("stream_id") or "unknown")

    now = datetime.datetime.utcnow()
    year = f"year={now.year:04d}"
    month = f"month={now.month:02d}"
    stream_part = f"stream={stream}"
    out_dir = os.path.join(ANALYSES_DIR, year, month, stream_part)
    os.makedirs(out_dir, exist_ok=True)

    key = make_analysis_key(source_platform, source_external_id, pipeline_version, stream)

    filename = f"{_now_utc_iso_filename()}__{key}.parquet"
    out_path = os.path.join(out_dir, filename)

    record = {
        "key": key,
        "ts_utc": now.isoformat(),
        "year": now.year,
        "month": now.month,
        "stream": stream,
        "narrator_id": narrator_id,
        "narrator_name": narrator_name,
        "source_platform": source_platform,
        "source_external_id": source_external_id,
        "source_title": source_title,
        "run_id": run_id,
        "pipeline_version": pipeline_version,
        "asr_model": asr_model,
        "scoring_model": scoring_model,

        # campos resumidos para queries comuns
        "final_score": float(payload["final_score"]),
        "classification": payload.get("classification"),
        "total_segments": payload.get("total_segments"),
        "total_duration": float(payload.get("total_duration") or 0.0),

        # alguns detalhes úteis
        "audio_total": float(payload.get("audio_score", {}).get("total_score") or 0.0),
        "text_total": float(payload.get("text_score", {}).get("total_score") or 0.0),

        # o json bruto completo:
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }

    df = pl.from_dicts([record])

    # lock por diretório para evitar escrita concorrente confusa
    lock_path = os.path.join(out_dir, ".lock")
    with portalocker.Lock(lock_path, timeout=10):
        # se já existe um arquivo com a mesma "key", removemos (idempotência por chave)
        # varre só aquele stream/partição de mês (rápido)
        for f in os.listdir(out_dir):
            if f.endswith(".parquet") and f.split("__")[-1].startswith(key):
                try:
                    os.remove(os.path.join(out_dir, f))
                except FileNotFoundError:
                    pass

        tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".parquet.tmp")
        os.close(tmp_fd)
        df.write_parquet(tmp_path)
        os.replace(tmp_path, out_path)

    return os.path.abspath(out_path)


if __name__ == "__main__":
    payload = {
        "stream_id": "teste_stream",
        "audio_score": {
            "vocal_dynamics": 9.2,
            "speech_pacing": 8.8,
            "emotion_audio": 7.5,
            "total_score": 83.1,
            "details": {
                "avg_pitch": 212.4,
                "avg_rms": 0.142,
                "avg_speech_rate": 2.64,
                "avg_snr": 0.23,
                "emotion_corr_raw": 0.19,
                "segments_used_for_emotion": 510
            }
        },
        "text_score": {
            "emotion_content": 8.5,
            "storytelling": 8.9,
            "game_rhythm": 8.3,
            "appropriate_fun": 8.4,
            "total_score": 85.2,
            "details": {
                "source": "llm",
                "justifications": {
                    "emotion": "Demonstra emoção equilibrada e contagiante durante os momentos de destaque do jogo.",
                    "storytelling": "Constrói uma narrativa fluida e coerente, mantendo o público engajado.",
                    "game_pace": "Acompanha bem o ritmo da partida, com variações adequadas de entonação.",
                    "appropriate_commentary": "Comentários informativos e divertidos, sem exageros."
                }
            }
        },
        "final_score": 84.15,
        "classification": "Excelente",
        "strengths": [
            "Boa modulação vocal e ritmo de fala natural.",
            "Narrativa fluida e coerente, prende o ouvinte."
        ],
        "improvements": [
            "Trabalhar emoção em lances decisivos para gerar maior impacto.",
            "Usar pausas curtas antes de momentos de clímax para criar tensão positiva."
        ],
        "total_segments": 510,
        "total_duration": 7200.0
    }

    out = write_analysis_parquet(
        source_platform="youtube",
        source_external_id="abcd1234",
        source_title="Final do Campeonato",
        run_id="c1b2-...-uuid",
        pipeline_version="v1.2.0",
        asr_model="whisper-large-v3",
        scoring_model="caze-scorer-v0",
        payload=payload,
        narrator_name="teste",
        narrator_id=None,
    )
    print("gravado:", out)