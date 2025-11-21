"""API Flask para disparar o pipeline de avaliação."""

import os
import platform

# Mesmos ajustes de ambiente do main.py para evitar contenção em libs nativas.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    from config import settings as _cfg

    if getattr(_cfg, "prefer_gpu", False) and platform.system() == "Darwin":
        os.environ.setdefault("CT2_USE_MPS", "1")
except Exception:  # pragma: no cover - ambiente pode não ter config completo
    pass

import json
import logging
import multiprocessing as mp
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, jsonify, request    # type: ignore

from stages import pipeline_core
from maintenance import purge_processed, purge_temp_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

JobDict = Dict[str, Any]
_jobs: Dict[str, JobDict] = {}
_jobs_lock = threading.Lock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_spawn_start_method() -> None:
    start_method = mp.get_start_method(allow_none=True)
    if start_method != "spawn":
        mp.set_start_method("spawn", force=True)


def _collect_score_payload(stream_id: str) -> Dict[str, Any]:
    from config import PROCESSED_DIR
    score_file = PROCESSED_DIR / stream_id / "score_result.json"
    payload: Dict[str, Any] = {"score_path": str(score_file)}
    if score_file.exists():
        try:
            with open(score_file, "r", encoding="utf-8") as fh:
                payload["score"] = json.load(fh)
        except Exception as exc:  # pragma: no cover - I/O defensivo
            logger.warning("Falha ao ler %s: %s", score_file, exc)
            payload["score"] = None
    else:
        payload["score"] = None
    return payload


def _snapshot(job_id: str) -> JobDict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return dict(job)


def _snapshot_all() -> List[JobDict]:
    with _jobs_lock:
        return [dict(job) for job in _jobs.values()]


def _update_job(job_id: str, **updates: Any) -> JobDict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        job.update(updates)
        return dict(job)


def _run_pipeline_job(job_id: str, url: str, stream_id: str, gemini_api_key: Optional[str]) -> None:
    _ensure_spawn_start_method()
    from config import TEMP_DIR
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    _update_job(job_id, status="running", started_at=_utcnow())

    try:
        pipeline_core.main_parallel(
            url=url,
            stream_id=stream_id,
            gemini_api_key=gemini_api_key,
        )
        payload = _collect_score_payload(stream_id)
        _update_job(job_id, status="finished", finished_at=_utcnow(), **payload)
    except Exception as exc:  # pragma: no cover - tratativa defensiva
        logger.exception("Job %s falhou", job_id)
        _update_job(job_id, status="failed", finished_at=_utcnow(), error=str(exc))


@app.get("/health")
def health_check():
    """Retorna status simples para probes."""
    return jsonify({"status": "ok", "jobs": len(_jobs)})


@app.post("/jobs")
def create_job():
    """
    Cria um novo job de pipeline. Exemplo de payload:
    {
        "url": "https://youtube.com/....",
        "stream_id": "narrador_xpto",
        "gemini_api_key": "opcional"
    }
    """
    data = request.get_json(silent=True) or {}
    url = data.get("url") or pipeline_core.DEFAULT_URL
    stream_id = data.get("stream_id") or pipeline_core.DEFAULT_STREAM_ID
    gemini_api_key = data.get("gemini_api_key")

    if not url:
        abort(400, description="Campo 'url' é obrigatório")
    if not stream_id:
        abort(400, description="Campo 'stream_id' é obrigatório")

    job_id = str(uuid.uuid4())
    job: JobDict = {
        "id": job_id,
        "status": "pending",
        "created_at": _utcnow(),
        "url": url,
        "stream_id": stream_id,
        "has_custom_gemini_api_key": bool(gemini_api_key),
    }

    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, url, stream_id, gemini_api_key),
        name=f"pipeline-{job_id}",
        daemon=True,
    )
    thread.start()

    return jsonify(job), 202


@app.get("/jobs")
def list_jobs():
    """Lista todos os jobs conhecidos na instância."""
    return jsonify(_snapshot_all())


@app.get("/jobs/<job_id>")
def get_job(job_id: str):
    """Retorna os metadados de um job específico."""
    try:
        job = _snapshot(job_id)
    except KeyError:
        abort(404, description="Job não encontrado")
    return jsonify(job)


@app.get("/jobs/<job_id>/score")
def get_job_score(job_id: str):
    """Retorna o score final (quando disponível)."""
    try:
        job = _snapshot(job_id)
    except KeyError:
        abort(404, description="Job não encontrado")

    score = job.get("score")
    if score is None:
        abort(404, description="Score ainda não disponível para este job")
    return jsonify(score)


@app.post("/maintenance/flush")
def flush_processed():
    """
    Remove diretórios antigos em processed/ (e opcionalmente arquivos/dirs de temp/).

    Payload opcional:
    {
        "max_age_days": 7,
        "keep_stream_ids": ["narrador_demo"],
        "processed_root": "processed",
        "apply": false,
        "measure_size": false,
        "clean_temp": false,
        "temp_root": "temp",
        "temp_max_age_days": 1,
        "temp_apply": false  # opcional; se ausente usa o mesmo valor de "apply"
    }
    """
    data = request.get_json(silent=True) or {}
    max_age_days = int(data.get("max_age_days", 7))
    keep_stream_ids = data.get("keep_stream_ids") or []
    if isinstance(keep_stream_ids, str):
        keep_stream_ids = [keep_stream_ids]
    processed_root = data.get("processed_root", "processed")
    apply_changes = bool(data.get("apply", False))
    measure_size = bool(data.get("measure_size", False))

    clean_temp = bool(data.get("clean_temp", False))
    temp_root = data.get("temp_root", "temp")
    temp_max_age_days = int(data.get("temp_max_age_days", 1))
    temp_apply_flag = data.get("temp_apply")
    apply_temp = apply_changes if temp_apply_flag is None else bool(temp_apply_flag)

    processed_summary = purge_processed(
        max_age_days=max_age_days,
        processed_root=processed_root,
        keep_stream_ids=keep_stream_ids,
        dry_run=not apply_changes,
        measure_size=measure_size,
    )
    result = {"processed": processed_summary}

    if clean_temp:
        temp_summary = purge_temp_dir(
            temp_root=temp_root,
            max_age_days=temp_max_age_days,
            dry_run=not apply_temp,
        )
        result["temp"] = temp_summary

    status_code = 200
    if any(summary.get("errors") for summary in result.values()):
        status_code = 207  # Multi-Status para indicar parcial
    return jsonify(result), status_code


def create_app():
    """Factory compatível com `flask run`."""
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
