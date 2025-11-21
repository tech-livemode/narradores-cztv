"""CLI dispatcher for the narration scoring pipeline."""

import os
import platform

# Reduz contendas em bibliotecas nativas durante paralelismo
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Ativa backend Metal (MPS) do CTranslate2 em Apple Silicon quando prefer_gpu=True
try:
    from config import settings as _cfg

    if getattr(_cfg, "prefer_gpu", False) and platform.system() == "Darwin":
        os.environ.setdefault("CT2_USE_MPS", "1")
except Exception:  # pragma: no cover - ambiente pode não ter config
    pass

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from stages import pipeline_core
from stages.cache_io import main_from_chunks, rescored_run, run_mvp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_URL: Optional[str] = None
DEFAULT_STREAM_ID: Optional[str] = None
DEFAULT_GEMINI_KEY: Optional[str] = None

pipeline_core.DEFAULT_URL = DEFAULT_URL
pipeline_core.DEFAULT_STREAM_ID = DEFAULT_STREAM_ID
pipeline_core.DEFAULT_GEMINI_KEY = DEFAULT_GEMINI_KEY

JOB_QUEUE = [
   #{"url": "youtube_url", "stream_id": "name_identifier", "gemini_api_key": ""}
]


def _parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de avaliação de narradores")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Executa apenas o estágio de score usando caches em processed/<stream_id>",
    )
    parser.add_argument(
        "--stream-id",
        type=str,
        default=None,
        help="Stream ID para re-score (necessário com --score-only)",
    )
    parser.add_argument(
        "--mvp",
        action="store_true",
        help="Modo ultra-rápido: reavalia múltiplos stream_ids só com caches (sem capturar/ASR)",
    )
    parser.add_argument(
        "--from-chunks",
        action="store_true",
        help="Roda ASR+score a partir de processed/<stream_id>/segment_*.wav (sem capturar/preprocessar)",
    )
    return parser.parse_known_args()


def _run_score_only(stream_id: str) -> None:
    ok = rescored_run(stream_id)
    if not ok:
        raise SystemExit(f"Falha ao reprocessar {stream_id} (veja logs acima)")
    raise SystemExit(0)


def _run_mvp_mode():
    success = run_mvp(JOB_QUEUE)
    if not success:
        raise SystemExit("Nada reavaliado no modo MVP (nenhum cache encontrado).")
    raise SystemExit(0)


def _dispatch():
    args, _unknown = _parse_args()

    if args.from_chunks:
        if not args.stream_id:
            raise SystemExit("--from-chunks requer --stream-id=<id>")
        main_from_chunks(args.stream_id)
        raise SystemExit(0)

    if args.mvp:
        _run_mvp_mode()

    if args.score_only:
        if not args.stream_id:
            raise SystemExit("--score-only requer --stream-id=<id>")
        _run_score_only(args.stream_id)

    start_method = mp.get_start_method(allow_none=True)
    if start_method != "spawn":
        mp.set_start_method("spawn", force=True)

    from config import TEMP_DIR
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    jobs = JOB_QUEUE if JOB_QUEUE else [
        {
            "url": DEFAULT_URL,
            "stream_id": DEFAULT_STREAM_ID,
            "gemini_api_key": DEFAULT_GEMINI_KEY,
        }
    ]

    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        logger.info("\n==============================")
        logger.info(f"🚀 Iniciando job {idx}/{total}")
        pipeline_core.main_parallel(
            url=job.get("url") or DEFAULT_URL,
            stream_id=job.get("stream_id") or DEFAULT_STREAM_ID,
            gemini_api_key=job.get("gemini_api_key", DEFAULT_GEMINI_KEY),
        )


def main():
    _dispatch()


if __name__ == "__main__":
    main()
