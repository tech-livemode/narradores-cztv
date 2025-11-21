"""Utilidades administrativas (ex.: limpeza do diretório processed)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def purge_processed(
    max_age_days: int,
    *,
    processed_root: Path | str = None,
    keep_stream_ids: Optional[Iterable[str]] = None,
    dry_run: bool = True,
    measure_size: bool = False,
) -> Dict[str, object]:
    from config import PROCESSED_DIR
    if processed_root is None:
        processed_root = PROCESSED_DIR
    """
    Remove diretórios de processed/ mais antigos do que `max_age_days`.

    Retorna um resumo com listas de removidos/preservados/erros que pode ser serializado em JSON.
    """

    root = Path(processed_root)
    keep_set = {sid for sid in (keep_stream_ids or []) if sid}
    now = datetime.now(timezone.utc)
    threshold = now - timedelta(days=max(0, max_age_days))

    summary: Dict[str, object] = {
        "dry_run": dry_run,
        "processed_root": str(root),
        "max_age_days": max_age_days,
        "threshold_utc": threshold.isoformat(),
        "removed": [],
        "skipped": [],
        "errors": [],
    }

    removed: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    if not root.exists():
        summary["message"] = "processed_root não encontrado"
        summary["removed"] = removed
        summary["skipped"] = skipped
        summary["errors"] = errors
        return summary

    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue

        stream_id = item.name
        try:
            mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
        except OSError as exc:
            errors.append({"stream_id": stream_id, "error": str(exc)})
            continue

        record = {
            "stream_id": stream_id,
            "last_modified": mtime.isoformat(),
        }
        if measure_size:
            try:
                record["size_bytes"] = _dir_size_bytes(item)
            except Exception as exc:  # pragma: no cover - cálculo opcional
                record["size_bytes"] = None
                record["size_error"] = str(exc)

        if keep_set and stream_id in keep_set:
            record["reason"] = "protected"
            skipped.append(record)
            continue

        if mtime >= threshold:
            record["reason"] = "newer_than_threshold"
            skipped.append(record)
            continue

        if dry_run:
            removed.append(record)
            continue

        try:
            shutil.rmtree(item)
            removed.append(record)
        except Exception as exc:  # pragma: no cover - tratativa defensiva
            record["error"] = str(exc)
            errors.append(record)

    summary["removed"] = removed
    summary["skipped"] = skipped
    summary["errors"] = errors
    return summary


def purge_temp_dir(
    *,
    temp_root: Path | str = None,
    max_age_days: int = 1,
    dry_run: bool = True,
) -> Dict[str, object]:
    from config import TEMP_DIR
    if temp_root is None:
        temp_root = TEMP_DIR
    """
    Remove arquivos/dirs em temp/ mais antigos do que `max_age_days`.
    """

    root = Path(temp_root)
    now = datetime.now(timezone.utc)
    threshold = now - timedelta(days=max(0, max_age_days))

    summary: Dict[str, object] = {
        "dry_run": dry_run,
        "temp_root": str(root),
        "max_age_days": max_age_days,
        "threshold_utc": threshold.isoformat(),
        "removed": [],
        "skipped": [],
        "errors": [],
    }

    removed: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    if not root.exists():
        summary["message"] = "temp_root não encontrado"
        summary["removed"] = removed
        summary["skipped"] = skipped
        summary["errors"] = errors
        return summary

    for item in root.iterdir():
        try:
            mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
        except OSError as exc:
            errors.append({"path": str(item), "error": str(exc)})
            continue

        record = {
            "path": str(item),
            "is_dir": item.is_dir(),
            "last_modified": mtime.isoformat(),
        }

        if mtime >= threshold:
            record["reason"] = "newer_than_threshold"
            skipped.append(record)
            continue

        if dry_run:
            removed.append(record)
            continue

        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed.append(record)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            record["error"] = str(exc)
            errors.append(record)

    summary["removed"] = removed
    summary["skipped"] = skipped
    summary["errors"] = errors
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpa caches antigos em processed/")
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Remove stream_ids cujo diretório não é tocado há mais dias que esse limite (default: 7).",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Stream IDs que nunca devem ser deletados (pode repetir a flag).",
    )
    parser.add_argument(
        "--processed-root",
        default="processed",
        help="Caminho base do diretório processed (default: ./processed).",
    )
    parser.add_argument(
        "--measure-size",
        action="store_true",
        help="Calcula o tamanho aproximado de cada diretório (custa mais tempo).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica de fato a limpeza. Sem essa flag, roda em modo dry-run.",
    )
    parser.add_argument(
        "--clean-temp",
        action="store_true",
        help="Também limpa arquivos/directórios antigos em temp/.",
    )
    parser.add_argument(
        "--temp-root",
        default="temp",
        help="Caminho do diretório temporário (default: ./temp).",
    )
    parser.add_argument(
        "--temp-max-age-days",
        type=int,
        default=1,
        help="Remove arquivos em temp/ não modificados há mais dias que esse limite (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    maintenance_summary: Dict[str, object] = {}
    processed_summary = purge_processed(
        max_age_days=args.max_age_days,
        processed_root=args.processed_root,
        keep_stream_ids=args.keep,
        dry_run=not args.apply,
        measure_size=args.measure_size,
    )
    maintenance_summary["processed"] = processed_summary

    if args.clean_temp:
        temp_summary = purge_temp_dir(
            temp_root=args.temp_root,
            max_age_days=args.temp_max_age_days,
            dry_run=not args.apply,
        )
        maintenance_summary["temp"] = temp_summary

    logger.info("Resumo manutenção:\n%s", json.dumps(maintenance_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
