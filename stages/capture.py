# Estagio 1: Captura do audio (ATUALIZADO + DASH-first + HLS paralelo)

import logging
import os
import subprocess
import uuid
import time
from datetime import datetime
from pathlib import Path

from config import settings, TEMP_DIR
from model.audiochunk import AudioChunk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def capture_audio(url: str, stream_id: str):
    temp_dir = TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4()}.pcm"

    try:
        _capture_vod(url, output_path=str(temp_path))

        if not temp_path.exists():
            logger.error(f"Arquivo de captura não encontrado: {temp_path}")
            return

        file_size = temp_path.stat().st_size
        logger.info(f"Arquivo de captura salvo: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        if file_size == 0:
            logger.warning("Arquivo de captura vazio")
            return

        chunk_seq = 0
        chunk_size = settings.sample_rate * settings.channels * (settings.bit_depth // 8) * settings.chunk_duration
        logger.info(f"Chunk cfg → {settings.sample_rate} Hz | {settings.channels} ch | {settings.bit_depth} bit | {settings.chunk_duration}s")

        with open(temp_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield AudioChunk(
                    stream_id=stream_id,
                    chunk_seq=chunk_seq,
                    audio_data=chunk,
                    sample_rate=settings.sample_rate,
                    channels=settings.channels,
                    timestamp=datetime.utcnow(),
                    metadata={"url": url}
                )
                chunk_seq += 1
                logger.debug(f"Captured chunk {chunk_seq} ({len(chunk)} bytes)")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Falha ao remover temp: {e}")

    logger.info(f"Capture completed: {chunk_seq} chunks")


# ------ Funções de uso interno ------

def _capture_vod(url: str, output_path: str = None, max_retries: int = 3) -> subprocess.Popen:
    from config import TEMP_DIR
    if output_path is None:
        output_path = str(TEMP_DIR / "output.wav")
    """
    Captura áudio do YouTube priorizando:
      1) DASH/M4A (mais rápido)
      2) HLS via download concorrente (yt-dlp -N 8) + ffmpeg de arquivo local
    Força runtime JS (EJS solver) e usa cookies (arquivo ou navegador).
    """

    # === Runtime JS para EJS (n-sig) ===
    js_runtime = (
        getattr(settings, "yt_js_runtime", None)
        or os.getenv("YT_DLP_JS_RUNTIME")
        or "/opt/homebrew/bin/node"
    )
    env = os.environ.copy()
    env["YT_DLP_JS_RUNTIME"] = js_runtime

    # === Cookies ===
    cookie_args = []
    cfg_file = getattr(settings, "yt_cookies_file", "") or os.getenv("YTDLP_COOKIES_FILE", "")
    cfg_browser = getattr(settings, "yt_cookies_browser", "") or os.getenv("YTDLP_BROWSER", "")

    cookies_file = Path(cfg_file).expanduser() if cfg_file else (Path.home() / ".config" / "yt-dlp" / "cookies.txt")
    cookies_added = False

    if cookies_file and cookies_file.exists():
        cookie_args = ["--cookies", str(cookies_file)]
        logger.info(f"🍪 Usando cookies do arquivo: {cookies_file}")
        cookies_added = True
    else:
        preferred = [cfg_browser] if cfg_browser else []
        browsers = preferred + [b for b in ["chrome", "brave", "firefox", "safari"] if b and b not in preferred]
        for browser in browsers:
            try:
                browser_path = {
                    "chrome": Path.home() / "Library/Application Support/Google/Chrome",
                    "safari": Path.home() / "Library/Safari",
                    "firefox": Path.home() / "Library/Application Support/Firefox",
                    "brave": Path.home() / "Library/Application Support/BraveSoftware/Brave-Browser",
                }.get(browser)
                if browser_path and browser_path.exists():
                    cookie_args = ["--cookies-from-browser", browser]
                    logger.info(f"🍪 Extraindo cookies do {browser.capitalize()}")
                    cookies_added = True
                    break
            except Exception as e:
                if browser == "safari" and "Operation not permitted" in str(e):
                    logger.warning("⚠️ Safari sem permissão; tentando próximo")
                    continue
                logger.warning(f"⚠️ Erro ao acessar cookies do {browser}: {e}")
        if not cookies_added:
            logger.info("ℹ️ Continuando sem cookies (pode limitar formatos)")

    # === Helpers ===
    def _extract_direct_url(clients_csv: str, prefer_m4a: bool = True) -> str:
        """
        Usa --get-url para obter uma URL direta.
        prefer_m4a=True tenta DASH/M4A primeiro; se não existir, cai para HLS.
        Retorna URL (string) ou vazia.
        """
        fmt = (
            "ba[ext=m4a]/bestaudio[ext=m4a]/"
            "bestaudio*[protocol*=m3u8]/best*[protocol*=m3u8]/"
            "bestaudio/best"
            if prefer_m4a else
            "bestaudio*[protocol*=m3u8]/best*[protocol*=m3u8]/bestaudio/best"
        )
        args = [
            "yt-dlp",
            "-q", "--no-warnings",
            "--no-check-certificates",
            "--sleep-requests", "1",
            "--extractor-retries", "5",
            "--fragment-retries", "5",
            "--retry-sleep", "3",
            "--extractor-args", f"youtube:ejs_runtime={js_runtime},player_client={clients_csv}",
            "-f", fmt,
            "--get-url", url,
        ] + cookie_args
        proc = subprocess.run(args, capture_output=True, text=True, env=env)
        out = (proc.stdout or "").strip()
        if not out:
            return ""
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        # preferir m4a se houver
        for ln in lines:
            if ".m4a" in ln:
                return ln
        # depois m3u8
        for ln in lines:
            if ".m3u8" in ln:
                return ln
        return lines[0] if lines else ""

    def _download_hls_to_file(clients_csv: str) -> Path | None:
        """
        Baixa HLS em arquivo local com concorrência (-N 8), depois converte.
        """
        from config import TEMP_DIR
        out_file = TEMP_DIR / f"{uuid.uuid4()}.m4a"
        fmt = "bestaudio*[protocol*=m3u8]/best*[protocol*=m3u8]"
        args = [
            "yt-dlp",
            "-N", "8",
            "--no-check-certificates",
            "--sleep-requests", "1",
            "--extractor-retries", "5",
            "--fragment-retries", "5",
            "--retry-sleep", "3",
            "--extractor-args", f"youtube:ejs_runtime={js_runtime},player_client={clients_csv}",
            "-f", fmt,
            "-o", str(out_file),
            url,
        ] + cookie_args
        logger.info("⬇️  Baixando HLS com concorrência (yt-dlp -N 8) para arquivo local…")
        proc = subprocess.run(args, capture_output=True, text=True, env=env)
        if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            logger.info(f"📦 HLS salvo em arquivo local: {out_file} ({out_file.stat().st_size/1_048_576:.2f} MB)")
            return out_file
        logger.warning(f"Falha no download HLS concorrente: rc={proc.returncode} | {proc.stderr.strip() if proc.stderr else ''}")
        return None

    def _ffmpeg_from_file(file_path: Path) -> subprocess.Popen:
        """Converte arquivo local (M4A/HLS baixado) para PCM."""
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", str(file_path),
            "-vn", "-sn",
            "-ar", str(settings.sample_rate),
            "-ac", str(settings.channels),
            "-acodec", "pcm_s16le",
        ]
        # FAST_MODE: corta X segundos para debug
        fast_mode = getattr(settings, "fast_mode", False)
        fast_clip = int(getattr(settings, "fast_clip_seconds", 0) or 0)
        if fast_mode and fast_clip > 0:
            ffmpeg_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-t", str(fast_clip), "-i", str(file_path),
                          "-vn", "-sn", "-ar", str(settings.sample_rate), "-ac", str(settings.channels), "-acodec", "pcm_s16le"]
        ffmpeg_cmd += ["-f", "s16le", output_path]

        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, ff_err = proc.communicate()
        if proc.returncode != 0:
            logger.error(f"FFmpeg(file) falhou: code {proc.returncode}: {(ff_err or b'').decode('utf-8','ignore')}")
        else:
            logger.info("✅ Conversão concluída a partir de arquivo local")
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        return proc

    def _try_ffmpeg_from_url(media_url: str) -> subprocess.Popen | None:
        """Converte URL direta (m4a ou m3u8). Em m3u8, é mais lento; preferir _download_hls_to_file."""
        if not media_url:
            return None
        if ".m3u8" in media_url:
            logger.info("🔁 URL é HLS (m3u8). Preferindo download concorrente para arquivo…")
            return None  # força caminho de download concorrente
        logger.info(f"🔗 URL direta (provável DASH/M4A): {media_url[:96]}…")
        return _ffmpeg_from_url(media_url, output_path, env=env)

    # === Estratégia: tentar M4A rápido; se não, HLS concorrente ===
    for attempt in range(max_retries):
        if attempt > 0:
            wait = 2 ** attempt
            logger.info(f"🔄 Retry {attempt+1}/{max_retries} aguardando {wait}s…")
            time.sleep(wait)

        # 1) Tenta WEB com M4A primeiro (DASH)
        direct = _extract_direct_url("web", prefer_m4a=True)
        if direct:
            proc = _try_ffmpeg_from_url(direct)
            if proc and proc.returncode == 0:
                logger.info("✅ Captura concluída (WEB/M4A)")
                return proc

        # 2) Se foi HLS ou não veio nada, baixa HLS com concorrência (WEB)
        hls_file = _download_hls_to_file("web")
        if hls_file:
            proc = _ffmpeg_from_file(hls_file)
            if proc and proc.returncode == 0:
                logger.info("✅ Captura concluída (WEB/HLS paralelo)")
                return proc

        # 3) WEB-like (mweb/web_embedded) — repetir lógica
        direct = _extract_direct_url("mweb,web_embedded,web", prefer_m4a=True)
        if direct:
            proc = _try_ffmpeg_from_url(direct)
            if proc and proc.returncode == 0:
                logger.info("✅ Captura concluída (WEB-like/M4A)")
                return proc

        hls_file = _download_hls_to_file("mweb,web_embedded,web")
        if hls_file:
            proc = _ffmpeg_from_file(hls_file)
            if proc and proc.returncode == 0:
                logger.info("✅ Captura concluída (WEB-like/HLS paralelo)")
                return proc

        logger.warning("⚠️ Tentativa sem sucesso (M4A ausente e HLS falhou ou muito limitado).")

    # === Último recurso: pipeline legado (pipe yt-dlp → ffmpeg) ===
    logger.info("🧯 Último recurso: pipeline legacy (pipe yt-dlp → ffmpeg) — pode ser lento em HLS")
    ytdlp_cmd = [
        "yt-dlp",
        "-q", "--no-warnings",
        "--no-check-certificates",
        "--sleep-requests", "1",
        "--extractor-retries", "5",
        "--fragment-retries", "5",
        "--retry-sleep", "3",
        "--extractor-args", f"youtube:ejs_runtime={js_runtime},player_client=web",
        "-f", "bestaudio/best",
        url,
    ] + cookie_args

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,http,https,tcp,tls,pipe,crypto",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_at_eof", "1",
        "-i", "pipe:0", "-vn",
        "-ar", str(settings.sample_rate),
        "-ac", str(settings.channels),
        "-acodec", "pcm_s16le",
        "-f", "s16le", output_path
    ]

    try:
        ytdlp_proc = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=ytdlp_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ytdlp_proc.stdout.close()

        try:
            ytdlp_proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            logger.warning("⏱️ yt-dlp excedeu o tempo limite (pipe); encerrando")
            try: ytdlp_proc.kill()
            except Exception: pass

        try:
            _, ff_err = ffmpeg_proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            logger.warning("⏱️ ffmpeg excedeu o tempo limite (pipe); encerrando")
            try: ffmpeg_proc.kill()
            except Exception: pass

        if ffmpeg_proc.returncode != 0:
            err_msg = (ff_err or b"").decode("utf-8", errors="ignore")
            logger.error(f"Pipeline legacy falhou: ffmpeg code {ffmpeg_proc.returncode}: {err_msg}")
        else:
            logger.info("✅ Captura concluída (pipeline legacy)")
        return ffmpeg_proc
    except Exception as e:
        logger.error(f"Erro no pipeline legacy: {e}")
        raise


def _ffmpeg_from_url(url: str, output_path: str = None, env: dict | None = None) -> subprocess.Popen:
    from config import TEMP_DIR
    if output_path is None:
        output_path = str(TEMP_DIR / "output.wav")
    """
    Converte URL direta (m4a OU m3u8) para PCM/WAV com FFmpeg.
    Para m3u8, adiciona flags de reconnect; FAST_MODE suporta corte.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
        "-y",
    ]
    fast_mode = getattr(settings, "fast_mode", False)
    fast_clip = int(getattr(settings, "fast_clip_seconds", 0) or 0)
    if fast_mode and fast_clip > 0:
        cmd += ["-t", str(fast_clip)]

    # Reconnects ajudam no HLS
    if ".m3u8" in url:
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_at_eof", "1"]

    cmd += [
        "-i", url,
        "-vn", "-sn",
        "-ar", str(settings.sample_rate),
        "-ac", str(settings.channels),
        "-acodec", "pcm_s16le",
        "-f", "s16le", output_path
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env or os.environ.copy())
    _, ff_err = proc.communicate()
    if proc.returncode != 0:
        logger.error(f"FFmpeg(URL) falhou: code {proc.returncode}: {(ff_err or b'').decode('utf-8','ignore')}")
    else:
        logger.info("✅ Conversão concluída a partir de URL direta")
    return proc