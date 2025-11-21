from dataclasses import dataclass, field
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv  # type: ignore

load_dotenv()

# Diretórios: usa /tmp no Cloud Run, ou variável de ambiente, ou padrão local
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/temp"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "/tmp/processed"))
DATA_LAKE_DIR = Path(os.getenv("DATA_LAKE_DIR", "/tmp/data_lake"))

# Criar diretórios se não existirem
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Settings:

    # Tokens
    hf_token: str = os.getenv("HF_TOKEN")

    # Parametros de audio
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_duration: int = 120  # Aumentado de 30s para 120s - reduz overhead

    # VAD parameters
    vad_threshold: float = 0.35
    min_speech_duration: float = 150    # em ms
    max_speech_duration: float = 30    # em s
    min_silence_duration: float = 400   # em ms

    # Preprocessing
    enable_rnnoise: bool = False
    loudnorm_target: float = -16.0
    rnnoise_snr_threshold: float = 15.0

    # Diarization
    enable_diarization: bool = False  # Diarização é MUITO lenta, desabilitar para testes rápidos

    # Diarization Leve
    enable_light_diarization = True
    light_diarization_k = 2
    light_diarization_narrator_share_hi = 0.60
    light_diarization_narrator_share_lo = 0.30
    light_diarization_weight_boost = 1.12   # aumenta peso da janela com narrador
    light_diarization_weight_cut = 0.90     # reduz peso da janela com comentarista 
    audio_emotion_top_energy_frac = 0.65    # filtra 65% mais energéticos por janela

    # ASR
    whisper_model: str = "medium"  # 'large' para produto final; 'medium' para velocidade (3-4x mais rápido)
    whisper_language: str = "pt"
    whisper_beam_size: int = 1  # 1 acelera bastante e é suficiente para GPU/CPU
    force_language_pt: bool = True
    whisper_cpu_threads: int = 6  # Limita threads do backend em CPU
    whisper_num_workers: int = 1  # Workers internos do decoder
    whisper_compute_type: str = "int8_float32"  # Perfil estável em CPU (usa int8 + float32)
    prefer_gpu: bool = False  # Prioriza CPU para evitar instabilidades em GPU
    use_subprocess_asr: bool = False  # Sequencial no processo principal
    min_asr_segment_duration: float = 0.5  # descarta segmentos muito curtos
    
    # Performance
    use_vad_filter: bool = False  # Já segmentamos antes; evita VAD interno do Whisper
    extract_formants: bool = False  # Formantes são lentos e não usados no score
    
    # Paralelização
    enable_parallel_asr: bool = False  # Desliga paralelismo no ASR
    allow_threaded_asr: bool = False  # True ativa inferência paralela dentro do processo (instável em CPU)
    max_workers_asr: int = 8  # Número de threads para ASR paralelo (aumentado de 4 para 8)
    asr_model_pool_size: int = 1  # Número de instâncias Whisper carregadas em paralelo
    asr_concurrency: int = 1  # Quantos chunks podem fazer ASR simultaneamente
    asr_processes: int = 1  # Mantém ASR sequencial/processo único
    enable_parallel_features: bool = False  # Desliga paralelismo na extração de features
    max_workers_features: int = 4  # Número de threads para features (aumentado de 2 para 4)
    
    # Paralelização de chunks
    enable_parallel_chunks: bool = False  # Ativa pré-processamento paralelo
    max_workers_chunks: int = 2  # Número de chunks processados em paralelo

    # yt-dlp cookies (autenticação YouTube)
    # Se 'yt_cookies_file' estiver definido e o arquivo existir, será usado (Netscape cookies.txt)
    # Caso contrário, se 'yt_cookies_browser' estiver definido, usa --cookies-from-browser <browser>
    # Valores comuns: "chrome", "brave", "firefox" ("safari" geralmente exige Full Disk Access)
    yt_cookies_file: str = ""  # Ex.: "/Users/<user>/.config/yt-dlp/cookies.txt"
    yt_cookies_browser: str = "chrome"  # Ex.: "chrome"
    capture_timeout_seconds: int = 90  # Tempo máximo por tentativa de download (yt-dlp/ffmpeg)
    
    # LLM Scoring (Gemini)
    enable_llm_scoring: bool = True  # Desabilitar reduz chamadas externas e acelera processamento
    # "AIzaSyDtB23uN6LMSBBu-3JvPVwvReMLv79IBeo"
    gemini_api_key_1: str = ""
    gemini_api_key: str = os.getenv("MAIN_GEMINI_TOKEN") # Adicione sua API key aqui ou use GEMINI_API_KEY env var
    llm_validation_only: bool = False  # Se True, usa LLM apenas para validação de conteúdo
    llm_full_scoring: bool = True  # Se True, usa LLM para scoring completo dos critérios
    
    # Correção de transcrição com LLM
    enable_transcription_correction: bool = True  # Desabilitar evita round-trips de LLM na etapa de texto
    correction_batch_size: int = 5  # Agrupa N segmentos por requisição (economiza tokens)

    # Email Configurations
    username: str = os.getenv("MAIL_SENDER_USERNAME") # email mesmo
    # se for usar um email google acesse -> https://myaccount.google.com/security
    # Busque por senhad de app -> crie um app e gere a senha, o valor será o password.
    password: str = os.getenv("MAIL_SENDER_PASSWORD")
    destinatarios: List[str] = field(default_factory=lambda: [
        ""
        ])     


settings = Settings()
