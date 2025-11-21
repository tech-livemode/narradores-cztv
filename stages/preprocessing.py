# Estagio 2: Preprocessamento do audio
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np # type: ignore
import soundfile as sf # type: ignore
import librosa # type: ignore
import torch # type: ignore
import subprocess
import tempfile
import matplotlib.pyplot as plt  # type: ignore

from config import settings
from model.audiochunk import AudioChunk
from model.processedsegment import ProcessedSegment
from model.vadsegment import VADSegment

logger = logging.getLogger(__name__)


# Variáveis globais para cache do Silero VAD
_silero_model = None
_silero_utils = None

def preprocess_chunk(
        chunk: AudioChunk,
        output_dir: Path
    ) -> List[ProcessedSegment]:
    """
    Realiza o preprocessamento do audio: normaliza, remove ruído, segmenta com VAD e extrai features.    
    """
    logger.debug(f"Iniciando o preprocessamento do chunk {chunk.chunk_seq}")

    # Converte os bytes para um array numpy
    audio = np.frombuffer(chunk.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Etapa 1: Normalização do audio
    audio_normalized = _normalize_loudness(audio, settings.sample_rate)

    # Etapa 2: Estima SNR
    snr = _estimate_snr(audio_normalized, settings.sample_rate)
    logger.debug(f"Estimated SNR: {snr:.2f} dB")

    # Etapa 3: Remove ruidos caso necessario
    """"
    audio_denoised = _denoise_rnnoise(audio_normalized, settings.sample_rate)
    old_normalized = audio_normalized
    audio_normalized = _normalize_loudness(audio_denoised, settings.sample_rate)
    plot_denoise_comparison(old_normalized, audio_normalized, settings.sample_rate)
    """
    
    if settings.enable_rnnoise and snr < settings.rnnoise_snr_threshold:
        logger.debug("Aplicando a remoção de ruído com RNNoise")
        audio_denoised = _denoise_rnnoise(audio_normalized, settings.sample_rate)
    else:
        audio_denoised = audio_normalized

    # Etapa 4: Segmentacao VAD
    vad_segments = _segment_with_vad(
        audio_denoised,
        settings.sample_rate,
        chunk.stream_id,
        chunk.chunk_seq
    )
    
    logger.info(f"Achou {len(vad_segments)} segmentos de fala no chunk {chunk.chunk_seq}")
    
    # Etapa 5: Extrai as features e salva os segmentos
    enable_parallel = getattr(settings, "enable_parallel_features", False)
    max_workers = getattr(settings, "max_workers_features", 2)
    
    # Processamento paralelo de features se habilitado e há múltiplos segmentos
    if enable_parallel and len(vad_segments) > 1:
        logger.debug(f"🚀 Extraindo features de {len(vad_segments)} segmentos em paralelo...")
        processed_segments = _process_segments_parallel(
            vad_segments, chunk, output_dir, max_workers
        )
    else:
        # Processamento sequencial (fallback)
        processed_segments = []
        for i, vad_seg in enumerate(vad_segments):
            # Converte novamente para o array de audio
            seg_audio = np.frombuffer(vad_seg.audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Extrai as features
            features = _extract_features(seg_audio, settings.sample_rate)

            # Salva no disco
            seg_dir = output_dir / chunk.stream_id
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_path = seg_dir / f"segment_{chunk.chunk_seq}_{i}.wav"

            sf.write(seg_path, seg_audio, settings.sample_rate)

            processed_segments.append(ProcessedSegment(
                segment_id=f"{chunk.chunk_seq}_{i}",
                stream_id=vad_seg.stream_id,
                chunk_seq=vad_seg.chunk_seq,
                start_time=vad_seg.start_time,
                end_time=vad_seg.end_time,
                audio_data=vad_seg.audio_data,
                features=features,
                local_path=str(seg_path),
                vad_confidence=vad_seg.confidence
            ))

    return processed_segments

# ------ Funções de uso interno ------

def _process_segments_parallel(
    vad_segments: List[VADSegment],
    chunk: AudioChunk,
    output_dir: Path,
    max_workers: int
) -> List[ProcessedSegment]:
    """
    Processa múltiplos segmentos VAD em paralelo.
    Extrai features e salva arquivos de forma concorrente.
    """
    processed_segments = []
    
    # Prepara diretório
    seg_dir = output_dir / chunk.stream_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_segment(args):
        i, vad_seg = args
        try:
            # Converte para array de audio
            seg_audio = np.frombuffer(vad_seg.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Extrai features
            features = _extract_features(seg_audio, settings.sample_rate)
            
            # Salva no disco
            seg_path = seg_dir / f"segment_{chunk.chunk_seq}_{i}.wav"
            sf.write(seg_path, seg_audio, settings.sample_rate)
            
            return ProcessedSegment(
                segment_id=f"{chunk.chunk_seq}_{i}",
                stream_id=vad_seg.stream_id,
                chunk_seq=vad_seg.chunk_seq,
                start_time=vad_seg.start_time,
                end_time=vad_seg.end_time,
                audio_data=vad_seg.audio_data,
                features=features,
                local_path=str(seg_path),
                vad_confidence=vad_seg.confidence
            )
        except Exception as e:
            logger.warning(f"Erro ao processar segmento {i}: {e}")
            return None
    
    # Processa em paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_segment, (i, seg)): i 
                   for i, seg in enumerate(vad_segments)}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                processed_segments.append(result)
    
    # Ordena por segment_id para manter ordem original
    processed_segments.sort(key=lambda x: x.segment_id)
    
    return processed_segments


def _normalize_loudness(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Normaliza o áudio usando o filtro `loudnorm` do FFmpeg.
    Se o processo falhar, aplica normalização simples por pico.
    """

    input_path = Path(tempfile.mktemp(suffix=".wav"))
    output_path = Path(tempfile.mktemp(suffix=".wav"))

    try:
        sf.write(input_path, audio, sr)

        ffmpeg_cmd = [
            "ffmpeg", "-i", input_path,
            "-af", f"loudnorm=I={settings.loudnorm_target}:TP=-1.5:LRA=8",
            "-ar", str(sr), "-y", output_path
        ]
        
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        
        # Lê o audio normalizado
        normalized_audio, _ = sf.read(output_path)
        logger.debug(f"Normalização de audio feita com sucesso (I={settings.loudnorm_target})")
        return normalized_audio
    
    except subprocess.CalledProcessError as e:
        logger.warning(f"FFmpeg loudnorm falhou: {e}. Usando normalização por pico.")
    except Exception as e:
        logger.warning(f"Erro inesperado na normalização: {e}. Usando fallback.")
    finally:
        # Limpa os temp
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    # Fallback simples: normalização por pico
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    return audio

def _estimate_snr(
        audio: np.ndarray, 
        sr: int
    ) -> float:
    """"
    Estima a relação Sinal-Ruído usando spectral rolloff.
    """

    try:
        # Calcula o rolloff espectral
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)

        # Calcula a energia RMS
        rms = librosa.feature.rms(y=audio)

        signal_power = np.mean(rms ** 2)
        noise_power = np.var(audio - librosa.effects.preemphasis(audio))

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 40.0
        
        return float(np.clip(snr, 0, 50))
    
    except Exception as e:
        logger.warning(f"Estimativa SNR falhou: {e}")
        return 20.0  # Valor default

def _denoise_rnnoise(
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
    """"
    Remove ruidos usando o DeepFilterNet.
    """
    try: 
        from df import enhance, init_df # type: ignore
        import torch    # type: ignore

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Converte o áudio numpy -> tensor 2D (batch, samples)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        model, df_state, _ = init_df(post_filter=True)
        enhanced_tensor = enhance(model, df_state, audio_tensor)

        # Converte de volta para numpy
        enhanced_tensor = enhanced_tensor.squeeze(0)
        enhanced = enhanced_tensor.detach().cpu().numpy()

        logger.info("✅ DeepFilterNet (df.enhance) aplicado com sucesso")
        return enhanced
    
    except Exception as e:
        logger.warning(f"DeepFilterNet failed ({e}), falling back to spectral subtraction")

    # Fallback: spectral subtraction
    D = librosa.stft(audio)
    magnitude, phase = np.abs(D), np.angle(D)
    noise_frames = int(0.5 * sr / 512)
    noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
    magnitude_denoised = np.maximum(magnitude - 2 * noise_profile, 0.1 * magnitude)
    D_denoised = magnitude_denoised * np.exp(1j * phase)
    audio_denoised = librosa.istft(D_denoised)

    return audio_denoised

def _segment_with_vad(
    audio: np.ndarray,
    sr: int,
    stream_id: str,
    chunk_seq: int
) -> List[VADSegment]:
    """
    Segmenta o audio usando SileroVAD, com fallback baseado em energia RMS.
    Também informa o número de segmentos e tempo total de fala.
    """
    global _silero_model, _silero_utils
    try:
        # Carrega o modelo silero VAD
        if _silero_model is None or _silero_utils is None:
            _silero_model, _silero_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
        model = _silero_model
        utils = _silero_utils
        
        (get_speech_timestamps, _, read_audio, *_) = utils
        
        # Normaliza a taxa de amostragem para 16 kHz: Necessario para o Silero VAD
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
        
        audio_tensor = torch.from_numpy(audio_16k).float()
        
        # Detecta os intervalos de fala
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=settings.vad_threshold,
            min_speech_duration_ms= settings.min_speech_duration,
            min_silence_duration_ms= settings.min_silence_duration,
            max_speech_duration_s=settings.max_speech_duration
        )
        
        segments = []
        total_speech_time = 0.0
        for i, ts in enumerate(speech_timestamps):
            start_sample = ts['start']
            end_sample = ts['end']
            
            seg_audio = audio_16k[start_sample:end_sample]
            seg_audio_int16 = (seg_audio * 32768).astype(np.int16)

            seg_start = start_sample / 16000.0
            seg_end = end_sample / 16000.0
            total_speech_time += (seg_end - seg_start)
            
            segments.append(VADSegment(
                stream_id=stream_id,
                chunk_seq=chunk_seq,
                start_time=seg_start,
                end_time=seg_end,
                audio_data=seg_audio_int16.tobytes(),
                confidence=settings.vad_threshold
            ))

            logger.info(f"SileroVAD: Detectou {len(segments)} segmentos, tempo total de fala: {total_speech_time:.2f}s no chunk {chunk_seq}")
        return segments
        
    except Exception as e:
        logger.error(f"Segmentação VAD falhou: {e}")

        # Fallback simples baseado em energia RMS
        try:
            logger.warning("Falling back to simple RMS-based segmentation.")
            # Parâmetros do fallback
            frame_size = int(0.03 * sr)  # 30 ms
            hop_size = int(0.015 * sr)   # 15 ms
            min_speech_frames = int(settings.min_speech_duration * sr // hop_size)
            min_silence_frames = int(settings.min_silence_duration * sr // hop_size)
            threshold = 0.2 * np.max(librosa.feature.rms(y=audio))  # 20% do pico RMS
            rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_size)[0]
            speech_mask = rms > threshold
            # Busca segmentos de fala
            segments = []
            in_speech = False
            start_frame = 0
            total_speech_time = 0.0
            for idx, val in enumerate(speech_mask):
                if val and not in_speech:
                    in_speech = True
                    start_frame = idx
                elif not val and in_speech:
                    end_frame = idx
                    # Checa duração mínima
                    if (end_frame - start_frame) >= min_speech_frames:
                        start_sample = start_frame * hop_size
                        end_sample = end_frame * hop_size
                        seg_audio = audio[start_sample:end_sample]
                        seg_audio_int16 = (seg_audio * 32768).astype(np.int16)
                        seg_start = start_sample / sr
                        seg_end = end_sample / sr
                        total_speech_time += (seg_end - seg_start)
                        segments.append(VADSegment(
                            stream_id=stream_id,
                            chunk_seq=chunk_seq,
                            start_time=seg_start,
                            end_time=seg_end,
                            audio_data=seg_audio_int16.tobytes(),
                            confidence=0.0
                        ))
                    in_speech = False
            # Caso termine em fala
            if in_speech:
                end_frame = len(speech_mask)
                if (end_frame - start_frame) >= min_speech_frames:
                    start_sample = start_frame * hop_size
                    end_sample = len(audio)
                    seg_audio = audio[start_sample:end_sample]
                    seg_audio_int16 = (seg_audio * 32768).astype(np.int16)
                    seg_start = start_sample / sr
                    seg_end = end_sample / sr
                    total_speech_time += (seg_end - seg_start)
                    segments.append(VADSegment(
                        stream_id=stream_id,
                        chunk_seq=chunk_seq,
                        start_time=seg_start,
                        end_time=seg_end,
                        audio_data=seg_audio_int16.tobytes(),
                        confidence=0.0
                    ))
            logger.info(f"RMS-fallback VAD: Detectou {len(segments)} segmentos, tempo total de fala: {total_speech_time:.2f}s no chunk {chunk_seq}")
            return segments
        except Exception as e2:
            logger.error(f"Fallback RMS-VAD falhou: {e2}")
            return []

def _extract_features(
        audio: np.ndarray,
        sr: int
    ) -> Optional[dict]:
    """"
    Extrai features do áudio para análise posterior.
    Features extraídas:
        - RMS (Root Mean Square): energia do sinal.
        - Pitch (F0): frequência fundamental estimada.
        - Formantes: frequências formantes principais.
        - Taxa de fala (speech rate): palavras por segundo estimadas.
        - SNR (Signal-to-Noise Ratio): relação sinal-ruído estimada.
        Retorna um dicionário com as features, mesmo que algumas falhem.
    """
    features = {}
    try:
        # RMS
        rms = np.mean(librosa.feature.rms(y=audio))
        features['rms'] = float(rms)
    except Exception as e:
        logger.warning(f"Falha ao extrair RMS: {e}")
        features['rms'] = None
    
    try:
        # Pitch (F0) usando librosa.pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        pitch = np.nanmean(f0)
        features['pitch'] = float(pitch) if not np.isnan(pitch) else None
    except Exception as e:
        logger.warning(f"Falha ao extrair Pitch: {e}")
        features['pitch'] = None
    
    # Formantes: desabilitado por padrão (lento e não usado no score)
    if settings.extract_formants:
        try:
            # Formantes: estimativa simples via LPC (Linear Predictive Coding)
            def lpc_coefficients(signal, order=12):
                import scipy.linalg # type: ignore
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                R = autocorr[:order+1]
                R_matrix = scipy.linalg.toeplitz(R[:-1])
                r = R[1:]
                a = np.linalg.solve(R_matrix, r)
                return np.concatenate(([1], -a))
                    
            a = lpc_coefficients(audio, order=12)    
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            angz = np.arctan2(np.imag(roots), np.real(roots))
            formants = sorted(angz * (sr / (2 * np.pi)))
            features['formants'] = [float(f) for f in formants[:4]]  # primeiros 4 formantes
        except Exception as e:
            logger.warning(f"Falha ao extrair formantes: {e}")
            features['formants'] = None
    else:
        features['formants'] = None

    try:
        # Taxa de fala (speech rate)
        speech_rate = _estimate_speech_rate(audio, sr)
        features['speech_rate'] = float(speech_rate)
    except Exception as e:
        logger.warning(f"Falha ao extrair taxa de fala: {e}")
        features['speech_rate'] = None
    
    try:
        # SNR
        snr = _estimate_snr(audio, sr)
        features['snr'] = float(snr)
    except Exception as e:
        logger.warning(f"Falha ao extrair SNR: {e}")
        features['snr'] = None
        
    return features


def _estimate_speech_rate(
        audio: np.ndarray, 
        sr: int
    ) -> float:
    """
    Estima a taxa de fala (palavras por segundo) baseada em detecção simples de pausas.
    """
    try:
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = 0.1 * np.max(energy)
        speech_frames = energy > threshold

        # Contar transições de silêncio para fala como palavras aproximadas
        transitions = np.diff(speech_frames.astype(int))
        word_count = np.sum(transitions == 1)
        duration_sec = len(audio) / sr
        
        if duration_sec > 0:
            rate = word_count / duration_sec
        else:
            rate = 0.0

        return rate
    except Exception as e:
        logger.warning(f"Falha ao estimar taxa de fala: {e}")
        return 0.0

# --------- Função de plotagem ---------
def plot_denoise_comparison(
    original_audio: np.ndarray,
    denoised_audio: np.ndarray,
    sr: int
) -> None:
    """
    Exibe um gráfico comparativo entre o áudio original e o áudio limpo.
    """
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt # type: ignore

    # Garante que ambos os sinais tenham o mesmo comprimento para comparação visual
    min_len = min(len(original_audio), len(denoised_audio))
    original_audio = original_audio[:min_len]
    denoised_audio = denoised_audio[:min_len]
    time = np.arange(min_len) / sr

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(time, original_audio, color='tab:blue', label='Original')
    axs[0].set_title('Áudio Original')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, which='both', linestyle=':', alpha=0.5)

    axs[1].plot(time, denoised_audio, color='tab:green', label='Denoised')
    axs[1].set_title('Áudio Após Remoção de Ruído')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, which='both', linestyle=':', alpha=0.5)

    fig.suptitle('Comparação: Áudio Original vs Áudio Limpo', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()