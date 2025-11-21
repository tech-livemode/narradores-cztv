# Estagio 3: 

from pathlib import Path
from typing import List
import tempfile
import logging
import warnings
import bisect
import os
from torch.nn.utils.rnn import pad_sequence  # type: ignore
try:
    import torchaudio  # type: ignore
    from torchaudio.functional import resample as ta_resample  # type: ignore
except Exception:
    torchaudio = None
    ta_resample = None

import torch # type: ignore
from pyannote.audio import Pipeline # type: ignore
from pyannote.audio.pipelines.utils.hook import ProgressHook # type: ignore
from transformers import AutoModel # type: ignore
from pyannote.audio.pipelines import SpeakerDiarization # type: ignore
from pyannote.audio import Model # type: ignore
from speechbrain.pretrained import EncoderClassifier  # type: ignore
import soundfile as sf  # type: ignore
import numpy as np  # type: ignore

from config import settings
from model.processedsegment import ProcessedSegment
from model.speakerembeding import SpeakerEmbedding


logger = logging.getLogger(__name__)

# Ignorar avisos de depreciação do torchaudio e pyannote
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio._backend.utils.info has been deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio._backend.common.AudioMetaData has been deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*function's implementation will be changed to use torchaudio.load_with_torchcodec.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*degrees of freedom is <= 0.*"
)

# TODO: Baixar python >= 3.10, Torch >= 2.9, Pyannote.audio ≥ 3.2

def diarizate_audio(
    segments: List[ProcessedSegment]
    ) -> List[ProcessedSegment]:
    # settings.gpu_device if torch.cuda.is_available() else 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Fast-path: desabilita gradiente e otimiza kernels
    torch.set_grad_enabled(False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=settings.hf_token
    )
    pipeline.instantiate({
        "clustering": {
            "method": "centroid",
            "threshold": 0.5
        }
    })
    pipeline.to(device)

    embedding_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}
    )

    # Diariza e extrai os embeddings (sem gradiente)
    with torch.no_grad():
        speaker_results = _diarizate_audio(segments, pipeline, embedding_model, device)

    # --- Interval index por chunk (reduz O(N×M) para ~O(N log M)) ---
    results_by_chunk = {}
    for r in speaker_results:
        key = r.get("chunk_dir", "")
        results_by_chunk.setdefault(key, []).append(r)
    # Ordena e pré-computa inícios por chunk
    chunk_index = {}
    for key, lst in results_by_chunk.items():
        lst.sort(key=lambda x: float(x["start"]))
        starts = [float(x["start"]) for x in lst]
        chunk_index[key] = (lst, starts)

    # Associa cada segmento VAD ao turno de diarização com maior sobreposição temporal
    updated_segments: List[ProcessedSegment] = []
    per_segment_embeddings: List[SpeakerEmbedding] = []

    for seg in segments:
        seg_start = float(getattr(seg, "start_time", 0.0))
        seg_end = float(getattr(seg, "end_time", 0.0))
        chunk_key = str(Path(seg.local_path).parent)

        best_res = None
        best_overlap = 0.0

        if chunk_key in chunk_index:
            lst, starts = chunk_index[chunk_key]
            # Posição do primeiro turno que pode começar após o início do segmento
            i = bisect.bisect_left(starts, seg_start)

            # Vasculha para a esquerda enquanto houver sobreposição (turno termina após o início do seg)
            j = i - 1
            while j >= 0 and float(lst[j]["end"]) > seg_start:
                t = lst[j]
                overlap = max(0.0, min(seg_end, float(t["end"])) - max(seg_start, float(t["start"])))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_res = t
                j -= 1

            # Vasculha para a direita enquanto o turno começar antes do fim do segmento
            k = i
            n = len(lst)
            while k < n and float(lst[k]["start"]) < seg_end:
                t = lst[k]
                overlap = max(0.0, min(seg_end, float(t["end"])) - max(seg_start, float(t["start"])))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_res = t
                k += 1
        else:
            logger.debug(f"Sem índice para chunk {chunk_key} (segmento {getattr(seg, 'segment_id', None)})")

        if best_res is not None:
            seg.speaker_label = best_res["speaker"]
            seg.embedding = best_res["embedding"]
            per_segment_embeddings.append(
                SpeakerEmbedding(
                    segment_id=seg.segment_id,
                    speaker_label=seg.speaker_label,
                    embedding=seg.embedding,
                )
            )
        else:
            # fallback se não houver sobreposição
            seg.speaker_label = "SPEAKER_00"
            seg.embedding = None

        updated_segments.append(seg)

    # Deduplicar speakers com base nos embeddings por segmento
    speaker_map = deduplicate_speakers(per_segment_embeddings)

    # Aplicar rótulos canônicos aos segmentos
    for seg in updated_segments:
        if getattr(seg, "embedding", None) is not None:
            seg.speaker_label = speaker_map.get(seg.segment_id, seg.speaker_label)

    logger.info(f"Diarização concluída: {len(updated_segments)} segmentos rotulados")
    return updated_segments

def _diarizate_audio(
        segments: List[ProcessedSegment], 
        pipe: Pipeline, 
        embedding_model, 
        device
    ):
    try:
        # Agrupar por diretório do chunk
        chunk_dirs = list({str(Path(seg.local_path).parent) for seg in segments})

        all_speaker_results = []

        for chunk_dir in chunk_dirs:
            logger.info(f"Diarizando chunk: {chunk_dir}")

            # Combinar todos os segmentos do diretório em um único array (somente fala)
            combined = []
            sr = None
            for seg in segments:
                if str(Path(seg.local_path).parent) == chunk_dir:
                    audio, sr = sf.read(seg.local_path, always_2d=False)
                    if audio.ndim == 2:
                        # downmix para mono
                        audio = audio.mean(axis=1)
                    combined.append(audio.astype(np.float32))

            if not combined:
                logger.warning(f"Nenhum segmento encontrado em {chunk_dir}")
                continue

            combined_audio = np.concatenate(combined).astype(np.float32)
            orig_sr = int(sr or 16000)

            # Resample para 16k (evita resample interno caro do pyannote)
            target_sr = 16000
            if orig_sr != target_sr:
                if ta_resample is not None and torchaudio is not None:
                    wav_t = torch.from_numpy(combined_audio).unsqueeze(0)
                    combined_audio = ta_resample(wav_t, orig_sr, target_sr).squeeze(0).cpu().numpy()
                    sr = target_sr
                else:
                    # fallback: mantém SR original (pyannote pode resamplear internamente)
                    sr = orig_sr
            else:
                sr = target_sr

            # Diarização em memória
            waveform_tensor = torch.tensor(combined_audio, dtype=torch.float32, device=device)
            diarization = pipe({"waveform": waveform_tensor.cpu(), "sample_rate": sr})

            # Coleta todos os turns e prepara batch de embeddings (sem I/O)
            turns = []
            segs_audio = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                s = int(turn.start * sr)
                e = int(turn.end * sr)
                if e <= s:
                    continue
                seg_tensor = torch.tensor(combined_audio[s:e], dtype=torch.float32)
                if seg_tensor.numel() == 0:
                    continue
                turns.append((turn, speaker))
                segs_audio.append(seg_tensor)

            # Codifica em lotes para controlar memória (ex.: 64)
            BATCH = int(os.getenv("EMBED_BATCH", "64"))
            if segs_audio:
                # pad e compute relative lengths por lote
                for b in range(0, len(segs_audio), BATCH):
                    batch_list = segs_audio[b:b+BATCH]
                    lengths = torch.tensor([x.numel() for x in batch_list], dtype=torch.float32)
                    max_len = int(lengths.max().item())
                    padded = pad_sequence(batch_list, batch_first=True)  # (B, T)
                    rel_lens = (lengths / max_len).unsqueeze(1)

                    batch_signal = padded.to(device)
                    with torch.no_grad():
                        batch_emb = embedding_model.encode_batch(batch_signal, relative_lengths=rel_lens.to(device))
                    batch_emb = batch_emb.squeeze(1).cpu().numpy()

                    # normaliza L2
                    norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
                    norms[norms == 0.0] = 1.0
                    batch_emb = batch_emb / norms

                    for idx, emb in enumerate(batch_emb):
                        t, speaker = turns[b + idx]
                        all_speaker_results.append({
                            "speaker": speaker,
                            "start": float(t.start),
                            "end": float(t.end),
                            "duration": float(t.end - t.start),
                            "embedding": emb.tolist(),
                            "chunk_dir": chunk_dir,
                        })

        return all_speaker_results

    except Exception as e:
        logger.warning(f"Falha ao realizar a diarização do áudio: {e}")

def _extract_embedding(audio_path: str, embedding_model, device):
    audio, sr = sf.read(audio_path)
    signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedding_model.encode_batch(signal)
        emb = emb.squeeze().cpu().numpy()
        emb = emb / np.linalg.norm(emb)
    return emb

# Depois refatorar
def deduplicate_speakers(
        embeddings: List[SpeakerEmbedding],
        threshold: float = 0.70
    ):

    if not embeddings:
        return {}

    clusters = []
    centroids = []
    mapping = {}

    for emb in embeddings:
        vec = np.array(emb.embedding, dtype=np.float32)
        found = False

        for idx, centroid in enumerate(centroids):
            denom = (np.linalg.norm(vec) * np.linalg.norm(centroid))
            if denom == 0:
                continue
            sim = np.dot(vec, centroid) / denom
            if sim > threshold:
                clusters[idx].append(emb)
                # atualização incremental do centróide
                k = len(clusters[idx])
                centroids[idx] = centroids[idx] + (vec - centroids[idx]) / k
                found = True
                break

        if not found:
            clusters.append([emb])
            centroids.append(vec.copy())

    for i, cluster in enumerate(clusters):
        real_id = f"SPEAKER_{i:02d}"
        for emb in cluster:
            mapping[emb.segment_id] = real_id

    logger.info(f"Deduplicated to {len(clusters)} unique speakers")
    return mapping
