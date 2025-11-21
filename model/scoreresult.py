from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class AudioScore:
    """Pontuação dos aspectos de áudio (40% do total)"""
    # Subcritérios
    vocal_dynamics: float  # 0-10: Variação de pitch, energia, expressividade
    speech_pacing: float   # 0-10: Ritmo da fala, pausas adequadas
    emotion_audio: float   # 0-10: Emoção transmitida pela voz
    
    # Score total do áudio (média ponderada)
    total_score: float     # 0-100
    
    # Detalhes técnicos
    details: Optional[Dict] = None


@dataclass
class TextScore:
    """Pontuação dos aspectos de texto/conteúdo (60% do total)"""
    # Subcritérios baseados nos requisitos
    emotion_content: float      # 0-10: Captura emoção dos momentos, transmite paixão
    storytelling: float         # 0-10: Conta história do jogo, contexto, significado
    game_rhythm: float          # 0-10: Dá ritmo ao jogo, menciona elenco
    appropriate_fun: float      # 0-10: Resenha apropriada (não comediante)
    
    # Score total do texto (média ponderada)
    total_score: float          # 0-100
    
    # Detalhes da análise
    details: Optional[Dict] = None


@dataclass
class FinalScore:
    """Resultado final da avaliação do narrador"""
    stream_id: str
    
    # Scores parciais
    audio_score: AudioScore
    text_score: TextScore
    
    # Score final ponderado: 40% áudio + 60% texto
    final_score: float  # 0-100
    
    # Classificação categórica
    classification: str  # Excelente, Bom, Regular, Abaixo da média, Insuficiente
    
    # Feedback estruturado
    strengths: list  # Pontos fortes
    improvements: list  # Áreas de melhoria
    
    # Metadados
    total_segments: int
    total_duration: float  # em segundos
