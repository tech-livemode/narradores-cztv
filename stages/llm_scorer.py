# Estágio de avaliação com LLM (Gemini)
import logging
import json
import os
from typing import Dict, List, Optional
import google.generativeai as genai  # type: ignore

from config import settings

logger = logging.getLogger(__name__)


def configure_llm():
    """
    Configura a API do Gemini.
    Adicione sua API key em config.py ou variável de ambiente.
    """
    api_key = (
        getattr(settings, 'gemini_api_key_1', None)
        or getattr(settings, 'gemini_api_key', None)
        or os.getenv('GEMINI_API_KEY')
    )
    if not api_key:
        logger.warning("⚠️ GEMINI_API_KEY não configurada. LLM scorer desabilitado.")
        return None
    
    genai.configure(api_key=api_key)
    #for m in genai.list_models():
    #    print(m.name)

    model = genai.GenerativeModel("gemini-2.5-flash")
    logger.info("✅ Gemini 1.5 Flash configurado")
    return model


def validate_sports_content_llm(text: str, duration: float, model=None) -> tuple:
    """
    Usa LLM para validar se o conteúdo é narração esportiva.
    Retorna: (is_sports, confidence, details)
    """
    if not model:
        logger.warning("LLM não disponível, usando fallback")
        return True, 0.5, {"reason": "llm_unavailable"}
    
    try:
        prompt = f"""Analise esta transcrição e determine se é uma NARRAÇÃO ESPORTIVA AO VIVO ou COMENTÁRIO DE JOGO.

TRANSCRIÇÃO:
{text[:2000]}  # Limita a 2000 chars

DURAÇÃO: {duration:.1f} segundos

INSTRUÇÕES:
- Retorne APENAS um JSON válido, sem texto adicional
- Seja criterioso: deve ser narração/comentário de evento esportivo REAL
- Notícias, entrevistas, podcasts sobre esporte NÃO são narrações

FORMATO DE RESPOSTA (JSON):
{{
  "is_sports_narration": true ou false,
  "confidence": número entre 0.0 e 1.0,
  "sport_detected": "futebol" ou "basquete" ou "vôlei" ou "outro" ou null,
  "content_type": "narração ao vivo" ou "comentário pós-jogo" ou "notícia" ou "podcast" ou "outro",
  "reasoning": "explicação breve em 1 frase"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Remove markdown code blocks se existirem
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        
        is_sports = result.get("is_sports_narration", False)
        confidence = float(result.get("confidence", 0.5))
        
        logger.info(f"🤖 LLM: is_sports={is_sports}, confidence={confidence:.2f}, type={result.get('content_type')}")
        
        return is_sports, confidence, result
        
    except Exception as e:
        logger.error(f"Erro ao validar com LLM: {e}")
        # Fallback: assume que é esporte com baixa confiança
        return True, 0.5, {"error": str(e)}


def evaluate_narration_criteria_llm(text: str, model=None) -> Optional[Dict]:
    """
    Usa LLM para avaliar os critérios de narração esportiva.
    Retorna dict com scores 0-10 para cada critério.
    """
    if not model:
        logger.warning("LLM não disponível")
        return None
    
    try:
        prompt = f"""Você é um avaliador de NARRAÇÃO AO VIVO DE FUTEBOL e precisa julgar se o trecho
        se aproxima do ESTILO MVP "LUISINHO" (narrador referência da plataforma). Valorize emoção
        contagiante, storytelling que guia o ouvinte e ritmo intenso. Não avalie clareza acadêmica:
        priorize energia, imediatismo e condução do lance com arcos narrativos.

        TRANSCRIÇÃO (amostrada):
        {text[:3000]}

        INSTRUÇÕES
        - Devolva APENAS JSON válido no formato definido abaixo (sem texto extra).
        - Use âncoras objetivas: 5=mediano, 7=bom, 9=excelente, 10=excepcional com forte evidência.
        - Penalize “modo comentarista” (análise longa, estatísticas, opinião extensa, off-topic).
        - Beneficie “estilo narrador”: verbos de ação, interjeições, imperativos, clímax (“gol”, “defendeu”).
        - Não penalize palavrões: trate-os como ênfase emocional (não contam para offtopic_ratio).
        - Interações/brincadeiras com comentaristas não são off-topic se estiverem ligadas ao lance, construção de clímax ou manutenção do ritmo.
        - Bonifique storytelling quando houver micro-histórias (pré/pós lance, protagonistas, contexto histórico), bem como "pré-clímax" que constroem expectativa ("segura o coração", "é agora", "vai pintar"). Valorize frases apaixonadas longas (≥15 palavras) com palavras como "gol", "torcida", "explode", "sensacional" e descrições vívidas com adjetivos/metáforas.
        - Quando uma análise curta servir para ancorar o lance (ex.: explicar a jogada anterior ou preparar o clímax), não reduza a emoção; apenas aplique penalização quando a análise domina o trecho.

        SINAIS QUE VOCÊ DEVE MEDIR NO PRÓPRIO TEXTO
        - action_verb_density_per_100w: contagem por 100 palavras de verbos de ação (regex base: chut|pass|cruz|marc|defend|atac|finaliz|arremat|lanç|dribl|corre|toca|cabece)
        - interjection_rate_per_100w: “gol”, “que lance”, “incrível”, “olha”, “vai”, “é agora”, “pra fora”, “defendeu”, etc., por 100 palavras
        - imperative_rate_per_100w: ocorrências de forma imperativa (“olha!”, “vem!”, “bate!”, “cruza!”, “marca!”) por 100 palavras
        - analysis_ratio: fração aproximada do texto dedicada à análise/estatística/opinião (0.00–1.00)
        - offtopic_ratio: fração aproximada do texto fora do lance (piadas/banter/assuntos alheios) (0.00–1.00)

        REGRAS DE NOTA (referência)
        - emotion: alta quando há interjeições/ênfase/variação e picos coerentes com lances.
        - storytelling: alta quando há contexto do jogo SEM virar comentário analítico longo; premie quando o narrador constrói expectativas ("é agora!", "segura o coração", etc.) ou referencia a jornada do jogo/jogadores.
        - game_pace: alta quando o texto acelera/desacelera com o lance, com muitos verbos de ação.
        - appropriate_commentary: equilíbrio; penalize comédia/off-topic excessivo.
        - style_alignment_luisinho: combine os sinais acima:
        • action_verb_density: ≤6→4–5; 8–12→6–7; 13–18→8–9; ≥19→9–10
        • interjection_rate: ≤2→4–5; 3–5→6–7; 6–9→8–9; ≥10→9–10
        • imperative_rate: ≤1→4–5; 2–3→6–7; 4–6→8–9; ≥7→9–10
        • analysis_ratio alto reduz estilo (penalize 1–3 pts se ≥0.35; 2–4 pts se ≥0.45)
        • offtopic_ratio alto reduz appropriate_commentary (≈1–2 pts se ≥0.18; escale pela aderência de estilo: penalidade *= (1 - style_score/10))

        FORMATO (JSON):
        {{
        "emotion": {{"score": 0, "justification": ""}},
        "storytelling": {{"score": 0, "justification": ""}},
        "game_pace": {{"score": 0, "justification": ""}},
        "appropriate_commentary": {{"score": 0, "justification": ""}},
        "style_alignment_luisinho": {{
            "score": 0,
            "signals": {{
            "action_verb_density_per_100w": 0,
            "interjection_rate_per_100w": 0,
            "imperative_rate_per_100w": 0,
            "analysis_ratio": 0.0,
            "offtopic_ratio": 0.0
            }}
        }}
        }}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Remove markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        
        logger.info("🤖 LLM avaliou critérios de texto com sucesso")
        return result
        
    except Exception as e:
        logger.error(f"Erro ao avaliar critérios com LLM: {e}")
        return None


def generate_personalized_feedback_llm(
    audio_score: Dict,
    text_score: Dict,
    final_score: float,
    classification: str,
    model=None
) -> tuple:
    """
    Usa LLM para gerar feedback personalizado e construtivo.
    Retorna: (strengths, improvements)
    """
    if not model:
        logger.warning("LLM não disponível para feedback")
        return [], []
    
    try:
        prompt = f"""Você é um coach de narradores esportivos. Gere feedback construtivo.

AVALIAÇÃO DO NARRADOR:
- Score Final: {final_score:.1f}/100 ({classification})

Áudio (40%):
- Dinâmica Vocal: {audio_score.get('vocal_dynamics', 0):.1f}/10
- Ritmo da Fala: {audio_score.get('speech_pacing', 0):.1f}/10
- Emoção (Áudio): {audio_score.get('emotion_audio', 0):.1f}/10

Texto (60%):
- Emoção (Conteúdo): {text_score.get('emotion_content', 0):.1f}/10
- Storytelling: {text_score.get('storytelling', 0):.1f}/10
- Ritmo do Jogo: {text_score.get('game_pace', 0):.1f}/10
- Resenha Apropriada: {text_score.get('appropriate_commentary', 0):.1f}/10

INSTRUÇÕES:
- Seja específico e construtivo
- Tom motivador, não crítico
- Identifique 2-3 pontos fortes REAIS (não invente se não houver)
- Sugira 2-4 melhorias PRÁTICAS e ACIONÁVEIS
- Retorne APENAS JSON válido

FORMATO (JSON):
{{
  "strengths": ["ponto forte 1", "ponto forte 2"],
  "improvements": ["melhoria prática 1", "melhoria prática 2", "melhoria prática 3"]
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Remove markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        
        strengths = result.get("strengths", [])
        improvements = result.get("improvements", [])
        
        logger.info(f"🤖 LLM gerou feedback: {len(strengths)} fortes, {len(improvements)} melhorias")
        
        return strengths, improvements
        
    except Exception as e:
        logger.error(f"Erro ao gerar feedback com LLM: {e}")
        return [], []


def hybrid_scoring_mode(text: str, duration: float, use_llm: bool = True) -> Dict:
    """
    Modo híbrido: combina regras simples + LLM para melhor resultado.
    
    1. Validação rápida com regras (elimina casos óbvios)
    2. Se passar, usa LLM para análise profunda
    3. Retorna resultado combinado
    """
    result = {
        "validation_method": "hybrid",
        "is_sports": False,
        "confidence": 0.0,
        "details": {}
    }
    
    # Passo 1: Validação rápida (regras)
    # Casos óbvios de não-esporte
    non_sports_obvious = [
        "podcast", "episódio", "inscreva-se", "like", "compartilhe",
        "status do whatsapp", "motivação", "reflexão"
    ]
    
    text_lower = text.lower()
    if any(word in text_lower for word in non_sports_obvious):
        if duration < 60:  # Conteúdo curto + palavras não-esportivas
            result["validation_method"] = "rules_reject"
            result["details"]["reason"] = "obvious_non_sports"
            logger.info("🚫 Regras: Conteúdo claramente não-esportivo")
            return result
    
    # Passo 2: Se passou validação inicial, usa LLM
    if use_llm:
        model = configure_llm()
        if model:
            is_sports, confidence, llm_details = validate_sports_content_llm(text, duration, model)
            result["is_sports"] = is_sports
            result["confidence"] = confidence
            result["details"] = llm_details
            result["validation_method"] = "llm"
            return result
    
    # Fallback: regras heurísticas (menos confiável)
    from stages.score import _validate_sports_content
    is_sports, confidence = _validate_sports_content(text, duration)
    result["is_sports"] = is_sports
    result["confidence"] = confidence
    result["validation_method"] = "rules_fallback"
    result["details"]["reason"] = "llm_unavailable"
    
    return result
