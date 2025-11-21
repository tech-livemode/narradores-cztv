# Caze Narrator Scoring Pipeline

Pipeline completo para capturar transmissões esportivas, processar áudio/texto e gerar uma avaliação 0–100 para narradores. O sistema combina métricas acústicas, heurísticas textuais e a opinião de um LLM (Gemini) para estimar proximidade com o narrador‑modelo ("MVP").

## Sumário
- [Arquitetura](#arquitetura)
- [Principais recursos](#principais-recursos)
- [Instalação](#instalação)
- [Configuração (.env)](#configuração-env)
- [Como executar](#como-executar)
- [API Flask](#api-flask)
- [Modos de operação](#modos-de-operação)
- [Feedback e notificações](#feedback-e-notificações)
- [Estrutura de diretórios](#estrutura-de-diretórios)

## Arquitetura
```
 captura → pré-processamento → diarização leve → ASR (+correções) → 
 seleção de janelas → LLM scoring → agregação + heurísticas → relatório/e-mail
```
- **stages/pipeline_core.py**: orquestra captura, pré-processamento e ASR (paralelo opcional).
- **stages/scoring/**: módulo principal de scoring (audio.py, text.py, windows.py, llm_aggregate.py etc.).
- **stages/llm_scorer.py**: integra com Gemini para validar conteúdo e avaliar critérios por janela.
- **stages/cache_io.py**: lida com caches (`processed/<stream_id>/`).
- **smtp_sender.py**: envia notificações quando um narrador ultrapassa o threshold de qualidade.

## Principais recursos
- **Captura flexível**: yt-dlp/streamlink (não incluso aqui) + chunks configuráveis.
- **Áudio calibrado**: dinâmica vocal, ritmo e emoção com normalização baseada em perfil.
- **Texto híbrido**: heurísticas + LLM windowed scoring com janelas de pico, uniforme, analítica, longform e "pre-peak".
- **Ponderação dinâmica**: mistura heurística/LLM conforme a proximidade com o narrador MVP.
- **Light diarization**: identifica janelas dominadas pelo narrador para priorizar trechos relevantes.
- **Reaproveitamento**: caches permitem re-run `--score-only` ou `--from-chunks` sem refazer captura.
- **Alertas**: envia e-mail quando o score final passa de 70.

## Instalação
### Pré-requisitos
- Python 3.9+
- FFmpeg (com filtro `loudnorm`)
- yt-dlp e streamlink (para captura)
- GPU CUDA opcional

### macOS
```bash
brew install ffmpeg yt-dlp streamlink
```

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-dev
pip install yt-dlp streamlink
```

### Python deps
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuração (.env)
Crie um arquivo `.env` na raiz com as credenciais. Exemplo:
```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
GEMINI_API_KEY=ya29.xxxxxxxxxxxxxxxxx
MAIL_SENDER_USERNAME=seuemail@gmail.com
MAIL_SENDER_PASSWORD=senha_app
```
> Variáveis adicionais podem ser lidas em `config.py` (por exemplo `GEMINI_API_KEY_1`, cookies do YouTube, etc.).

## Como executar
### Pipeline completo (captura → score)
```bash
python main.py
```
Utiliza `JOB_QUEUE` (definida em `main.py`) ou parâmetros padrão.

### Score apenas (usando caches existentes)
```bash
python main.py --score-only --stream-id luisinho_MVP
```
Lê `processed/<stream_id>/segments.json` e `transcripts.json`, reaplica o scorer e gera `score_result_rescored.json`.

### Reprocessar a partir dos WAVs (sem recapturar)
```bash
python main.py --from-chunks --stream-id luisinho_MVP
```
Roda ASR em `processed/<stream_id>/segment_*_*.wav`, salva novos caches e score.

### Modo MVP (re-score em lote dos caches listados em `JOB_QUEUE`)
```bash
python main.py --mvp
```

## API Flask
Uma API REST opcional permite disparar o pipeline remotamente e acompanhar o status de cada job.

### Instanciando
```bash
export FLASK_APP=api:create_app  # ou use python api.py
flask run --host 0.0.0.0 --port 8000
```
> A API roda na mesma venv usada pelo CLI. Certifique-se de ter instalado `Flask` via `pip install -r requirements.txt`.

### Endpoints
| Método | Caminho | Descrição |
|--------|---------|-----------|
| `GET`  | `/health` | Probe simples (quantidade de jobs na instância). |
| `POST` | `/jobs` | Cria um job novo. Espera JSON com `url`, `stream_id` e (opcional) `gemini_api_key`. Retorna `202 Accepted` com o ID do job. |
| `GET`  | `/jobs` | Lista todos os jobs conhecidos. |
| `GET`  | `/jobs/<id>` | Mostra metadados e status do job (`pending`, `running`, `finished`, `failed`). |
| `GET`  | `/jobs/<id>/score` | Retorna o JSON de `processed/<stream_id>/score_result.json` após a conclusão. |
| `POST` | `/maintenance/flush` | Limpa diretórios de `processed/` e, opcionalmente, `temp/` (dry-run por padrão). |

### Exemplo de requisição
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
        "url": "https://www.youtube.com/watch?v=xxxxx",
        "stream_id": "narrador_demo",
        "gemini_api_key": "opcional"
      }'
```
Use o `id` retornado para acompanhar o progresso:
```bash
curl http://localhost:8000/jobs/<id>
curl http://localhost:8000/jobs/<id>/score
```

## Manutenção dos diretórios `processed/` e `temp/`
Com o uso contínuo, `processed/` e `temp/` podem crescer rapidamente. O módulo `maintenance.py` oferece uma limpeza segura por idade (dry-run por padrão) para ambos.

### Linha de comando
Dry-run (apenas relata o que seria removido):
```bash
python maintenance.py --max-age-days 7
```
Aplicando de fato e preservando alguns IDs:
```bash
python maintenance.py --max-age-days 7 --apply --keep narrador_vip --keep narrador_teste
```
- Inclua `--clean-temp` para também varrer o diretório temporário. Exemplo apagando arquivos de `temp/` com mais de 1 dia:
```bash
python maintenance.py --max-age-days 14 --clean-temp --temp-max-age-days 1 --apply
```

### Via API
O mesmo fluxo pode ser disparado remotamente:
```bash
curl -X POST http://localhost:8000/maintenance/flush \
  -H "Content-Type: application/json" \
  -d '{
        "max_age_days": 14,
        "keep_stream_ids": ["narrador_vip"],
        "apply": true,
        "clean_temp": true,
        "temp_max_age_days": 1
      }'
```
Adicione `"temp_apply": true` se quiser aplicar apenas na pasta temporária; sem `"apply": true`/`"temp_apply": true`, o endpoint roda em modo dry-run e apenas retorna o relatório.

## Modos de operação
| Modo | Descrição |
|------|-----------|
| padrão | captura streaming → chunks → preprocessa → ASR → scoring |
| `--from-chunks` | pula captura/preprocess, usa WAVs já extraídos |
| `--score-only` | pula captura, preprocess e ASR; usa segments/transcripts cacheados |
| `--mvp` | reavalia toda a JOB_QUEUE usando caches |

## Feedback e notificações
- Relatório completo via `print_score_report` (CLI).
- E-mail automático (smtp_sender) quando `final_score >= 70` informando narrador, score e stats.

## Estrutura de diretórios
```
Caze/
├── main.py                # CLI principal
├── config.py              # Settings + defaults
├── smtp_sender.py         # utilitário de e-mail
├── stages/
│   ├── pipeline_core.py   # captura + preprocess + ASR
│   ├── cache_io.py        # caches e reprocessamentos rápidos
│   ├── scoring/           # módulos de scoring (audio, text, windows, entrypoint, etc.)
│   └── llm_scorer.py      # prompts e integração Gemini
├── processed/<stream_id>/ # caches: segments.json, transcripts.json, score_result*.json
└── requirements.txt
```

---
