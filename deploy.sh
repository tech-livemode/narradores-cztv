#!/bin/bash
# Script de deploy para Google Cloud Run

set -e

PROJECT_ID="${GCP_PROJECT_ID:-seu-project-id}"
IMAGE_NAME="narradores-caze"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="narradores-caze"

echo "🚀 Iniciando deploy do $SERVICE_NAME para Cloud Run..."

# Verificar se está autenticado
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Erro: Você precisa estar autenticado no GCP"
    echo "Execute: gcloud auth login"
    exit 1
fi

# Configurar projeto
gcloud config set project "$PROJECT_ID"

# Build e push da imagem
echo "📦 Construindo e enviando imagem Docker..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy no Cloud Run
echo "🚀 Fazendo deploy no Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --memory 8Gi \
  --cpu 4 \
  --timeout 3600 \
  --max-instances 10 \
  --min-instances 0 \
  --allow-unauthenticated \
  --set-env-vars PORT=8080 \
  --set-secrets HF_TOKEN=HF_TOKEN:latest,MAIN_GEMINI_TOKEN=MAIN_GEMINI_TOKEN:latest,MAIL_SENDER_USERNAME=MAIL_SENDER_USERNAME:latest,MAIL_SENDER_PASSWORD=MAIL_SENDER_PASSWORD:latest

echo "✅ Deploy concluído!"
echo "🌐 URL do serviço:"
gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)"


