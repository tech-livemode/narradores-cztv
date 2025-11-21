FROM python:3.11-slim

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar gunicorn para produção
RUN pip install --no-cache-dir gunicorn

# Copiar código da aplicação
COPY . .

# Expor porta (Cloud Run define PORT automaticamente via variável de ambiente)
ENV PORT=8080

# Executar API Flask com gunicorn
# --workers 1: Cloud Run gerencia escalonamento horizontal
# --threads 8: Permite múltiplas requisições simultâneas na mesma instância
# --timeout 3600: 60 minutos (máximo do Cloud Run)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 3600 api:app

