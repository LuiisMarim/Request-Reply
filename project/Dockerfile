FROM python:3.10-slim

# Instala dependências de sistema
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "apps.cli.main", "--help"]
