# 🧠 TEA Biomarker Pipeline

Sistema para extração de **expressões faciais/microexpressões** e **padrões biomarcadores de TEA** usando **Visão Computacional + Deep Learning + FairML + XAI**.  
Arquitetura baseada em **Clean Architecture + SOLID**.

---

## 🚀 Setup

```bash
# Clonar repositório
git clone <repo_url>
cd project

# Instalar dependências
python -m pip install -r requirements.txt
````

Ou via **Docker**:

```bash
docker build -t tea-biomarker .
docker run -it --rm tea-biomarker
```

---

## 🛠️ CLI - Comandos

```bash
# Validação de dados
python -m apps.cli.main validate-data \
  --input datasets/raw --out datasets/processed --report artifacts/data_quality.html

# Extração de features
python -m apps.cli.main extract-features \
  --input datasets/processed --out datasets/features --profile medium

# Treino + XAI + Fairness
python -m apps.cli.main train \
  --features datasets/features --models-dir models \
  --profile medium --enable-xai --audit-fairness --protocol loso

# Inferência
python -m apps.cli.main infer \
  --input datasets/processed --models-dir models --out artifacts/reports

# Benchmark
python -m apps.cli.main benchmark --profile medium --iterations 1000
```

---

## 📂 Estrutura

* `domain/` → entidades e interfaces (contratos).
* `application/` → casos de uso (orquestração).
* `infrastructure/` → implementações (visão, modelos, XAI, fairness).
* `interfaces/` → CLI, API e reporters.
* `scripts/` → utilitários (validate, audit, benchmark).
* `datasets/` → dados brutos e processados.
* `artifacts/` → relatórios, métricas, figuras.

---

## 📊 Métricas e Relatórios

* **HTML + JSON**: qualidade de dados, fairness, XAI.
* **Model Card** e **Data Card**.
* Benchmarks com perfis: `low`, `medium`, `high`.

---

## ⚖️ Nota Ética

> Ferramenta de apoio à decisão; **não substitui avaliação multiprofissional**.
> Requer **aprovação ética** e conformidade **LGPD/GDPR**.
> Uso sob supervisão de profissional de saúde.

---

## 📈 Próximos Passos

* Suporte a vídeos curtos e pooling temporal robusto.
* Compressão via TensorRT.
* Domain adaptation em datasets clínicos heterogêneos.

