# LLMOps Pipeline
### End-to-End LLM Fine-tuning, Evaluation, Serving & Monitoring

<p align="center">
  <img src="https://img.shields.io/badge/LoRA-Fine--tuning-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/QLoRA-4bit_Quantization-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MLflow-Experiment_Tracking-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/Grafana-Dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Model_Hub-FFD21E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

<p align="center">
  <a href="https://github.com/djism/llmops-pipeline/actions/workflows/ci.yml">
    <img src="https://github.com/djism/llmops-pipeline/actions/workflows/ci.yml/badge.svg" alt="LLMOps CI" />
  </a>
  &nbsp;
  <a href="https://huggingface.co/dhananjay9624/phi3-medical-qa">
    <img src="https://img.shields.io/badge/🤗_Model-phi3--medical--qa-yellow?style=flat-square" />
  </a>
</p>

---

> **Most projects use LLMs. This project builds one.** Full lifecycle — dataset curation, LoRA fine-tuning, automated evaluation with a hard quality gate, production serving, and real-time monitoring with Prometheus + Grafana.

---

## What This Is

A production-grade **LLMOps pipeline** that fine-tunes `microsoft/Phi-3-mini-4k-instruct` on a medical QA dataset using **LoRA (QLoRA)**, evaluates it with automated metrics, gates deployment on quality thresholds, serves it via FastAPI, and monitors it in production with Prometheus.

**The fine-tuned model is live on HuggingFace Hub:**
→ [dhananjay9624/phi3-medical-qa](https://huggingface.co/dhananjay9624/phi3-medical-qa)

---

## The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DATA                                                 │
│                                                                 │
│  MedAlpaca dataset (10,178 medical QA samples)                  │
│  → Filter → Format for Phi-3 chat template                     │
│  → Split 85/15 train/eval → Save to disk                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — TRAINING (Google Colab T4 GPU)                       │
│                                                                 │
│  Base model: Phi-3-mini-4k-instruct (3.8B params)               │
│  4-bit quantization (QLoRA) → fits in 15GB VRAM                 │
│  LoRA adapters: rank=16, alpha=32, target Q/K/V/O projections   │
│  Trains only ~1% of parameters → same quality, fraction of cost │
│  All experiments tracked with MLflow                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — EVALUATION GATE (runs in CI on every push)           │
│                                                                 │
│  ROUGE-L ≥ 0.25        → content coverage check                 │
│  BLEU ≥ 0.15           → precision check                        │
│  Hallucination ≤ 10%   → safety check                           │
│                                                                 │
│  ALL THREE must pass → exit code 0 → pipeline continues         │
│  ANY ONE fails       → exit code 1 → pipeline blocked           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — SERVING                                              │
│                                                                 │
│  LoRA adapters pushed to HuggingFace Hub                        │
│  FastAPI inference endpoint — POST /generate                    │
│  Containerized with Docker                                      │
│  Swagger docs at /docs                                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5 — MONITORING                                           │
│                                                                 │
│  Prometheus scrapes /metrics every 15 seconds                   │
│  Tracks: request count, latency histogram, error rate,          │
│          token count, active requests, model status             │
│  Grafana dashboards visualize all metrics                       │
│  Alerts fire when latency or error rate crosses threshold       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why LoRA? Why Not Full Fine-tuning?

Full fine-tuning Phi-3-mini (3.8B params) requires:
- ~30GB VRAM (impossible on free hardware)
- Hours of compute
- Storing a full 7GB model checkpoint per experiment

**LoRA injects small adapter matrices into the attention layers — only ~0.5% of parameters are trained.** The base model stays frozen. The result:

```
Full fine-tuning:  3,800,000,000 trainable params → 30GB VRAM
LoRA (r=16):          20,000,000 trainable params → 15GB VRAM
                                                   ✅ Free T4 GPU
```

Quality difference: minimal. Compute difference: 10x.

---

## The Evaluation Gate — The Critical Design Decision

Most LLM projects test manually. This pipeline tests automatically on every push and **blocks bad models from reaching production.**

```python
# From src/evaluation/eval_gate.py
# Exit code 0 = pass → CI continues → model gets deployed
# Exit code 1 = fail → CI stops   → model gets blocked

gate_rouge      = avg_rouge_l >= 0.25        # ✅ or ❌
gate_bleu       = avg_bleu >= 0.15           # ✅ or ❌
gate_hallucinate = hallucination_rate <= 0.10 # ✅ or ❌

passed = gate_rouge AND gate_bleu AND gate_hallucinate
sys.exit(0 if passed else 1)
```

The hallucination detector checks if the model introduces **medical entities** (drug names, dosages, clinical terms) that weren't in the reference answer. In a medical domain, this matters.

---

## Experiment Tracking With MLflow

Every training run is tracked:

```
Run: lora_finetune_16r
├── Parameters
│   ├── base_model: microsoft/Phi-3-mini-4k-instruct
│   ├── lora_r: 16
│   ├── lora_alpha: 32
│   ├── learning_rate: 0.0002
│   ├── num_epochs: 3
│   └── dataset_train: 1700
├── Metrics (per epoch)
│   ├── epoch_train_loss: 2.18 → 1.54 → 1.12
│   └── epoch_eval_loss:  2.45 → 1.79 → 1.34
└── Evaluation
    ├── eval_rouge_l: 0.312
    ├── eval_bleu: 0.198
    ├── eval_hallucination_rate: 0.06
    └── eval_gate_passed: True
```

Compare runs, track regressions, reproduce any experiment.

---

## Prometheus Metrics

Every inference request is instrumented:

| Metric | Type | What It Tracks |
|---|---|---|
| `llmops_requests_total` | Counter | Total requests by endpoint + status |
| `llmops_request_latency_ms` | Histogram | Latency distribution (8 buckets) |
| `llmops_tokens_generated` | Histogram | Token count distribution |
| `llmops_active_requests` | Gauge | Currently processing |
| `llmops_errors_total` | Counter | Errors by type |
| `llmops_model_loaded` | Gauge | Model health (1=ready, 0=down) |
| `llmops_model_info` | Gauge | Model ID + type labels |

---

## Tech Stack

| Stage | Technology |
|---|---|
| **Base Model** | microsoft/Phi-3-mini-4k-instruct (3.8B) |
| **Fine-tuning** | LoRA/QLoRA via HuggingFace PEFT |
| **Quantization** | BitsAndBytes 4-bit (QLoRA) |
| **Training** | TRL SFTTrainer on Google Colab T4 |
| **Experiment Tracking** | MLflow |
| **Dataset** | MedAlpaca medical_meadow_medqa (10K samples) |
| **Evaluation** | ROUGE-L, BLEU, Hallucination Rate |
| **Model Registry** | HuggingFace Hub |
| **Serving** | FastAPI + Uvicorn |
| **Monitoring** | Prometheus + Grafana |
| **Containerization** | Docker + docker-compose |
| **CI/CD** | GitHub Actions — evaluation gate on every push |

**Total compute cost: $0** — Google Colab free T4, HuggingFace free tier, open source everything.

---

## Project Structure

```
llmops-pipeline/
├── src/
│   ├── data/
│   │   └── dataset_builder.py       # MedAlpaca loading + Phi-3 formatting
│   ├── training/
│   │   ├── fine_tune.py             # LoRA fine-tuning pipeline
│   │   ├── mlflow_logger.py         # Experiment tracking wrapper
│   │   └── colab_trainer.py         # Generates ready-to-use Colab notebook
│   ├── evaluation/
│   │   ├── evaluator.py             # ROUGE-L, BLEU, hallucination rate
│   │   └── eval_gate.py             # CI gate — exit 0/1 based on thresholds
│   ├── serving/
│   │   └── api.py                   # FastAPI inference endpoint
│   └── monitoring/
│       └── metrics.py               # Prometheus metrics definitions
├── tests/
│   └── test_pipeline.py             # 13 unit tests
├── phi3_medical_qa_finetuning.ipynb # Colab notebook (auto-generated)
├── prometheus.yml                   # Prometheus scrape config
├── docker-compose.yml               # API + Prometheus + Grafana
├── Dockerfile
└── config.py
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker Desktop
- [HuggingFace account](https://huggingface.co) + token (free)
- Google Colab account (free, for training)

### 1. Clone and setup

```bash
git clone https://github.com/djism/llmops-pipeline.git
cd llmops-pipeline

python -m venv llmops-env
source llmops-env/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add HF_TOKEN to .env
```

### 3. Build dataset

```bash
python src/data/dataset_builder.py
```

### 4. Fine-tune on Colab (free T4 GPU)

```bash
# Generate the notebook
python src/training/colab_trainer.py

# Then upload phi3_medical_qa_finetuning.ipynb to colab.research.google.com
# Runtime → T4 GPU → Run all (~45 minutes)
```

### 5. Run evaluation gate

```bash
python src/evaluation/evaluator.py --mock --samples 50
python src/evaluation/eval_gate.py --results-file eval_results.json
```

### 6. Start serving + monitoring

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI inference | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Metrics endpoint | http://localhost:8000/metrics |

### 7. Run tests

```bash
pytest tests/ -v
```

---

## API Reference

### `POST /generate`

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the first-line treatment for type 2 diabetes?"}'
```

```json
{
  "question": "What is the first-line treatment for type 2 diabetes?",
  "answer": "The first-line treatment is metformin combined with lifestyle modifications...",
  "model_id": "dhananjay9624/phi3-medical-qa",
  "latency_ms": 847.32,
  "tokens_generated": 94
}
```

---

## Test Results

```
tests/test_pipeline.py::test_config_imports               PASSED
tests/test_pipeline.py::test_sample_dataset_loads         PASSED
tests/test_pipeline.py::test_format_for_training          PASSED
tests/test_pipeline.py::test_dataset_split                PASSED
tests/test_pipeline.py::test_rouge_l_perfect_match        PASSED
tests/test_pipeline.py::test_rouge_l_no_match             PASSED
tests/test_pipeline.py::test_bleu_perfect_match           PASSED
tests/test_pipeline.py::test_hallucination_rate_zero      PASSED
tests/test_pipeline.py::test_hallucination_rate_detected  PASSED
tests/test_pipeline.py::test_evaluation_mock_mode         PASSED
tests/test_pipeline.py::test_eval_gate_pass               PASSED
tests/test_pipeline.py::test_eval_gate_fail               PASSED
tests/test_pipeline.py::test_prometheus_metrics           PASSED

13 passed in 2.49s
```

---

## The Design Decision That Matters

**Why a hard evaluation gate instead of manual review?**

Manual review doesn't scale and doesn't catch regressions. Every time you retrain with new data or changed hyperparameters, you need to know: *is this model better or worse than the last one?*

The gate enforces this automatically. If ROUGE-L drops from 0.32 to 0.18 after a training change, the CI pipeline exits with code 1 and blocks deployment — no human review needed, no production incident.

This is the difference between a research experiment and a production system.

---

## Author

**Dhananjay Sharma**
M.S. Data Science, SUNY Stony Brook (May 2026)

<p>
  <a href="https://www.linkedin.com/in/dsharma2496/">LinkedIn</a> ·
  <a href="https://djism.github.io/">Portfolio</a> ·
  <a href="https://github.com/djism">GitHub</a>
</p>

---

<p align="center">
  <i>Most projects use LLMs. This one builds, evaluates, deploys, and monitors one.</i>
</p>