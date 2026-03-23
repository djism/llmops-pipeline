"""
FastAPI serving endpoint for the fine-tuned Phi-3-mini model.

Exposes:
    POST /generate  — generate answer for a medical question
    GET  /health    — service health + model info
    GET  /metrics   — Prometheus metrics endpoint
    GET  /examples  — example questions to try
"""

import sys
import os
import time
import ssl
import certifi
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

ssl._create_default_https_context = ssl.create_default_context
os.environ['SSL_CERT_FILE'] = certifi.where()

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    validate_config, API_HOST, API_PORT,
    HF_TOKEN, HF_FINETUNED_REPO,
    MAX_NEW_TOKENS, TEMPERATURE
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.monitoring.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_REQUESTS,
    ERROR_COUNT, MODEL_INFO, record_request
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Medical question to answer",
        examples=["What is the first-line treatment for type 2 diabetes?"]
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Override default max tokens"
    )


class GenerateResponse(BaseModel):
    question: str
    answer: str
    model_id: str
    latency_ms: float
    tokens_generated: int


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
    total_requests: int
    message: str


# ── Model state ───────────────────────────────────────────────────────────────

class ModelState:
    pipe = None
    model_id = None
    request_count = 0
    is_loaded = False


state = ModelState()


def load_model():
    """Loads the fine-tuned model on startup."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_id = HF_FINETUNED_REPO
        print(f"⏳ Loading model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )

        state.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False
        )
        state.model_id = model_id
        state.is_loaded = True

        # Update Prometheus model info
        MODEL_INFO.labels(
            model_id=model_id,
            model_type="fine-tuned-lora"
        ).set(1)

        print(f"✅ Model loaded: {model_id}")

    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
        print("   API will run but /generate will return errors")
        state.model_id = HF_FINETUNED_REPO
        state.is_loaded = False


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 55)
    print("  LLMOps Serving API — Starting Up")
    print("=" * 55)
    validate_config()
    load_model()
    print("✅ API ready\n")
    yield
    print("\n👋 LLMOps API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLMOps Serving API",
    description="""
    Production serving endpoint for fine-tuned Phi-3-mini medical QA model.

    **Model**: Phi-3-mini-4k-instruct fine-tuned with LoRA on MedAlpaca dataset

    **Pipeline**:
    - Dataset: MedAlpaca medical QA (2,000 samples)
    - Fine-tuning: QLoRA (4-bit) on Google Colab T4
    - Evaluation: ROUGE-L, BLEU, Hallucination Rate
    - Monitoring: Prometheus + Grafana
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "LLMOps Serving API",
        "model": state.model_id,
        "loaded": state.is_loaded,
        "docs": "/docs",
        "health": "/health",
        "generate": "POST /generate",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if state.is_loaded else "degraded",
        model_id=state.model_id or "not loaded",
        model_loaded=state.is_loaded,
        total_requests=state.request_count,
        message="Model ready for inference" if state.is_loaded
        else "Model not loaded — check logs"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate an answer for a medical question.
    Uses the fine-tuned Phi-3-mini model.
    """
    if not state.is_loaded or state.pipe is None:
        ERROR_COUNT.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health for status."
        )

    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        prompt = f"<|user|>\n{request.question}<|end|>\n<|assistant|>\n"

        max_tokens = request.max_new_tokens or MAX_NEW_TOKENS
        output = state.pipe(
            prompt,
            max_new_tokens=max_tokens,
            return_full_text=False
        )

        answer = output[0]["generated_text"].strip()
        latency_ms = (time.time() - start_time) * 1000
        tokens = len(answer.split())

        # Update state and metrics
        state.request_count += 1
        record_request(
            endpoint="/generate",
            latency_ms=latency_ms,
            success=True
        )

        return GenerateResponse(
            question=request.question,
            answer=answer,
            model_id=state.model_id,
            latency_ms=round(latency_ms, 2),
            tokens_generated=tokens
        )

    except Exception as e:
        ERROR_COUNT.labels(error_type="inference_error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/examples")
async def examples():
    """Example questions to test the model."""
    return {
        "examples": [
            "What is the first-line treatment for type 2 diabetes mellitus?",
            "What are the classic symptoms of myocardial infarction?",
            "What is the mechanism of action of beta-blockers?",
            "What is the treatment for anaphylaxis?",
            "How does heparin work as an anticoagulant?",
            "What are the components of the Glasgow Coma Scale?",
            "What is the treatment protocol for STEMI?"
        ],
        "note": "POST /generate with any medical question"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", API_PORT))
    print(f"\n🚀 Starting LLMOps Serving API on http://{API_HOST}:{port}")
    print(f"📖 Docs at http://localhost:{port}/docs\n")
    uvicorn.run(
        "src.serving.api:app",
        host=API_HOST,
        port=port,
        reload=False,
        log_level="info"
    )