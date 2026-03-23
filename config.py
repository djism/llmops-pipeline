import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MODELS_DIR = BASE_DIR / "models"

# Create dirs if they don't exist
for d in [DATASET_DIR, EXPERIMENTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
HF_FINETUNED_REPO = os.getenv("HF_FINETUNED_REPO", "dhananjay9624/phi3-medical-qa")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(EXPERIMENTS_DIR))
MLFLOW_EXPERIMENT_NAME = "phi3-medical-qa-finetuning"

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "model_id": HF_MODEL_ID,
    "max_seq_length": 512,
    "lora_r": 16,               # LoRA rank
    "lora_alpha": 32,           # LoRA scaling
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "output_dir": str(MODELS_DIR / "phi3-medical-qa"),
}

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "hf_dataset": "medalpaca/medical_meadow_medqa",
    "train_split": 0.85,
    "eval_split": 0.15,
    "max_samples": 2000,        # keep manageable for free GPU
    "sample_file": str(DATASET_DIR / "sample.json"),
    "train_file": str(DATASET_DIR / "train.json"),
    "eval_file": str(DATASET_DIR / "eval.json"),
}

# ── Evaluation thresholds ─────────────────────────────────────────────────────
# CI evaluation gate — model only gets promoted if it passes ALL thresholds
EVAL_THRESHOLDS = {
    "min_rouge_l": float(os.getenv("MIN_ROUGE_L", "0.25")),
    "min_bleu": float(os.getenv("MIN_BLEU", "0.15")),
    "max_hallucination_rate": float(os.getenv("MAX_HALLUCINATION_RATE", "0.10")),
}

# ── Serving ───────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# ── Monitoring ────────────────────────────────────────────────────────────────
PROMETHEUS_PORT = 8001
METRICS_INTERVAL = 15           # seconds between metric scrapes


# ── Validation ────────────────────────────────────────────────────────────────
def validate_config():
    errors = []
    if not HF_TOKEN:
        errors.append("HF_TOKEN is missing from .env")
    if errors:
        raise EnvironmentError("\n".join(errors))
    print("✅ Config validated successfully")
    print(f"   Base model     : {HF_MODEL_ID}")
    print(f"   Fine-tuned repo: {HF_FINETUNED_REPO}")
    print(f"   MLflow URI     : {MLFLOW_TRACKING_URI}")
    print(f"   Dataset dir    : {DATASET_DIR}")
    print(f"   Eval thresholds: ROUGE-L≥{EVAL_THRESHOLDS['min_rouge_l']} | "
          f"BLEU≥{EVAL_THRESHOLDS['min_bleu']} | "
          f"Hallucination≤{EVAL_THRESHOLDS['max_hallucination_rate']}")


if __name__ == "__main__":
    validate_config()