import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, TRAINING_CONFIG

import mlflow
import mlflow.pytorch


def setup_mlflow() -> None:
    """
    Configures MLflow tracking URI and experiment.
    Call once before any logging.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"✅ MLflow configured")
    print(f"   Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"   Experiment   : {MLFLOW_EXPERIMENT_NAME}")


class MLflowLogger:
    """
    Wraps MLflow for clean experiment tracking throughout
    the LLMOps pipeline.

    Tracks:
    - Training hyperparameters (LoRA config, learning rate, etc.)
    - Training metrics (loss per epoch)
    - Evaluation metrics (ROUGE-L, BLEU, hallucination rate)
    - Model artifacts (adapter weights path)
    - Dataset stats (sample counts, split sizes)
    - Run metadata (duration, model ID, status)
    """

    def __init__(self, run_name: Optional[str] = None):
        setup_mlflow()
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = None
        self.start_time = None

    def start_run(self) -> None:
        """Starts a new MLflow run."""
        self.run = mlflow.start_run(run_name=self.run_name)
        self.start_time = time.time()
        print(f"\n🚀 MLflow run started: {self.run_name}")
        print(f"   Run ID: {self.run.info.run_id}")

    def log_training_config(self) -> None:
        """Logs all training hyperparameters."""
        mlflow.log_params({
            "base_model": TRAINING_CONFIG["model_id"],
            "lora_r": TRAINING_CONFIG["lora_r"],
            "lora_alpha": TRAINING_CONFIG["lora_alpha"],
            "lora_dropout": TRAINING_CONFIG["lora_dropout"],
            "target_modules": str(TRAINING_CONFIG["target_modules"]),
            "num_epochs": TRAINING_CONFIG["num_train_epochs"],
            "batch_size": TRAINING_CONFIG["per_device_train_batch_size"],
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "warmup_ratio": TRAINING_CONFIG["warmup_ratio"],
            "max_seq_length": TRAINING_CONFIG["max_seq_length"],
        })
        print(f"   ✅ Training config logged to MLflow")

    def log_dataset_stats(self, stats: dict) -> None:
        """Logs dataset statistics."""
        mlflow.log_params({
            "dataset_total": stats.get("total", 0),
            "dataset_train": stats.get("train", 0),
            "dataset_eval": stats.get("eval", 0),
            "dataset_source": stats.get("source", "medalpaca/medical_meadow_medqa"),
        })
        print(f"   ✅ Dataset stats logged to MLflow")

    def log_training_step(self, step: int, loss: float, lr: float) -> None:
        """Logs metrics for a single training step."""
        mlflow.log_metrics({
            "train_loss": loss,
            "learning_rate": lr,
        }, step=step)

    def log_epoch_metrics(self, epoch: int, train_loss: float,
                          eval_loss: Optional[float] = None) -> None:
        """Logs metrics at end of each epoch."""
        metrics = {"epoch_train_loss": train_loss}
        if eval_loss is not None:
            metrics["epoch_eval_loss"] = eval_loss

        mlflow.log_metrics(metrics, step=epoch)
        print(f"   Epoch {epoch}: train_loss={train_loss:.4f}"
              + (f" | eval_loss={eval_loss:.4f}" if eval_loss else ""))

    def log_evaluation_results(self, results: dict) -> None:
        """
        Logs final evaluation metrics.
        This is the most important log — determines if model gets promoted.
        """
        eval_metrics = {
            "eval_rouge_l": results.get("rouge_l", 0),
            "eval_bleu": results.get("bleu", 0),
            "eval_hallucination_rate": results.get("hallucination_rate", 1.0),
            "eval_avg_response_length": results.get("avg_response_length", 0),
            "eval_sample_count": results.get("sample_count", 0),
        }
        mlflow.log_metrics(eval_metrics)

        # Log pass/fail status
        passed = results.get("passed_gate", False)
        mlflow.log_param("eval_gate_passed", str(passed))

        print(f"\n📊 Evaluation results logged to MLflow:")
        print(f"   ROUGE-L            : {results.get('rouge_l', 0):.4f}")
        print(f"   BLEU               : {results.get('bleu', 0):.4f}")
        print(f"   Hallucination rate : {results.get('hallucination_rate', 1.0):.4f}")
        print(f"   Gate passed        : {passed}")

    def log_model_artifact(self, model_path: str) -> None:
        """Logs the fine-tuned model adapter as an artifact."""
        mlflow.log_artifact(model_path)
        mlflow.log_param("model_artifact_path", model_path)
        print(f"   ✅ Model artifact logged: {model_path}")

    def log_model_pushed(self, hf_repo: str) -> None:
        """Records that model was pushed to HuggingFace Hub."""
        mlflow.log_param("hf_repo", hf_repo)
        mlflow.log_param("model_status", "pushed_to_hub")
        print(f"   ✅ Model push recorded: {hf_repo}")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        Ends the MLflow run.
        Status: FINISHED, FAILED, KILLED
        """
        if self.start_time:
            duration = round(time.time() - self.start_time, 1)
            mlflow.log_metric("total_duration_seconds", duration)
            print(f"\n⏱️  Run duration: {duration}s")

        mlflow.end_run(status=status)
        print(f"✅ MLflow run ended: {status}")

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)


def get_best_run() -> Optional[dict]:
    """
    Returns the best run from the experiment based on ROUGE-L score.
    Used to compare new runs against historical best.
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.eval_rouge_l > 0",
            order_by=["metrics.eval_rouge_l DESC"],
            max_results=1
        )

        if not runs:
            return None

        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "run_name": best.info.run_name,
            "rouge_l": best.data.metrics.get("eval_rouge_l", 0),
            "bleu": best.data.metrics.get("eval_bleu", 0),
            "hallucination_rate": best.data.metrics.get("eval_hallucination_rate", 1.0),
        }
    except Exception as e:
        print(f"⚠️  Could not fetch best run: {e}")
        return None


if __name__ == "__main__":
    print("Testing MLflow Logger...\n")

    # Test full logging flow
    print("=" * 55)
    print("TEST 1: Full run logging")
    print("=" * 55)

    with MLflowLogger(run_name="test_run_001") as logger:
        # Log config
        logger.log_training_config()

        # Log dataset
        logger.log_dataset_stats({
            "total": 2000,
            "train": 1700,
            "eval": 300,
            "source": "medalpaca/medical_meadow_medqa"
        })

        # Simulate training steps
        import math
        for epoch in range(1, 4):
            # Simulate decreasing loss
            train_loss = 2.5 * math.exp(-0.3 * epoch)
            eval_loss = 2.8 * math.exp(-0.25 * epoch)
            logger.log_epoch_metrics(epoch, train_loss, eval_loss)

        # Log evaluation results
        logger.log_evaluation_results({
            "rouge_l": 0.312,
            "bleu": 0.198,
            "hallucination_rate": 0.06,
            "avg_response_length": 87,
            "sample_count": 300,
            "passed_gate": True
        })

    print("\n" + "=" * 55)
    print("TEST 2: Fetch best run")
    print("=" * 55)
    best = get_best_run()
    if best:
        print(f"   Best run: {best['run_name']}")
        print(f"   ROUGE-L : {best['rouge_l']:.4f}")
        print(f"   BLEU    : {best['bleu']:.4f}")
    else:
        print("   No previous runs found")

    print("\n" + "=" * 55)
    print("TEST 3: View in MLflow UI")
    print("=" * 55)
    print(f"   Run: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"   Then open: http://localhost:5000")

    print("\n✅ MLflow Logger working correctly!")