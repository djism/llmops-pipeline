import sys
import json
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_config_imports():
    from config import (
        HF_MODEL_ID, MLFLOW_EXPERIMENT_NAME,
        EVAL_THRESHOLDS, TRAINING_CONFIG
    )
    assert HF_MODEL_ID == "microsoft/Phi-3-mini-4k-instruct"
    assert EVAL_THRESHOLDS["min_rouge_l"] == 0.25
    assert EVAL_THRESHOLDS["min_bleu"] == 0.15
    assert EVAL_THRESHOLDS["max_hallucination_rate"] == 0.10
    assert TRAINING_CONFIG["lora_r"] == 16


def test_sample_dataset_loads():
    from src.data.dataset_builder import _get_sample_dataset
    samples = _get_sample_dataset()
    assert len(samples) == 10
    assert "instruction" in samples[0]
    assert "output" in samples[0]
    assert len(samples[0]["instruction"]) > 20


def test_format_for_training():
    from src.data.dataset_builder import format_for_training
    sample = {
        "instruction": "What is metformin used for?",
        "input": "",
        "output": "Metformin is used for type 2 diabetes."
    }
    formatted = format_for_training(sample)
    assert "text" in formatted
    assert "<|user|>" in formatted["text"]
    assert "<|assistant|>" in formatted["text"]
    assert "metformin" in formatted["text"].lower()


def test_dataset_split():
    from src.data.dataset_builder import split_dataset, _get_sample_dataset, format_for_training
    samples = [format_for_training(s) for s in _get_sample_dataset()]
    train, eval_set = split_dataset(samples, train_ratio=0.8)
    assert len(train) + len(eval_set) == len(samples)
    assert len(train) > len(eval_set)


def test_rouge_l_perfect_match():
    from src.evaluation.evaluator import compute_rouge_l
    text = "The first-line treatment is metformin combined with lifestyle changes."
    score = compute_rouge_l(text, text)
    assert score == 1.0


def test_rouge_l_no_match():
    from src.evaluation.evaluator import compute_rouge_l
    score = compute_rouge_l("apple orange banana", "completely different text here")
    assert score < 0.3


def test_bleu_perfect_match():
    from src.evaluation.evaluator import compute_bleu
    text = "Metformin is the first line treatment for diabetes."
    score = compute_bleu(text, text)
    assert score > 0.9


def test_hallucination_rate_zero():
    from src.evaluation.evaluator import compute_hallucination_rate
    generated = ["Metformin is used for type 2 diabetes management."]
    reference = ["Metformin is used for type 2 diabetes management."]
    rate = compute_hallucination_rate(generated, reference)
    assert rate == 0.0


def test_hallucination_rate_detected():
    from src.evaluation.evaluator import compute_hallucination_rate
    # Generated introduces a drug dosage not in reference
    generated = ["Give metformin 500mg twice daily for diabetes."]
    reference = ["Metformin is used for type 2 diabetes."]
    rate = compute_hallucination_rate(generated, reference)
    assert rate > 0.0


def test_evaluation_mock_mode():
    from src.evaluation.evaluator import run_evaluation
    from src.data.dataset_builder import _get_sample_dataset, format_for_training
    samples = [format_for_training(s) for s in _get_sample_dataset()[:5]]
    results = run_evaluation(
        eval_samples=samples,
        mock_mode=True,
        max_eval_samples=5
    )
    assert results["rouge_l"] == 1.0
    assert results["bleu"] == 1.0
    assert results["hallucination_rate"] == 0.0
    assert results["passed_gate"] is True


def test_eval_gate_pass():
    from src.evaluation.eval_gate import run_gate
    import tempfile, os
    results = {
        "rouge_l": 0.35,
        "bleu": 0.22,
        "hallucination_rate": 0.05,
        "passed_gate": True,
        "mock_mode": True,
        "sample_count": 10,
        "gate_details": {
            "rouge_l": {"value": 0.35, "threshold": 0.25, "passed": True},
            "bleu": {"value": 0.22, "threshold": 0.15, "passed": True},
            "hallucination": {"value": 0.05, "threshold": 0.10, "passed": True}
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        tmp_path = f.name
    try:
        passed = run_gate(results_file=tmp_path)
        assert passed is True
    finally:
        os.unlink(tmp_path)


def test_eval_gate_fail():
    from src.evaluation.eval_gate import run_gate
    import tempfile, os
    results = {
        "rouge_l": 0.10,
        "bleu": 0.05,
        "hallucination_rate": 0.25,
        "passed_gate": False,
        "mock_mode": True,
        "sample_count": 10,
        "gate_details": {
            "rouge_l": {"value": 0.10, "threshold": 0.25, "passed": False},
            "bleu": {"value": 0.05, "threshold": 0.15, "passed": False},
            "hallucination": {"value": 0.25, "threshold": 0.10, "passed": False}
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        tmp_path = f.name
    try:
        passed = run_gate(results_file=tmp_path)
        assert passed is False
    finally:
        os.unlink(tmp_path)


def test_prometheus_metrics():
    from src.monitoring.metrics import record_request, get_metrics_text
    record_request("/generate", latency_ms=350.0, success=True, tokens=80)
    metrics_text = get_metrics_text()
    assert "llmops_requests_total" in metrics_text
    assert "llmops_request_latency_ms" in metrics_text