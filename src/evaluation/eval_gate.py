"""
Evaluation Gate — CI/CD model promotion script.

This script is called by GitHub Actions after every push.
It runs the evaluation suite and exits with:
    0 → model passes all thresholds → pipeline continues → deploy
    1 → model fails any threshold  → pipeline stops → no deploy

WHY AN EVALUATION GATE?
------------------------
Without a gate, every code change could silently degrade model quality.
The gate enforces minimum quality standards as hard constraints — the same
way a test suite prevents broken code from reaching production.

This is the difference between a research pipeline and a production pipeline.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVAL_THRESHOLDS
from src.evaluation.evaluator import run_evaluation


def run_gate(
    results_file: str = None,
    mock_mode: bool = False,
    use_base_model: bool = False
) -> bool:
    """
    Runs evaluation gate.

    Args:
        results_file: Path to pre-computed eval_results.json
                      If provided, skips re-running evaluation.
        mock_mode: Run in mock mode (for CI without GPU)
        use_base_model: Use base model for baseline comparison

    Returns:
        True if gate passes, False if it fails.
        Exits with code 0 (pass) or 1 (fail) for CI.
    """
    print("\n" + "=" * 55)
    print("  LLMOps — Evaluation Gate")
    print("=" * 55)
    print(f"\n  Thresholds:")
    print(f"    ROUGE-L           ≥ {EVAL_THRESHOLDS['min_rouge_l']}")
    print(f"    BLEU              ≥ {EVAL_THRESHOLDS['min_bleu']}")
    print(f"    Hallucination Rate ≤ {EVAL_THRESHOLDS['max_hallucination_rate']}")

    # ── Load or run evaluation ────────────────────────────────────────────────
    if results_file and Path(results_file).exists():
        print(f"\n📂 Loading pre-computed results from {results_file}")
        with open(results_file) as f:
            results = json.load(f)
    else:
        print(f"\n🔄 Running evaluation...")
        results = run_evaluation(
            mock_mode=mock_mode,
            use_base_model=use_base_model,
            max_eval_samples=50
        )

    # ── Check gate ────────────────────────────────────────────────────────────
    passed = results.get("passed_gate", False)
    gate_details = results.get("gate_details", {})

    print(f"\n{'='*55}")
    print(f"  GATE DECISION")
    print(f"{'='*55}")

    all_checks = [
        ("ROUGE-L",
         gate_details.get("rouge_l", {}).get("value", 0),
         EVAL_THRESHOLDS["min_rouge_l"],
         gate_details.get("rouge_l", {}).get("passed", False),
         "≥"),
        ("BLEU",
         gate_details.get("bleu", {}).get("value", 0),
         EVAL_THRESHOLDS["min_bleu"],
         gate_details.get("bleu", {}).get("passed", False),
         "≥"),
        ("Hallucination Rate",
         gate_details.get("hallucination", {}).get("value", 1.0),
         EVAL_THRESHOLDS["max_hallucination_rate"],
         gate_details.get("hallucination", {}).get("passed", False),
         "≤"),
    ]

    for name, value, threshold, check_passed, op in all_checks:
        icon = "✅ PASS" if check_passed else "❌ FAIL"
        print(f"  {icon}  {name:<22}: {value:.4f} "
              f"(required {op}{threshold})")

    print()
    if passed:
        print(f"  ✅ GATE PASSED — Model approved for deployment")
        print(f"     Exiting with code 0")
    else:
        print(f"  ❌ GATE FAILED — Model blocked from deployment")
        print(f"     Exiting with code 1")
        print(f"\n  To fix: retrain with adjusted hyperparameters or more data")
        print(f"  Check MLflow UI for experiment comparison: mlflow ui")

    print(f"{'='*55}\n")

    # ── Save gate result ──────────────────────────────────────────────────────
    gate_result = {
        "passed": passed,
        "metrics": {
            "rouge_l": results.get("rouge_l"),
            "bleu": results.get("bleu"),
            "hallucination_rate": results.get("hallucination_rate"),
        },
        "thresholds": EVAL_THRESHOLDS,
        "sample_count": results.get("sample_count"),
        "mock_mode": results.get("mock_mode", False)
    }

    gate_path = Path("gate_result.json")
    with open(gate_path, "w") as f:
        json.dump(gate_result, f, indent=2)
    print(f"Gate result saved to: {gate_path}")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLMOps Evaluation Gate — runs metrics and blocks bad models"
    )
    parser.add_argument(
        "--results-file",
        help="Path to pre-computed eval_results.json"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock mode for CI without GPU"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Evaluate base model (for baseline comparison)"
    )
    args = parser.parse_args()

    passed = run_gate(
        results_file=args.results_file,
        mock_mode=args.mock,
        use_base_model=args.use_base_model
    )

    # Exit code for CI — 0 = pass, 1 = fail
    sys.exit(0 if passed else 1)