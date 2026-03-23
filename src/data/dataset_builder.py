import sys
import json
import random
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import DATASET_CONFIG, DATASET_DIR


def load_medqa_dataset(max_samples: int = None) -> list[dict]:
    """
    Loads the MedAlpaca medical QA dataset from HuggingFace.
    This is a curated medical question-answer dataset derived from
    medical board exam questions and clinical scenarios.

    Each sample has:
        - instruction: the medical question
        - input: additional context (often empty)
        - output: the correct answer with explanation
    """
    try:
        from datasets import load_dataset

        max_samples = max_samples or DATASET_CONFIG["max_samples"]

        print(f"📥 Loading medical QA dataset from HuggingFace...")
        print(f"   Dataset: {DATASET_CONFIG['hf_dataset']}")
        print(f"   Max samples: {max_samples}")

        dataset = load_dataset(
            DATASET_CONFIG["hf_dataset"],
            split="train"
        )

        print(f"   ✅ Loaded {len(dataset)} total samples")

        # Convert to list of dicts
        samples = []
        for item in dataset:
            sample = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", "")
            }
            # Filter out empty or very short samples
            if (len(sample["instruction"]) > 20 and
                    len(sample["output"]) > 20):
                samples.append(sample)

        # Shuffle and limit
        random.shuffle(samples)
        samples = samples[:max_samples]
        print(f"   ✅ {len(samples)} valid samples after filtering")
        return samples

    except Exception as e:
        print(f"   ⚠️  Could not load from HuggingFace: {e}")
        print(f"   Falling back to sample dataset...")
        return _get_sample_dataset()


def _get_sample_dataset() -> list[dict]:
    """
    Built-in sample medical QA dataset for testing without internet.
    Used as fallback and for unit tests.
    """
    return [
        {
            "instruction": "What is the first-line treatment for type 2 diabetes mellitus?",
            "input": "",
            "output": "The first-line treatment for type 2 diabetes mellitus is metformin, combined with lifestyle modifications including diet changes and increased physical activity. Metformin is preferred due to its efficacy, low cost, weight neutrality, and favorable cardiovascular profile."
        },
        {
            "instruction": "What are the classic symptoms of myocardial infarction?",
            "input": "",
            "output": "Classic symptoms of myocardial infarction include crushing chest pain radiating to the left arm or jaw, shortness of breath, diaphoresis (sweating), nausea, and vomiting. The pain typically lasts more than 20 minutes and is not relieved by nitroglycerin."
        },
        {
            "instruction": "What is the mechanism of action of beta-blockers?",
            "input": "",
            "output": "Beta-blockers competitively block catecholamines at beta-adrenergic receptors. Beta-1 blockade reduces heart rate and contractility, lowering cardiac output and blood pressure. This makes them useful in hypertension, angina, heart failure, and arrhythmias."
        },
        {
            "instruction": "How do you calculate creatinine clearance using the Cockcroft-Gault formula?",
            "input": "",
            "output": "The Cockcroft-Gault formula is: CrCl = [(140 - age) × weight in kg] / (72 × serum creatinine). For females, multiply the result by 0.85. This estimates glomerular filtration rate and is used for drug dosing adjustments in renal impairment."
        },
        {
            "instruction": "What is the treatment for anaphylaxis?",
            "input": "",
            "output": "The treatment for anaphylaxis is immediate intramuscular epinephrine (0.3-0.5 mg of 1:1000 solution) in the lateral thigh. Additional measures include IV fluids, antihistamines (diphenhydramine), corticosteroids (methylprednisolone), and supplemental oxygen. The patient should be monitored for biphasic reactions."
        },
        {
            "instruction": "What are the components of the Glasgow Coma Scale?",
            "input": "",
            "output": "The Glasgow Coma Scale (GCS) has three components: Eye opening (1-4 points), Verbal response (1-5 points), and Motor response (1-6 points). Maximum score is 15 (normal), minimum is 3 (deep coma). A score of 8 or less indicates severe brain injury requiring airway protection."
        },
        {
            "instruction": "What is the pathophysiology of septic shock?",
            "input": "",
            "output": "Septic shock involves uncontrolled systemic inflammation triggered by infection. Bacterial toxins activate immune cells releasing cytokines (TNF-alpha, IL-1, IL-6), causing vasodilation, increased vascular permeability, decreased systemic vascular resistance, and distributive shock. This leads to end-organ hypoperfusion despite elevated cardiac output."
        },
        {
            "instruction": "What are the diagnostic criteria for systemic lupus erythematosus?",
            "input": "",
            "output": "SLE diagnosis requires meeting 4 of 11 ACR criteria: malar rash, discoid rash, photosensitivity, oral ulcers, arthritis, serositis, renal disorder (proteinuria >0.5g/day), neurological disorder, hematologic disorder, immunologic disorder (anti-dsDNA, anti-Sm, antiphospholipid antibodies), and positive ANA."
        },
        {
            "instruction": "How does heparin work as an anticoagulant?",
            "input": "",
            "output": "Heparin works by binding to antithrombin III, dramatically increasing its ability to inhibit thrombin and Factor Xa. This prevents conversion of fibrinogen to fibrin and stops clot propagation. Unfractionated heparin inhibits both thrombin and Xa, while low molecular weight heparins preferentially inhibit Factor Xa."
        },
        {
            "instruction": "What is the treatment protocol for STEMI?",
            "input": "",
            "output": "STEMI treatment follows a door-to-balloon time goal of <90 minutes. Primary PCI is preferred. If PCI unavailable within 120 minutes, fibrinolysis with tPA or streptokinase is used. Adjunct therapy includes aspirin, P2Y12 inhibitor (ticagrelor or clopidogrel), anticoagulation, and beta-blockers. ICU monitoring is essential post-intervention."
        }
    ]


def format_for_training(sample: dict) -> dict:
    """
    Formats a raw sample into the instruction-following format
    used for Phi-3 fine-tuning.

    Phi-3 uses a specific chat template:
    <|user|> question <|end|> <|assistant|> answer <|end|>
    """
    instruction = sample.get("instruction", "")
    context = sample.get("input", "")
    answer = sample.get("output", "")

    if context:
        user_content = f"{instruction}\n\nContext: {context}"
    else:
        user_content = instruction

    formatted = {
        "text": f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n{answer}<|end|>",
        "instruction": instruction,
        "input": context,
        "output": answer
    }
    return formatted


def split_dataset(
    samples: list[dict],
    train_ratio: float = None,
) -> tuple[list[dict], list[dict]]:
    """
    Splits samples into train and evaluation sets.
    Returns (train_samples, eval_samples)
    """
    train_ratio = train_ratio or DATASET_CONFIG["train_split"]
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    train = samples[:split_idx]
    eval_set = samples[split_idx:]

    return train, eval_set


def build_and_save_dataset(max_samples: int = None) -> dict:
    """
    Full dataset pipeline:
    1. Load from HuggingFace (or fallback)
    2. Format for Phi-3 training
    3. Split into train/eval
    4. Save to disk
    5. Save a small sample for quick testing

    Returns stats dict.
    """
    print("\n" + "=" * 55)
    print("  LLMOps — Dataset Builder")
    print("=" * 55)

    # Load raw data
    raw_samples = load_medqa_dataset(max_samples)

    # Format for training
    print(f"\n🔄 Formatting {len(raw_samples)} samples for Phi-3...")
    formatted = [format_for_training(s) for s in raw_samples]
    print(f"   ✅ Formatting complete")

    # Split
    print(f"\n✂️  Splitting dataset...")
    train, eval_set = split_dataset(formatted)
    print(f"   Train : {len(train)} samples")
    print(f"   Eval  : {len(eval_set)} samples")

    # Save
    print(f"\n💾 Saving to disk...")

    train_path = Path(DATASET_CONFIG["train_file"])
    eval_path = Path(DATASET_CONFIG["eval_file"])
    sample_path = Path(DATASET_CONFIG["sample_file"])

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    print(f"   ✅ Train saved: {train_path}")

    with open(eval_path, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"   ✅ Eval saved: {eval_path}")

    # Save 10 samples for quick testing
    with open(sample_path, "w") as f:
        json.dump(formatted[:10], f, indent=2)
    print(f"   ✅ Sample saved: {sample_path}")

    stats = {
        "total": len(formatted),
        "train": len(train),
        "eval": len(eval_set),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "sample_file": str(sample_path)
    }

    print(f"\n✅ Dataset ready!")
    return stats


if __name__ == "__main__":
    print("Testing Dataset Builder...\n")

    # Test with sample data first (no internet needed)
    print("TEST 1: Sample dataset (no internet)")
    samples = _get_sample_dataset()
    print(f"   Sample count: {len(samples)}")
    print(f"   First question: {samples[0]['instruction'][:60]}...")

    print("\nTEST 2: Format for training")
    formatted = format_for_training(samples[0])
    print(f"   Formatted text preview:")
    print(f"   {formatted['text'][:200]}...")

    print("\nTEST 3: Dataset split")
    train, eval_set = split_dataset(samples)
    print(f"   Train: {len(train)} | Eval: {len(eval_set)}")

    print("\nTEST 4: Build and save (using sample data)")
    # Use small sample for test
    import random
    random.seed(42)
    formatted_all = [format_for_training(s) for s in samples]
    train_s, eval_s = split_dataset(formatted_all)

    sample_path = DATASET_DIR / "sample.json"
    with open(sample_path, "w") as f:
        json.dump(formatted_all[:5], f, indent=2)
    print(f"   ✅ Sample saved to {sample_path}")

    print("\nTEST 5: Try loading from HuggingFace (requires internet)")
    try:
        hf_samples = load_medqa_dataset(max_samples=20)
        print(f"   ✅ Loaded {len(hf_samples)} from HuggingFace")
    except Exception as e:
        print(f"   ⚠️  HuggingFace load skipped: {e}")

    print("\n✅ Dataset Builder working correctly!")