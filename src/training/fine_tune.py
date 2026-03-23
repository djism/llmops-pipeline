"""
Fine-tuning script for Phi-3-mini using LoRA (Low-Rank Adaptation).

WHY LoRA?
---------
Full fine-tuning a 3.8B parameter model requires ~15GB VRAM and hours of compute.
LoRA fine-tunes only a small set of adapter weights (~1% of total parameters)
injected into the attention layers — same quality improvement, fraction of the cost.
This runs on a free Google Colab T4 GPU in under an hour.

HOW IT WORKS:
    1. Load base model in 4-bit quantization (QLoRA) to fit in GPU memory
    2. Inject LoRA adapters into Q, K, V, O projection layers
    3. Train only the adapter weights — base model stays frozen
    4. Save adapters separately — can be merged with base model later
    5. Push adapters to HuggingFace Hub for serving

This script is designed to run on Google Colab.
Run locally only if you have a GPU with 8GB+ VRAM.
"""

import sys
import json
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    TRAINING_CONFIG, DATASET_CONFIG, HF_TOKEN,
    HF_FINETUNED_REPO, MLFLOW_TRACKING_URI
)


def check_gpu():
    """Check if GPU is available for training."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {gpu_name} ({vram:.1f}GB VRAM)")
            return True
        else:
            print("⚠️  No GPU detected — training will be very slow on CPU")
            print("   Recommended: Run on Google Colab (free T4 GPU)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed with CUDA support")
        return False


def load_training_data() -> tuple:
    """Loads train and eval datasets from disk."""
    train_path = Path(DATASET_CONFIG["train_file"])
    eval_path = Path(DATASET_CONFIG["eval_file"])

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}\n"
            "Run: python src/data/dataset_builder.py first"
        )

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    print(f"✅ Loaded {len(train_data)} train + {len(eval_data)} eval samples")
    return train_data, eval_data


def run_finetuning():
    """
    Full LoRA fine-tuning pipeline.
    Requires GPU — designed for Google Colab T4.
    """
    print("\n" + "=" * 55)
    print("  LLMOps — LoRA Fine-tuning")
    print("=" * 55)

    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("\n💡 To run fine-tuning:")
        print("   1. Open Google Colab: colab.research.google.com")
        print("   2. Runtime → Change runtime type → T4 GPU")
        print("   3. Upload this script or use the Colab notebook")
        print("   4. Run the cells in order")
        print("\n   Alternatively run: python src/training/colab_trainer.py")
        print("   to generate the ready-to-use Colab notebook")
        return None

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        import torch
        import mlflow
    except ImportError as e:
        print(f"❌ Missing training dependency: {e}")
        print("   Install: pip install -r requirements_training.txt")
        return None

    from src.training.mlflow_logger import MLflowLogger
    from src.data.dataset_builder import build_and_save_dataset
    from datasets import Dataset

    # ── Step 1: Prepare dataset ───────────────────────────────────────────────
    print("\n📂 Step 1: Loading dataset...")
    train_data, eval_data = load_training_data()

    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # ── Step 2: Load model in 4-bit (QLoRA) ──────────────────────────────────
    print(f"\n🤖 Step 2: Loading base model ({TRAINING_CONFIG['model_id']})...")
    print("   Using 4-bit quantization (QLoRA) to fit in GPU memory...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TRAINING_CONFIG["model_id"],
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )
    print(f"   ✅ Base model loaded")

    # ── Step 3: Configure LoRA ────────────────────────────────────────────────
    print(f"\n🔧 Step 3: Configuring LoRA adapters...")
    print(f"   Rank (r)        : {TRAINING_CONFIG['lora_r']}")
    print(f"   Alpha           : {TRAINING_CONFIG['lora_alpha']}")
    print(f"   Target modules  : {TRAINING_CONFIG['target_modules']}")
    print(f"   Trainable params: ~1% of total model parameters")

    lora_config = LoraConfig(
        r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        target_modules=TRAINING_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Step 4: Training arguments ────────────────────────────────────────────
    print(f"\n⚙️  Step 4: Setting training arguments...")
    output_dir = TRAINING_CONFIG["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",           # we handle MLflow manually
        fp16=True,
        dataloader_pin_memory=False,
    )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    print(f"\n🏋️  Step 5: Training...")

    with MLflowLogger(run_name=f"lora_finetune_{TRAINING_CONFIG['lora_r']}r") as logger:
        logger.log_training_config()
        logger.log_dataset_stats({
            "total": len(train_data) + len(eval_data),
            "train": len(train_data),
            "eval": len(eval_data)
        })

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=TRAINING_CONFIG["max_seq_length"],
            packing=False,
        )

        print("   Starting training loop...")
        trainer.train()

        # Log epoch metrics from trainer history
        for i, log in enumerate(trainer.state.log_history):
            if "loss" in log and "eval_loss" in log:
                logger.log_epoch_metrics(
                    epoch=i + 1,
                    train_loss=log["loss"],
                    eval_loss=log["eval_loss"]
                )

        # ── Step 6: Save model ────────────────────────────────────────────────
        print(f"\n💾 Step 6: Saving LoRA adapters...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"   ✅ Adapters saved to: {output_dir}")
        logger.log_model_artifact(output_dir)

        # ── Step 7: Push to HuggingFace Hub ───────────────────────────────────
        print(f"\n🚀 Step 7: Pushing to HuggingFace Hub...")
        model.push_to_hub(HF_FINETUNED_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_FINETUNED_REPO, token=HF_TOKEN)
        print(f"   ✅ Model pushed to: https://huggingface.co/{HF_FINETUNED_REPO}")
        logger.log_model_pushed(HF_FINETUNED_REPO)

    print("\n✅ Fine-tuning complete!")
    return output_dir


if __name__ == "__main__":
    print("LLMOps — Fine-tuning Script\n")
    print("This script runs LoRA fine-tuning of Phi-3-mini on medical QA data.")
    print("Requires GPU — designed for Google Colab T4 (free).\n")

    check_gpu()

    print("\nTo run fine-tuning:")
    print("  Option 1: python src/training/fine_tune.py  (if you have GPU)")
    print("  Option 2: Use Google Colab (see colab_trainer.py for notebook)")
    print("\nTo skip fine-tuning and use base model for evaluation:")
    print("  python src/evaluation/evaluator.py --use-base-model")