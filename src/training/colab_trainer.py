"""
Generates a Google Colab notebook for fine-tuning Phi-3-mini.
Run this script locally to produce the .ipynb file,
then upload it to Colab and run with a free T4 GPU.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TRAINING_CONFIG, HF_FINETUNED_REPO


def generate_colab_notebook() -> str:
    """Generates a complete Colab notebook for fine-tuning."""

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# LLMOps Pipeline — Phi-3-mini Medical QA Fine-tuning\n",
                    "\n",
                    "**Runtime**: Make sure you have **T4 GPU** selected\n",
                    "Runtime → Change runtime type → T4 GPU\n",
                    "\n",
                    "This notebook fine-tunes `microsoft/Phi-3-mini-4k-instruct` on medical QA data using **LoRA (QLoRA)**."
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 1: Install dependencies\n",
                    "!pip install -q transformers peft trl bitsandbytes accelerate datasets mlflow huggingface-hub evaluate rouge-score nltk"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 2: Configuration\n",
                    "import os\n",
                    "from google.colab import userdata\n",
                    "\n",
                    "# Set your HuggingFace token as a Colab secret named HF_TOKEN\n",
                    "HF_TOKEN = userdata.get('HF_TOKEN')\n",
                    f"BASE_MODEL = '{TRAINING_CONFIG['model_id']}'\n",
                    f"FINETUNED_REPO = '{HF_FINETUNED_REPO}'\n",
                    f"LORA_R = {TRAINING_CONFIG['lora_r']}\n",
                    f"LORA_ALPHA = {TRAINING_CONFIG['lora_alpha']}\n",
                    f"NUM_EPOCHS = {TRAINING_CONFIG['num_train_epochs']}\n",
                    f"LEARNING_RATE = {TRAINING_CONFIG['learning_rate']}\n",
                    f"MAX_SEQ_LENGTH = {TRAINING_CONFIG['max_seq_length']}\n",
                    "\n",
                    "print(f'Base model: {BASE_MODEL}')\n",
                    "print(f'Fine-tuned repo: {FINETUNED_REPO}')"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 3: Load and prepare dataset\n",
                    "from datasets import load_dataset\n",
                    "import json, random\n",
                    "\n",
                    "print('Loading medical QA dataset...')\n",
                    "dataset = load_dataset('medalpaca/medical_meadow_medqa', split='train')\n",
                    "print(f'Loaded {len(dataset)} samples')\n",
                    "\n",
                    "def format_sample(sample):\n",
                    "    instruction = sample.get('instruction', '')\n",
                    "    output = sample.get('output', '')\n",
                    "    return {'text': f'<|user|>\\n{instruction}<|end|>\\n<|assistant|>\\n{output}<|end|>'}\n",
                    "\n",
                    "# Filter and format\n",
                    "samples = [s for s in dataset if len(s.get('instruction','')) > 20 and len(s.get('output','')) > 20]\n",
                    "random.shuffle(samples)\n",
                    "samples = samples[:2000]  # Keep manageable\n",
                    "\n",
                    "from datasets import Dataset\n",
                    "formatted = Dataset.from_list([format_sample(s) for s in samples])\n",
                    "\n",
                    "# Split 85/15\n",
                    "split = formatted.train_test_split(test_size=0.15, seed=42)\n",
                    "train_dataset = split['train']\n",
                    "eval_dataset = split['test']\n",
                    "\n",
                    "print(f'Train: {len(train_dataset)} | Eval: {len(eval_dataset)}')"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 4: Load model with 4-bit quantization (QLoRA)\n",
                    "import torch\n",
                    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
                    "\n",
                    "bnb_config = BitsAndBytesConfig(\n",
                    "    load_in_4bit=True,\n",
                    "    bnb_4bit_use_double_quant=True,\n",
                    "    bnb_4bit_quant_type='nf4',\n",
                    "    bnb_4bit_compute_dtype=torch.bfloat16\n",
                    ")\n",
                    "\n",
                    "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)\n",
                    "tokenizer.pad_token = tokenizer.eos_token\n",
                    "tokenizer.padding_side = 'right'\n",
                    "\n",
                    "model = AutoModelForCausalLM.from_pretrained(\n",
                    "    BASE_MODEL,\n",
                    "    quantization_config=bnb_config,\n",
                    "    device_map='auto',\n",
                    "    token=HF_TOKEN\n",
                    ")\n",
                    "print('Base model loaded!')"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 5: Apply LoRA\n",
                    "from peft import LoraConfig, get_peft_model, TaskType\n",
                    "\n",
                    "lora_config = LoraConfig(\n",
                    "    r=LORA_R,\n",
                    "    lora_alpha=LORA_ALPHA,\n",
                    f"    lora_dropout={TRAINING_CONFIG['lora_dropout']},\n",
                    f"    target_modules={TRAINING_CONFIG['target_modules']},\n",
                    "    task_type=TaskType.CAUSAL_LM,\n",
                    "    bias='none'\n",
                    ")\n",
                    "\n",
                    "model = get_peft_model(model, lora_config)\n",
                    "model.print_trainable_parameters()\n",
                    "print('LoRA applied!')"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 6: Train\n",
                    "from transformers import TrainingArguments\n",
                    "from trl import SFTTrainer\n",
                    "\n",
                    "training_args = TrainingArguments(\n",
                    "    output_dir='./phi3-medical-qa',\n",
                    "    num_train_epochs=NUM_EPOCHS,\n",
                    "    per_device_train_batch_size=4,\n",
                    "    gradient_accumulation_steps=4,\n",
                    "    learning_rate=LEARNING_RATE,\n",
                    "    warmup_ratio=0.03,\n",
                    "    evaluation_strategy='epoch',\n",
                    "    save_strategy='epoch',\n",
                    "    load_best_model_at_end=True,\n",
                    "    fp16=True,\n",
                    "    logging_steps=10,\n",
                    "    report_to='none',\n",
                    ")\n",
                    "\n",
                    "trainer = SFTTrainer(\n",
                    "    model=model,\n",
                    "    args=training_args,\n",
                    "    train_dataset=train_dataset,\n",
                    "    eval_dataset=eval_dataset,\n",
                    "    tokenizer=tokenizer,\n",
                    "    dataset_text_field='text',\n",
                    "    max_seq_length=MAX_SEQ_LENGTH,\n",
                    ")\n",
                    "\n",
                    "print('Starting training...')\n",
                    "trainer.train()\n",
                    "print('Training complete!')"
                ],
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Cell 7: Save and push to HuggingFace Hub\n",
                    "trainer.save_model('./phi3-medical-qa')\n",
                    "tokenizer.save_pretrained('./phi3-medical-qa')\n",
                    "\n",
                    "model.push_to_hub(FINETUNED_REPO, token=HF_TOKEN)\n",
                    "tokenizer.push_to_hub(FINETUNED_REPO, token=HF_TOKEN)\n",
                    "\n",
                    "print(f'Model pushed to: https://huggingface.co/{FINETUNED_REPO}')"
                ],
                "outputs": [],
                "execution_count": None
            }
        ]
    }

    output_path = Path("phi3_medical_qa_finetuning.ipynb")
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"✅ Colab notebook generated: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Go to colab.research.google.com")
    print(f"  2. File → Upload notebook → select {output_path}")
    print(f"  3. Runtime → Change runtime type → T4 GPU")
    print(f"  4. Add HF_TOKEN to Colab Secrets (key icon in left sidebar)")
    print(f"  5. Runtime → Run all")
    print(f"  6. Model will be pushed to: huggingface.co/{HF_FINETUNED_REPO}")

    return str(output_path)


if __name__ == "__main__":
    print("Generating Colab notebook for Phi-3-mini fine-tuning...\n")
    path = generate_colab_notebook()
    print(f"\n✅ Done! Upload {path} to Google Colab to start training.")