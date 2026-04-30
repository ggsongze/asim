#!/usr/bin/env python3
"""SFT: Distill PPO actions into Qwen3-8B via LoRA fine-tuning.

Trains the LLM to output PPO-quality setpoints given knot-style prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/songze/asim")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if SHARED_SITE_PACKAGES.exists() and str(SHARED_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(SHARED_SITE_PACKAGES))

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str,
                        default=str(PROJECT_ROOT / "result/gspo/ppo_sft_dataset.jsonl"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "result/gspo/qwen3_sft_ppo_distill"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset

    print(f"Loading base model: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}", flush=True)
    raw_data = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))
    print(f"  {len(raw_data)} samples loaded", flush=True)

    # Build training samples: chat format -> tokenized
    class SFTDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.samples = []
            for item in data:
                messages = [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": "/no_think\n" + item["user_prompt"]},
                    {"role": "assistant", "content": item["output"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                encoded = tokenizer(
                    text, truncation=True, max_length=max_length,
                    padding="max_length", return_tensors="pt",
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)

                # Build labels: mask everything except assistant response
                labels = input_ids.clone()
                # Find assistant response start
                assistant_text = item["output"]
                assistant_tokens = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
                # Mask all tokens before the last len(assistant_tokens) tokens
                response_len = len(assistant_tokens)
                labels[:-response_len] = -100
                labels[attention_mask == 0] = -100

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    print("Tokenizing dataset...", flush=True)
    dataset = SFTDataset(raw_data, tokenizer, args.max_seq_length)
    print(f"  {len(dataset)} tokenized samples", flush=True)

    # Split train/val
    val_size = min(500, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)

    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        report_to="wandb",
        run_name="qwen3_sft_ppo_distill",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting SFT training...", flush=True)
    trainer.train()

    # Save final
    print(f"Saving to {output_dir / 'final'}", flush=True)
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
