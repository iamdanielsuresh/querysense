#!/usr/bin/env python3
"""Train Qwen-style models with QLoRA for Text2SQL.

Examples:
1) Stage 1 (new adapter)
   python finetuning/train_qlora.py \
     --model-name Qwen/Qwen3-8B \
     --train-file finetuning/data/stage1_train.jsonl \
     --eval-file finetuning/data/stage1_valid.jsonl \
     --output-dir finetuning/runs/qwen3-8b-stage1

2) Stage 2 (continue from existing adapter)
   python finetuning/train_qlora.py \
     --model-name Qwen/Qwen3-8B \
     --adapter-path finetuning/runs/qwen3-8b-stage1 \
     --train-file finetuning/data/stage2_mixed_train.jsonl \
     --eval-file finetuning/data/stage2_mixed_valid.jsonl \
     --output-dir finetuning/runs/qwen3-8b-stage2
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA trainer for Text2SQL")

    # I/O
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional existing LoRA adapter path to continue training (stage 2)",
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    # Training knobs
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)

    parser.add_argument("--max-seq-length", type=int, default=3072)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)

    # LoRA knobs
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")

    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are an expert Text-to-SQL system. "
            "Generate one executable SQL query that answers the user's question. "
            "Return SQL only."
        ),
    )

    return parser.parse_args()


def build_user_prompt(example: Dict) -> str:
    parts = [
        f"Dialect: {example.get('dialect', 'unknown')}",
        f"Database: {example.get('db_id', 'unknown')}",
        "Question:",
        example.get("question", "").strip(),
    ]

    schema = (example.get("schema_context") or "").strip()
    if schema:
        parts.extend(["Schema context:", schema])

    external = (example.get("external_knowledge") or "").strip()
    if external:
        parts.extend(["External knowledge:", external])

    parts.extend(
        [
            "Rules:",
            "- Output only SQL.",
            "- Use only tables/columns implied by the context.",
            "- Keep SQL syntactically valid for the given dialect.",
        ]
    )
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if args.eval_file and not args.eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    print("=" * 80)
    print("QLoRA Training Configuration")
    print("=" * 80)
    print(f"Model              : {args.model_name}")
    print(f"Adapter path       : {args.adapter_path}")
    print(f"Train file         : {args.train_file}")
    print(f"Eval file          : {args.eval_file}")
    print(f"Output dir         : {args.output_dir}")
    print(f"Epochs             : {args.num_train_epochs}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"Batch size (dev)   : {args.per_device_train_batch_size}")
    print(f"Grad accumulation  : {args.gradient_accumulation_steps}")
    print(f"Max seq length     : {args.max_seq_length}")
    print(f"bf16               : {args.bf16}")

    data_files = {"train": str(args.train_file)}
    if args.eval_file:
        data_files["validation"] = str(args.eval_file)

    raw_ds = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(example: Dict) -> Dict[str, str]:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": build_user_prompt(example)},
            {"role": "assistant", "content": (example.get("sql", "") or "").strip()},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = raw_ds["train"].map(to_text, remove_columns=raw_ds["train"].column_names)
    eval_ds: Optional[object] = None
    if "validation" in raw_ds:
        eval_ds = raw_ds["validation"].map(to_text, remove_columns=raw_ds["validation"].column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = None
    if args.adapter_path:
        if not args.adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            str(args.adapter_path),
            is_trainable=True,
        )
        print(f"Loaded existing adapter for continuation: {args.adapter_path}")
    else:
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"Creating new adapter with target_modules={target_modules}")

    has_eval = eval_ds is not None and len(eval_ds) > 0

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=bool(has_eval),
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file) if args.eval_file else None,
        "output_dir": str(args.output_dir),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 80)
    print("Training complete")
    print(f"Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
