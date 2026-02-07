#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model for standalone inference/export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter_path}")
    peft_model = PeftModel.from_pretrained(base, str(args.adapter_path))

    print("Merging adapter into base weights...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged.save_pretrained(str(args.output_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_path), use_fast=True, trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output_dir))

    print("Done.")


if __name__ == "__main__":
    main()
