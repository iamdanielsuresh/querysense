#!/usr/bin/env python3
"""Run single-question inference for a fine-tuned Text2SQL model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_optional_text(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def build_user_prompt(
    question: str,
    dialect: str,
    db_id: str,
    schema_context: str,
    external_knowledge: str,
) -> str:
    parts = [
        f"Dialect: {dialect}",
        f"Database: {db_id}",
        "Question:",
        question.strip(),
    ]

    if schema_context.strip():
        parts.extend(["Schema context:", schema_context.strip()])

    if external_knowledge.strip():
        parts.extend(["External knowledge:", external_knowledge.strip()])

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
    parser = argparse.ArgumentParser(description="Single-prompt inference for fine-tuned Text2SQL model")
    parser.add_argument("--model-path", type=str, required=True, help="Base model, adapter folder, or merged model")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--dialect", type=str, default="sqlite")
    parser.add_argument("--db-id", type=str, default="unknown_db")
    parser.add_argument("--schema-file", type=Path, default=None, help="Optional text file with schema context")
    parser.add_argument("--external-knowledge-file", type=Path, default=None, help="Optional text file")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are an expert Text-to-SQL system. "
            "Generate one executable SQL query that answers the user's question. "
            "Return SQL only."
        ),
    )
    args = parser.parse_args()

    schema_context = load_optional_text(args.schema_file)
    external_knowledge = load_optional_text(args.external_knowledge_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [
        {"role": "system", "content": args.system_prompt},
        {
            "role": "user",
            "content": build_user_prompt(
                question=args.question,
                dialect=args.dialect,
                db_id=args.db_id,
                schema_context=schema_context,
                external_knowledge=external_knowledge,
            ),
        },
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[0][prompt_len:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print("=" * 80)
    print("Generated SQL")
    print("=" * 80)
    print(sql)


if __name__ == "__main__":
    main()
