#!/usr/bin/env python3
"""Prepare supervised fine-tuning datasets for Text2SQL QLoRA.

Outputs JSONL files under finetuning/data/ with fields:
- id, source, dialect, db_id, question, schema_context, external_knowledge, sql
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_split(key: str, val_ratio: float) -> str:
    """Deterministic train/valid split from hash(key)."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    threshold = int(val_ratio * 10000)
    return "valid" if bucket < threshold else "train"


def trim_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_spider1_schema_map(tables_path: Path, max_schema_chars: int) -> Dict[str, str]:
    schemas = read_json(tables_path)
    out: Dict[str, str] = {}

    for db in schemas:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        col_names = db["column_names_original"]
        col_types = db["column_types"]

        by_table: Dict[str, List[str]] = {t: [] for t in table_names}
        for idx, (table_idx, col_name) in enumerate(col_names):
            if table_idx == -1:
                continue
            table = table_names[table_idx]
            col_type = col_types[idx] if idx < len(col_types) else "text"
            by_table[table].append(f"{col_name} ({col_type})")

        lines = []
        for table in table_names:
            cols = by_table.get(table, [])
            lines.append(f"{table}: " + ", ".join(cols))

        out[db_id] = trim_text("\n".join(lines), max_schema_chars)

    return out


def prepare_spider1(
    spider1_dir: Path,
    max_schema_chars: int,
    val_ratio: float,
) -> Tuple[List[dict], List[dict]]:
    schema_map = build_spider1_schema_map(spider1_dir / "tables.json", max_schema_chars)

    rows: List[dict] = []
    train_files = [
        ("train_spider", spider1_dir / "train_spider.json"),
        ("train_others", spider1_dir / "train_others.json"),
    ]

    for split_name, path in train_files:
        examples = read_json(path)
        for idx, ex in enumerate(examples):
            db_id = ex["db_id"]
            question = ex["question"].strip()
            sql = ex["query"].strip()
            schema_context = schema_map.get(db_id, "")
            rows.append(
                {
                    "id": f"spider1::{split_name}::{idx}",
                    "source": "spider1",
                    "dialect": "sqlite",
                    "db_id": db_id,
                    "question": question,
                    "schema_context": schema_context,
                    "external_knowledge": "",
                    "sql": sql,
                }
            )

    train_rows: List[dict] = []
    valid_rows: List[dict] = []
    for row in rows:
        split = stable_split(row["db_id"] + "::" + row["id"], val_ratio)
        if split == "valid":
            valid_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, valid_rows


def read_ddl_summary_for_db(db_root: Path, db_name: str, max_schema_chars: int) -> str:
    """Load and compress all DDL.csv files found under a DB directory."""
    db_dir = db_root / db_name
    if not db_dir.exists():
        return ""

    ddl_files = sorted(db_dir.rglob("DDL.csv"))
    snippets: List[str] = []

    for ddl in ddl_files:
        try:
            with ddl.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                ddl_lines: List[str] = []
                for row in reader:
                    table_name = row.get("table_name", "")
                    ddl_text = row.get("DDL", "")
                    ddl_lines.append(f"[{table_name}] {trim_text(ddl_text, 600)}")
                if ddl_lines:
                    relative = ddl.relative_to(db_dir)
                    snippets.append(f"DB file: {relative}\n" + "\n".join(ddl_lines))
        except Exception:
            continue

    return trim_text("\n\n".join(snippets), max_schema_chars)


def build_spider2_db_map(spider2_resource_databases: Path, max_schema_chars: int) -> Dict[str, Tuple[str, str]]:
    """Map db_id(lower) -> (dialect, schema_summary)."""
    mapping: Dict[str, Tuple[str, str]] = {}

    dialect_roots = {
        "bigquery": spider2_resource_databases / "bigquery",
        "snowflake": spider2_resource_databases / "snowflake",
        "sqlite": spider2_resource_databases / "sqlite",
    }

    for dialect, root in dialect_roots.items():
        if not root.exists():
            continue
        for db_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            db_name = db_dir.name
            summary = read_ddl_summary_for_db(root, db_name, max_schema_chars)
            mapping[db_name.lower()] = (dialect, summary)

    return mapping


def load_external_knowledge(doc_dir: Path, filename: Optional[str], max_doc_chars: int) -> str:
    if not filename:
        return ""
    doc_path = doc_dir / filename
    if not doc_path.exists():
        return ""
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    return trim_text(text, max_doc_chars)


def prepare_spider2(
    spider2_dir: Path,
    max_schema_chars: int,
    max_doc_chars: int,
    val_ratio: float,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    jsonl_path = spider2_dir / "spider2-lite.jsonl"
    gold_sql_dir = spider2_dir / "evaluation_suite" / "gold" / "sql"
    doc_dir = spider2_dir / "resource" / "documents"
    db_root = spider2_dir / "resource" / "databases"

    db_map = build_spider2_db_map(db_root, max_schema_chars)

    rows: List[dict] = []
    tasks = read_jsonl(jsonl_path)

    stats = {
        "total_tasks": len(tasks),
        "with_gold_sql": 0,
        "missing_gold_sql": 0,
        "missing_schema": 0,
        "with_external_knowledge": 0,
    }

    for task in tasks:
        instance_id = task["instance_id"]
        sql_path = gold_sql_dir / f"{instance_id}.sql"
        if not sql_path.exists():
            stats["missing_gold_sql"] += 1
            continue

        stats["with_gold_sql"] += 1

        db_id = task["db"]
        mapped = db_map.get(db_id.lower())
        if mapped:
            dialect, schema_context = mapped
        else:
            dialect, schema_context = "unknown", ""
            stats["missing_schema"] += 1

        ext = load_external_knowledge(doc_dir, task.get("external_knowledge"), max_doc_chars)
        if ext:
            stats["with_external_knowledge"] += 1

        rows.append(
            {
                "id": f"spider2::{instance_id}",
                "source": "spider2_lite",
                "dialect": dialect,
                "db_id": db_id,
                "question": task["question"].strip(),
                "schema_context": schema_context,
                "external_knowledge": ext,
                "sql": sql_path.read_text(encoding="utf-8", errors="ignore").strip(),
            }
        )

    train_rows: List[dict] = []
    valid_rows: List[dict] = []
    for row in rows:
        split = stable_split(row["id"], val_ratio)
        if split == "valid":
            valid_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, valid_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT datasets for QLoRA")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to project root",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory for prepared JSONL files",
    )
    parser.add_argument("--spider1-val-ratio", type=float, default=0.05)
    parser.add_argument("--spider2-val-ratio", type=float, default=0.1)
    parser.add_argument("--max-schema-chars", type=int, default=12000)
    parser.add_argument("--max-doc-chars", type=int, default=4000)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    spider1_dir = project_root / "src" / "evaluvation" / "spider1.0"
    spider2_dir = project_root / "src" / "evaluvation" / "spider2_lite" / "spider2-lite"

    if not spider1_dir.exists():
        raise FileNotFoundError(f"Spider1 directory not found: {spider1_dir}")
    if not spider2_dir.exists():
        raise FileNotFoundError(f"Spider2-lite directory not found: {spider2_dir}")

    spider1_train, spider1_valid = prepare_spider1(
        spider1_dir=spider1_dir,
        max_schema_chars=args.max_schema_chars,
        val_ratio=args.spider1_val_ratio,
    )

    spider2_train, spider2_valid, spider2_stats = prepare_spider2(
        spider2_dir=spider2_dir,
        max_schema_chars=args.max_schema_chars,
        max_doc_chars=args.max_doc_chars,
        val_ratio=args.spider2_val_ratio,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "spider1_train.jsonl": spider1_train,
        "spider1_valid.jsonl": spider1_valid,
        "spider2_train.jsonl": spider2_train,
        "spider2_valid.jsonl": spider2_valid,
        "stage1_train.jsonl": spider1_train,
        "stage1_valid.jsonl": spider1_valid,
        "stage2_spider2_only_train.jsonl": spider2_train,
        "stage2_spider2_only_valid.jsonl": spider2_valid,
        "stage2_mixed_train.jsonl": spider1_train + spider2_train,
        "stage2_mixed_valid.jsonl": spider1_valid + spider2_valid,
    }

    print("=" * 72)
    print("Prepared datasets")
    print("=" * 72)
    for name, rows in files.items():
        n = write_jsonl(out_dir / name, rows)
        print(f"{name:<36} {n:>8}")

    print("-" * 72)
    print("Spider2 stats")
    for k, v in spider2_stats.items():
        print(f"{k:<28} {v}")


if __name__ == "__main__":
    main()
