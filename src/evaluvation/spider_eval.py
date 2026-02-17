"""
Spider 1.0 Evaluation Runner — Architecture-Parity Edition

This evaluator executes the SAME logical pipeline as the runtime system
(pipeline.py), ensuring benchmark numbers reflect production behaviour.

Runtime path:   retrieval → decomposition → generation → execution → retry
Evaluation path: retrieval → decomposition → generation → execution → retry
                 (each step toggleable via ablation modes)

Ablation modes:
  baseline         Full schema → generate_sql
  +retrieval       SchemaRetriever → generate_sql
  +decomposition   Full schema → QueryDecomposer → generate_sql
  +retry           Full schema → generate_sql → retry on exec failure
  full             SchemaRetriever → QueryDecomposer → generate_sql + retry

Result comparison uses multiset (bag) semantics to avoid false
positives/negatives from duplicate-row or ordering differences.

Usage:
    python src/evaluvation/spider_eval.py --mode full --limit 100
    python src/evaluvation/spider_eval.py --mode all --output results/ablation
    python src/evaluvation/spider_eval.py --mode baseline --output results/baseline
"""

import sys
import json
import sqlite3
import argparse
import csv
import time
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.sql_generator_azure import generate_sql
from src.generator.prompts import get_active_versions
from src.generator.retry import classify_error, build_retry_prompt
from src.generator.prompt_registry import (
    validate_active_versions,
    record_metric_delta,
    save_registry_snapshot,
    get_registered_version,
)
from src.decomposition.query_decomposer import QueryDecomposer
from src.retriever.schema_retriever import SchemaRetriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPIDER_DIR = PROJECT_ROOT / "src" / "evaluvation" / "spider1.0"
DEV_JSON = SPIDER_DIR / "dev.json"
TABLES_JSON = SPIDER_DIR / "tables.json"
DATABASE_DIR = SPIDER_DIR / "database"

ABLATION_MODES = {
    "baseline":       {"retrieval": False, "decomposition": False, "retry": False},
    "+retrieval":     {"retrieval": True,  "decomposition": False, "retry": False},
    "+decomposition": {"retrieval": False, "decomposition": True,  "retry": False},
    "+retry":         {"retrieval": False, "decomposition": False, "retry": True},
    "full":           {"retrieval": True,  "decomposition": True,  "retry": True},
}


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------
class FailureCategory:
    """Exhaustive failure taxonomy for result classification."""
    SUCCESS           = "success"
    GENERATION_ERROR  = "generation_error"
    EXECUTION_ERROR   = "execution_error"
    GOLD_ERROR        = "gold_execution_error"
    WRONG_COLUMNS     = "wrong_columns"
    EMPTY_RESULT      = "empty_result"
    EXTRA_ROWS        = "extra_rows"
    MISSING_ROWS      = "missing_rows"
    WRONG_VALUES      = "wrong_values"


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_spider_tables() -> Dict[str, Dict]:
    """
    Load all database schemas from tables.json.

    Returns items in the same dict format as ``schema_loader.load_schema``
    so that ``SchemaRetriever`` and ``generate_sql`` receive identical
    shapes in evaluation and production.
    """
    with open(TABLES_JSON, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]

        columns = []
        for i, (table_idx, col_name) in enumerate(db["column_names_original"]):
            if table_idx == -1:
                continue
            col_type = (
                db["column_types"][i] if i < len(db["column_types"]) else "text"
            )
            table = table_names[table_idx]
            # `text` field mirrors schema_loader format for embedding
            columns.append({
                "table": table,
                "column": col_name,
                "type": col_type,
                "text": f"{table}.{col_name} {col_name} {col_type}",
            })

        schemas[db_id] = {
            "db_id": db_id,
            "tables": table_names,
            "columns": columns,
            "foreign_keys": db.get("foreign_keys", []),
            "primary_keys": db.get("primary_keys", []),
        }

    return schemas


def load_spider_dev() -> List[Dict]:
    """Load the Spider dev set questions."""
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_db_path(db_id: str) -> Path:
    """Return the SQLite database file for *db_id*."""
    return DATABASE_DIR / db_id / f"{db_id}.sqlite"


# ═══════════════════════════════════════════════════════════════════════
# SQL EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def execute_sql_sqlite(
    db_path: Path, sql: str, timeout: int = 30
) -> Tuple[bool, Any]:
    """Execute SQL on a SQLite database and return (success, result|error)."""
    if not db_path.exists():
        return False, f"Database not found: {db_path}"
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.text_factory = str
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════════════
# RESULT COMPARISON  (multiset / bag semantics — fixes P1 duplicate bug)
# ═══════════════════════════════════════════════════════════════════════

def _normalize_value(val):
    """Normalise a single cell value for comparison."""
    if val is None:
        return None
    if isinstance(val, float):
        return round(val, 5)
    if isinstance(val, str):
        return val.strip().lower()
    return val


def _normalize_row(row: tuple) -> tuple:
    return tuple(_normalize_value(v) for v in row)


def compare_results(
    gold_result: List[Tuple], pred_result: List[Tuple]
) -> Tuple[bool, str]:
    """
    Compare two SQL result sets using **multiset** semantics.

    Improvements over the previous set-based comparator:
    * Duplicate rows are counted (``Counter`` instead of ``set``).
    * Column-count mismatches are detected separately.
    * Failure is classified into a taxonomy for the report.
    """
    if not gold_result and not pred_result:
        return True, FailureCategory.SUCCESS

    gold_norm = Counter(_normalize_row(r) for r in gold_result)
    pred_norm = Counter(_normalize_row(r) for r in pred_result)

    if gold_norm == pred_norm:
        return True, FailureCategory.SUCCESS

    # --- classify the mismatch ---
    if gold_result and pred_result:
        if len(gold_result[0]) != len(pred_result[0]):
            return False, FailureCategory.WRONG_COLUMNS

    gold_total = sum(gold_norm.values())
    pred_total = sum(pred_norm.values())

    if pred_total == 0 and gold_total > 0:
        return False, FailureCategory.EMPTY_RESULT
    if pred_total > gold_total:
        return False, FailureCategory.EXTRA_ROWS
    if pred_total < gold_total:
        return False, FailureCategory.MISSING_ROWS
    return False, FailureCategory.WRONG_VALUES


# ═══════════════════════════════════════════════════════════════════════
# RETRIEVER CACHE  (one SchemaRetriever per database)
# ═══════════════════════════════════════════════════════════════════════

class RetrieverCache:
    """
    Lazily creates and caches a ``SchemaRetriever`` per database so that
    each db's embeddings are computed only once across the evaluation run.
    """

    def __init__(self):
        self._retrievers: Dict[str, SchemaRetriever] = {}

    def get(self, db_id: str, schema_items: List[Dict]) -> SchemaRetriever:
        if db_id not in self._retrievers:
            self._retrievers[db_id] = SchemaRetriever(schema_items)
        return self._retrievers[db_id]


# ═══════════════════════════════════════════════════════════════════════
# SINGLE-EXAMPLE EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_single(
    example: Dict,
    schema: Dict,
    db_path: Path,
    *,
    retriever_cache: Optional[RetrieverCache],
    decomposer: Optional[QueryDecomposer],
    mode_config: Dict,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate one question through the pipeline.

    The pipeline mirrors ``pipeline.py``::

        1. Retrieval   (SchemaRetriever.retrieve)   — if enabled
        2. Decomposition (QueryDecomposer.process)  — if enabled
        3. Generation  (generate_sql)
        4. Execution   (SQLite)
        5. Retry       (error-guided re-generation) — if enabled
    """
    question = example["question"]
    gold_sql = example["query"]
    db_id = example["db_id"]

    result = {
        "question": question,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "pred_sql": None,
        "gold_success": False,
        "pred_success": False,
        "execution_match": False,
        "failure_category": None,
        "error": None,
        "retried": False,
        "complexity": None,
        "complexity_score": None,
        "retrieval_count": len(schema["columns"]),
    }

    use_retrieval = mode_config["retrieval"]
    use_decomposition = mode_config["decomposition"]
    use_retry = mode_config["retry"]

    # ── Step 1: Schema retrieval (same as pipeline.py Step 3-4) ──
    if use_retrieval and retriever_cache is not None:
        retriever = retriever_cache.get(db_id, schema["columns"])
        scored_results = retriever.retrieve(question, top_k=10)
        schema_items = [item for _, item in scored_results]
        result["retrieval_count"] = len(schema_items)
    else:
        schema_items = schema["columns"]

    # ── Step 2: Decomposition (same as pipeline.py Step 5) ──
    gen_question = question
    if use_decomposition and decomposer is not None:
        try:
            decomp = decomposer.process(question, schema_items, dialect="sqlite")
            gen_question = decomp["enhanced_question"]
            result["complexity"] = decomp["complexity"]
            result["complexity_score"] = decomp["complexity_score"]
        except Exception as e:
            result["error"] = f"Decomposition error: {e}"
            gen_question = question

    # ── Step 3: SQL generation (same as pipeline.py Step 6) ──
    try:
        pred_sql = generate_sql(
            question=gen_question,
            schema_items=schema_items,
            dialect="sqlite",
        )
        result["pred_sql"] = pred_sql
    except Exception as e:
        result["error"] = f"Generation error: {e}"
        result["failure_category"] = FailureCategory.GENERATION_ERROR
        return result

    # ── Step 4: Execute gold SQL ──
    gold_success, gold_result = execute_sql_sqlite(db_path, gold_sql)
    result["gold_success"] = gold_success
    if not gold_success:
        result["error"] = f"Gold SQL execution failed: {gold_result}"
        result["failure_category"] = FailureCategory.GOLD_ERROR
        return result

    # ── Step 5: Execute predicted SQL ──
    pred_success, pred_result = execute_sql_sqlite(db_path, pred_sql)
    result["pred_success"] = pred_success

    if not pred_success:
        # ── Step 6: Retry (same as pipeline.py Step 8) ──
        if use_retry:
            result["retried"] = True
            error_class = classify_error(str(pred_result), dialect="sqlite")
            result["error_class"] = error_class.name
            retry_question = build_retry_prompt(
                question, str(pred_result), error_class, failed_sql=pred_sql,
            )
            try:
                pred_sql = generate_sql(
                    question=retry_question,
                    schema_items=schema_items,
                    dialect="sqlite",
                )
                result["pred_sql"] = pred_sql
                pred_success, pred_result = execute_sql_sqlite(db_path, pred_sql)
                result["pred_success"] = pred_success
            except Exception:
                pass  # keep original failure

        if not result["pred_success"]:
            result["error"] = f"Predicted SQL execution failed: {pred_result}"
            result["failure_category"] = FailureCategory.EXECUTION_ERROR
            return result

    # ── Step 7: Compare results (multiset semantics) ──
    match, failure_detail = compare_results(gold_result, pred_result)
    result["execution_match"] = match
    result["failure_category"] = failure_detail

    if verbose:
        status = "✓" if match else "✗"
        print(f"  {status} [{db_id}] {question[:60]}...")

    return result


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[Dict], mode_name: str) -> Dict:
    """Return a comprehensive metrics dict from raw results."""
    total = len(results)
    if total == 0:
        return {"mode": mode_name, "total_examples": 0}

    execution_correct = sum(r["execution_match"] for r in results)
    pred_executed = sum(r["pred_success"] for r in results)
    generation_errors = sum(1 for r in results if r["pred_sql"] is None)
    retried = sum(1 for r in results if r.get("retried"))

    # Failure category counts
    failure_counts = Counter(
        r["failure_category"]
        for r in results
        if r["failure_category"] != FailureCategory.SUCCESS
    )

    # Per-database accuracy
    db_results: Dict[str, Dict] = {}
    for r in results:
        db_id = r["db_id"]
        db_results.setdefault(db_id, {"total": 0, "correct": 0})
        db_results[db_id]["total"] += 1
        if r["execution_match"]:
            db_results[db_id]["correct"] += 1

    # Complexity breakdown
    cx_results: Dict[str, Dict] = {}
    for r in results:
        cx = r.get("complexity") or "n/a"
        cx_results.setdefault(cx, {"total": 0, "correct": 0})
        cx_results[cx]["total"] += 1
        if r["execution_match"]:
            cx_results[cx]["correct"] += 1

    prompt_versions = get_active_versions()
    exec_err_count = failure_counts.get(FailureCategory.EXECUTION_ERROR, 0)

    return {
        "mode": mode_name,
        "total_examples": total,
        "execution_correct": execution_correct,
        "execution_accuracy_pct": round(execution_correct / total * 100, 2),
        "valid_sql_count": pred_executed,
        "valid_sql_rate_pct": round(pred_executed / total * 100, 2),
        "generation_errors": generation_errors,
        "execution_errors": exec_err_count,
        "execution_error_rate_pct": round(exec_err_count / total * 100, 2),
        "retried_count": retried,
        "failure_categories": dict(failure_counts),
        "per_database": {
            db_id: {
                "total": info["total"],
                "correct": info["correct"],
                "accuracy_pct": round(
                    info["correct"] / info["total"] * 100, 1
                ),
            }
            for db_id, info in sorted(db_results.items())
        },
        "complexity_breakdown": {
            cx: {
                "total": info["total"],
                "correct": info["correct"],
                "accuracy_pct": (
                    round(info["correct"] / info["total"] * 100, 1)
                    if info["total"] > 0
                    else 0
                ),
            }
            for cx, info in cx_results.items()
        },
        "prompt_version_sqlite": prompt_versions["sqlite"],
        "prompt_version_bigquery": prompt_versions["bigquery"],
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_report(all_metrics: Dict[str, Dict], output_dir: Path) -> Path:
    """Produce a Markdown report with parity statement and ablation comparison."""
    lines = [
        "# Spider 1.0 Evaluation Report — Architecture-Parity Edition",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Architecture Parity",
        "",
        "This evaluation executes the **same logical pipeline** as the runtime",
        "system (`pipeline.py`).  Every component is the same class / function;",
        "only the execution backend differs (SQLite vs BigQuery).",
        "",
        "| Step | Runtime (`pipeline.py`) | Evaluation (`spider_eval.py`) |",
        "|------|------------------------|-------------------------------|",
        "| Schema loading | `schema_loader.load_schema()` | `load_spider_tables()` (same dict format) |",
        "| Retrieval | `SchemaRetriever.retrieve()` | `SchemaRetriever.retrieve()` (same class) |",
        "| Decomposition | `QueryDecomposer.process()` | `QueryDecomposer.process()` (same class) |",
        "| Generation | `generate_sql()` | `generate_sql()` (same function) |",
        "| Execution | `bq_executor.execute_sql()` | `execute_sql_sqlite()` (dialect differs) |",
        "| Retry | Error-guided re-generation | Error-guided re-generation (same logic) |",
        "",
        "> The retrieval, decomposition, generation, and retry code paths",
        "> are identical between evaluation and runtime.",
        "",
        "---",
        "",
        "## 2. Dataset Coverage",
        "",
    ]

    first = next(iter(all_metrics.values()))
    lines += [
        f"- **Benchmark:** Spider 1.0 dev set",
        f"- **Examples evaluated:** {first['total_examples']}",
        f"- **Unique databases:** {len(first.get('per_database', {}))}",
        "",
        "---",
        "",
        "## 3. Ablation Results",
        "",
        "| Mode | EX (%) | Valid SQL (%) | Exec Err (%) | Gen Errors | Retries |",
        "|------|--------|---------------|--------------|------------|---------|",
    ]
    for mode, m in all_metrics.items():
        lines.append(
            f"| {mode} "
            f"| {m['execution_accuracy_pct']} "
            f"| {m['valid_sql_rate_pct']} "
            f"| {m['execution_error_rate_pct']} "
            f"| {m['generation_errors']} "
            f"| {m.get('retried_count', 0)} |"
        )
    lines.append("")

    # Failure breakdown per mode
    for mode, m in all_metrics.items():
        total = m["total_examples"]
        fc = m.get("failure_categories", {})
        if not fc:
            continue
        lines += [
            f"### Failure breakdown — `{mode}`",
            "",
            "| Category | Count | % |",
            "|----------|------:|--:|",
        ]
        for cat, cnt in sorted(fc.items(), key=lambda x: -x[1]):
            pct = round(cnt / total * 100, 1) if total else 0
            lines.append(f"| {cat} | {cnt} | {pct}% |")
        lines.append("")

        # Complexity sub-table (only when decomposition was on)
        cx = m.get("complexity_breakdown", {})
        if cx and any(k != "n/a" for k in cx):
            lines += [
                f"#### Complexity breakdown — `{mode}`",
                "",
                "| Complexity | Total | Correct | Accuracy |",
                "|------------|------:|--------:|---------:|",
            ]
            for label, info in cx.items():
                lines.append(
                    f"| {label} | {info['total']} | {info['correct']} "
                    f"| {info['accuracy_pct']}% |"
                )
            lines.append("")

    # Methodology section
    lines += [
        "---",
        "",
        "## 4. Methodology",
        "",
        "- **Execution Accuracy (EX):** Predicted SQL returns the same result",
        "  set as gold SQL when executed on the benchmark database.",
        "- **Result comparison:** Multiset (bag) semantics — duplicate rows are",
        "  preserved, row order is ignored.  Floats rounded to 5 decimal places.",
        "  Strings are case-normalised and trimmed.",
        "- **Valid SQL Rate:** % of predictions that execute without error.",
        "- **Execution Error Rate:** % of predictions that fail to execute.",
        "",
        "### Ablation mode definitions",
        "",
        "| Mode | Retrieval | Decomposition | Retry |",
        "|------|:---------:|:-------------:|:-----:|",
    ]
    for mode, cfg in ABLATION_MODES.items():
        r = "✓" if cfg["retrieval"] else "—"
        d = "✓" if cfg["decomposition"] else "—"
        t = "✓" if cfg["retry"] else "—"
        lines.append(f"| {mode} | {r} | {d} | {t} |")
    lines.append("")

    report_text = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_evaluation(
    mode_name: str,
    limit: Optional[int] = None,
    verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict, List[Dict]]:
    """Run Spider evaluation in a single ablation mode."""
    mode_config = ABLATION_MODES[mode_name]

    print("=" * 60)
    print(f"  Spider 1.0 Evaluation — mode: {mode_name}")
    print("=" * 60)

    # ── Validate prompt registry ──
    validate_active_versions()

    prompt_versions = get_active_versions()
    print(f"\n  Prompt Version : SQLite={prompt_versions['sqlite']}, BigQuery={prompt_versions['bigquery']}")
    print(f"  Retrieval      : {'ON' if mode_config['retrieval'] else 'OFF'}")
    print(f"  Decomposition  : {'ON' if mode_config['decomposition'] else 'OFF'}")
    print(f"  Retry          : {'ON' if mode_config['retry'] else 'OFF'}")

    # ── Load data ──
    print("\n[1/5] Loading Spider data...")
    schemas = load_spider_tables()
    dev_data = load_spider_dev()
    print(f"      {len(schemas)} database schemas, {len(dev_data)} dev examples")

    if limit:
        dev_data = dev_data[:limit]
        print(f"      Limited to {limit} examples")

    # ── Init pipeline components ──
    print("[2/5] Initialising pipeline components...")
    retriever_cache = RetrieverCache() if mode_config["retrieval"] else None
    decomposer = None
    if mode_config["decomposition"]:
        decomposer = QueryDecomposer(cot_enabled=True, decompose_enabled=True)
        print("      QueryDecomposer ready")
    if retriever_cache:
        print("      SchemaRetriever cache ready (lazy per db)")

    # ── Evaluate ──
    print(f"\n[3/5] Evaluating {len(dev_data)} examples...")
    results: List[Dict] = []
    t0 = time.time()

    for i, example in enumerate(dev_data):
        db_id = example["db_id"]
        schema = schemas.get(db_id)
        if schema is None:
            print(f"  WARNING: schema missing for {db_id}, skipping")
            continue

        db_path = get_db_path(db_id)

        res = evaluate_single(
            example,
            schema,
            db_path,
            retriever_cache=retriever_cache,
            decomposer=decomposer,
            mode_config=mode_config,
            verbose=verbose,
        )
        results.append(res)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = sum(r["execution_match"] for r in results) / len(results) * 100
            print(f"      {i+1}/{len(dev_data)}  EX={acc:.1f}%  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s")

    # ── Metrics ──
    print("[4/5] Computing metrics...")
    metrics = compute_metrics(results, mode_name)

    # ── Save outputs ──
    if output_dir:
        print(f"[5/5] Saving to {output_dir}/...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Raw CSV
        csv_path = output_dir / "raw_results.csv"
        fieldnames = [
            "question", "db_id", "gold_sql", "pred_sql",
            "gold_success", "pred_success", "execution_match",
            "failure_category", "error", "error_class", "retried",
            "complexity", "complexity_score", "retrieval_count",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        # Summary JSON
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Error analysis JSON
        error_analysis = {
            "failure_categories": metrics["failure_categories"],
            "errors": [
                {
                    "question": r["question"][:120],
                    "db_id": r["db_id"],
                    "category": r["failure_category"],
                    "error": r.get("error"),
                    "gold_sql": r["gold_sql"][:200],
                    "pred_sql": (r["pred_sql"] or "")[:200],
                }
                for r in results
                if r["failure_category"] != FailureCategory.SUCCESS
            ],
        }
        err_path = output_dir / "error_analysis.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2)

        # Registry snapshot — records which prompt versions were active
        save_registry_snapshot(output_dir / "prompt_registry_snapshot.json")
    else:
        print("[5/5] Skipping file output (use --output)")

    # ── Console summary ──
    print("\n" + "=" * 60)
    print(f"  RESULTS: {mode_name}")
    print("=" * 60)
    print(f"  Total Examples:       {metrics['total_examples']}")
    print(
        f"  Execution Accuracy:   {metrics['execution_accuracy_pct']}%  "
        f"({metrics['execution_correct']}/{metrics['total_examples']})"
    )
    print(
        f"  Valid SQL Rate:       {metrics['valid_sql_rate_pct']}%  "
        f"({metrics['valid_sql_count']}/{metrics['total_examples']})"
    )
    print(f"  Execution Error Rate: {metrics['execution_error_rate_pct']}%")
    print(f"  Generation Errors:    {metrics['generation_errors']}")
    if metrics.get("retried_count"):
        print(f"  Retried:              {metrics['retried_count']}")

    fc = metrics.get("failure_categories", {})
    if fc:
        print("\n  Failure Categories:")
        for cat, cnt in sorted(fc.items(), key=lambda x: -x[1]):
            print(f"    {cat:<25} {cnt}")

    print("=" * 60)
    return metrics, results


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Spider 1.0 Evaluation — Architecture-Parity Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation modes
  baseline         Full schema -> generate_sql
  +retrieval       SchemaRetriever -> generate_sql
  +decomposition   Full schema -> QueryDecomposer -> generate_sql
  +retry           Full schema -> generate_sql -> retry on failure
  full             SchemaRetriever -> QueryDecomposer -> generate_sql + retry
  all              Run ALL modes and produce comparison report

Prompt A/B testing
  python src/evaluvation/spider_eval.py --prompt-ab V2,V3 --dialect bigquery --mode baseline --limit 50

Examples
  python src/evaluvation/spider_eval.py --mode full --limit 50
  python src/evaluvation/spider_eval.py --mode all --output results/ablation
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=list(ABLATION_MODES.keys()) + ["all"],
        help="Ablation mode (default: full)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max examples (None = all)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (relative to project root)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce verbosity"
    )
    parser.add_argument(
        "--prompt-ab",
        type=str,
        default=None,
        help=(
            "Run A/B comparison between two prompt versions (comma-separated). "
            "E.g. --prompt-ab V2,V3  Requires --dialect."
        ),
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="sqlite",
        choices=["sqlite", "bigquery"],
        help="Dialect for --prompt-ab comparison (default: sqlite)",
    )

    args = parser.parse_args()
    output_base = Path(PROJECT_ROOT / args.output) if args.output else None

    # ── Prompt A/B mode ─────────────────────────────────────────────
    if args.prompt_ab:
        import src.generator.prompts as prompts_module

        versions = [v.strip() for v in args.prompt_ab.split(",")]
        if len(versions) != 2:
            parser.error("--prompt-ab requires exactly two versions, e.g. V2,V3")
        dialect = args.dialect
        version_a, version_b = versions

        # Validate both versions are registered
        for v in versions:
            pv = get_registered_version(dialect, v)
            if pv is None:
                parser.error(
                    f"Prompt version {dialect}/{v} is not in the prompt registry. "
                    f"Register it in src/generator/prompt_registry.py first."
                )

        ab_metrics: Dict[str, Dict] = {}
        for ver in versions:
            # Activate the prompt version
            if dialect == "sqlite":
                prompts_module.ACTIVE_SQLITE_VERSION = ver
            else:
                prompts_module.ACTIVE_BIGQUERY_VERSION = ver

            tag = f"{dialect}_{ver}"
            ver_output = output_base / tag if output_base else None
            print(f"\n{'#' * 60}")
            print(f"  A/B RUN: {dialect} prompt {ver}")
            print(f"{'#' * 60}")

            metrics, _ = run_evaluation(
                mode_name=args.mode,
                limit=args.limit,
                verbose=not args.quiet,
                output_dir=ver_output,
            )
            ab_metrics[tag] = metrics

        # ── A/B comparison ──
        ma = ab_metrics[f"{dialect}_{version_a}"]
        mb = ab_metrics[f"{dialect}_{version_b}"]
        delta = mb["execution_accuracy_pct"] - ma["execution_accuracy_pct"]

        print("\n" + "=" * 80)
        print("  PROMPT A/B COMPARISON")
        print("=" * 80)
        print(f"  Dialect   : {dialect}")
        print(f"  Version A : {version_a}  EX = {ma['execution_accuracy_pct']}%")
        print(f"  Version B : {version_b}  EX = {mb['execution_accuracy_pct']}%")
        print(f"  Delta     : {delta:+.2f}%")
        print("=" * 80)

        # Record metric delta
        record_metric_delta(dialect, version_b, {
            "baseline_version": version_a,
            "execution_accuracy_delta_pct": round(delta, 2),
            "execution_accuracy_a_pct": ma["execution_accuracy_pct"],
            "execution_accuracy_b_pct": mb["execution_accuracy_pct"],
            "eval_examples": ma["total_examples"],
            "eval_mode": args.mode,
        })

        if output_base:
            save_registry_snapshot(output_base / "prompt_registry_snapshot.json")
            # Save A/B comparison report
            ab_report = output_base / "ab_comparison.json"
            with open(ab_report, "w", encoding="utf-8") as f:
                json.dump({
                    "dialect": dialect,
                    "version_a": version_a,
                    "version_b": version_b,
                    "metrics_a": ma,
                    "metrics_b": mb,
                    "delta_pct": round(delta, 2),
                }, f, indent=2)
            print(f"\nA/B report saved to: {ab_report}")

    elif args.mode == "all":
        all_metrics: Dict[str, Dict] = {}
        for mode_name in ABLATION_MODES:
            mode_output = output_base / mode_name if output_base else None
            metrics, _ = run_evaluation(
                mode_name=mode_name,
                limit=args.limit,
                verbose=not args.quiet,
                output_dir=mode_output,
            )
            all_metrics[mode_name] = metrics
            print()

        # Comparison report
        if output_base:
            rpt = generate_report(all_metrics, output_base)
            print(f"\nComparison report saved to: {rpt}")

        # Console comparison
        print("\n" + "=" * 80)
        print("  ABLATION COMPARISON")
        print("=" * 80)
        hdr = f"  {'Mode':<20} {'EX%':<10} {'Valid SQL%':<12} {'Exec Err%':<12} {'Gen Err':<10}"
        print(hdr)
        print("-" * 80)
        for mn, m in all_metrics.items():
            print(
                f"  {mn:<20} {m['execution_accuracy_pct']:<10} "
                f"{m['valid_sql_rate_pct']:<12} "
                f"{m['execution_error_rate_pct']:<12} "
                f"{m['generation_errors']:<10}"
            )
        print("=" * 80)
    else:
        mode_output = output_base if output_base else None
        metrics, results = run_evaluation(
            mode_name=args.mode,
            limit=args.limit,
            verbose=not args.quiet,
            output_dir=mode_output,
        )

        if output_base:
            rpt = generate_report({args.mode: metrics}, output_base)
            print(f"\nReport saved to: {rpt}")

        # Legacy summary location
        summary_path = SPIDER_DIR / "evaluation_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
