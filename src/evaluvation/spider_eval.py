"""
Spider 1.0 Evaluation Runner

This script evaluates the Text2SQL pipeline on the Spider benchmark dataset.
It measures Execution Accuracy (EX) - whether the predicted SQL returns the
same results as the gold SQL when executed on the database.

Usage:
    python src/evaluvation/spider_eval.py [--limit N] [--output results.csv]
"""

import sys
import json
import sqlite3
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.sql_generator_azure import generate_sql
from src.generator.prompts import get_active_versions
from src.decomposition.query_decomposer import QueryDecomposer

# Decomposer instance (initialized lazily based on --decompose flag)
_decomposer = None


# -----------------------------
# Paths
# -----------------------------
SPIDER_DIR = PROJECT_ROOT / "src" / "evaluvation" / "spider1.0"
DEV_JSON = SPIDER_DIR / "dev.json"
TABLES_JSON = SPIDER_DIR / "tables.json"
DATABASE_DIR = SPIDER_DIR / "database"


def load_spider_tables() -> Dict[str, Dict]:
    """
    Load all database schemas from tables.json.
    Returns a dict mapping db_id -> schema info.
    """
    with open(TABLES_JSON, "r", encoding="utf-8") as f:
        tables_data = json.load(f)
    
    schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        
        # Build table name list
        table_names = db["table_names_original"]
        
        # Build column info: list of {table, column, type}
        columns = []
        for i, (table_idx, col_name) in enumerate(db["column_names_original"]):
            if table_idx == -1:  # Skip the * column
                continue
            columns.append({
                "table": table_names[table_idx],
                "column": col_name,
                "type": db["column_types"][i] if i < len(db["column_types"]) else "text",
                "text": f"{table_names[table_idx]}.{col_name} ({db['column_types'][i] if i < len(db['column_types']) else 'text'})"
            })
        
        schemas[db_id] = {
            "db_id": db_id,
            "tables": table_names,
            "columns": columns,
            "foreign_keys": db.get("foreign_keys", []),
            "primary_keys": db.get("primary_keys", [])
        }
    
    return schemas


def load_spider_dev() -> List[Dict]:
    """Load the Spider dev set questions."""
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_db_path(db_id: str) -> Path:
    """Get the SQLite database path for a given db_id."""
    return DATABASE_DIR / db_id / f"{db_id}.sqlite"


def execute_sql_sqlite(db_path: Path, sql: str, timeout: int = 30) -> Tuple[bool, Any]:
    """
    Execute SQL on a SQLite database.
    
    Returns:
        (success: bool, result: list of tuples or error string)
    """
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


def normalize_result(result: List[Tuple]) -> set:
    """
    Normalize SQL results for comparison.
    - Convert to set of tuples (order-independent)
    - Handle None values
    - Round floats for comparison
    """
    normalized = set()
    for row in result:
        normalized_row = []
        for val in row:
            if val is None:
                normalized_row.append(None)
            elif isinstance(val, float):
                # Round floats to avoid precision issues
                normalized_row.append(round(val, 5))
            else:
                normalized_row.append(val)
        normalized.add(tuple(normalized_row))
    return normalized


def compare_results(gold_result: List[Tuple], pred_result: List[Tuple]) -> bool:
    """
    Compare two SQL result sets.
    Returns True if they are equivalent (order-independent).
    """
    gold_normalized = normalize_result(gold_result)
    pred_normalized = normalize_result(pred_result)
    return gold_normalized == pred_normalized


def evaluate_single(
    example: Dict,
    schema: Dict,
    db_path: Path,
    verbose: bool = False,
    use_decomposition: bool = False
) -> Dict:
    """
    Evaluate a single question.
    
    Returns a dict with:
        - question: str
        - db_id: str
        - gold_sql: str
        - pred_sql: str
        - gold_success: bool
        - pred_success: bool
        - execution_match: bool (main metric)
        - error: str or None
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
        "error": None
    }
    
    # Step 1: Generate SQL using our pipeline
    try:
        # Optionally apply query decomposition + CoT
        gen_question = question
        if use_decomposition and _decomposer is not None:
            decomp = _decomposer.process(question, schema["columns"], dialect="sqlite")
            gen_question = decomp["enhanced_question"]
            result["complexity"] = decomp["complexity"]
            result["complexity_score"] = decomp["complexity_score"]

        # Pass the full schema columns to the generator
        pred_sql = generate_sql(
            question=gen_question,
            schema_items=schema["columns"],
            dialect="sqlite"
        )
        result["pred_sql"] = pred_sql
    except Exception as e:
        result["error"] = f"Generation error: {str(e)}"
        return result
    
    # Step 2: Execute gold SQL
    gold_success, gold_result = execute_sql_sqlite(db_path, gold_sql)
    result["gold_success"] = gold_success
    
    if not gold_success:
        result["error"] = f"Gold SQL execution failed: {gold_result}"
        return result
    
    # Step 3: Execute predicted SQL
    pred_success, pred_result = execute_sql_sqlite(db_path, pred_sql)
    result["pred_success"] = pred_success
    
    if not pred_success:
        result["error"] = f"Predicted SQL execution failed: {pred_result}"
        return result
    
    # Step 4: Compare results
    result["execution_match"] = compare_results(gold_result, pred_result)
    
    if verbose:
        status = "✓" if result["execution_match"] else "✗"
        print(f"  {status} Q: {question[:60]}...")
        if not result["execution_match"]:
            print(f"      Gold: {gold_sql[:80]}...")
            print(f"      Pred: {pred_sql[:80]}...")
    
    return result


def run_evaluation(
    limit: Optional[int] = None,
    verbose: bool = True,
    output_csv: Optional[str] = None,
    use_decomposition: bool = False
) -> Dict:
    """
    Run the full Spider evaluation.
    
    Args:
        limit: Max number of examples to evaluate (None = all)
        verbose: Print progress
        output_csv: Path to save detailed results
    
    Returns:
        Summary dict with accuracy metrics
    """
    print("=" * 60)
    print("  Spider 1.0 Evaluation")
    print("=" * 60)
    
    # Show active prompt version
    prompt_versions = get_active_versions()
    print(f"\n  Prompt Version: SQLite={prompt_versions['sqlite']}")
    print(f"  Decomposition: {'ON' if use_decomposition else 'OFF'}")
    
    # Load data
    print("\n[1/4] Loading Spider data...")
    schemas = load_spider_tables()
    dev_data = load_spider_dev()
    print(f"      Loaded {len(schemas)} database schemas")
    print(f"      Loaded {len(dev_data)} dev examples")
    
    if limit:
        dev_data = dev_data[:limit]
        print(f"      Limiting to {limit} examples")
    
    # Run evaluation
    print(f"\n[2/4] Running evaluation on {len(dev_data)} examples...")
    results = []
    
    for i, example in enumerate(dev_data):
        db_id = example["db_id"]
        schema = schemas.get(db_id)
        
        if schema is None:
            print(f"  Warning: Schema not found for {db_id}, skipping")
            continue
        
        db_path = get_db_path(db_id)
        
        result = evaluate_single(example, schema, db_path, verbose=False,
                                use_decomposition=use_decomposition)
        results.append(result)
        
        # Progress update
        if verbose and (i + 1) % 50 == 0:
            current_acc = sum(r["execution_match"] for r in results) / len(results) * 100
            print(f"      Progress: {i+1}/{len(dev_data)} ({current_acc:.1f}% EX so far)")
    
    # Calculate metrics
    print("\n[3/4] Calculating metrics...")
    
    total = len(results)
    execution_correct = sum(r["execution_match"] for r in results)
    pred_executed = sum(r["pred_success"] for r in results)
    generation_errors = sum(1 for r in results if r["pred_sql"] is None)
    
    execution_accuracy = execution_correct / total * 100 if total > 0 else 0
    valid_sql_rate = pred_executed / total * 100 if total > 0 else 0
    
    # Get active prompt versions for tracking
    prompt_versions = get_active_versions()
    
    summary = {
        "total_examples": total,
        "execution_correct": execution_correct,
        "execution_accuracy": round(execution_accuracy, 2),
        "valid_sql_count": pred_executed,
        "valid_sql_rate": round(valid_sql_rate, 2),
        "generation_errors": generation_errors,
        "prompt_version_sqlite": prompt_versions["sqlite"],
        "prompt_version_bigquery": prompt_versions["bigquery"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Save detailed results to CSV
    if output_csv:
        print(f"\n[4/4] Saving results to {output_csv}...")
        output_path = PROJECT_ROOT / output_csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "db_id", "gold_sql", "pred_sql",
                "gold_success", "pred_success", "execution_match", "error"
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"      Saved {len(results)} results")
    else:
        print("\n[4/4] Skipping CSV output (use --output to save)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Prompt Version:       SQLite {prompt_versions['sqlite']}")
    print(f"  Total Examples:       {total}")
    print(f"  Execution Accuracy:   {execution_accuracy:.2f}%  ({execution_correct}/{total})")
    print(f"  Valid SQL Rate:       {valid_sql_rate:.2f}%  ({pred_executed}/{total})")
    print(f"  Generation Errors:    {generation_errors}")
    print("=" * 60)
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Spider 1.0 Evaluation")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit number of examples (for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (relative to project root)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--decompose", action="store_true",
                        help="Enable query decomposition + chain-of-thought")
    
    args = parser.parse_args()

    # Initialize decomposer if requested
    global _decomposer
    if args.decompose:
        _decomposer = QueryDecomposer(cot_enabled=True, decompose_enabled=True)
        print("Query decomposition enabled")
    
    summary, results = run_evaluation(
        limit=args.limit,
        verbose=not args.quiet,
        output_csv=args.output,
        use_decomposition=args.decompose
    )
    
    # Save summary to JSON
    summary_path = SPIDER_DIR / "evaluation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
