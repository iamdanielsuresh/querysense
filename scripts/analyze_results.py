"""
Error Analysis Script for Spider Evaluation Results

Analyzes failure patterns and generates formatted reports.

Usage:
    python scripts/analyze_results.py [results_folder_name]
    
    Default: spider1.0_baseline_v1
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Get results folder from command line or use default
if len(sys.argv) > 1:
    RESULTS_FOLDER = sys.argv[1]
else:
    RESULTS_FOLDER = "spider1.0_baseline_v1"

RESULTS_DIR = PROJECT_ROOT / "results" / RESULTS_FOLDER


def analyze_errors():
    """Categorize and analyze all errors from the evaluation."""
    
    with open(RESULTS_DIR / "raw_results.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Basic stats
    total = len(rows)
    correct = sum(1 for r in rows if r['execution_match'] == 'True')
    failed = [r for r in rows if r['execution_match'] == 'False']

    # Categorize errors
    error_categories = defaultdict(list)

    for r in failed:
        pred_sql = r['pred_sql'] or ''
        gold_sql = r['gold_sql'] or ''
        error = r['error'] or ''
        
        # SQL execution failed
        if r['pred_success'] == 'False':
            if 'no such column' in error.lower():
                error_categories['wrong_column'].append(r)
            elif 'no such table' in error.lower():
                error_categories['wrong_table'].append(r)
            elif 'syntax error' in error.lower():
                error_categories['syntax_error'].append(r)
            else:
                error_categories['other_exec_error'].append(r)
        else:
            # Logic/semantic errors - SQL runs but returns wrong results
            
            # Check for quoting issues (using "column" instead of column)
            if '"' in pred_sql and '"' not in gold_sql:
                error_categories['quoting_issue'].append(r)
            # Wrong aggregation function
            elif 'count(*)' in gold_sql.lower() and 'count(*)' not in pred_sql.lower():
                error_categories['wrong_aggregation'].append(r)
            # Missing JOIN
            elif 'join' in gold_sql.lower() and 'join' not in pred_sql.lower():
                error_categories['missing_join'].append(r)
            # Extra unnecessary JOIN
            elif 'join' not in gold_sql.lower() and 'join' in pred_sql.lower():
                error_categories['extra_join'].append(r)
            # Missing ORDER BY
            elif 'order by' in gold_sql.lower() and 'order by' not in pred_sql.lower():
                error_categories['missing_order_by'].append(r)
            # Missing GROUP BY
            elif 'group by' in gold_sql.lower() and 'group by' not in pred_sql.lower():
                error_categories['missing_group_by'].append(r)
            # Missing LIMIT
            elif 'limit' in gold_sql.lower() and 'limit' not in pred_sql.lower():
                error_categories['missing_limit'].append(r)
            # Wrong WHERE clause
            elif 'where' in gold_sql.lower() and 'where' not in pred_sql.lower():
                error_categories['missing_where'].append(r)
            else:
                error_categories['other_logic_error'].append(r)

    return total, correct, failed, error_categories


def generate_report(total, correct, failed, error_categories):
    """Generate formatted markdown report."""
    
    sorted_categories = sorted(error_categories.items(), key=lambda x: -len(x[1]))
    
    report = []
    report.append("# Spider 1.0 Evaluation Report - Baseline v1")
    report.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\n**Model:** Azure OpenAI (Spider fine-tuned)")
    report.append(f"\n**Dialect:** SQLite")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Summary Metrics")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total Examples | {total} |")
    report.append(f"| Correct | {correct} |")
    report.append(f"| Failed | {len(failed)} |")
    report.append(f"| **Execution Accuracy** | **{correct/total*100:.2f}%** |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Architecture")
    report.append("")
    report.append("```")
    report.append("Question → Full Schema (all columns) → LLM (zero-shot) → SQL")
    report.append("```")
    report.append("")
    report.append("**Key characteristics:**")
    report.append("- No schema retrieval (full schema passed)")
    report.append("- Zero-shot prompting (no examples)")
    report.append("- Single generation attempt (no retry)")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Error Analysis")
    report.append("")
    report.append("| Error Category | Count | % of Errors |")
    report.append("|----------------|-------|-------------|")
    
    for category, items in sorted_categories:
        pct = len(items) / len(failed) * 100
        report.append(f"| {category} | {len(items)} | {pct:.1f}% |")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Error Categories Explained")
    report.append("")
    
    category_descriptions = {
        'quoting_issue': 'Model uses double quotes around column names (e.g., "column") when not needed',
        'wrong_column': 'SQL references a column that does not exist in the database',
        'wrong_table': 'SQL references a table that does not exist',
        'syntax_error': 'SQL has syntax errors that prevent execution',
        'wrong_aggregation': 'Wrong aggregate function (e.g., COUNT(id) instead of COUNT(*))',
        'missing_join': 'Gold SQL has JOIN but predicted SQL does not',
        'extra_join': 'Predicted SQL has unnecessary JOIN',
        'missing_order_by': 'Missing ORDER BY clause',
        'missing_group_by': 'Missing GROUP BY clause',
        'missing_limit': 'Missing LIMIT clause',
        'missing_where': 'Missing WHERE clause',
        'other_exec_error': 'Other SQL execution errors',
        'other_logic_error': 'Other semantic/logic errors'
    }
    
    for category, items in sorted_categories:
        desc = category_descriptions.get(category, 'Uncategorized error')
        report.append(f"### {category.upper()} ({len(items)} errors)")
        report.append(f"\n**Description:** {desc}")
        report.append("")
        
        # Show top 3 examples
        report.append("**Examples:**")
        for i, r in enumerate(items[:3], 1):
            report.append(f"\n{i}. **Question:** {r['question']}")
            report.append(f"   - Gold: `{r['gold_sql'][:100]}{'...' if len(r['gold_sql']) > 100 else ''}`")
            pred = r['pred_sql'][:100] if r['pred_sql'] else 'None'
            report.append(f"   - Pred: `{pred}{'...' if r['pred_sql'] and len(r['pred_sql']) > 100 else ''}`")
            if r['error']:
                report.append(f"   - Error: {r['error'][:80]}")
        report.append("")
    
    report.append("---")
    report.append("")
    report.append("## Recommendations for Improvement")
    report.append("")
    report.append("Based on error analysis:")
    report.append("")
    
    # Generate recommendations based on top errors
    if 'quoting_issue' in error_categories and len(error_categories['quoting_issue']) > 20:
        report.append("1. **Fix quoting in prompt** - Add explicit instruction: 'Do not use double quotes around column names'")
    if 'missing_join' in error_categories and len(error_categories['missing_join']) > 10:
        report.append("2. **Add JOIN examples** - Include few-shot examples with JOINs")
    if 'wrong_column' in error_categories and len(error_categories['wrong_column']) > 5:
        report.append("3. **Schema retrieval** - Use semantic retrieval instead of full schema")
    if 'other_logic_error' in error_categories and len(error_categories['other_logic_error']) > 50:
        report.append("4. **Few-shot prompting** - Add 3-5 examples of question→SQL pairs")
    
    return "\n".join(report)


def main():
    print("Analyzing errors...")
    total, correct, failed, error_categories = analyze_errors()
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total} | Correct: {correct} | Failed: {len(failed)}")
    print(f"Execution Accuracy: {correct/total*100:.2f}%")
    print(f"\n{'Error Category':<25} {'Count':<8} {'% of Errors'}")
    print("-" * 50)
    
    sorted_categories = sorted(error_categories.items(), key=lambda x: -len(x[1]))
    for category, items in sorted_categories:
        pct = len(items) / len(failed) * 100
        print(f"{category:<25} {len(items):<8} {pct:.1f}%")
    
    # Generate and save report
    report = generate_report(total, correct, failed, error_categories)
    report_path = RESULTS_DIR / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Save analysis JSON
    analysis = {
        "total": total,
        "correct": correct,
        "failed": len(failed),
        "execution_accuracy": round(correct/total*100, 2),
        "error_breakdown": {cat: len(items) for cat, items in sorted_categories},
        "timestamp": datetime.now().isoformat()
    }
    
    analysis_path = RESULTS_DIR / "error_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis JSON saved to: {analysis_path}")


if __name__ == "__main__":
    main()
