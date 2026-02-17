import sys
import numpy as np
from pathlib import Path

# Allow `python src/pipeline.py` to resolve imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retriever.schema_loader import load_schema
from src.retriever.schema_retriever import SchemaRetriever
from src.generator.sql_generator_azure import generate_sql
from src.generator.retry import classify_error, build_retry_prompt, SQLErrorClass
from src.executor.bq_executor import BigQueryExecutor
from src.decomposition.orchestrator import PlannerExecutorOrchestrator


def _divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Lazy initialisation – no side effects at import time.
# ---------------------------------------------------------------------------

_schema = None
_retriever = None
_orchestrator = None
_executor = None


def _ensure_initialised():
    """Initialise schema, retriever, orchestrator and executor once on first use."""
    global _schema, _retriever, _orchestrator, _executor
    if _schema is not None:
        return

    # ── Step 1: Schema Loading ──────────────────────────────────
    enriched_path = PROJECT_ROOT / "data" / "schemas" / "bigcommerce_schema_enriched.csv"
    fallback_path = PROJECT_ROOT / "data" / "schemas" / "bigcommerce_schema_cleaned.csv"
    schema_path = enriched_path if enriched_path.exists() else fallback_path
    _schema = load_schema(str(schema_path))
    print(f"[Step 1] Schema loaded: {len(_schema)} columns from {schema_path.name}")

    # ── Step 2: Embedding schema columns ────────────────────────
    _retriever = SchemaRetriever(_schema)
    print(f"[Step 2] Schema embeddings computed: matrix shape = {_retriever.schema_embeddings.shape}")
    if _retriever._bm25 is not None:
        print(f"         BM25 index built: {len(_retriever._bm25_corpus)} documents")
    else:
        print(f"         BM25 disabled (install rank_bm25 for hybrid retrieval)")

    # ── Step 2b: Planner-Executor Orchestrator ──────────────────
    _orchestrator = PlannerExecutorOrchestrator(
        dialect="bigquery",
        enable_planner=True,
        planner_confidence_threshold=0.6,
    )
    print(f"[Step 2b] Planner-executor orchestrator ready (planner enabled)")

    # ── Executor (config-driven, with guardrails) ───────────────
    _executor = BigQueryExecutor()
    print("[Step 2c] BigQuery executor ready (safety guardrails enabled)")


def answer(question):
    """
    End-to-end Text-to-SQL pipeline with verbose output.
    """
    _ensure_initialised()
    assert _retriever is not None
    assert _orchestrator is not None
    assert _executor is not None

    # ── Step 3: Embed the question ──────────────────────────
    _divider("Step 3 — Embedding the Question")
    print(f"Question          : {question}")
    scored_results = _retriever.retrieve(question, top_k=10)
    q_emb = _retriever._last_question_embedding[0]
    emb_question = _retriever._embedding_question
    if emb_question != question:
        print(f"Embedding query   : {emb_question}  (temporal phrases stripped)")
    print(f"Embedding (first 10): {np.round(q_emb[:10], 4).tolist()}")
    print(f"Embedding shape     : {q_emb.shape}")

    # ── Step 4: Retrieval & Similarity Scores ───────────────
    _divider("Step 4 — Schema Retrieval (Top-k Similarity)")
    print(f"{'Rank':<5} {'Score':<8} {'Table':<25} {'Column':<25} {'Type'}")
    print("-" * 75)
    for rank, (score, item) in enumerate(scored_results, 1):
        print(f"{rank:<5} {score:<8.4f} {item['table']:<25} {item['column']:<25} {item['type']}")

    # Unpack items for downstream use
    retrieved_schema = [item for _, item in scored_results]

    # ── Step 5: Planner-Executor / Decomposition ───────────
    _divider("Step 5 — Query Planning & SQL Generation")
    result_bundle = _orchestrator.process(
        question, retrieved_schema, generate_sql_fn=generate_sql,
    )

    used_planner = result_bundle.get("used_planner", False)
    print(f"Complexity     : {result_bundle['complexity']} "
          f"(score={result_bundle['complexity_score']})")
    if result_bundle.get("complexity_signals"):
        print(f"Signals        : {', '.join(result_bundle['complexity_signals'])}")

    if used_planner:
        plan = result_bundle.get("plan")
        confidence = plan.confidence if plan else 0.0
        print(f"Path           : PLANNER (confidence={confidence:.2f})")
        if plan and plan.steps:
            print(f"Plan steps     : {len(plan.steps)}")
            for step in plan.steps:
                print(f"  {step.step_number}. [{step.step_type.value}] {step.description}")
        if result_bundle.get("elapsed_planning_s"):
            print(f"Planning time  : {result_bundle['elapsed_planning_s']}s")
        if result_bundle.get("elapsed_compilation_s"):
            print(f"Compile time   : {result_bundle['elapsed_compilation_s']}s")
    else:
        print(f"Path           : FALLBACK (direct generation)")
        if result_bundle.get("failure_stage"):
            print(f"Fallback reason: [{result_bundle['failure_stage']}] "
                  f"{result_bundle.get('failure_detail', '')}")
        if result_bundle.get("sub_questions"):
            print(f"Sub-questions  :")
            for i, sq in enumerate(result_bundle["sub_questions"], 1):
                print(f"  {i}. {sq}")

    if result_bundle.get("reasoning"):
        print(f"Reasoning      :\n{result_bundle['reasoning']}")

    sql = result_bundle.get("sql", "")

    # ── Step 6: Display SQL ─────────────────────────────────
    _divider("Step 6 — Generated SQL")
    schema_context = "\n".join(
        f"  {s['table']}.{s['column']} ({s['type']})" for s in retrieved_schema
    )
    print(f"Schema context sent to LLM:\n{schema_context}")
    print(f"\nGenerated SQL:\n{sql}")

    # ── Step 7: Guarded Execution ─────────────────────────
    _divider("Step 7 — BigQuery Execution (with guardrails)")
    result = _executor.execute(sql)

    # Surface block / cost information
    if result.get("blocked"):
        print(f"Status : BLOCKED")
        for reason in result.get("block_reasons", []):
            print(f"  Reason: {reason}")
    elif result["success"]:
        print("Status : SUCCESS")
        if result.get("estimated_bytes"):
            print(f"Estimated bytes: {result['estimated_bytes']:,}")
        if result.get("elapsed_seconds"):
            print(f"Elapsed: {result['elapsed_seconds']}s")
        for row in result["rows"][:10]:
            print(f"  {row}")
        if len(result["rows"]) > 10:
            print(f"  ... ({len(result['rows'])} rows total)")
    else:
        print(f"Status : FAILED")
        print(f"Error  : {result['error']}")

        # ── Step 8: Error-class-guided retry ─────
        _divider("Step 8 — Error-Class-Guided Retry")
        error_class = classify_error(result["error"], dialect="bigquery")
        print(f"Error class : {error_class.name}")
        retry_question = build_retry_prompt(
            question, result["error"], error_class, failed_sql=sql,
        )
        sql = generate_sql(retry_question, retrieved_schema)
        print(f"Retry SQL:\n{sql}")
        result = _executor.execute(sql)

        if result.get("blocked"):
            print(f"\nRetry Status : BLOCKED")
            for reason in result.get("block_reasons", []):
                print(f"  Reason: {reason}")
        elif result["success"]:
            print("\nRetry Status : SUCCESS")
            for row in result["rows"][:10]:
                print(f"  {row}")
        else:
            print(f"\nRetry Status : FAILED")
            print(f"Error        : {result['error']}")

    return {
        "question": question,
        "sql": sql,
        "result": result,
        "used_planner": used_planner,
        "plan": result_bundle.get("plan"),
        "failure_stage": result_bundle.get("failure_stage"),
        "failure_detail": result_bundle.get("failure_detail"),
    }


if __name__ == "__main__":
    q = str(input("\nAsk a question: "))
    output = answer(q)
