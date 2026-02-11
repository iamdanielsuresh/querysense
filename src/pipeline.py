import sys
import numpy as np
from pathlib import Path

# Allow `python src/pipeline.py` to resolve imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retriever.schema_loader import load_schema
from src.retriever.schema_retriever import SchemaRetriever
from src.generator.sql_generator_azure import generate_sql
from src.executor.bq_executor import execute_sql


def _divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Step 1: Schema Loading ──────────────────────────────────
# Prefer the enriched schema (with descriptions) if available
enriched_path = PROJECT_ROOT / "data" / "schemas" / "bigcommerce_schema_enriched.csv"
fallback_path = PROJECT_ROOT / "data" / "schemas" / "bigcommerce_schema_cleaned.csv"
schema_path = enriched_path if enriched_path.exists() else fallback_path
schema = load_schema(str(schema_path))
print(f"[Step 1] Schema loaded: {len(schema)} columns from {schema_path.name}")

# ── Step 2: Embedding schema columns ────────────────────────
retriever = SchemaRetriever(schema)
print(f"[Step 2] Schema embeddings computed: matrix shape = {retriever.schema_embeddings.shape}")
if retriever._bm25 is not None:
    print(f"         BM25 index built: {len(retriever._bm25_corpus)} documents")
else:
    print(f"         BM25 disabled (install rank_bm25 for hybrid retrieval)")


def answer(question):
    """
    End-to-end Text-to-SQL pipeline with verbose output.
    """

    # ── Step 3: Embed the question ──────────────────────────
    _divider("Step 3 — Embedding the Question")
    print(f"Question          : {question}")
    scored_results = retriever.retrieve(question, top_k=10)
    q_emb = retriever._last_question_embedding[0]
    emb_question = retriever._embedding_question
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

    # ── Step 5: SQL Generation ──────────────────────────────
    _divider("Step 5 — SQL Generation")
    schema_context = "\n".join(
        f"  {s['table']}.{s['column']} ({s['type']})" for s in retrieved_schema
    )
    print(f"Schema context sent to LLM:\n{schema_context}")
    sql = generate_sql(question, retrieved_schema)
    print(f"\nGenerated SQL:\n{sql}")

    # ── Step 6: Execution ───────────────────────────────────
    _divider("Step 6 — BigQuery Execution")
    result = execute_sql(sql)

    if result["success"]:
        print("Status : SUCCESS")
        for row in result["rows"][:10]:
            print(f"  {row}")
        if len(result["rows"]) > 10:
            print(f"  ... ({len(result['rows'])} rows total)")
    else:
        print(f"Status : FAILED")
        print(f"Error  : {result['error']}")

        # ── Step 7: Retry ───────────────────────────────────
        _divider("Step 7 — Execution-Guided Retry")
        retry_question = question + "\nPrevious SQL error: " + result["error"]
        sql = generate_sql(retry_question, retrieved_schema)
        print(f"Retry SQL:\n{sql}")
        result = execute_sql(sql)

        if result["success"]:
            print("\nRetry Status : SUCCESS")
            for row in result["rows"][:10]:
                print(f"  {row}")
        else:
            print(f"\nRetry Status : FAILED")
            print(f"Error        : {result['error']}")

    return {
        "question": question,
        "sql": sql,
        "result": result
    }


if __name__ == "__main__":
    q = str(input("\nAsk a question: "))
    output = answer(q)
