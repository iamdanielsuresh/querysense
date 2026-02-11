"""Quick test for the enhanced schema retriever."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever.schema_loader import load_schema
from src.retriever.schema_retriever import SchemaRetriever

# Load enriched schema
schema = load_schema(str(PROJECT_ROOT / "data/schemas/bigcommerce_schema_enriched.csv"))
print(f"Loaded {len(schema)} columns")
print(f"Sample item: {schema[0]}")
print()

# Create retriever
retriever = SchemaRetriever(schema)
print(f"Embeddings shape: {retriever.schema_embeddings.shape}")
print(f"BM25 active: {retriever._bm25 is not None}")
print(f"Semantic weight: {retriever.semantic_weight}")
print(f"BM25 weight: {retriever.bm25_weight}")
print(f"Table priors: {len(retriever.table_priors)} keywords")
print()

# Test retrieval
queries = [
    "What is the total revenue last month?",
    "Which customers placed the most orders?",
    "Show me top selling products by quantity",
    "How many coupons were redeemed?",
]

for q in queries:
    results = retriever.retrieve(q)
    print(f"Query: {q}")
    print(f"Retrieved {len(results)} columns:")
    for i, (score, item) in enumerate(results):
        desc = item.get("description", "")[:50]
        print(f"  {i+1}. [{score:.4f}] {item['table']}.{item['column']} ({item['type']}) {desc}")
    print()

# Test caching - second load should be fast
print("Testing cache reload...")
import time
start = time.time()
retriever2 = SchemaRetriever(schema)
elapsed = time.time() - start
print(f"Second load time: {elapsed:.3f}s (should be faster with cache)")
print(f"Cache dir exists: {retriever2._cache_dir.exists()}")
