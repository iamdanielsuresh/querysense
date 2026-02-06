import sys
from pathlib import Path

# Allow direct execution via `python src/retriever/test_retriever.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.retriever.schema_loader import load_schema
from src.retriever.schema_retriever import SchemaRetriever


def main():
    schema_path = PROJECT_ROOT / "data" / "schemas" / "bigcommerce_schema_1.csv"
    schema = load_schema(str(schema_path))
    retriever = SchemaRetriever(schema)

    question = "What is the total revenue last month ?"
    results = retriever.retrieve(question, top_k=10)

    print("Top retrieved schema column:\n")
    for result in results:
        print(result["text"])


if __name__ == "__main__":
    main()
