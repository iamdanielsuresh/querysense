import csv
from pathlib import Path

def load_schema(csv_path):
    schema_items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = f"{row['table_name']}.{row['column_name']} {row['column_name']} {row['data_type']}"
            schema_items.append({
                "table": row["table_name"],
                "column": row["column_name"],
                "type":row["data_type"],
                "text": text
            })

    return schema_items


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    schema_path = project_root / "data" / "schemas" / "bigcommerce_schema_1.csv"
    schema = load_schema(str(schema_path))
    print(f"Loaded {len(schema)} schema columns")
    print(schema[:5])
