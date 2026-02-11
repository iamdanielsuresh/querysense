import csv
from pathlib import Path


def load_schema(csv_path):
    """
    Load schema from a CSV file.

    Supports both the original 3-column format (table_name, column_name, data_type)
    and the enriched 4-column format (+ description).

    When descriptions are present, they are included in the embedding text
    to improve semantic retrieval quality.
    """
    schema_items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_description = "description" in fieldnames

        for row in reader:
            table = row["table_name"]
            column = row["column_name"]
            dtype = row["data_type"]
            description = row.get("description", "").strip() if has_description else ""

            # Build embedding text:
            # With description: "orders.total_inc_tax total_inc_tax FLOAT64 — Total order amount including tax (revenue)"
            # Without:          "orders.total_inc_tax total_inc_tax FLOAT64"
            if description:
                text = f"{table}.{column} {column} {dtype} — {description}"
            else:
                text = f"{table}.{column} {column} {dtype}"

            item = {
                "table": table,
                "column": column,
                "type": dtype,
                "text": text,
            }
            if description:
                item["description"] = description

            schema_items.append(item)

    return schema_items


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    # Test original schema
    schema_path = project_root / "data" / "schemas" / "bigcommerce_schema_1.csv"
    schema = load_schema(str(schema_path))
    print(f"Loaded {len(schema)} schema columns (original)")
    print(schema[:3])

    # Test enriched schema
    enriched_path = project_root / "data" / "schemas" / "bigcommerce_schema_enriched.csv"
    if enriched_path.exists():
        schema_e = load_schema(str(enriched_path))
        print(f"\nLoaded {len(schema_e)} schema columns (enriched)")
        print(schema_e[:3])
