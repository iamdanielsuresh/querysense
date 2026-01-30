import csv

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
    schema = load_schema("/Users/daniel/Projects/text2sql/bquxjob_26d42422_19c09963dbb.csv")
    print(f"Loaded {len(schema)} schema coloums")
    print(schema[:5])
