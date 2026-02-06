
#script to evaluate the model:
#python evaluation/evaluate_model.py --dev_file dev.json --tables_file tables.json --db_dir database
import json
import sqlite3
import os
import argparse
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

# Configuration - use environment variables
ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "")
# API_VERSION = "2024-12-01-preview"

def load_tables(tables_file):
    """Loads schema information from tables.json."""
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    
    # Index by db_id for easy access
    tables_by_db = {t['db_id']: t for t in tables}
    return tables_by_db

def get_schema_string(table_data):
    """
    Formats schema for the prompt, matching the training format.
    Format:
    Table: [table_name], columns: [col1, col2, ...]
    """
    schema_lines = []
    table_names = table_data['table_names_original']
    column_names = table_data['column_names_original']
    
    # Group columns by table index
    columns_by_table = {}
    for i, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1: continue # Skip *
        if table_idx not in columns_by_table:
            columns_by_table[table_idx] = []
        columns_by_table[table_idx].append(col_name)
        
    for table_idx, table_name in enumerate(table_names):
        cols = columns_by_table.get(table_idx, [])
        schema_lines.append(f"Table: {table_name}, columns: [{', '.join(cols)}]")
        
    return "\n".join(schema_lines)

def get_sql_query(client, model_name, question, db_schema):
    """Queries the model."""
    system_message = f"""You are a helpful assistant that translates natural language questions into SQL queries.
The database schema is as follows:
{db_schema}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            temperature=1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying model: {e}")
        return "SELECT *" # Fallback

def execute_sql(db_path, sql):
    """Executes SQL and returns results."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        # print(f"Execution error: {e}") # Optional: verbose logging
        return None

def normalize_sql(sql):
    """Basic normalization for string comparison."""
    if not sql: return ""
    return " ".join(sql.replace("\n", " ").split()).lower().strip(';')

def eval_exact_match(pred, gold):
    """Checks for exact string match (normalized)."""
    return normalize_sql(pred) == normalize_sql(gold)

def eval_exec_match(db_path, pred, gold):
    """Compares execution results."""
    pred_res = execute_sql(db_path, pred)
    gold_res = execute_sql(db_path, gold)
    
    # If both failed (None), we count it as no match (or match if you prefer, but usually no)
    # If one failed and other didn't, no match.
    if pred_res is None or gold_res is None:
        return False
        
    # Compare results (set comparison to ignore order if order doesn't matter, 
    # but for some queries order matters. Spider eval usually treats results as sets 
    # unless ORDER BY is involved. For simplicity here, we compare lists directly first, 
    # then sets if lists don't match to be lenient).
    if pred_res == gold_res:
        return True
    
    try:
        return set(pred_res) == set(gold_res)
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file", default="dev.json", help="Path to dev.json")
    parser.add_argument("--tables_file", default="tables.json", help="Path to tables.json")
    parser.add_argument("--db_dir", default="database", help="Path to database directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file")
    
    # Model Configuration
    parser.add_argument("--provider", choices=["azure", "ollama", "openai"], default="azure", help="Model provider")
    parser.add_argument("--model", help="Model deployment name (Azure) or model name (Ollama/OpenAI)")
    parser.add_argument("--endpoint", help="Azure Endpoint or Base URL")
    parser.add_argument("--api_key", help="API Key")
    parser.add_argument("--api_version", default="2024-12-01-preview", help="API Version (Azure)")
    
    args = parser.parse_args()

    # Defaults from hardcoded constants if not provided (for backward compatibility/convenience)
    # You can update these constants at the top of the file for your most frequent model
    endpoint = args.endpoint or ENDPOINT
    api_key = args.api_key or API_KEY
    model_name = args.model or DEPLOYMENT_NAME
    api_version = args.api_version or API_VERSION

    # Initialize client
    if args.provider == "azure":
        if not api_key or "YOUR_API_KEY" in api_key:
             print("WARNING: API_KEY not set. Please provide --api_key or edit script.")
        
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        print(f"Using Azure OpenAI: {model_name}")
        
    elif args.provider == "ollama":
        client = OpenAI(
            base_url=endpoint or "http://localhost:11434/v1",
            api_key="ollama"
        )
        model_name = args.model or "codellama:7b" # Default for Ollama if not specified
        print(f"Using Ollama: {model_name}")

    elif args.provider == "openai":
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )
        print(f"Using OpenAI Compatible: {model_name}")

    tables_by_db = load_tables(args.tables_file)
    
    with open(args.dev_file, 'r') as f:
        data = json.load(f)
        
    if args.limit:
        data = data[:args.limit]
        
    results = []
    exact_matches = 0
    exec_matches = 0
    
    print(f"Starting evaluation on {len(data)} examples...")
    
    for item in tqdm(data):
        db_id = item['db_id']
        question = item['question']
        gold_sql = item['query']
        
        # Get Schema
        if db_id not in tables_by_db:
            print(f"Missing schema for {db_id}, skipping.")
            continue
            
        schema_str = get_schema_string(tables_by_db[db_id])
        
        # Generate SQL
        pred_sql = get_sql_query(client, model_name, question, schema_str)
        
        # Evaluate Exact Match
        is_exact = eval_exact_match(pred_sql, gold_sql)
        if is_exact: exact_matches += 1
        
        # Evaluate Execution Match
        db_path = os.path.join(args.db_dir, db_id, f"{db_id}.sqlite")
        is_exec = False
        if os.path.exists(db_path):
            is_exec = eval_exec_match(db_path, pred_sql, gold_sql)
            if is_exec: exec_matches += 1
        else:
            print(f"Database not found: {db_path}")

        results.append({
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": is_exact,
            "exec_match": is_exec,
            "db_id": db_id
        })
        
    # Summary
    total = len(data)
    em_score = exact_matches / total if total > 0 else 0
    ex_score = exec_matches / total if total > 0 else 0
    
    print("\n--- Evaluation Results ---")
    print(f"Total Examples: {total}")
    print(f"Exact Match Accuracy: {em_score:.2%}")
    print(f"Execution Accuracy:   {ex_score:.2%}")
    
    with open(args.output, 'w') as f:
        json.dump({
            "metrics": {"exact_match": em_score, "execution_accuracy": ex_score},
            "details": results
        }, f, indent=2)
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()
