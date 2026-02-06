"""
SQL Generator using Azure OpenAI (Spider-fine-tuned model)

This module:
- Takes a natural language question
- Takes retrieved schema columns
- Uses a fine-tuned GPT model on Azure
- Outputs BigQuery-compatible SQL
"""
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
BQ_DATASET = os.getenv("BQ_DATASET", "")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)



def generate_sql(question, schema_items):
    """
    Generate BigQuery-compatible SQL using a fine-tuned Azure GPT model.

    Inputs:
    - question (str): natural language analytics question
    - schema_items (list): output of SchemaRetriever (top-k columns)

    Output:
    - sql (str): generated SQL query
    """

    # -----------------------------
    # Build schema context
    # -----------------------------
    # Convert retrieved schema columns into a compact text block
    # This is the ONLY schema the model sees
    # If a dataset prefix is configured, qualify table names
    if BQ_DATASET:
        schema_context = "\n".join(
            f"{BQ_DATASET}.{s['table']}.{s['column']} ({s['type']})"
            for s in schema_items
        )
    else:
        schema_context = "\n".join(
            f"{s['table']}.{s['column']} ({s['type']})"
            for s in schema_items
        )

    # -----------------------------
    # Prompt design
    # -----------------------------
    # We:
    # - lock dialect to BigQuery
    # - forbid hallucination
    # - force SQL-only output
    prompt = f"""
You are an expert data engineer.

Task:
Generate a VALID Google BigQuery SQL query.

Question:
{question}

Available schema columns:
{schema_context}

Rules:
- Use backticks (`) around table names
- ALWAYS qualify table names with the full dataset path shown above (e.g. `dataset.table`)
- Do NOT invent tables or columns
- Use JOINs only if required
- For date/time filtering on TIMESTAMP columns, cast to DATE first: e.g. DATE(column) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
- NEVER use TIMESTAMP_SUB with MONTH or YEAR intervals (BigQuery does not support it)
- Output ONLY the SQL query
"""

    # -----------------------------
    # Call Azure OpenAI Chat Completion
    # -----------------------------
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": "You generate correct and executable BigQuery SQL only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1024,
        temperature=0.0,   # deterministic SQL
        top_p=1.0
    )

    # Extract SQL text
    sql = response.choices[0].message.content.strip()

    return sql