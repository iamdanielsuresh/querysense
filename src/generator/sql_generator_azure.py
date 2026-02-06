"""
SQL Generator using Azure OpenAI (Spider-fine-tuned model)

This module:
- Takes a natural language question
- Takes retrieved schema columns
- Uses a fine-tuned GPT model on Azure
- Outputs SQL compatible with specified dialect (BigQuery or SQLite)

Prompt versions are managed in prompts.py for easy A/B testing.
"""
import os
import re
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.generator.prompts import get_sqlite_prompt, get_bigquery_prompt, get_active_versions

# Load environment variables from .env file
load_dotenv()
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
BQ_DATASET = os.getenv("BQ_DATASET", "")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def _clean_sql_output(sql: str) -> str:
    """Remove markdown code blocks and extra whitespace from LLM output."""
    # Remove ```sql ... ``` or ``` ... ```
    sql = re.sub(r"^```(?:sql)?\s*", "", sql.strip())
    sql = re.sub(r"\s*```$", "", sql.strip())
    return sql.strip()


def generate_sql(question, schema_items, dialect="bigquery"):
    """
    Generate SQL using a fine-tuned Azure GPT model.

    Inputs:
    - question (str): natural language analytics question
    - schema_items (list): output of SchemaRetriever (top-k columns)
    - dialect (str): "bigquery" or "sqlite"

    Output:
    - sql (str): generated SQL query
    """

    # -----------------------------
    # Build schema context based on dialect
    # -----------------------------
    if dialect == "sqlite":
        # For SQLite: table.column (type)
        schema_context = "\n".join(
            f"{s['table']}.{s['column']} ({s['type']})"
            for s in schema_items
        )
        prompt_template = get_sqlite_prompt()
        prompt = prompt_template.format(
            question=question,
            schema_context=schema_context
        )
        system_message = "You generate correct and executable SQLite SQL only."
    else:
        # For BigQuery: dataset.table.column (type)
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
        prompt_template = get_bigquery_prompt()
        prompt = prompt_template.format(
            question=question,
            schema_context=schema_context
        )
        system_message = "You generate correct and executable BigQuery SQL only."

    # -----------------------------
    # Call Azure OpenAI Chat Completion
    # -----------------------------
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": system_message
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

    # Extract SQL text and clean up
    sql = response.choices[0].message.content.strip()
    sql = _clean_sql_output(sql)

    return sql