"""
Versioned Prompt Templates for SQL Generation

Each prompt version is tracked for reproducibility and thesis documentation.
Change the ACTIVE_* variables to switch between versions.

Version History:
- V1: Baseline zero-shot prompts
- V2: Quoting fix + few-shot examples (SQLite only)
"""

# =============================================================================
# ACTIVE PROMPT SELECTION
# Change these to switch prompt versions for evaluation
# =============================================================================
ACTIVE_SQLITE_VERSION = "V2"  # Options: "V1", "V2"
ACTIVE_BIGQUERY_VERSION = "V1"  # Options: "V1"


# =============================================================================
# SQLITE PROMPTS
# =============================================================================

SQLITE_V1 = """You are an expert data engineer.

Task:
Generate a VALID SQLite SQL query.

Question:
{question}

Database schema:
{schema_context}

Rules:
- Use standard SQLite syntax
- Do NOT use backticks for identifiers; use double quotes only if needed for special characters
- Do NOT invent tables or columns not listed in the schema
- Use JOINs only if required
- For date filtering use SQLite date functions: date(), datetime(), strftime()
- Output ONLY the SQL query, no explanations
"""

SQLITE_V2 = """You are an expert SQL query writer.

Task: Generate a valid SQLite query for the given question.

Question:
{question}

Schema:
{schema_context}

Rules:
1. Output ONLY the SQL query - no explanations, no markdown
2. Do NOT use double quotes around column or table names
3. Do NOT prefix column names with table names inside quotes (wrong: "table.column")
4. Use column names EXACTLY as shown in schema
5. Use JOINs when data spans multiple tables
6. For aggregations, use appropriate GROUP BY

Examples:

Q: How many singers are there?
Schema: singer.singer_id (number), singer.name (text), singer.age (number)
SQL: SELECT count(*) FROM singer

Q: What is the name of the singer with the highest age?
Schema: singer.singer_id (number), singer.name (text), singer.age (number)
SQL: SELECT name FROM singer ORDER BY age DESC LIMIT 1

Q: Find all concert names and the stadium names where they were held.
Schema: concert.concert_id (number), concert.concert_name (text), concert.stadium_id (number), stadium.stadium_id (number), stadium.name (text)
SQL: SELECT T1.concert_name, T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id

Q: What are the countries with singers above age 20?
Schema: singer.singer_id (number), singer.name (text), singer.country (text), singer.age (number)
SQL: SELECT DISTINCT country FROM singer WHERE age > 20

Q: Show the stadium name and number of concerts in each stadium.
Schema: concert.concert_id (number), concert.stadium_id (number), stadium.stadium_id (number), stadium.name (text)
SQL: SELECT T2.name, count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id

Now generate SQL for:
Question: {question}
SQL:"""


# =============================================================================
# BIGQUERY PROMPTS
# =============================================================================

BIGQUERY_V1 = """You are an expert data engineer.

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


# =============================================================================
# PROMPT GETTER FUNCTIONS
# =============================================================================

def get_sqlite_prompt() -> str:
    """Get the currently active SQLite prompt template."""
    prompts = {
        "V1": SQLITE_V1,
        "V2": SQLITE_V2,
    }
    return prompts[ACTIVE_SQLITE_VERSION]


def get_bigquery_prompt() -> str:
    """Get the currently active BigQuery prompt template."""
    prompts = {
        "V1": BIGQUERY_V1,
    }
    return prompts[ACTIVE_BIGQUERY_VERSION]


def get_active_versions() -> dict:
    """Return currently active prompt versions (for logging/tracking)."""
    return {
        "sqlite": ACTIVE_SQLITE_VERSION,
        "bigquery": ACTIVE_BIGQUERY_VERSION,
    }
