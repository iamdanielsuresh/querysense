"""
Versioned Prompt Templates for SQL Generation

Each prompt version is tracked for reproducibility and thesis documentation.
Change the ACTIVE_* variables to switch between versions.

Version History:
- V1: Baseline zero-shot prompts
- V2 (SQLite): Quoting fix + few-shot examples
- V2 (BigQuery): Enhanced with aliasing rules + few-shot examples to fix column ambiguity errors
- V3 (BigQuery): Fixed incorrect order_product→category_tree join; uses product_category junction table.
                  Changed category label from search_keywords to name.
"""

# =============================================================================
# ACTIVE PROMPT SELECTION
# Change these to switch prompt versions for evaluation
# =============================================================================
ACTIVE_SQLITE_VERSION = "V2"  # Options: "V1", "V2"
ACTIVE_BIGQUERY_VERSION = "V3"  # Options: "V1", "V2", "V3"


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

BIGQUERY_V2 = """You are an expert SQL query writer for Google BigQuery.

Task: Generate a valid BigQuery SQL query for the given question.

Question:
{question}

Available schema columns:
{schema_context}

CRITICAL RULES:
1. Output ONLY the SQL query - no explanations, no markdown
2. Use backticks (`) around table references: `dataset.table`
3. ALWAYS create table aliases: `dataset.table` AS t1
4. ALWAYS reference columns using aliases: t1.column_name, t2.column_name
5. NEVER put table.column inside backticks — WRONG: `orders.id`, RIGHT: t1.id
6. NEVER reference original table names after aliasing — use t1, t2, etc.
7. Do NOT invent tables or columns not in the schema
8. For TIMESTAMP columns, use DATE() to cast: DATE(timestamp_col)
9. Use DATE_TRUNC for month/year grouping: DATE_TRUNC(DATE(col), MONTH)
10. NEVER use TIMESTAMP_SUB with MONTH or YEAR intervals

Examples:

Q: What is the total revenue for all orders?
Schema:
nb-sandbox.dataset.orders.id (INT64)
nb-sandbox.dataset.orders.total_inc_tax (FLOAT64)
SQL:
SELECT SUM(total_inc_tax) AS total_revenue
FROM `nb-sandbox.dataset.orders`

Q: Show product names and their order quantities
Schema:
nb-sandbox.dataset.order_product.order_id (INT64)
nb-sandbox.dataset.order_product.product_id (INT64)
nb-sandbox.dataset.order_product.quantity (INT64)
nb-sandbox.dataset.products.product_id (INT64)
nb-sandbox.dataset.products.name (STRING)
SQL:
SELECT t2.name, t1.quantity
FROM `nb-sandbox.dataset.order_product` AS t1
JOIN `nb-sandbox.dataset.products` AS t2 ON t1.product_id = t2.product_id

Q: Count orders per month in 2024
Schema:
nb-sandbox.dataset.orders.id (INT64)
nb-sandbox.dataset.orders.date_created (TIMESTAMP)
SQL:
SELECT DATE_TRUNC(DATE(date_created), MONTH) AS month, COUNT(*) AS order_count
FROM `nb-sandbox.dataset.orders`
WHERE DATE(date_created) >= '2024-01-01' AND DATE(date_created) < '2025-01-01'
GROUP BY month
ORDER BY month

Q: Show revenue per category with category names
Schema:
nb-sandbox.dataset.orders.id (INT64)
nb-sandbox.dataset.orders.total_inc_tax (FLOAT64)
nb-sandbox.dataset.order_product.order_id (INT64)
nb-sandbox.dataset.order_product.product_id (INT64)
nb-sandbox.dataset.category_tree.category_id (INT64)
nb-sandbox.dataset.category_tree.search_keywords (STRING)
SQL:
SELECT t3.search_keywords, SUM(t1.total_inc_tax) AS revenue
FROM `nb-sandbox.dataset.orders` AS t1
JOIN `nb-sandbox.dataset.order_product` AS t2 ON t1.id = t2.order_id
JOIN `nb-sandbox.dataset.category_tree` AS t3 ON t2.product_id = t3.category_id
GROUP BY t3.search_keywords

Now generate SQL for:
Question: {question}
SQL:"""

# BigQuery V3 — fixes the incorrect product_id→category_id join from V2.
# The revenue-per-category example now routes through the product_category
# junction table to correctly resolve products to categories.
BIGQUERY_V3 = """You are an expert SQL query writer for Google BigQuery.

Task: Generate a valid BigQuery SQL query for the given question.

Question:
{question}

Available schema columns:
{schema_context}

CRITICAL RULES:
1. Output ONLY the SQL query - no explanations, no markdown
2. Use backticks (`) around table references: `dataset.table`
3. ALWAYS create table aliases: `dataset.table` AS t1
4. ALWAYS reference columns using aliases: t1.column_name, t2.column_name
5. NEVER put table.column inside backticks — WRONG: `orders.id`, RIGHT: t1.id
6. NEVER reference original table names after aliasing — use t1, t2, etc.
7. Do NOT invent tables or columns not in the schema
8. For TIMESTAMP columns, use DATE() to cast: DATE(timestamp_col)
9. Use DATE_TRUNC for month/year grouping: DATE_TRUNC(DATE(col), MONTH)
10. NEVER use TIMESTAMP_SUB with MONTH or YEAR intervals
11. When joining products to categories, use the product_category junction table

Examples:

Q: What is the total revenue for all orders?
Schema:
nb-sandbox.dataset.orders.id (INT64)
nb-sandbox.dataset.orders.total_inc_tax (FLOAT64)
SQL:
SELECT SUM(total_inc_tax) AS total_revenue
FROM `nb-sandbox.dataset.orders`

Q: Show product names and their order quantities
Schema:
nb-sandbox.dataset.order_product.order_id (INT64)
nb-sandbox.dataset.order_product.product_id (INT64)
nb-sandbox.dataset.order_product.quantity (INT64)
nb-sandbox.dataset.products.product_id (INT64)
nb-sandbox.dataset.products.name (STRING)
SQL:
SELECT t2.name, t1.quantity
FROM `nb-sandbox.dataset.order_product` AS t1
JOIN `nb-sandbox.dataset.products` AS t2 ON t1.product_id = t2.product_id

Q: Count orders per month in 2024
Schema:
nb-sandbox.dataset.orders.id (INT64)
nb-sandbox.dataset.orders.date_created (TIMESTAMP)
SQL:
SELECT DATE_TRUNC(DATE(date_created), MONTH) AS month, COUNT(*) AS order_count
FROM `nb-sandbox.dataset.orders`
WHERE DATE(date_created) >= '2024-01-01' AND DATE(date_created) < '2025-01-01'
GROUP BY month
ORDER BY month

Q: Show revenue per category with category names
Schema:
nb-sandbox.dataset.orders.id (INT64)
nb-sandbox.dataset.orders.total_inc_tax (FLOAT64)
nb-sandbox.dataset.order_product.order_id (INT64)
nb-sandbox.dataset.order_product.product_id (INT64)
nb-sandbox.dataset.product_category.product_id (INT64)
nb-sandbox.dataset.product_category.category_id (INT64)
nb-sandbox.dataset.category_tree.category_id (INT64)
nb-sandbox.dataset.category_tree.name (STRING)
SQL:
SELECT t4.name AS category_name, SUM(t1.total_inc_tax) AS revenue
FROM `nb-sandbox.dataset.orders` AS t1
JOIN `nb-sandbox.dataset.order_product` AS t2 ON t1.id = t2.order_id
JOIN `nb-sandbox.dataset.product_category` AS t3 ON t2.product_id = t3.product_id
JOIN `nb-sandbox.dataset.category_tree` AS t4 ON t3.category_id = t4.category_id
GROUP BY t4.name

Now generate SQL for:
Question: {question}
SQL:"""


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
        "V2": BIGQUERY_V2,
        "V3": BIGQUERY_V3,
    }
    return prompts[ACTIVE_BIGQUERY_VERSION]


def get_active_versions() -> dict:
    """Return currently active prompt versions (for logging/tracking)."""
    return {
        "sqlite": ACTIVE_SQLITE_VERSION,
        "bigquery": ACTIVE_BIGQUERY_VERSION,
    }
