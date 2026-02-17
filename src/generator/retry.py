"""
Error-class-based retry prompts for SQL generation.

Instead of blindly appending raw error messages to the question, this module
classifies execution errors into an enum and selects targeted guidance for
each error class.  The retry prompt is constructed from:

    original question + schema context + error class hint

No domain-specific or hardcoded column-level hints are used.

Usage:
    from src.generator.retry import classify_error, build_retry_prompt

    error_class = classify_error(error_message, dialect="bigquery")
    retry_q = build_retry_prompt(question, error_message, error_class)
"""
from __future__ import annotations

import re
from enum import Enum, auto
from typing import Optional


# ── Error taxonomy ─────────────────────────────────────────────────────

class SQLErrorClass(Enum):
    """Dialect-agnostic error taxonomy covering the most common SQL
    execution failures observed in text-to-SQL systems."""

    SYNTAX_ERROR        = auto()  # malformed SQL
    NO_SUCH_TABLE       = auto()  # table does not exist
    NO_SUCH_COLUMN      = auto()  # column does not exist / ambiguous
    AMBIGUOUS_COLUMN    = auto()  # column reference is ambiguous
    TYPE_MISMATCH       = auto()  # incompatible operand types
    GROUP_BY_ERROR      = auto()  # non-aggregated column not in GROUP BY
    JOIN_ERROR          = auto()  # join condition mismatch / missing key
    FUNCTION_ERROR      = auto()  # unknown or misused SQL function
    TIMEOUT             = auto()  # query exceeded time / resource limit
    PERMISSION_ERROR    = auto()  # access denied / auth failure
    EMPTY_QUERY         = auto()  # empty or null SQL submitted
    UNKNOWN             = auto()  # unclassified


# ── Classification rules ───────────────────────────────────────────────
# Each rule is (compiled_regex, error_class).  First match wins.

_CLASSIFICATION_RULES = [
    # Empty / null query
    (re.compile(r"empty query|no query|query is empty", re.I),
     SQLErrorClass.EMPTY_QUERY),

    # Table not found
    (re.compile(r"no such table|table .* not found|not found: table|"
                r"does not exist.*table|table .* was not found|"
                r"unknown table", re.I),
     SQLErrorClass.NO_SUCH_TABLE),

    # Ambiguous column (check before generic "no such column")
    (re.compile(r"ambiguous column|ambiguous.*reference|column reference .* is ambiguous", re.I),
     SQLErrorClass.AMBIGUOUS_COLUMN),

    # Column not found
    (re.compile(r"no such column|column .* not found|"
                r"Unrecognized name|unknown column|"
                r"does not exist.*column|field .* not found|"
                r"name .* not found inside", re.I),
     SQLErrorClass.NO_SUCH_COLUMN),

    # Type mismatch
    (re.compile(r"type mismatch|incompatible type|cannot (cast|convert)|"
                r"operand type|invalid.*input.*type|"
                r"No matching signature for", re.I),
     SQLErrorClass.TYPE_MISMATCH),

    # GROUP BY issues
    (re.compile(r"not in GROUP BY|must appear in the GROUP BY|"
                r"not.*aggregate.*function.*GROUP BY|"
                r"SELECT list.*not in GROUP BY", re.I),
     SQLErrorClass.GROUP_BY_ERROR),

    # Join issues
    (re.compile(r"join.*error|cannot.*join|"
                r"join key.*type|invalid join", re.I),
     SQLErrorClass.JOIN_ERROR),

    # Function errors
    (re.compile(r"unknown function|no.*function.*matches|"
                r"function .* not (found|recognized)|"
                r"TIMESTAMP_SUB.*MONTH|TIMESTAMP_SUB.*YEAR|"
                r"Interval.*not supported|"
                r"wrong number of arguments", re.I),
     SQLErrorClass.FUNCTION_ERROR),

    # Timeout / resource
    (re.compile(r"timeout|timed out|resources exceeded|"
                r"deadline exceeded|cancelled", re.I),
     SQLErrorClass.TIMEOUT),

    # Permission / auth
    (re.compile(r"access denied|permission|unauthorized|"
                r"forbidden|not authorized", re.I),
     SQLErrorClass.PERMISSION_ERROR),

    # Generic syntax errors — keep last among parseable errors
    (re.compile(r"syntax error|parse error|unexpected token|"
                r"mismatched input|expecting|near \"|"
                r"Syntax error in SQL", re.I),
     SQLErrorClass.SYNTAX_ERROR),
]


def classify_error(
    error_message: str,
    dialect: str = "bigquery",  # reserved for future dialect-specific rules
) -> SQLErrorClass:
    """Classify a SQL execution error message into a ``SQLErrorClass``."""
    if not error_message or not error_message.strip():
        return SQLErrorClass.EMPTY_QUERY
    for pattern, cls in _CLASSIFICATION_RULES:
        if pattern.search(error_message):
            return cls
    return SQLErrorClass.UNKNOWN


# ── Retry guidance per error class ─────────────────────────────────────
# Each value is a concise, schema-agnostic instruction appended to the
# retry prompt.  No column names, table names, or domain-specific
# content appears here.

_RETRY_GUIDANCE: dict[SQLErrorClass, str] = {
    SQLErrorClass.SYNTAX_ERROR: (
        "The previous SQL had a syntax error.  "
        "Double-check keyword order, parentheses, commas, and quoting.  "
        "Regenerate a syntactically valid query."
    ),
    SQLErrorClass.NO_SUCH_TABLE: (
        "The previous SQL referenced a table that does not exist.  "
        "Use ONLY the tables listed in the schema below.  "
        "Check for typos and correct fully-qualified table paths."
    ),
    SQLErrorClass.NO_SUCH_COLUMN: (
        "The previous SQL referenced a column that does not exist.  "
        "Use ONLY the column names listed in the schema below.  "
        "Ensure column names match exactly (case-sensitive) and are not "
        "confused with columns from other tables."
    ),
    SQLErrorClass.AMBIGUOUS_COLUMN: (
        "A column name was ambiguous because it exists in multiple tables.  "
        "Prefix every column reference with the correct table alias "
        "(e.g., t1.column_name) to resolve ambiguity."
    ),
    SQLErrorClass.TYPE_MISMATCH: (
        "The previous SQL had a type mismatch.  "
        "Check that comparison and join operands have compatible types.  "
        "Cast values explicitly where necessary (e.g., CAST, DATE(), SAFE_CAST)."
    ),
    SQLErrorClass.GROUP_BY_ERROR: (
        "The previous SQL had a GROUP BY error.  "
        "Every non-aggregated column in the SELECT list must appear in GROUP BY.  "
        "Verify that all selected columns are either aggregated or grouped."
    ),
    SQLErrorClass.JOIN_ERROR: (
        "The previous SQL had a join error.  "
        "Ensure join keys reference the correct foreign-key relationships "
        "as shown in the schema.  Do not join on semantically unrelated columns."
    ),
    SQLErrorClass.FUNCTION_ERROR: (
        "The previous SQL used an unsupported or incorrectly called function.  "
        "Use only functions available in the target SQL dialect.  "
        "For BigQuery: avoid TIMESTAMP_SUB with MONTH/YEAR; use DATE_SUB with "
        "DATE types.  Check the number and type of arguments."
    ),
    SQLErrorClass.TIMEOUT: (
        "The previous query timed out or exceeded resource limits.  "
        "Simplify the query: reduce joins, add WHERE filters, or use "
        "LIMIT to restrict output size."
    ),
    SQLErrorClass.PERMISSION_ERROR: (
        "Access was denied.  The query may reference a table or dataset "
        "outside the allowed scope.  Use only the tables in the provided schema."
    ),
    SQLErrorClass.EMPTY_QUERY: (
        "No SQL was generated.  Please generate a complete SQL query "
        "that answers the question using the schema provided."
    ),
    SQLErrorClass.UNKNOWN: (
        "The previous SQL failed with the error shown below.  "
        "Analyse the error message, fix the issue, and regenerate."
    ),
}


def build_retry_prompt(
    question: str,
    error_message: str,
    error_class: SQLErrorClass,
    *,
    failed_sql: Optional[str] = None,
) -> str:
    """Build a structured retry prompt for the LLM.

    Parameters
    ----------
    question : str
        The original natural-language question.
    error_message : str
        The raw error message from execution.
    error_class : SQLErrorClass
        The classified error category.
    failed_sql : str, optional
        The SQL that failed (included to help the model avoid the same mistake).

    Returns
    -------
    str
        A prompt string to use as the ``question`` argument to ``generate_sql``.
    """
    guidance = _RETRY_GUIDANCE.get(error_class, _RETRY_GUIDANCE[SQLErrorClass.UNKNOWN])

    parts = [question]
    parts.append(f"\n--- RETRY CONTEXT ---")
    parts.append(f"Error class: {error_class.name}")
    parts.append(f"Error message: {error_message}")
    if failed_sql:
        parts.append(f"Failed SQL:\n{failed_sql}")
    parts.append(f"\nGuidance: {guidance}")
    parts.append("Generate a corrected SQL query.")

    return "\n".join(parts)
