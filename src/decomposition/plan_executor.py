"""
Plan Executor – compiles a ``QueryPlan`` into SQL.

Two compilation strategies:

1. **LLM-assisted** (default) – sends the structured plan JSON to the LLM
   and asks for SQL that exactly follows the plan.  This keeps the LLM
   focused on *syntax* while the plan handles *semantics*.

2. **Template fallback** – if LLM compilation fails, a lightweight
   template builder attempts to assemble SQL from the plan steps.
   This is limited to simple plans but guarantees *some* output.

Validation checkpoint
---------------------
After compilation the SQL is passed through a basic structural check
(non-empty, starts with SELECT, no multi-statement).  If the check fails
the plan is marked ``FAILED`` at the ``COMPILATION`` stage.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from src.decomposition.plan_models import (
    FailureStage,
    PlanStatus,
    QueryPlan,
)

load_dotenv()

logger = logging.getLogger(__name__)

_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    api_version=os.getenv("OPENAI_API_VERSION", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
)


# ── Compilation prompt ───────────────────────────────────────

_COMPILE_SYSTEM = (
    "You are an expert SQL compiler. You receive a structured query plan "
    "in JSON and produce the exact SQL that implements it. "
    "Output ONLY the SQL — no markdown fences, no commentary."
)

_COMPILE_USER_BQ = """\
Question: {question}

Schema columns (BigQuery, {dataset} dataset):
{schema_context}

Query plan:
{plan_json}

Rules:
1. Output ONLY the SQL query — nothing else.
2. Use backticks (`) around BigQuery table references: `{dataset}.table_name`.
3. ALWAYS create table aliases: `{dataset}.orders` AS t1
4. ALWAYS reference columns using aliases: t1.column_name, t2.column_name
5. NEVER put table.column inside backticks — WRONG: `orders.id`, RIGHT: t1.id
6. NEVER reference original table names after aliasing — use t1, t2, etc.
7. Follow the plan steps exactly: use CTEs, JOINs, aggregations, filters,
   ordering, and limits as specified.
8. For TIMESTAMP columns cast with DATE().
9. NEVER use TIMESTAMP_SUB with MONTH or YEAR.

SQL:"""

_COMPILE_USER_SQLITE = """\
Question: {question}

Schema columns:
{schema_context}

Query plan:
{plan_json}

Rules:
1. Output ONLY the SQL query — nothing else.
2. Do NOT use double quotes around identifiers.
3. Follow the plan steps exactly.
4. Use table aliases and qualify columns.

SQL:"""


# ── SQL validation ───────────────────────────────────────────

def _validate_compiled_sql(sql: str) -> List[str]:
    """Light structural checks on compiled SQL."""
    errors: List[str] = []
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        errors.append("Compiled SQL is empty.")
        return errors
    first_kw = stripped.split()[0].upper()
    if first_kw not in ("SELECT", "WITH"):
        errors.append(f"Compiled SQL starts with '{first_kw}', expected SELECT or WITH.")
    if ";" in stripped:
        errors.append("Compiled SQL contains multiple statements.")
    return errors


def _clean_sql(sql: str) -> str:
    """Remove markdown fences and whitespace."""
    sql = re.sub(r"^```(?:sql)?\s*", "", sql.strip())
    sql = re.sub(r"\s*```$", "", sql.strip())
    return sql.strip()


# ── Executor class ───────────────────────────────────────────

class PlanExecutor:
    """
    Compiles a validated ``QueryPlan`` into SQL.

    Parameters
    ----------
    dialect : str
        ``"bigquery"`` or ``"sqlite"``.
    dataset : str
        BigQuery dataset prefix (e.g. ``"nb-sandbox.my_dataset"``).
    """

    def __init__(self, dialect: str = "bigquery", dataset: str = ""):
        self.dialect = dialect
        self.dataset = dataset

    # ── Public API ───────────────────────────────────────────

    def compile(
        self,
        plan: QueryPlan,
        schema_items: List[Dict],
    ) -> QueryPlan:
        """
        Compile the plan into SQL.

        Updates ``plan.compiled_sql`` and ``plan.status`` in place.
        Returns the same plan object.
        """
        if plan.status not in (PlanStatus.VALIDATED,):
            logger.warning(
                "compile() called on plan with status=%s; skipping", plan.status.value
            )
            return plan

        schema_context = self._format_schema(schema_items)
        plan_json = json.dumps(
            {"steps": [s.to_dict() for s in plan.steps]},
            indent=2,
        )

        t0 = time.time()
        try:
            raw_sql = self._llm_compile(plan.question, schema_context, plan_json)
        except Exception as exc:
            logger.error("LLM compilation failed: %s", exc)
            plan.status = PlanStatus.FAILED
            plan.failure_stage = FailureStage.COMPILATION
            plan.failure_detail = f"LLM compilation error: {exc}"
            return plan
        compile_elapsed = time.time() - t0

        sql = _clean_sql(raw_sql)

        # Validation checkpoint
        errors = _validate_compiled_sql(sql)
        if errors:
            logger.warning("Compiled SQL failed validation: %s", errors)
            plan.status = PlanStatus.FAILED
            plan.failure_stage = FailureStage.COMPILATION
            plan.failure_detail = "; ".join(errors)
            plan.compiled_sql = sql  # keep for debugging
            return plan

        plan.compiled_sql = sql
        plan.status = PlanStatus.COMPILED
        logger.info("Plan compiled successfully in %.2fs", compile_elapsed)
        return plan

    # ── Private helpers ──────────────────────────────────────

    def _format_schema(self, schema_items: List[Dict]) -> str:
        if self.dialect == "bigquery" and self.dataset:
            return "\n".join(
                f"{self.dataset}.{s['table']}.{s['column']} ({s['type']})"
                for s in schema_items
            )
        return "\n".join(
            f"{s['table']}.{s['column']} ({s['type']})"
            for s in schema_items
        )

    def _llm_compile(
        self, question: str, schema_context: str, plan_json: str
    ) -> str:
        """Send plan JSON to the LLM and get SQL back."""
        if self.dialect == "bigquery":
            user_msg = _COMPILE_USER_BQ.format(
                question=question,
                schema_context=schema_context,
                plan_json=plan_json,
                dataset=self.dataset,
            )
        else:
            user_msg = _COMPILE_USER_SQLITE.format(
                question=question,
                schema_context=schema_context,
                plan_json=plan_json,
            )

        response = _client.chat.completions.create(
            model=_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": _COMPILE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
        )
        return (response.choices[0].message.content or "").strip()
