"""
Query Planner – produces structured ``QueryPlan`` objects.

The planner calls the LLM *once* to emit a JSON plan (tables, joins,
aggregations, filters, ordering) instead of free-form reasoning text.
A deterministic validator checks the plan against the retrieved schema
before anything is compiled to SQL.

Confidence gate
---------------
If the planner self-reports low confidence (< threshold) **or** validation
fails, the pipeline falls back to direct SQL generation.
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
    AggregationSpec,
    FailureStage,
    JoinSpec,
    PlanStatus,
    PlanStep,
    QueryPlan,
    StepType,
)
from src.decomposition.query_decomposer import classify_complexity

load_dotenv()

logger = logging.getLogger(__name__)

_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    api_version=os.getenv("OPENAI_API_VERSION", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
)

# ── Planning prompt ──────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You are an expert SQL query planner. Given a natural-language question "
    "and a database schema, produce a structured JSON query plan. "
    "Do NOT produce SQL. Produce ONLY valid JSON — no markdown fences, "
    "no commentary."
)

_PLANNER_USER = """\
Question: {question}

Available schema columns:
{schema_context}

Produce a JSON object with exactly these keys:

{{
  "confidence": <float 0-1, how confident you are this plan is correct>,
  "tables_needed": ["table1", "table2", ...],
  "steps": [
    {{
      "step_number": 1,
      "description": "human-readable intent of this step",
      "step_type": "<scan|join|aggregate|filter|subquery|cte|order|union>",
      "tables": ["table1"],
      "columns": ["col1", "col2"],
      "joins": [
        {{"left_table": "t1", "right_table": "t2",
          "left_column": "id", "right_column": "t1_id",
          "join_type": "INNER"}}
      ],
      "aggregations": [
        {{"function": "SUM", "column": "amount", "alias": "total_amount"}}
      ],
      "filters": ["status = 'completed'"],
      "group_by": ["category"],
      "order_by": ["total_amount DESC"],
      "limit": null,
      "cte_name": null,
      "depends_on": []
    }}
  ]
}}

Rules:
1. "steps" must be ordered so that earlier steps are referenced by later ones.
2. Only use tables and columns from the schema above.
3. For simple queries use 1-2 steps. For complex queries use 2-5 steps.
4. join_type must be one of: INNER, LEFT, RIGHT, FULL.
5. aggregation function must be one of: COUNT, SUM, AVG, MIN, MAX, COUNT_DISTINCT.
6. step_type must be one of: scan, join, aggregate, filter, subquery, cte, order, union.
7. Set confidence lower if the question is ambiguous or the schema may not cover it.
"""


# ── LLM helper ───────────────────────────────────────────────

def _call_planner_llm(question: str, schema_context: str) -> str:
    """Call Azure OpenAI and return raw text."""
    user_msg = _PLANNER_USER.format(
        question=question,
        schema_context=schema_context,
    )
    response = _client.chat.completions.create(
        model=_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1200,
        temperature=0.0,
        top_p=1.0,
    )
    return (response.choices[0].message.content or "").strip()


# ── JSON parsing ─────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """Strip markdown fences and leading/trailing junk."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _parse_plan_json(raw: str) -> Dict[str, Any]:
    """Best-effort parse of LLM JSON output."""
    cleaned = _clean_json(raw)
    return json.loads(cleaned)


# ── Plan construction ────────────────────────────────────────

_STEP_TYPE_MAP = {v.value: v for v in StepType}


def _build_plan(question: str, data: Dict[str, Any], complexity_info: dict) -> QueryPlan:
    """Convert parsed JSON into a typed QueryPlan."""
    steps: List[PlanStep] = []
    for s in data.get("steps", []):
        joins = [
            JoinSpec(
                left_table=j.get("left_table", ""),
                right_table=j.get("right_table", ""),
                left_column=j.get("left_column", ""),
                right_column=j.get("right_column", ""),
                join_type=j.get("join_type", "INNER"),
            )
            for j in s.get("joins", [])
        ]
        aggregations = [
            AggregationSpec(
                function=a.get("function", "COUNT"),
                column=a.get("column", ""),
                alias=a.get("alias"),
            )
            for a in s.get("aggregations", [])
        ]
        step_type_str = s.get("step_type", "scan").lower()
        step_type = _STEP_TYPE_MAP.get(step_type_str, StepType.SCAN)
        steps.append(
            PlanStep(
                step_number=s.get("step_number", len(steps) + 1),
                description=s.get("description", ""),
                step_type=step_type,
                tables=s.get("tables", []),
                columns=s.get("columns", []),
                joins=joins,
                aggregations=aggregations,
                filters=s.get("filters", []),
                group_by=s.get("group_by", []),
                order_by=s.get("order_by", []),
                limit=s.get("limit"),
                cte_name=s.get("cte_name"),
                depends_on=s.get("depends_on", []),
            )
        )

    return QueryPlan(
        question=question,
        complexity=complexity_info["complexity"],
        complexity_score=complexity_info["score"],
        confidence=float(data.get("confidence", 0.5)),
        steps=steps,
        tables_needed=data.get("tables_needed", []),
        status=PlanStatus.CREATED,
    )


# ── Validation ───────────────────────────────────────────────

def validate_plan(plan: QueryPlan, schema_items: List[Dict]) -> QueryPlan:
    """
    Deterministic checks against the retrieved schema.

    Modifies ``plan.validation_errors`` and ``plan.status`` in place.
    """
    known_tables = {s["table"] for s in schema_items}
    known_columns = {(s["table"], s["column"]) for s in schema_items}
    errors: List[str] = []

    if not plan.steps:
        errors.append("Plan has no steps.")

    for step in plan.steps:
        # Check tables exist
        for tbl in step.tables:
            if tbl not in known_tables:
                errors.append(f"Step {step.step_number}: unknown table '{tbl}'.")

        # Check joins reference known tables
        for j in step.joins:
            if j.left_table not in known_tables:
                errors.append(
                    f"Step {step.step_number}: join references unknown table '{j.left_table}'."
                )
            if j.right_table not in known_tables:
                errors.append(
                    f"Step {step.step_number}: join references unknown table '{j.right_table}'."
                )

        # Check depends_on references exist
        existing_steps = {s.step_number for s in plan.steps}
        for dep in step.depends_on:
            if dep not in existing_steps:
                errors.append(
                    f"Step {step.step_number}: depends_on step {dep} does not exist."
                )
            if dep >= step.step_number:
                errors.append(
                    f"Step {step.step_number}: depends_on step {dep} is not earlier."
                )

    plan.validation_errors = errors
    plan.status = PlanStatus.VALIDATED if not errors else PlanStatus.FAILED
    if errors:
        plan.failure_stage = FailureStage.PLAN_VALIDATION
        plan.failure_detail = "; ".join(errors[:5])
        logger.warning("Plan validation failed: %s", errors)

    return plan


# ── Public API ───────────────────────────────────────────────

class QueryPlanner:
    """
    Produces a structured ``QueryPlan`` from a natural-language question.

    Parameters
    ----------
    confidence_threshold : float
        Minimum planner confidence to proceed with plan compilation.
        Below this the pipeline should fall back to direct generation.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def plan(
        self,
        question: str,
        schema_items: List[Dict],
    ) -> QueryPlan:
        """
        Generate and validate a query plan.

        Returns a ``QueryPlan`` whose ``status`` is one of:
          * ``VALIDATED`` – ready for compilation
          * ``FAILED`` – validation failed (check ``validation_errors``)
          * ``FALLBACK`` – planner confidence below threshold

        The caller decides whether to compile or fall back.
        """
        schema_context = "\n".join(
            f"{s['table']}.{s['column']} ({s['type']})" for s in schema_items
        )

        # Classify complexity (reuses existing heuristics)
        cx = classify_complexity(question)

        t0 = time.time()
        try:
            raw = _call_planner_llm(question, schema_context)
            data = _parse_plan_json(raw)
        except Exception as exc:
            logger.error("Planner LLM/parse error: %s", exc)
            plan = QueryPlan(
                question=question,
                complexity=cx["complexity"],
                complexity_score=cx["score"],
                confidence=0.0,
                status=PlanStatus.FAILED,
                failure_stage=FailureStage.PLANNING,
                failure_detail=str(exc),
            )
            return plan
        planning_elapsed = time.time() - t0

        plan = _build_plan(question, data, cx)
        plan = validate_plan(plan, schema_items)

        # Confidence gate
        if plan.confidence < self.confidence_threshold and plan.status != PlanStatus.FAILED:
            plan.status = PlanStatus.FALLBACK
            plan.failure_stage = FailureStage.PLANNING
            plan.failure_detail = (
                f"Planner confidence {plan.confidence:.2f} < threshold "
                f"{self.confidence_threshold:.2f}"
            )
            logger.info("Planner confidence too low – will fall back: %.2f", plan.confidence)

        logger.info(
            "Plan created: status=%s confidence=%.2f steps=%d elapsed=%.2fs",
            plan.status.value, plan.confidence, len(plan.steps), planning_elapsed,
        )
        return plan
