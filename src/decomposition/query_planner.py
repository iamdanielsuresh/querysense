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

def _resolve_column(
    col: str,
    step_tables: List[str],
    known_columns: set,
) -> bool:
    """Return True if *col* can be resolved against the schema.

    Handles both qualified (``table.column``) and bare column names.
    A qualified name must reference a table visible in this step.
    A bare name is valid if it exists in **any** of the step's tables.
    """
    if "." in col:
        tbl, c = col.split(".", 1)
        return tbl in step_tables and (tbl, c) in known_columns
    return any((t, col) in known_columns for t in step_tables)


def _detect_depends_on_cycle(steps: List[PlanStep]) -> Optional[List[int]]:
    """Return a cycle path if dependency graph has a cycle, else ``None``.

    Uses iterative DFS with three-colour marking (white/grey/black).
    """
    adj: Dict[int, List[int]] = {s.step_number: list(s.depends_on) for s in steps}
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[int, int] = {n: WHITE for n in adj}
    parent: Dict[int, Optional[int]] = {n: None for n in adj}

    for start in adj:
        if colour[start] != WHITE:
            continue
        stack = [start]
        while stack:
            node = stack[-1]
            if colour[node] == WHITE:
                colour[node] = GREY
                for dep in adj.get(node, []):
                    if dep not in colour:
                        continue  # dangling ref – caught elsewhere
                    if colour[dep] == GREY:
                        # Found cycle – reconstruct path
                        cycle = [dep, node]
                        p = parent.get(node)
                        while p is not None and p != dep:
                            cycle.append(p)
                            p = parent.get(p)
                        cycle.append(dep)
                        return list(reversed(cycle))
                    if colour[dep] == WHITE:
                        parent[dep] = node
                        stack.append(dep)
            else:
                colour[node] = BLACK
                stack.pop()
    return None


def validate_plan(plan: QueryPlan, schema_items: List[Dict]) -> QueryPlan:
    """
    Deterministic schema-aware checks.

    Validation classes
    ------------------
    1. **Table existence** – every referenced table must be in the schema.
    2. **Column existence** – every referenced column must exist in one of
       the step's tables (or be fully qualified ``table.col``).
    3. **Join-key existence** – join columns must exist on their respective
       sides.
    4. **Dependency ordering** – ``depends_on`` must reference earlier steps
       only, with no forward references, dangling refs, or cycles.
    5. **Aggregation / GROUP BY consistency** – non-aggregated columns in a
       step that contains aggregations must appear in ``group_by``.

    Modifies ``plan.validation_errors`` and ``plan.status`` in place.
    """
    known_tables = {s["table"] for s in schema_items}
    known_columns = {(s["table"], s["column"]) for s in schema_items}
    # Also build a per-table column set for fast lookup
    table_columns: Dict[str, set] = {}
    for s in schema_items:
        table_columns.setdefault(s["table"], set()).add(s["column"])

    errors: List[str] = []

    if not plan.steps:
        errors.append("Plan has no steps.")

    existing_steps = {s.step_number for s in plan.steps}

    for step in plan.steps:
        pfx = f"Step {step.step_number}"

        # ── 1. Table existence ──────────────────────────────
        for tbl in step.tables:
            if tbl not in known_tables:
                errors.append(f"{pfx}: unknown table '{tbl}'.")

        # Collect all tables visible in this step (explicit + join sides)
        visible_tables = [t for t in step.tables if t in known_tables]
        for j in step.joins:
            if j.left_table in known_tables and j.left_table not in visible_tables:
                visible_tables.append(j.left_table)
            if j.right_table in known_tables and j.right_table not in visible_tables:
                visible_tables.append(j.right_table)

        # ── 2. Column existence ─────────────────────────────
        for col in step.columns:
            if not _resolve_column(col, visible_tables, known_columns):
                errors.append(f"{pfx}: column '{col}' not found in tables {visible_tables}.")

        # ── 3. Join-key existence ───────────────────────────
        for j in step.joins:
            if j.left_table not in known_tables:
                errors.append(f"{pfx}: join references unknown table '{j.left_table}'.")
            else:
                left_cols = table_columns.get(j.left_table, set())
                if j.left_column not in left_cols:
                    errors.append(
                        f"{pfx}: join key '{j.left_column}' does not exist "
                        f"in table '{j.left_table}'."
                    )
            if j.right_table not in known_tables:
                errors.append(f"{pfx}: join references unknown table '{j.right_table}'.")
            else:
                right_cols = table_columns.get(j.right_table, set())
                if j.right_column not in right_cols:
                    errors.append(
                        f"{pfx}: join key '{j.right_column}' does not exist "
                        f"in table '{j.right_table}'."
                    )

        # ── 4a. Dependency ordering (per-step) ─────────────
        for dep in step.depends_on:
            if dep not in existing_steps:
                errors.append(f"{pfx}: depends_on step {dep} does not exist.")
            elif dep >= step.step_number:
                errors.append(f"{pfx}: depends_on step {dep} is not earlier.")

        # ── 5. Aggregation / GROUP BY consistency ───────────
        if step.aggregations:
            agg_columns = {a.column for a in step.aggregations}
            # Columns that are selected but not aggregated must be grouped
            for col in step.columns:
                bare = col.split(".", 1)[-1] if "." in col else col
                if bare not in agg_columns and bare not in step.group_by:
                    errors.append(
                        f"{pfx}: column '{col}' is neither aggregated nor in group_by."
                    )
            # group_by entries must also resolve against schema
            for gb in step.group_by:
                if not _resolve_column(gb, visible_tables, known_columns):
                    errors.append(
                        f"{pfx}: group_by column '{gb}' not found in tables {visible_tables}."
                    )

    # ── 4b. Dependency cycle detection (global) ─────────────
    if plan.steps:
        cycle = _detect_depends_on_cycle(plan.steps)
        if cycle is not None:
            errors.append(
                f"Dependency cycle detected: {' -> '.join(str(s) for s in cycle)}."
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
