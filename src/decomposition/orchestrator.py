"""
Planner-Executor Orchestrator
==============================

Top-level coordinator that replaces direct use of ``QueryDecomposer``
in the pipeline.

Flow
----
1. Classify complexity (reuses existing heuristics).
2. **Complex path** → call ``QueryPlanner`` to produce a structured plan:
   a. If plan is validated & high-confidence → ``PlanExecutor.compile()`` → SQL.
   b. If plan fails validation or confidence is low → **fallback** to direct generation.
3. **Simple path** → chain-of-thought hint + direct ``generate_sql()``.

Outputs a ``PipelineResult`` that carries:
* The plan object (if planner was used).
* ``used_planner`` flag.
* Failure taxonomy (``FailureStage``).

This makes the planner path measurable independently in evaluation.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from src.decomposition.plan_models import (
    FailureStage,
    PipelineResult,
    PlanStatus,
    QueryPlan,
)
from src.decomposition.query_planner import QueryPlanner
from src.decomposition.plan_executor import PlanExecutor
from src.decomposition.query_decomposer import (
    QueryDecomposer,
    classify_complexity,
)

logger = logging.getLogger(__name__)

BQ_DATASET = os.getenv("BQ_DATASET", "")


class PlannerExecutorOrchestrator:
    """
    Unified entry-point for the decomposition + generation stage.

    Replaces the old ``QueryDecomposer.process() → generate_sql()`` pair
    with an explicit planner → compiler → fallback pipeline.

    Parameters
    ----------
    dialect : str
        ``"bigquery"`` or ``"sqlite"``.
    dataset : str
        BigQuery dataset prefix (used for schema formatting).
    planner_confidence_threshold : float
        Below this the planner path is skipped in favour of direct generation.
    enable_planner : bool
        Master switch. ``False`` disables the planner entirely (useful for
        A/B testing or if LLM plan quality is too low).
    """

    def __init__(
        self,
        dialect: str = "bigquery",
        dataset: str = "",
        planner_confidence_threshold: float = 0.6,
        enable_planner: bool = True,
    ):
        self.dialect = dialect
        self.dataset = dataset or BQ_DATASET
        self.enable_planner = enable_planner

        self._planner = QueryPlanner(
            confidence_threshold=planner_confidence_threshold,
        )
        self._compiler = PlanExecutor(dialect=dialect, dataset=self.dataset)

        # Legacy decomposer kept for the fallback path
        self._decomposer = QueryDecomposer(
            cot_enabled=True, decompose_enabled=True,
        )

    # ── Public API ───────────────────────────────────────────

    def process(
        self,
        question: str,
        schema_items: List[Dict],
        generate_sql_fn=None,
    ) -> Dict[str, Any]:
        """
        End-to-end: question → SQL (with plan or fallback).

        Parameters
        ----------
        question : str
        schema_items : list[dict]
            Retrieved schema columns.
        generate_sql_fn : callable | None
            ``generate_sql(question, schema_items, dialect)`` for the
            fallback path. If *None* only the plan path is attempted.

        Returns
        -------
        dict  with keys compatible with the old ``QueryDecomposer.process()``
              return shape **plus** planner-specific metadata:

            - complexity, complexity_score, complexity_signals
            - sub_questions (empty list in planner path)
            - reasoning
            - enhanced_question
            - sql              (compiled or fallback-generated)
            - plan             (QueryPlan or None)
            - used_planner     (bool)
            - failure_stage    (str or None)
            - failure_detail   (str or None)
            - elapsed_planning_s, elapsed_compilation_s
        """
        cx = classify_complexity(question)
        is_complex = cx["complexity"] == "complex"

        # ── Planner path (complex + enabled) ─────────────────
        if is_complex and self.enable_planner:
            return self._run_planner_path(question, schema_items, cx, generate_sql_fn)

        # ── Simple / planner-disabled → direct generation ────
        return self._run_fallback_path(question, schema_items, cx, generate_sql_fn)

    # ── Planner path ─────────────────────────────────────────

    def _run_planner_path(
        self,
        question: str,
        schema_items: List[Dict],
        cx: dict,
        generate_sql_fn,
    ) -> Dict[str, Any]:
        """Structured planning → compilation → SQL."""

        # 1. Plan
        t0 = time.time()
        plan = self._planner.plan(question, schema_items)
        elapsed_planning = time.time() - t0

        # 2. Check plan viability
        if plan.status not in (PlanStatus.VALIDATED,):
            logger.info(
                "Planner path not viable (status=%s); falling back.",
                plan.status.value,
            )
            result = self._run_fallback_path(question, schema_items, cx, generate_sql_fn)
            result["plan"] = plan
            result["failure_stage"] = (
                plan.failure_stage.value if plan.failure_stage else None
            )
            result["failure_detail"] = plan.failure_detail
            result["elapsed_planning_s"] = round(elapsed_planning, 3)
            return result

        # 3. Compile plan → SQL
        t1 = time.time()
        plan = self._compiler.compile(plan, schema_items)
        elapsed_compilation = time.time() - t1

        if plan.status != PlanStatus.COMPILED or not plan.compiled_sql:
            logger.info("Compilation failed; falling back.")
            result = self._run_fallback_path(question, schema_items, cx, generate_sql_fn)
            result["plan"] = plan
            result["failure_stage"] = (
                plan.failure_stage.value if plan.failure_stage else "compilation"
            )
            result["failure_detail"] = plan.failure_detail
            result["elapsed_planning_s"] = round(elapsed_planning, 3)
            result["elapsed_compilation_s"] = round(elapsed_compilation, 3)
            return result

        # 4. Build result
        return {
            "complexity": cx["complexity"],
            "complexity_score": cx["score"],
            "complexity_signals": cx["signals"],
            "sub_questions": [],
            "reasoning": self._plan_to_reasoning(plan),
            "enhanced_question": question,
            "sql": plan.compiled_sql,
            "plan": plan,
            "used_planner": True,
            "failure_stage": None,
            "failure_detail": None,
            "elapsed_planning_s": round(elapsed_planning, 3),
            "elapsed_compilation_s": round(elapsed_compilation, 3),
        }

    # ── Fallback path (legacy decomposer + direct gen) ───────

    def _run_fallback_path(
        self,
        question: str,
        schema_items: List[Dict],
        cx: dict,
        generate_sql_fn,
    ) -> Dict[str, Any]:
        """Falls back to the old CoT decomposer + direct SQL generation."""
        decomp = self._decomposer.process(question, schema_items, dialect=self.dialect)

        sql = ""
        failure_stage = None
        failure_detail = None
        if generate_sql_fn is not None:
            try:
                sql = generate_sql_fn(decomp["enhanced_question"], schema_items, self.dialect)
            except Exception as exc:
                logger.error("Fallback generation failed: %s", exc)
                failure_stage = FailureStage.FALLBACK_GENERATION.value
                failure_detail = str(exc)

        return {
            **decomp,
            "sql": sql,
            "plan": None,
            "used_planner": False,
            "failure_stage": failure_stage,
            "failure_detail": failure_detail,
            "elapsed_planning_s": None,
            "elapsed_compilation_s": None,
        }

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _plan_to_reasoning(plan: QueryPlan) -> str:
        """Convert a structured plan into human-readable reasoning text."""
        lines = [f"Query plan (confidence={plan.confidence:.2f}):"]
        for step in plan.steps:
            parts = [f"Step {step.step_number}: {step.description}"]
            if step.tables:
                parts.append(f"  Tables: {', '.join(step.tables)}")
            if step.joins:
                for j in step.joins:
                    parts.append(
                        f"  Join: {j.left_table}.{j.left_column} "
                        f"{j.join_type} {j.right_table}.{j.right_column}"
                    )
            if step.aggregations:
                aggs = ", ".join(
                    f"{a.function}({a.column})" for a in step.aggregations
                )
                parts.append(f"  Aggregations: {aggs}")
            if step.filters:
                parts.append(f"  Filters: {'; '.join(step.filters)}")
            if step.group_by:
                parts.append(f"  Group by: {', '.join(step.group_by)}")
            if step.order_by:
                parts.append(f"  Order by: {', '.join(step.order_by)}")
            if step.limit is not None:
                parts.append(f"  Limit: {step.limit}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)
