"""
Query Plan Data Structures
==========================

Typed representations for the planner-executor architecture.

The flow is:

    question  ──►  QueryPlan  ──►  PlanExecutor  ──►  SQL + validation
                   (structured)     (compiles plan)

Every plan step is explicit, inspectable, and serialisable so that:

* Evaluation can distinguish *planning* failures from *generation* failures.
* The plan object is logged/stored for reproducibility.
* A confidence score lets the pipeline fall back to direct generation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────

class PlanStatus(str, Enum):
    """Lifecycle of a plan."""
    CREATED = "created"
    VALIDATED = "validated"
    COMPILED = "compiled"
    EXECUTED = "executed"
    FAILED = "failed"
    FALLBACK = "fallback"          # fell back to direct generation


class StepType(str, Enum):
    """Semantic type of a plan step."""
    SCAN = "scan"                  # simple table scan / filter
    JOIN = "join"                  # joining two+ tables
    AGGREGATE = "aggregate"        # GROUP BY / window
    FILTER = "filter"              # WHERE / HAVING clause
    SUBQUERY = "subquery"          # correlated or uncorrelated sub-select
    CTE = "cte"                    # common table expression
    ORDER = "order"                # final ordering / limit
    UNION = "union"                # UNION / UNION ALL


class FailureStage(str, Enum):
    """Where in the pipeline did the failure happen?"""
    PLANNING = "planning"
    PLAN_VALIDATION = "plan_validation"
    COMPILATION = "compilation"
    EXECUTION = "execution"
    FALLBACK_GENERATION = "fallback_generation"


# ── Step / Join / Aggregation ────────────────────────────────

@dataclass
class JoinSpec:
    """A single join clause inside a plan step."""
    left_table: str
    right_table: str
    left_column: str
    right_column: str
    join_type: str = "INNER"       # INNER | LEFT | RIGHT | FULL

    def to_dict(self) -> dict:
        return {
            "left": f"{self.left_table}.{self.left_column}",
            "right": f"{self.right_table}.{self.right_column}",
            "type": self.join_type,
        }


@dataclass
class AggregationSpec:
    """An aggregation operation."""
    function: str                  # COUNT, SUM, AVG, MIN, MAX, COUNT_DISTINCT
    column: str                    # column or expression to aggregate
    alias: Optional[str] = None   # output alias

    def to_dict(self) -> dict:
        return {"function": self.function, "column": self.column, "alias": self.alias}


@dataclass
class PlanStep:
    """
    One logical step in the query plan.

    Steps are ordered; each step may reference outputs of earlier steps
    (e.g. a CTE name produced by step 1 used in step 3).
    """
    step_number: int
    description: str               # human-readable intent
    step_type: StepType
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    joins: List[JoinSpec] = field(default_factory=list)
    aggregations: List[AggregationSpec] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)       # free-form WHERE text
    group_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    cte_name: Optional[str] = None  # if this step defines a CTE
    depends_on: List[int] = field(default_factory=list)     # step_numbers

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "step": self.step_number,
            "description": self.description,
            "type": self.step_type.value,
            "tables": self.tables,
            "columns": self.columns,
        }
        if self.joins:
            d["joins"] = [j.to_dict() for j in self.joins]
        if self.aggregations:
            d["aggregations"] = [a.to_dict() for a in self.aggregations]
        if self.filters:
            d["filters"] = self.filters
        if self.group_by:
            d["group_by"] = self.group_by
        if self.order_by:
            d["order_by"] = self.order_by
        if self.limit is not None:
            d["limit"] = self.limit
        if self.cte_name:
            d["cte_name"] = self.cte_name
        if self.depends_on:
            d["depends_on"] = self.depends_on
        return d


# ── Query Plan ───────────────────────────────────────────────

@dataclass
class QueryPlan:
    """
    Structured plan produced by the planner for a single question.

    The plan captures *what* the SQL should do (semantically) without
    containing the literal SQL yet.  The ``PlanExecutor`` compiles the
    plan into SQL.
    """
    question: str
    complexity: str                             # "simple" | "complex"
    complexity_score: float
    confidence: float                           # planner self-assessed 0-1
    steps: List[PlanStep] = field(default_factory=list)
    tables_needed: List[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.CREATED
    created_at: float = field(default_factory=time.time)
    validation_errors: List[str] = field(default_factory=list)

    # Populated after compilation
    compiled_sql: Optional[str] = None

    # Populated after execution
    execution_result: Optional[dict] = None

    # For failure taxonomy
    failure_stage: Optional[FailureStage] = None
    failure_detail: Optional[str] = None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.6

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "complexity": self.complexity,
            "complexity_score": self.complexity_score,
            "confidence": self.confidence,
            "status": self.status.value,
            "tables_needed": self.tables_needed,
            "steps": [s.to_dict() for s in self.steps],
            "compiled_sql": self.compiled_sql,
            "validation_errors": self.validation_errors,
            "failure_stage": self.failure_stage.value if self.failure_stage else None,
            "failure_detail": self.failure_detail,
        }


# ── Pipeline Result ──────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Full result returned by the planner-executor pipeline.

    Carries enough metadata for evaluation to distinguish planning path
    vs. fallback path and to classify failures.
    """
    question: str
    sql: str
    plan: Optional[QueryPlan]
    used_planner: bool             # False ⇒ direct-generation fallback
    success: bool
    execution_result: Optional[dict] = None
    failure_stage: Optional[FailureStage] = None
    failure_detail: Optional[str] = None
    elapsed_planning: Optional[float] = None
    elapsed_compilation: Optional[float] = None
    elapsed_execution: Optional[float] = None

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "question": self.question,
            "sql": self.sql,
            "used_planner": self.used_planner,
            "success": self.success,
            "failure_stage": self.failure_stage.value if self.failure_stage else None,
            "failure_detail": self.failure_detail,
            "elapsed_planning_s": self.elapsed_planning,
            "elapsed_compilation_s": self.elapsed_compilation,
            "elapsed_execution_s": self.elapsed_execution,
        }
        if self.plan:
            d["plan"] = self.plan.to_dict()
        return d
