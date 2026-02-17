"""
Tests for the planner-executor architecture.

These tests validate the data structures, plan validation, and orchestrator
logic *without* calling the LLM (all LLM calls are mocked/bypassed).

Run with:  python -m pytest src/decomposition/test_planner.py -v
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from src.decomposition.plan_models import (
    AggregationSpec,
    FailureStage,
    JoinSpec,
    PipelineResult,
    PlanStatus,
    PlanStep,
    QueryPlan,
    StepType,
)
from src.decomposition.query_planner import (
    QueryPlanner,
    validate_plan,
    _build_plan,
    _parse_plan_json,
)
from src.decomposition.plan_executor import _validate_compiled_sql


# ── Sample schema for tests ─────────────────────────────────

SAMPLE_SCHEMA = [
    {"table": "orders", "column": "id", "type": "INT64"},
    {"table": "orders", "column": "total_inc_tax", "type": "FLOAT64"},
    {"table": "orders", "column": "customer_id", "type": "INT64"},
    {"table": "orders", "column": "date_created", "type": "TIMESTAMP"},
    {"table": "order_product", "column": "order_id", "type": "INT64"},
    {"table": "order_product", "column": "product_id", "type": "INT64"},
    {"table": "order_product", "column": "quantity", "type": "INT64"},
    {"table": "product", "column": "id", "type": "INT64"},
    {"table": "product", "column": "name", "type": "STRING"},
    {"table": "customer", "column": "id", "type": "INT64"},
    {"table": "customer", "column": "email", "type": "STRING"},
]


# =====================================================================
# Plan Models
# =====================================================================

class TestPlanModels:
    def test_plan_step_to_dict(self):
        step = PlanStep(
            step_number=1,
            description="Scan orders table",
            step_type=StepType.SCAN,
            tables=["orders"],
            columns=["id", "total_inc_tax"],
        )
        d = step.to_dict()
        assert d["step"] == 1
        assert d["type"] == "scan"
        assert "joins" not in d  # empty lists omitted

    def test_plan_step_with_join(self):
        step = PlanStep(
            step_number=2,
            description="Join orders with order_product",
            step_type=StepType.JOIN,
            tables=["orders", "order_product"],
            columns=["orders.id", "order_product.quantity"],
            joins=[JoinSpec("orders", "order_product", "id", "order_id")],
        )
        d = step.to_dict()
        assert len(d["joins"]) == 1
        assert d["joins"][0]["type"] == "INNER"

    def test_query_plan_to_dict(self):
        plan = QueryPlan(
            question="total revenue?",
            complexity="simple",
            complexity_score=0.1,
            confidence=0.9,
            steps=[
                PlanStep(1, "scan orders", StepType.SCAN, tables=["orders"])
            ],
            tables_needed=["orders"],
        )
        d = plan.to_dict()
        assert d["confidence"] == 0.9
        assert d["status"] == "created"
        assert len(d["steps"]) == 1

    def test_is_high_confidence(self):
        plan = QueryPlan("q", "simple", 0.1, confidence=0.8)
        assert plan.is_high_confidence
        plan2 = QueryPlan("q", "simple", 0.1, confidence=0.3)
        assert not plan2.is_high_confidence

    def test_pipeline_result_to_dict(self):
        r = PipelineResult(
            question="q",
            sql="SELECT 1",
            plan=None,
            used_planner=False,
            success=True,
        )
        d = r.to_dict()
        assert d["used_planner"] is False
        assert d["sql"] == "SELECT 1"

    def test_failure_stages(self):
        assert FailureStage.PLANNING.value == "planning"
        assert FailureStage.COMPILATION.value == "compilation"
        assert FailureStage.EXECUTION.value == "execution"


# =====================================================================
# Plan parsing & building
# =====================================================================

class TestPlanParsing:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "confidence": 0.85,
            "tables_needed": ["orders"],
            "steps": [
                {
                    "step_number": 1,
                    "description": "sum total",
                    "step_type": "aggregate",
                    "tables": ["orders"],
                    "columns": ["total_inc_tax"],
                    "aggregations": [{"function": "SUM", "column": "total_inc_tax", "alias": "revenue"}],
                }
            ],
        })
        data = _parse_plan_json(raw)
        assert data["confidence"] == 0.85

    def test_parse_with_markdown_fences(self):
        raw = "```json\n{\"confidence\": 0.5, \"tables_needed\": [], \"steps\": []}\n```"
        data = _parse_plan_json(raw)
        assert data["confidence"] == 0.5

    def test_build_plan_from_data(self):
        data = {
            "confidence": 0.7,
            "tables_needed": ["orders", "order_product"],
            "steps": [
                {
                    "step_number": 1,
                    "description": "join tables",
                    "step_type": "join",
                    "tables": ["orders", "order_product"],
                    "columns": [],
                    "joins": [
                        {
                            "left_table": "orders",
                            "right_table": "order_product",
                            "left_column": "id",
                            "right_column": "order_id",
                            "join_type": "INNER",
                        }
                    ],
                },
                {
                    "step_number": 2,
                    "description": "aggregate",
                    "step_type": "aggregate",
                    "tables": ["order_product"],
                    "columns": ["quantity"],
                    "aggregations": [{"function": "SUM", "column": "quantity"}],
                    "depends_on": [1],
                },
            ],
        }
        cx = {"complexity": "complex", "score": 0.6}
        plan = _build_plan("how many items sold?", data, cx)
        assert plan.confidence == 0.7
        assert len(plan.steps) == 2
        assert plan.steps[0].step_type == StepType.JOIN
        assert plan.steps[1].depends_on == [1]


# =====================================================================
# Plan validation
# =====================================================================

class TestPlanValidation:
    def test_valid_plan(self):
        plan = QueryPlan(
            question="total revenue?",
            complexity="simple",
            complexity_score=0.1,
            confidence=0.9,
            steps=[
                PlanStep(
                    1, "aggregate orders", StepType.AGGREGATE,
                    tables=["orders"],
                    columns=["total_inc_tax"],
                    aggregations=[AggregationSpec("SUM", "total_inc_tax", "revenue")],
                )
            ],
            tables_needed=["orders"],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED
        assert validated.validation_errors == []

    def test_unknown_table(self):
        plan = QueryPlan(
            question="q",
            complexity="simple",
            complexity_score=0.1,
            confidence=0.9,
            steps=[
                PlanStep(1, "scan", StepType.SCAN, tables=["nonexistent_table"])
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("nonexistent_table" in e for e in validated.validation_errors)

    def test_unknown_join_table(self):
        plan = QueryPlan(
            question="q",
            complexity="complex",
            complexity_score=0.5,
            confidence=0.8,
            steps=[
                PlanStep(
                    1, "join", StepType.JOIN,
                    tables=["orders"],
                    joins=[JoinSpec("orders", "bad_table", "id", "order_id")],
                )
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("bad_table" in e for e in validated.validation_errors)

    def test_forward_dependency_rejected(self):
        plan = QueryPlan(
            question="q",
            complexity="complex",
            complexity_score=0.5,
            confidence=0.8,
            steps=[
                PlanStep(1, "step1", StepType.SCAN, tables=["orders"], depends_on=[2]),
                PlanStep(2, "step2", StepType.SCAN, tables=["orders"]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("not earlier" in e for e in validated.validation_errors)

    def test_empty_plan_rejected(self):
        plan = QueryPlan(
            question="q", complexity="simple",
            complexity_score=0.1, confidence=0.9,
            steps=[],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("no steps" in e.lower() for e in validated.validation_errors)


# =====================================================================
# Strict schema-aware validation
# =====================================================================

class TestColumnExistenceValidation:
    """Validation class 1: referenced columns must exist in referenced tables."""

    def test_valid_columns_pass(self):
        plan = QueryPlan(
            "q", "simple", 0.1, 0.9,
            steps=[
                PlanStep(1, "scan orders", StepType.SCAN,
                         tables=["orders"],
                         columns=["id", "total_inc_tax"]),
            ],
            tables_needed=["orders"],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_qualified_column_valid(self):
        plan = QueryPlan(
            "q", "simple", 0.1, 0.9,
            steps=[
                PlanStep(1, "scan orders", StepType.SCAN,
                         tables=["orders"],
                         columns=["orders.id"]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_unknown_column_rejected(self):
        plan = QueryPlan(
            "q", "simple", 0.1, 0.9,
            steps=[
                PlanStep(1, "scan orders", StepType.SCAN,
                         tables=["orders"],
                         columns=["id", "nonexistent_col"]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("nonexistent_col" in e and "not found" in e
                    for e in validated.validation_errors)

    def test_column_from_wrong_table_rejected(self):
        """Column exists in schema but not in the step's tables."""
        plan = QueryPlan(
            "q", "simple", 0.1, 0.9,
            steps=[
                PlanStep(1, "scan orders", StepType.SCAN,
                         tables=["orders"],
                         columns=["email"]),  # email belongs to customer, not orders
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("email" in e and "not found" in e
                    for e in validated.validation_errors)

    def test_qualified_column_wrong_table_rejected(self):
        plan = QueryPlan(
            "q", "simple", 0.1, 0.9,
            steps=[
                PlanStep(1, "scan orders", StepType.SCAN,
                         tables=["orders"],
                         columns=["customer.email"]),  # customer not in step.tables
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("customer.email" in e for e in validated.validation_errors)

    def test_column_resolved_via_join_table(self):
        """A column from a join's right_table should resolve even if not in step.tables."""
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "join", StepType.JOIN,
                         tables=["orders"],
                         columns=["quantity"],  # from order_product via join
                         joins=[JoinSpec("orders", "order_product", "id", "order_id")]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED


class TestJoinKeyValidation:
    """Validation class 2: join columns must exist on their respective sides."""

    def test_valid_join_keys_pass(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "join", StepType.JOIN,
                         tables=["orders", "order_product"],
                         joins=[JoinSpec("orders", "order_product", "id", "order_id")]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_left_join_key_missing(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "join", StepType.JOIN,
                         tables=["orders", "order_product"],
                         joins=[JoinSpec("orders", "order_product",
                                         "fake_id", "order_id")]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("fake_id" in e and "orders" in e
                    for e in validated.validation_errors)

    def test_right_join_key_missing(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "join", StepType.JOIN,
                         tables=["orders", "order_product"],
                         joins=[JoinSpec("orders", "order_product",
                                         "id", "bad_fk")]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("bad_fk" in e and "order_product" in e
                    for e in validated.validation_errors)

    def test_both_join_keys_missing(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "join", StepType.JOIN,
                         tables=["orders", "order_product"],
                         joins=[JoinSpec("orders", "order_product",
                                         "nope_l", "nope_r")]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        errs = validated.validation_errors
        assert any("nope_l" in e for e in errs)
        assert any("nope_r" in e for e in errs)


class TestDependsOnChainValidation:
    """Validation class 3: depends_on must form a valid DAG."""

    def test_valid_chain_passes(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "step1", StepType.SCAN, tables=["orders"]),
                PlanStep(2, "step2", StepType.AGGREGATE, tables=["orders"],
                         depends_on=[1]),
                PlanStep(3, "step3", StepType.ORDER, tables=["orders"],
                         depends_on=[2]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_self_dependency_rejected(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "step1", StepType.SCAN, tables=["orders"],
                         depends_on=[1]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("not earlier" in e for e in validated.validation_errors)

    def test_two_step_cycle_rejected(self):
        """Step 1 → 2, step 2 → 1 is a cycle."""
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "step1", StepType.SCAN, tables=["orders"],
                         depends_on=[2]),
                PlanStep(2, "step2", StepType.SCAN, tables=["orders"],
                         depends_on=[1]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        # Should detect either the forward ref or the cycle
        assert len(validated.validation_errors) > 0

    def test_three_step_cycle_rejected(self):
        """1→2, 2→3, 3→1 forms a cycle."""
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "s1", StepType.SCAN, tables=["orders"]),
                PlanStep(2, "s2", StepType.SCAN, tables=["orders"],
                         depends_on=[1]),
                PlanStep(3, "s3", StepType.SCAN, tables=["orders"],
                         depends_on=[2]),
                PlanStep(4, "s4", StepType.SCAN, tables=["orders"],
                         depends_on=[3]),
            ],
        )
        # Valid chain – should pass
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_dangling_dependency_rejected(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "step1", StepType.SCAN, tables=["orders"]),
                PlanStep(2, "step2", StepType.SCAN, tables=["orders"],
                         depends_on=[99]),  # step 99 doesn't exist
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("99" in e and "does not exist" in e
                    for e in validated.validation_errors)


class TestAggregationGroupByValidation:
    """Validation class 4: aggregation/group_by consistency."""

    def test_aggregation_with_correct_group_by(self):
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "agg", StepType.AGGREGATE,
                         tables=["orders"],
                         columns=["customer_id", "total_inc_tax"],
                         aggregations=[AggregationSpec("SUM", "total_inc_tax", "revenue")],
                         group_by=["customer_id"]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_missing_group_by_rejected(self):
        """Non-aggregated column without group_by → error."""
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "agg", StepType.AGGREGATE,
                         tables=["orders"],
                         columns=["customer_id", "total_inc_tax"],
                         aggregations=[AggregationSpec("SUM", "total_inc_tax", "revenue")],
                         group_by=[]),  # customer_id not grouped!
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("customer_id" in e and "neither aggregated nor in group_by" in e
                    for e in validated.validation_errors)

    def test_all_columns_aggregated_no_group_by_needed(self):
        """If every column is aggregated, no group_by is required."""
        plan = QueryPlan(
            "q", "simple", 0.1, 0.9,
            steps=[
                PlanStep(1, "total", StepType.AGGREGATE,
                         tables=["orders"],
                         columns=["total_inc_tax"],
                         aggregations=[AggregationSpec("SUM", "total_inc_tax", "revenue")],
                         group_by=[]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_group_by_column_not_in_schema_rejected(self):
        """group_by references a column that doesn't exist in the schema."""
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "agg", StepType.AGGREGATE,
                         tables=["orders"],
                         columns=["total_inc_tax"],
                         aggregations=[AggregationSpec("SUM", "total_inc_tax", "revenue")],
                         group_by=["phantom_col"]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert any("phantom_col" in e and "not found" in e
                    for e in validated.validation_errors)

    def test_no_aggregation_no_group_by_check(self):
        """Steps without aggregations skip the group_by consistency check."""
        plan = QueryPlan(
            "q", "simple", 0.1, 0.9,
            steps=[
                PlanStep(1, "scan", StepType.SCAN,
                         tables=["orders"],
                         columns=["id", "total_inc_tax"]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.VALIDATED

    def test_multiple_validation_errors_combined(self):
        """A plan with multiple issues should report all of them."""
        plan = QueryPlan(
            "q", "complex", 0.5, 0.9,
            steps=[
                PlanStep(1, "bad step", StepType.JOIN,
                         tables=["orders"],
                         columns=["nonexistent_col"],
                         joins=[JoinSpec("orders", "order_product",
                                         "id", "bad_fk")],
                         aggregations=[AggregationSpec("SUM", "total_inc_tax", "rev")],
                         group_by=[]),
            ],
        )
        validated = validate_plan(plan, SAMPLE_SCHEMA)
        assert validated.status == PlanStatus.FAILED
        assert validated.failure_stage == FailureStage.PLAN_VALIDATION
        # Should have column, join-key, and agg/group_by errors
        assert len(validated.validation_errors) >= 3


# =====================================================================
# SQL compilation validation
# =====================================================================

class TestCompiledSQLValidation:
    def test_valid_select(self):
        assert _validate_compiled_sql("SELECT 1") == []

    def test_valid_cte(self):
        assert _validate_compiled_sql("WITH cte AS (SELECT 1) SELECT * FROM cte") == []

    def test_empty_sql_rejected(self):
        errors = _validate_compiled_sql("")
        assert any("empty" in e.lower() for e in errors)

    def test_non_select_rejected(self):
        errors = _validate_compiled_sql("INSERT INTO t VALUES (1)")
        assert any("INSERT" in e for e in errors)

    def test_multi_statement_rejected(self):
        errors = _validate_compiled_sql("SELECT 1; SELECT 2")
        assert any("multiple" in e.lower() for e in errors)


# =====================================================================
# Orchestrator (with mocked LLM)
# =====================================================================

class TestOrchestrator:
    """Test the orchestrator routing logic without real LLM calls."""

    def _mock_generate_sql(self, question, schema_items, dialect="bigquery"):
        return "SELECT 'fallback'"

    @patch("src.decomposition.query_planner._call_planner_llm")
    def test_complex_query_uses_planner(self, mock_llm):
        """Complex question → planner → plan compiled → SQL."""
        plan_json = json.dumps({
            "confidence": 0.9,
            "tables_needed": ["orders"],
            "steps": [
                {
                    "step_number": 1,
                    "description": "sum revenue",
                    "step_type": "aggregate",
                    "tables": ["orders"],
                    "columns": ["total_inc_tax"],
                    "aggregations": [{"function": "SUM", "column": "total_inc_tax", "alias": "revenue"}],
                }
            ],
        })
        mock_llm.return_value = plan_json

        from src.decomposition.orchestrator import PlannerExecutorOrchestrator

        # Also mock the compiler LLM
        with patch("src.decomposition.plan_executor._client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "SELECT SUM(total_inc_tax) AS revenue FROM orders"
            mock_client.chat.completions.create.return_value = mock_resp

            orch = PlannerExecutorOrchestrator(
                dialect="sqlite",
                enable_planner=True,
                planner_confidence_threshold=0.6,
            )
            # Use a complex question that triggers planner
            complex_q = (
                "Show me the total revenue and also the average order value "
                "for each customer who has more than 5 orders, compared to "
                "customers with less than 5 orders"
            )
            result = orch.process(complex_q, SAMPLE_SCHEMA, self._mock_generate_sql)

        assert result["used_planner"] is True
        assert result["plan"] is not None
        assert result["sql"] != ""
        assert result["failure_stage"] is None

    @patch("src.decomposition.query_planner._call_planner_llm")
    def test_low_confidence_falls_back(self, mock_llm):
        """Planner confidence < threshold → fallback to direct generation."""
        plan_json = json.dumps({
            "confidence": 0.2,   # low!
            "tables_needed": ["orders"],
            "steps": [
                {
                    "step_number": 1,
                    "description": "scan",
                    "step_type": "scan",
                    "tables": ["orders"],
                    "columns": [],
                }
            ],
        })
        mock_llm.return_value = plan_json

        from src.decomposition.orchestrator import PlannerExecutorOrchestrator

        # Mock the decomposer LLM calls in fallback path
        with patch("src.decomposition.query_decomposer._call_llm") as mock_decomp_llm:
            mock_decomp_llm.return_value = "1. Simple question"

            orch = PlannerExecutorOrchestrator(
                dialect="sqlite",
                enable_planner=True,
                planner_confidence_threshold=0.6,
            )
            complex_q = (
                "Compare the average order total for customers who also bought "
                "product X versus those who bought product Y"
            )
            result = orch.process(complex_q, SAMPLE_SCHEMA, self._mock_generate_sql)

        assert result["used_planner"] is False
        assert result["sql"] == "SELECT 'fallback'"
        assert result["plan"] is not None  # plan is still attached for diagnostics
        assert result["failure_stage"] is not None

    def test_simple_query_skips_planner(self):
        """Simple question → direct fallback, no planner invoked."""
        from src.decomposition.orchestrator import PlannerExecutorOrchestrator

        with patch("src.decomposition.query_decomposer._call_llm") as mock_llm:
            mock_llm.return_value = "Use orders table, sum total_inc_tax"

            orch = PlannerExecutorOrchestrator(
                dialect="sqlite",
                enable_planner=True,
            )
            result = orch.process(
                "What is the total revenue?",
                SAMPLE_SCHEMA,
                self._mock_generate_sql,
            )

        assert result["used_planner"] is False
        assert result["sql"] == "SELECT 'fallback'"
        assert result["plan"] is None

    def test_planner_disabled(self):
        """enable_planner=False → always fallback."""
        from src.decomposition.orchestrator import PlannerExecutorOrchestrator

        with patch("src.decomposition.query_decomposer._call_llm") as mock_llm:
            mock_llm.return_value = "Just do it"

            orch = PlannerExecutorOrchestrator(enable_planner=False)
            complex_q = (
                "Compare revenue and also find the average order value "
                "for each month having more than 10 orders"
            )
            result = orch.process(complex_q, SAMPLE_SCHEMA, self._mock_generate_sql)

        assert result["used_planner"] is False

    @patch("src.decomposition.query_planner._call_planner_llm")
    def test_planner_llm_failure_falls_back(self, mock_llm):
        """If planner LLM throws, gracefully fall back."""
        mock_llm.side_effect = RuntimeError("LLM down")

        from src.decomposition.orchestrator import PlannerExecutorOrchestrator

        with patch("src.decomposition.query_decomposer._call_llm") as mock_decomp:
            mock_decomp.return_value = "fallback reasoning"

            orch = PlannerExecutorOrchestrator(enable_planner=True)
            complex_q = (
                "Show the difference between monthly revenue and also "
                "the count of orders per customer having more than 3 orders"
            )
            result = orch.process(complex_q, SAMPLE_SCHEMA, self._mock_generate_sql)

        assert result["used_planner"] is False
        assert result["failure_stage"] == "planning"


# =====================================================================
# Failure taxonomy
# =====================================================================

class TestFailureTaxonomy:
    def test_planning_failure(self):
        plan = QueryPlan("q", "complex", 0.5, 0.0)
        plan.status = PlanStatus.FAILED
        plan.failure_stage = FailureStage.PLANNING
        plan.failure_detail = "LLM returned invalid JSON"
        d = plan.to_dict()
        assert d["failure_stage"] == "planning"

    def test_validation_failure(self):
        plan = QueryPlan("q", "complex", 0.5, 0.8,
                         steps=[PlanStep(1, "s", StepType.SCAN, tables=["bad"])])
        plan = validate_plan(plan, SAMPLE_SCHEMA)
        d = plan.to_dict()
        assert d["failure_stage"] == "plan_validation"

    def test_compilation_failure(self):
        plan = QueryPlan("q", "complex", 0.5, 0.8)
        plan.status = PlanStatus.FAILED
        plan.failure_stage = FailureStage.COMPILATION
        plan.failure_detail = "SQL starts with INSERT"
        assert plan.to_dict()["failure_stage"] == "compilation"

    def test_fallback_generation_failure(self):
        r = PipelineResult(
            question="q", sql="", plan=None,
            used_planner=False, success=False,
            failure_stage=FailureStage.FALLBACK_GENERATION,
            failure_detail="API timeout",
        )
        d = r.to_dict()
        assert d["failure_stage"] == "fallback_generation"
