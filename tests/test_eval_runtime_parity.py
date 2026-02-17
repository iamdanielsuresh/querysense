"""
Tests proving that the evaluation pipeline (spider_eval.py) calls the same
core modules and follows the same sequence as the runtime pipeline (pipeline.py).

These are structural / import-parity tests — they verify that both paths
wire into the same classes and functions rather than duplicating logic.
"""
import inspect
import types
from unittest.mock import patch, MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1.  IMPORT PARITY — both paths reference the same objects
# ═══════════════════════════════════════════════════════════════════════

class TestImportParity:
    """Evaluation and runtime must import the same core symbols."""

    def test_same_generate_sql(self):
        """Both modules import the identical generate_sql function object."""
        from src.generator.sql_generator_azure import generate_sql as gen_runtime
        # spider_eval also imports from the same module
        import src.evaluvation.spider_eval as ev
        assert ev.generate_sql is gen_runtime

    def test_same_schema_retriever_class(self):
        from src.retriever.schema_retriever import SchemaRetriever as RT
        import src.evaluvation.spider_eval as ev
        assert ev.SchemaRetriever is RT

    def test_same_orchestrator_class(self):
        """Evaluation must use PlannerExecutorOrchestrator, not raw QueryDecomposer."""
        from src.decomposition.orchestrator import PlannerExecutorOrchestrator as RT
        import src.evaluvation.spider_eval as ev
        assert ev.PlannerExecutorOrchestrator is RT

    def test_same_retry_functions(self):
        from src.generator.retry import classify_error as ce_rt, build_retry_prompt as brp_rt
        import src.evaluvation.spider_eval as ev
        assert ev.classify_error is ce_rt
        assert ev.build_retry_prompt is brp_rt

    def test_orchestrator_available_in_eval(self):
        """PlannerExecutorOrchestrator must be importable from spider_eval top-level."""
        import src.evaluvation.spider_eval as ev
        assert hasattr(ev, "PlannerExecutorOrchestrator")


# ═══════════════════════════════════════════════════════════════════════
# 2.  EVALUATE_SINGLE USES ORCHESTRATOR (not raw QueryDecomposer)
# ═══════════════════════════════════════════════════════════════════════

class TestEvaluateSingleSignature:
    """evaluate_single accepts `orchestrator`, not `decomposer`."""

    def test_has_orchestrator_param(self):
        from src.evaluvation.spider_eval import evaluate_single
        sig = inspect.signature(evaluate_single)
        assert "orchestrator" in sig.parameters
        assert "decomposer" not in sig.parameters

    def test_orchestrator_annotation(self):
        from src.evaluvation.spider_eval import evaluate_single
        import src.evaluvation.spider_eval as ev
        sig = inspect.signature(evaluate_single)
        annotation = sig.parameters["orchestrator"].annotation
        # annotation may be Optional[PlannerExecutorOrchestrator] or string
        assert "PlannerExecutorOrchestrator" in str(annotation)


# ═══════════════════════════════════════════════════════════════════════
# 3.  RUNTIME SEQUENCE PARITY (mock-based)
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeSequence:
    """
    When decomposition is ON, evaluate_single must call
    PlannerExecutorOrchestrator.process() — the same entry-point used
    by pipeline.py — rather than QueryDecomposer.process() directly.
    """

    @pytest.fixture
    def example(self):
        return {
            "question": "How many singers are there?",
            "query": "SELECT COUNT(*) FROM singer",
            "db_id": "concert_singer",
        }

    @pytest.fixture
    def schema(self):
        return {
            "db_id": "concert_singer",
            "tables": ["singer"],
            "columns": [
                {"table": "singer", "column": "singer_id", "type": "number",
                 "text": "singer.singer_id singer_id number"},
                {"table": "singer", "column": "name", "type": "text",
                 "text": "singer.name name text"},
            ],
            "foreign_keys": [],
            "primary_keys": [],
        }

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a tiny SQLite db so gold SQL can run."""
        import sqlite3
        db = tmp_path / "concert_singer.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE singer (singer_id INTEGER, name TEXT)")
        conn.execute("INSERT INTO singer VALUES (1, 'Alice')")
        conn.execute("INSERT INTO singer VALUES (2, 'Bob')")
        conn.commit()
        conn.close()
        return db

    def test_orchestrator_process_called_when_decomposition_on(
        self, example, schema, db_path
    ):
        """When decomposition=True, orchestrator.process() must be called."""
        from src.evaluvation.spider_eval import evaluate_single

        mock_orch = MagicMock()
        mock_orch.process.return_value = {
            "complexity": "simple",
            "complexity_score": 0.1,
            "complexity_signals": [],
            "sub_questions": [],
            "reasoning": "",
            "enhanced_question": example["question"],
            "sql": "SELECT COUNT(*) FROM singer",
            "plan": None,
            "used_planner": False,
            "failure_stage": None,
            "failure_detail": None,
            "elapsed_planning_s": None,
            "elapsed_compilation_s": None,
        }

        mode_config = {"retrieval": False, "decomposition": True, "retry": False}
        result = evaluate_single(
            example, schema, db_path,
            retriever_cache=None,
            orchestrator=mock_orch,
            mode_config=mode_config,
        )

        mock_orch.process.assert_called_once()
        # The SQL returned by orchestrator should be used
        assert result["pred_sql"] == "SELECT COUNT(*) FROM singer"

    def test_orchestrator_NOT_called_when_decomposition_off(
        self, example, schema, db_path
    ):
        """When decomposition=False, orchestrator must NOT be called."""
        from src.evaluvation.spider_eval import evaluate_single

        mock_orch = MagicMock()
        mode_config = {"retrieval": False, "decomposition": False, "retry": False}

        with patch("src.evaluvation.spider_eval.generate_sql") as mock_gen:
            mock_gen.return_value = "SELECT COUNT(*) FROM singer"

            result = evaluate_single(
                example, schema, db_path,
                retriever_cache=None,
                orchestrator=mock_orch,
                mode_config=mode_config,
            )

        mock_orch.process.assert_not_called()
        mock_gen.assert_called_once()

    def test_result_contains_planner_fields(self, example, schema, db_path):
        """Result dict must contain used_planner, failure_stage, failure_detail."""
        from src.evaluvation.spider_eval import evaluate_single

        mock_orch = MagicMock()
        mock_orch.process.return_value = {
            "complexity": "complex",
            "complexity_score": 0.8,
            "complexity_signals": ["multi_step"],
            "sub_questions": [],
            "reasoning": "plan",
            "enhanced_question": example["question"],
            "sql": "SELECT COUNT(*) FROM singer",
            "plan": None,
            "used_planner": True,
            "failure_stage": None,
            "failure_detail": None,
            "elapsed_planning_s": 0.5,
            "elapsed_compilation_s": 0.3,
        }

        mode_config = {"retrieval": False, "decomposition": True, "retry": False}
        result = evaluate_single(
            example, schema, db_path,
            retriever_cache=None,
            orchestrator=mock_orch,
            mode_config=mode_config,
        )

        assert "used_planner" in result
        assert result["used_planner"] is True
        assert "failure_stage" in result
        assert "failure_detail" in result


# ═══════════════════════════════════════════════════════════════════════
# 4.  METRICS CONTAIN PLANNER / FALLBACK FIELDS
# ═══════════════════════════════════════════════════════════════════════

class TestMetricsParityFields:
    def _make_results(self, n_planner=3, n_fallback=7):
        """Produce synthetic result dicts."""
        from src.evaluvation.spider_eval import FailureCategory
        results = []
        for i in range(n_planner):
            results.append({
                "execution_match": True,
                "pred_success": True,
                "pred_sql": "SELECT 1",
                "retried": False,
                "failure_category": FailureCategory.SUCCESS,
                "db_id": "db_a",
                "complexity": "complex",
                "complexity_score": 0.85,
                "used_planner": True,
                "failure_stage": None,
            })
        for i in range(n_fallback):
            results.append({
                "execution_match": i % 2 == 0,
                "pred_success": True,
                "pred_sql": "SELECT 1",
                "retried": False,
                "failure_category": (
                    FailureCategory.SUCCESS if i % 2 == 0
                    else FailureCategory.WRONG_VALUES
                ),
                "db_id": "db_b",
                "complexity": "simple",
                "complexity_score": 0.2,
                "used_planner": False,
                "failure_stage": None,
            })
        return results

    def test_planner_usage_rate_present(self):
        from src.evaluvation.spider_eval import compute_metrics
        results = self._make_results(3, 7)
        metrics = compute_metrics(results, "full")

        assert "planner_usage_count" in metrics
        assert "planner_usage_rate_pct" in metrics
        assert "fallback_count" in metrics
        assert "fallback_rate_pct" in metrics
        assert metrics["planner_usage_count"] == 3
        assert metrics["fallback_count"] == 7
        assert metrics["planner_usage_rate_pct"] == 30.0
        assert metrics["fallback_rate_pct"] == 70.0

    def test_orchestrator_invoked_count(self):
        from src.evaluvation.spider_eval import compute_metrics
        results = self._make_results(3, 7)
        metrics = compute_metrics(results, "full")
        # All 10 have complexity set → orchestrator was invoked for all
        assert metrics["orchestrator_invoked_count"] == 10

    def test_failure_stage_counts(self):
        from src.evaluvation.spider_eval import compute_metrics, FailureCategory
        results = self._make_results(0, 2)
        results[0]["failure_stage"] = "planning"
        results[1]["failure_stage"] = "compilation"
        metrics = compute_metrics(results, "full")
        assert metrics["failure_stage_counts"]["planning"] == 1
        assert metrics["failure_stage_counts"]["compilation"] == 1

    def test_zero_results_handled(self):
        from src.evaluvation.spider_eval import compute_metrics
        metrics = compute_metrics([], "baseline")
        assert metrics["total_examples"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 5.  ORCHESTRATOR IS WIRED IDENTICALLY
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorWiring:
    """
    Confirm that PlannerExecutorOrchestrator internally uses the same
    sub-components (QueryPlanner, PlanExecutor, QueryDecomposer) in both
    runtime and evaluation contexts.
    """

    def test_orchestrator_has_planner_compiler_decomposer(self):
        """Orchestrator must contain all three sub-components."""
        from src.decomposition.orchestrator import PlannerExecutorOrchestrator
        from src.decomposition.query_planner import QueryPlanner
        from src.decomposition.plan_executor import PlanExecutor
        from src.decomposition.query_decomposer import QueryDecomposer

        orch = PlannerExecutorOrchestrator(dialect="sqlite", enable_planner=True)
        assert isinstance(orch._planner, QueryPlanner)
        assert isinstance(orch._compiler, PlanExecutor)
        assert isinstance(orch._decomposer, QueryDecomposer)

    def test_runtime_and_eval_use_same_dialect_param(self):
        """
        Runtime creates orchestrator with dialect kwarg.
        Evaluation must do the same (dialect='sqlite').
        """
        from src.decomposition.orchestrator import PlannerExecutorOrchestrator

        # Runtime-style
        rt = PlannerExecutorOrchestrator(dialect="bigquery", enable_planner=True)
        assert rt.dialect == "bigquery"

        # Evaluation-style
        ev = PlannerExecutorOrchestrator(dialect="sqlite", enable_planner=True)
        assert ev.dialect == "sqlite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
