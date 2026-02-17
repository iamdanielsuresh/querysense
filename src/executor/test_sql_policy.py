"""
Tests for the SQL policy checker and execution guardrails.

Run with:  python -m pytest src/executor/test_sql_policy.py -v
"""
from __future__ import annotations

import pytest

from src.executor.sql_policy import (
    PolicyVerdict,
    VerdictStatus,
    check_sql_policy,
    _extract_statement_types,
)


# =====================================================================
# Statement-type extraction
# =====================================================================

class TestExtractStatementTypes:
    def test_simple_select(self):
        assert _extract_statement_types("SELECT 1") == ["SELECT"]

    def test_cte_select(self):
        types = _extract_statement_types(
            "WITH cte AS (SELECT id FROM t) SELECT * FROM cte"
        )
        assert "SELECT" in types

    def test_insert(self):
        assert "INSERT" in _extract_statement_types("INSERT INTO t VALUES (1)")

    def test_drop(self):
        assert "DROP" in _extract_statement_types("DROP TABLE foo")

    def test_mixed(self):
        types = _extract_statement_types("DELETE FROM t; SELECT 1")
        assert "DELETE" in types
        assert "SELECT" in types


# =====================================================================
# Policy verdicts — allowed queries
# =====================================================================

class TestAllowedQueries:
    def test_simple_select(self):
        v = check_sql_policy("SELECT * FROM orders")
        assert v.is_allowed
        assert v.status == VerdictStatus.ALLOWED
        assert v.reasons == []

    def test_select_with_trailing_semicolon(self):
        v = check_sql_policy("SELECT 1;")
        assert v.is_allowed

    def test_cte_select(self):
        sql = """
        WITH monthly AS (
            SELECT DATE_TRUNC(date_created, MONTH) AS m, SUM(total) AS t
            FROM orders
            GROUP BY 1
        )
        SELECT * FROM monthly ORDER BY m
        """
        v = check_sql_policy(sql)
        assert v.is_allowed

    def test_explain_allowed(self):
        v = check_sql_policy("EXPLAIN SELECT 1")
        assert v.is_allowed

    def test_subquery(self):
        sql = "SELECT * FROM (SELECT id FROM orders) sub"
        v = check_sql_policy(sql)
        assert v.is_allowed


# =====================================================================
# Policy verdicts — blocked queries
# =====================================================================

class TestBlockedQueries:
    def test_empty_query(self):
        v = check_sql_policy("")
        assert not v.is_allowed
        assert "empty" in v.reasons[0].lower()

    def test_whitespace_only(self):
        v = check_sql_policy("   \n  ")
        assert not v.is_allowed

    def test_insert(self):
        v = check_sql_policy("INSERT INTO orders VALUES (1)")
        assert not v.is_allowed
        assert any("INSERT" in r for r in v.reasons)

    def test_update(self):
        v = check_sql_policy("UPDATE orders SET total = 0")
        assert not v.is_allowed

    def test_delete(self):
        v = check_sql_policy("DELETE FROM orders WHERE id = 1")
        assert not v.is_allowed

    def test_drop_table(self):
        v = check_sql_policy("DROP TABLE orders")
        assert not v.is_allowed

    def test_create_table(self):
        v = check_sql_policy("CREATE TABLE foo (id INT)")
        assert not v.is_allowed

    def test_alter_table(self):
        v = check_sql_policy("ALTER TABLE orders ADD COLUMN x INT")
        assert not v.is_allowed

    def test_merge(self):
        v = check_sql_policy("MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET x=1")
        assert not v.is_allowed

    def test_truncate(self):
        v = check_sql_policy("TRUNCATE TABLE orders")
        assert not v.is_allowed

    def test_grant(self):
        v = check_sql_policy("GRANT SELECT ON orders TO user1")
        assert not v.is_allowed

    def test_multi_statement(self):
        v = check_sql_policy("SELECT 1; SELECT 2")
        assert not v.is_allowed
        assert any("multiple" in r.lower() for r in v.reasons)

    def test_query_too_long(self):
        long_sql = "SELECT " + "x, " * 6000 + "1"
        v = check_sql_policy(long_sql, max_query_length=100)
        assert not v.is_allowed
        assert any("length" in r.lower() for r in v.reasons)

    def test_blocked_keywords(self):
        v = check_sql_policy(
            "SELECT * FROM orders",
            blocked_keywords=["orders"],
        )
        assert not v.is_allowed
        assert any("ORDERS" in r for r in v.reasons)


# =====================================================================
# Verdict serialization
# =====================================================================

class TestVerdictSerialization:
    def test_to_dict_allowed(self):
        v = check_sql_policy("SELECT 1")
        d = v.to_dict()
        assert d["status"] == "allowed"
        assert d["reasons"] == []
        assert "sql_length" in d

    def test_to_dict_blocked(self):
        v = check_sql_policy("DROP TABLE x")
        d = v.to_dict()
        assert d["status"] == "blocked"
        assert len(d["reasons"]) > 0


# =====================================================================
# Execution envelope (unit test for the helper)
# =====================================================================

# These tests require google-cloud-bigquery to be installed since they
# import from bq_executor.  Skip gracefully in lightweight test envs.
try:
    from src.executor.bq_executor import _execution_envelope, load_execution_config, _resolve_env_vars
    _HAS_BQ = True
except ImportError:
    _HAS_BQ = False


@pytest.mark.skipif(not _HAS_BQ, reason="google-cloud-bigquery not installed")
class TestExecutionEnvelope:
    def test_blocked_envelope(self):
        env = _execution_envelope(
            success=False,
            sql="DROP TABLE x",
            blocked=True,
            block_reasons=["not allowed"],
            error="blocked",
        )
        assert env["success"] is False
        assert env["blocked"] is True
        assert env["block_reasons"] == ["not allowed"]
        assert "rows" not in env

    def test_success_envelope(self):
        env = _execution_envelope(
            success=True,
            sql="SELECT 1",
            rows=[{"col": 1}],
            estimated_bytes=1024,
            elapsed_seconds=0.5,
        )
        assert env["success"] is True
        assert env["blocked"] is False
        assert env["rows"] == [{"col": 1}]
        assert env["estimated_bytes"] == 1024


# =====================================================================
# Config loading
# =====================================================================

@pytest.mark.skipif(not _HAS_BQ, reason="google-cloud-bigquery not installed")
class TestConfigLoading:
    def test_load_default_config(self):
        cfg = load_execution_config()
        assert "sql_policy" in cfg
        assert "cost_guard" in cfg
        assert "timeout" in cfg

    def test_env_var_resolution(self):
        import os

        os.environ["_TEST_SQL_SAFETY_VAR"] = "hello"
        assert _resolve_env_vars("${_TEST_SQL_SAFETY_VAR}") == "hello"
        assert _resolve_env_vars("${_MISSING_VAR:fallback}") == "fallback"
        assert _resolve_env_vars("${_MISSING_VAR}") == ""
        del os.environ["_TEST_SQL_SAFETY_VAR"]
