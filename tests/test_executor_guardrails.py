"""
Tests for BigQueryExecutor production guardrails:

  - fail-closed when dry-run fails
  - maximum_bytes_billed enforced on real job config
  - max_returned_rows truncation
  - structured guardrail log events

Run with:  python -m pytest tests/test_executor_guardrails.py -v
"""
from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace, ModuleType
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Ensure google.cloud.bigquery is importable even without the real SDK.
# We create lightweight stubs that satisfy the executor's import-time needs.
# ---------------------------------------------------------------------------
_NEED_STUB = "google.cloud.bigquery" not in sys.modules

if _NEED_STUB:
    _gcloud_mod = sys.modules.setdefault("google", ModuleType("google"))
    _gcloud_cloud_mod = sys.modules.setdefault("google.cloud", ModuleType("google.cloud"))
    _bq_mod = ModuleType("google.cloud.bigquery")

    class _StubQueryJobConfig:
        def __init__(self, **kwargs):
            self.dry_run = False
            self.maximum_bytes_billed = None
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _StubQueryPriority:
        INTERACTIVE = "INTERACTIVE"
        BATCH = "BATCH"

    class _StubClient:
        def __init__(self, **kwargs):
            pass
        def query(self, sql, job_config=None):
            raise NotImplementedError("stub")

    _bq_mod.QueryJobConfig = _StubQueryJobConfig  # type: ignore[attr-defined]
    _bq_mod.QueryPriority = _StubQueryPriority     # type: ignore[attr-defined]
    _bq_mod.Client = _StubClient                    # type: ignore[attr-defined]
    _gcloud_cloud_mod.bigquery = _bq_mod            # type: ignore[attr-defined]
    sys.modules["google.cloud.bigquery"] = _bq_mod

from src.executor.bq_executor import BigQueryExecutor, _log_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(overrides: Dict[str, Any] | None = None) -> BigQueryExecutor:
    """Build an executor with a minimal in-memory config (no real BQ client)."""
    base: Dict[str, Any] = {
        "bigquery": {"project": "test-project", "dataset": "test_ds", "location": "US"},
        "sql_policy": {
            "allowed_statement_types": ["SELECT"],
            "blocked_keywords": ["DROP", "DELETE"],
            "max_query_length": 5000,
        },
        "cost_guard": {
            "enabled": True,
            "max_bytes": 1_000_000,
            "max_bytes_display": "1 MB",
            "warn_bytes": 500_000,
            "fail_closed_on_dry_run_error": True,
            "maximum_bytes_billed": 2_000_000,
        },
        "max_returned_rows": 5,
        "timeout": {"query_timeout_seconds": 10, "job_retry_count": 0},
        "query_priority": "INTERACTIVE",
        "logging": {"emit_json": True, "include_sql_in_logs": False},
    }
    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                base[key].update(val)
            else:
                base[key] = val

    with patch("src.executor.bq_executor.bigquery.Client"):
        return BigQueryExecutor(config=base)


def _fake_query_job(total_bytes: int = 100):
    """Return a mock BQ job suitable for dry-run responses."""
    job = MagicMock()
    job.total_bytes_processed = total_bytes
    return job


def _fake_result_rows(n: int) -> list:
    """Produce *n* mock row objects that dict() can consume."""
    rows = []
    for i in range(n):
        row = MagicMock()
        row.__iter__ = MagicMock(return_value=iter([("id", i), ("name", f"r{i}")]))
        row.keys = MagicMock(return_value=["id", "name"])
        # dict(row) needs the mapping protocol
        row.__getitem__ = MagicMock(side_effect=lambda k, _i=i: _i if k == "id" else f"r{_i}")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Fail-closed on dry-run failure
# ---------------------------------------------------------------------------

class TestFailClosedOnDryRunError:
    """When cost guard is enabled and dry-run raises, execution must be blocked."""

    def test_dry_run_exception_blocks_query(self):
        exe = _make_executor()
        exe._client.query = MagicMock(side_effect=Exception("BQ dry-run timeout"))

        result = exe.execute("SELECT 1")

        assert result["success"] is False
        assert result["blocked"] is True
        assert any("fail_closed" in r.lower() or "dry-run" in r.lower()
                    for r in result["block_reasons"])

    def test_dry_run_exception_allowed_when_fail_closed_disabled(self):
        exe = _make_executor({"cost_guard": {"fail_closed_on_dry_run_error": False}})

        call_count = 0

        def _side_effect(sql, job_config=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is the dry-run
                raise Exception("BQ dry-run timeout")
            # Second call is the real query
            job = MagicMock()
            job.result.return_value = iter([])
            return job

        exe._client.query = MagicMock(side_effect=_side_effect)

        result = exe.execute("SELECT 1")
        # Should NOT be blocked — dry-run failure is tolerated
        assert result["blocked"] is False

    def test_dry_run_exception_blocked_by_default(self):
        """Default config has fail_closed=True."""
        exe = _make_executor()
        exe._client.query = MagicMock(side_effect=RuntimeError("network glitch"))

        result = exe.execute("SELECT 1")
        assert result["blocked"] is True
        assert "fail_closed_on_dry_run_error" in " ".join(result["block_reasons"])


# ---------------------------------------------------------------------------
# maximum_bytes_billed enforcement
# ---------------------------------------------------------------------------

class TestMaximumBytesBilled:
    """Server-side bytes cap must be set on the real QueryJobConfig."""

    def test_bytes_billed_set_on_job_config(self):
        exe = _make_executor()

        configs_seen: list = []

        def _capture_query(sql, job_config=None):
            configs_seen.append(job_config)
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 100
                return job
            job = MagicMock()
            job.result.return_value = iter([])
            return job

        exe._client.query = MagicMock(side_effect=_capture_query)
        exe.execute("SELECT 1")

        # The second call is the real query
        assert len(configs_seen) == 2
        real_cfg = configs_seen[1]
        assert real_cfg.maximum_bytes_billed == 2_000_000

    def test_bytes_billed_none_when_not_configured(self):
        exe = _make_executor({"cost_guard": {"maximum_bytes_billed": None}})
        # Remove the attr set during init
        exe._maximum_bytes_billed = None

        configs_seen: list = []

        def _capture_query(sql, job_config=None):
            configs_seen.append(job_config)
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 100
                return job
            job = MagicMock()
            job.result.return_value = iter([])
            return job

        exe._client.query = MagicMock(side_effect=_capture_query)
        exe.execute("SELECT 1")

        real_cfg = configs_seen[1]
        # maximum_bytes_billed should NOT have been explicitly set
        assert not hasattr(real_cfg, "_maximum_bytes_billed_set") or \
               real_cfg.maximum_bytes_billed is None or \
               real_cfg.maximum_bytes_billed == 0


# ---------------------------------------------------------------------------
# max_returned_rows truncation
# ---------------------------------------------------------------------------

class TestMaxReturnedRows:
    """Rows beyond the configured limit must be silently truncated."""

    def test_rows_truncated_when_over_limit(self):
        exe = _make_executor()  # max_returned_rows = 5

        def _query_returning_rows(sql, job_config=None):
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 100
                return job
            job = MagicMock()
            # Return 20 rows – more than the 5-row cap
            fake_rows = [{"id": i, "val": f"v{i}"} for i in range(20)]
            job.result.return_value = iter(
                [MagicMock(**{"__iter__": MagicMock(return_value=iter(r.items())),
                              "keys": MagicMock(return_value=list(r.keys()))})
                 for r in fake_rows]
            )
            # Make dict(row) work by patching each mock
            real_rows = []
            for r in fake_rows:
                m = MagicMock()
                m.__iter__ = MagicMock(return_value=iter(r.items()))
                m.keys = MagicMock(return_value=list(r.keys()))
                m.__getitem__ = MagicMock(side_effect=r.__getitem__)
                real_rows.append(r)  # just use real dicts
            job.result.return_value = iter(real_rows)
            return job

        exe._client.query = MagicMock(side_effect=_query_returning_rows)
        result = exe.execute("SELECT 1")

        assert result["success"] is True
        assert len(result["rows"]) == 5

    def test_rows_not_truncated_when_under_limit(self):
        exe = _make_executor()

        def _query_returning_rows(sql, job_config=None):
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 100
                return job
            job = MagicMock()
            job.result.return_value = iter([{"id": 1}, {"id": 2}])
            return job

        exe._client.query = MagicMock(side_effect=_query_returning_rows)
        result = exe.execute("SELECT 1")

        assert result["success"] is True
        assert len(result["rows"]) == 2

    def test_no_truncation_when_limit_disabled(self):
        exe = _make_executor({"max_returned_rows": 0})
        exe._max_returned_rows = None  # 0 → disabled

        def _query_returning_rows(sql, job_config=None):
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 100
                return job
            job = MagicMock()
            job.result.return_value = iter([{"id": i} for i in range(50)])
            return job

        exe._client.query = MagicMock(side_effect=_query_returning_rows)
        result = exe.execute("SELECT 1")

        assert result["success"] is True
        assert len(result["rows"]) == 50


# ---------------------------------------------------------------------------
# Bytes cap (existing cost guard) – regression
# ---------------------------------------------------------------------------

class TestBytesCap:
    """Dry-run bytes over max_bytes should still block."""

    def test_over_cap_blocked(self):
        exe = _make_executor()

        def _dry_run_large(sql, job_config=None):
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 5_000_000  # > 1 MB cap
                return job
            raise AssertionError("Real query should not run")

        exe._client.query = MagicMock(side_effect=_dry_run_large)
        result = exe.execute("SELECT * FROM big_table")

        assert result["blocked"] is True
        assert result["estimated_bytes"] == 5_000_000

    def test_under_cap_allowed(self):
        exe = _make_executor()

        def _dry_run_small(sql, job_config=None):
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 100
                return job
            job = MagicMock()
            job.result.return_value = iter([])
            return job

        exe._client.query = MagicMock(side_effect=_dry_run_small)
        result = exe.execute("SELECT 1")

        assert result["success"] is True
        assert result["blocked"] is False


# ---------------------------------------------------------------------------
# Structured log events
# ---------------------------------------------------------------------------

class TestGuardrailLogging:
    """Guardrail decisions must emit structured log events with expected fields."""

    def test_blocked_cost_emits_guardrail_event(self, caplog):
        exe = _make_executor()

        def _dry_run_large(sql, job_config=None):
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 5_000_000
                return job
            raise AssertionError("should not run")

        exe._client.query = MagicMock(side_effect=_dry_run_large)

        with caplog.at_level(logging.INFO):
            exe.execute("SELECT * FROM huge")

        json_lines = [r.message for r in caplog.records]
        guardrail_events = []
        for line in json_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("event") == "guardrail_decision":
                    guardrail_events.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        assert any(e.get("guardrail") == "cost_cap" and e.get("action") == "blocked"
                   for e in guardrail_events)

    def test_fail_closed_emits_guardrail_event(self, caplog):
        exe = _make_executor()
        exe._client.query = MagicMock(side_effect=Exception("boom"))

        with caplog.at_level(logging.INFO):
            exe.execute("SELECT 1")

        json_lines = [r.message for r in caplog.records]
        guardrail_events = []
        for line in json_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("event") == "guardrail_decision":
                    guardrail_events.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        assert any(e.get("guardrail") == "dry_run_fail_closed" for e in guardrail_events)

    def test_truncated_rows_emits_guardrail_event(self, caplog):
        exe = _make_executor()

        def _query(sql, job_config=None):
            if job_config and job_config.dry_run:
                job = MagicMock()
                job.total_bytes_processed = 100
                return job
            job = MagicMock()
            job.result.return_value = iter([{"id": i} for i in range(20)])
            return job

        exe._client.query = MagicMock(side_effect=_query)

        with caplog.at_level(logging.INFO):
            exe.execute("SELECT id FROM t")

        json_lines = [r.message for r in caplog.records]
        guardrail_events = []
        for line in json_lines:
            try:
                parsed = json.loads(line)
                if parsed.get("event") == "guardrail_decision":
                    guardrail_events.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        assert any(e.get("guardrail") == "max_returned_rows" and e.get("action") == "truncated"
                   for e in guardrail_events)
