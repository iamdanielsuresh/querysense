"""
BigQuery Executor – safe, config-driven query execution.

All queries pass through:
  1. SQL policy check  (read-only allow-list, blocked keywords)
  2. Dry-run cost estimation  (bytes-processed cap)
  3. Timeout & priority controls
  4. Structured result / error envelope

No hardcoded project IDs.  Tenant/project resolved from config or env vars.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from google.cloud import bigquery

from src.executor.sql_policy import check_sql_policy, PolicyVerdict, VerdictStatus

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "execution_config.yaml"

_ENV_VAR_RE = None  # lazy-compiled


def _resolve_env_vars(value: str) -> str:
    """Replace ``${VAR}`` or ``${VAR:default}`` in a string."""
    import re
    global _ENV_VAR_RE
    if _ENV_VAR_RE is None:
        _ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")

    def _sub(m):
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default if default is not None else "")
    return _ENV_VAR_RE.sub(_sub, value)


def _walk_resolve(obj):
    """Recursively resolve env vars in a nested dict/list."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(v) for v in obj]
    return obj


def load_execution_config(path: Optional[Path] = None) -> dict:
    """Load and env-resolve the execution config YAML."""
    cfg_path = path or _CONFIG_PATH
    if not cfg_path.exists():
        logger.warning("Execution config not found at %s – using defaults", cfg_path)
        return {}
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _walk_resolve(raw)


# ---------------------------------------------------------------------------
# Structured execution result
# ---------------------------------------------------------------------------

def _execution_envelope(
    *,
    success: bool,
    sql: str,
    rows: Optional[List[dict]] = None,
    error: Optional[str] = None,
    blocked: bool = False,
    block_reasons: Optional[List[str]] = None,
    estimated_bytes: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Uniform result envelope returned by every execution path."""
    env = {
        "success": success,
        "sql": sql,
        "blocked": blocked,
        "block_reasons": block_reasons or [],
        "estimated_bytes": estimated_bytes,
        "elapsed_seconds": elapsed_seconds,
    }
    if rows is not None:
        env["rows"] = rows
    if error is not None:
        env["error"] = error
    return env


def _log_event(event: str, data: dict, *, emit_json: bool = True):
    """Emit a structured log line (JSON to stderr)."""
    payload = {"event": event, "ts": time.time(), **data}
    if emit_json:
        logger.info(json.dumps(payload, default=str))
    else:
        logger.info("%s: %s", event, data)


# ---------------------------------------------------------------------------
# Executor class
# ---------------------------------------------------------------------------

class BigQueryExecutor:
    """
    Config-driven BigQuery executor with safety guardrails.

    Parameters
    ----------
    config : dict | None
        Pre-loaded config dict.  If *None*, loads from the default YAML path.
    project : str | None
        Override project ID (takes precedence over config / env).
    dataset : str | None
        Override dataset (takes precedence over config / env).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        project: Optional[str] = None,
        dataset: Optional[str] = None,
    ):
        self._cfg = config or load_execution_config()

        bq_cfg = self._cfg.get("bigquery", {})
        self.project = project or bq_cfg.get("project") or os.getenv("BQ_PROJECT", "")
        self.dataset = dataset or bq_cfg.get("dataset") or os.getenv("BQ_DATASET", "")
        self.location = bq_cfg.get("location") or os.getenv("BQ_LOCATION", "US")

        # Infer project from dataset if BQ_PROJECT was not set explicitly
        # (BQ_DATASET often looks like "project.dataset")
        if not self.project and "." in self.dataset:
            self.project = self.dataset.split(".")[0]

        if not self.project:
            raise ValueError(
                "BigQuery project ID must be set via config, constructor arg, "
                "or the BQ_PROJECT environment variable."
            )

        self._client = bigquery.Client(project=self.project, location=self.location)

        # Policy settings
        pol = self._cfg.get("sql_policy", {})
        self._allowed_types: List[str] = pol.get("allowed_statement_types", ["SELECT"])
        self._blocked_keywords: List[str] = pol.get("blocked_keywords", [])
        self._max_query_length: int = int(pol.get("max_query_length", 10_000))

        # Cost guard
        cg = self._cfg.get("cost_guard", {})
        self._cost_guard_enabled: bool = cg.get("enabled", True)
        self._max_bytes: int = int(cg.get("max_bytes", 1_073_741_824))
        self._max_bytes_display: str = cg.get("max_bytes_display", "1 GB")
        self._warn_bytes: int = int(cg.get("warn_bytes", 536_870_912))

        # Timeout & priority
        to = self._cfg.get("timeout", {})
        self._query_timeout: int = int(to.get("query_timeout_seconds", 30))
        self._job_retry_count: int = int(to.get("job_retry_count", 0))
        self._priority: str = self._cfg.get("query_priority", "INTERACTIVE")

        # Logging prefs
        log_cfg = self._cfg.get("logging", {})
        self._emit_json: bool = log_cfg.get("emit_json", True)
        self._include_sql_in_logs: bool = log_cfg.get("include_sql_in_logs", False)

    # ── Public API ───────────────────────────────────────────

    def execute(self, sql: str) -> Dict[str, Any]:
        """
        Full guarded execution pipeline:
          policy check → dry-run cost → real execution → envelope.

        Returns a structured dict.  On any rejection the dict contains
        ``blocked=True`` and ``block_reasons`` with human-readable strings.
        """
        sql = (sql or "").strip()
        log_sql = sql if self._include_sql_in_logs else "<redacted>"

        # 1. Policy check
        verdict = check_sql_policy(
            sql,
            allowed_types=self._allowed_types,
            blocked_keywords=self._blocked_keywords,
            max_query_length=self._max_query_length,
        )
        if not verdict.is_allowed:
            _log_event("query_blocked_policy", {
                "sql": log_sql,
                "reasons": verdict.reasons,
            }, emit_json=self._emit_json)
            return _execution_envelope(
                success=False,
                sql=sql,
                blocked=True,
                block_reasons=verdict.reasons,
                error="Query blocked by SQL policy: " + "; ".join(verdict.reasons),
            )

        # 2. Dry-run cost estimation
        if self._cost_guard_enabled:
            dry_result = self._dry_run(sql)
            if dry_result is not None:
                estimated_bytes = dry_result.get("estimated_bytes", 0)
                if estimated_bytes > self._max_bytes:
                    reason = (
                        f"Estimated bytes ({estimated_bytes:,}) exceed cost cap "
                        f"({self._max_bytes_display})."
                    )
                    _log_event("query_blocked_cost", {
                        "sql": log_sql,
                        "estimated_bytes": estimated_bytes,
                        "max_bytes": self._max_bytes,
                    }, emit_json=self._emit_json)
                    return _execution_envelope(
                        success=False,
                        sql=sql,
                        blocked=True,
                        block_reasons=[reason],
                        estimated_bytes=estimated_bytes,
                        error="Query blocked by cost guard: " + reason,
                    )
                if estimated_bytes > self._warn_bytes:
                    logger.warning(
                        "Cost warning: query will process ~%s bytes", f"{estimated_bytes:,}"
                    )
            else:
                estimated_bytes = None
        else:
            estimated_bytes = None

        # 3. Real execution
        return self._run_query(sql, estimated_bytes=estimated_bytes)

    # ── Internal helpers ─────────────────────────────────────

    def _dry_run(self, sql: str) -> Optional[dict]:
        """Run a BigQuery dry-run and return estimated bytes, or None on error."""
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            job = self._client.query(sql, job_config=job_config)
            return {"estimated_bytes": job.total_bytes_processed}
        except Exception as exc:
            logger.warning("Dry-run failed (query will still be attempted): %s", exc)
            return None

    def _run_query(
        self, sql: str, *, estimated_bytes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute the query with timeout and priority."""
        try:
            job_config = bigquery.QueryJobConfig(
                use_query_cache=True,
                priority=(
                    bigquery.QueryPriority.BATCH
                    if self._priority.upper() == "BATCH"
                    else bigquery.QueryPriority.INTERACTIVE
                ),
            )

            t0 = time.time()
            job = self._client.query(sql, job_config=job_config)
            rows_iter = job.result(timeout=self._query_timeout)
            rows = [dict(row) for row in rows_iter]
            elapsed = time.time() - t0

            _log_event("query_success", {
                "elapsed": round(elapsed, 3),
                "row_count": len(rows),
                "estimated_bytes": estimated_bytes,
            }, emit_json=self._emit_json)

            return _execution_envelope(
                success=True,
                sql=sql,
                rows=rows,
                estimated_bytes=estimated_bytes,
                elapsed_seconds=round(elapsed, 3),
            )

        except Exception as exc:
            elapsed = time.time() - t0 if "t0" in dir() else None
            error_msg = str(exc)
            _log_event("query_error", {
                "error": error_msg,
                "estimated_bytes": estimated_bytes,
            }, emit_json=self._emit_json)

            return _execution_envelope(
                success=False,
                sql=sql,
                error=error_msg,
                estimated_bytes=estimated_bytes,
                elapsed_seconds=round(elapsed, 3) if elapsed else None,
            )


# ---------------------------------------------------------------------------
# Backwards-compatible module-level function
# ---------------------------------------------------------------------------
# Lazily initialised so import-time side effects are eliminated.

_default_executor: Optional[BigQueryExecutor] = None


def _get_default_executor() -> BigQueryExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = BigQueryExecutor()
    return _default_executor


def execute_sql(sql: str) -> Dict[str, Any]:
    """
    Drop-in replacement for the old ``execute_sql``.

    Returns the same ``{"success": bool, "rows": [...], "error": ...}``
    shape, now enriched with ``blocked``, ``block_reasons``,
    ``estimated_bytes``, and ``elapsed_seconds``.
    """
    return _get_default_executor().execute(sql)