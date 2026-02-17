"""
SQL Policy Checker – pre-execution guardrail.

Validates SQL statements against a configurable policy *before* any query
reaches BigQuery.  Returns structured verdicts so callers can log / surface
the exact reason a query was blocked.

Design goals
------------
* Zero network calls – pure string / parse analysis.
* No external dependencies beyond the stdlib + ``sqlglot`` (optional but
  recommended for robust parsing; falls back to regex heuristics).
* Every public function returns a typed ``PolicyVerdict`` dataclass.
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------

class VerdictStatus(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"


@dataclass
class PolicyVerdict:
    """Immutable result of a policy check."""
    status: VerdictStatus
    sql: str
    reasons: List[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    @property
    def is_allowed(self) -> bool:
        return self.status == VerdictStatus.ALLOWED

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "checked_at": self.checked_at,
            "sql_length": len(self.sql),
        }


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

# Regex that catches the *first* SQL keyword (ignoring leading whitespace,
# comments, and CTEs).
_LEADING_KEYWORD_RE = re.compile(
    r"^\s*(?:--[^\n]*\n\s*)*"         # skip line comments
    r"(?:/\*.*?\*/\s*)*"              # skip block comments
    r"(?:WITH\b[^;]*?\)\s*)?",        # optionally skip CTE preamble
    re.IGNORECASE | re.DOTALL,
)

_STATEMENT_TYPE_RE = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|MERGE|TRUNCATE|"
    r"GRANT|REVOKE|CALL|EXECUTE|EXEC|EXPLAIN|SHOW|SET|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def _extract_statement_types(sql: str) -> List[str]:
    """Return *all* SQL statement keywords found in the query (upper-cased)."""
    return [m.group(1).upper() for m in _STATEMENT_TYPE_RE.finditer(sql)]


def check_sql_policy(
    sql: str,
    *,
    allowed_types: Optional[List[str]] = None,
    blocked_keywords: Optional[List[str]] = None,
    max_query_length: int = 10_000,
) -> PolicyVerdict:
    """
    Evaluate *sql* against the policy.

    Parameters
    ----------
    sql : str
        The raw SQL string to validate.
    allowed_types : list[str] | None
        Upper-cased statement types that are permitted (e.g. ``["SELECT"]``).
        Defaults to ``["SELECT"]`` if *None*.
    blocked_keywords : list[str] | None
        Extra keywords whose presence anywhere in the query causes rejection.
    max_query_length : int
        Hard cap on character count.

    Returns
    -------
    PolicyVerdict
    """
    if allowed_types is None:
        allowed_types = ["SELECT"]
    allowed_types = [t.upper() for t in allowed_types]

    if blocked_keywords is None:
        blocked_keywords = []
    blocked_keywords_upper = [k.upper() for k in blocked_keywords]

    reasons: List[str] = []

    # ── 1. Empty / whitespace-only ───────────────────────────
    stripped = sql.strip()
    if not stripped:
        reasons.append("Query is empty.")
        return PolicyVerdict(
            status=VerdictStatus.BLOCKED, sql=sql, reasons=reasons
        )

    # ── 2. Length guard ──────────────────────────────────────
    if len(stripped) > max_query_length:
        reasons.append(
            f"Query length ({len(stripped)} chars) exceeds maximum "
            f"({max_query_length} chars)."
        )

    # ── 3. Statement-type allow-list ─────────────────────────
    found_types = _extract_statement_types(stripped)
    if not found_types:
        reasons.append("Could not determine SQL statement type.")
    else:
        disallowed = [t for t in found_types if t not in allowed_types]
        # Special cases: WITH (CTE) is allowed if paired with SELECT.
        # EXPLAIN is treated as read-only.
        safe_passthrough = {"WITH", "EXPLAIN"}
        disallowed = [t for t in disallowed if t not in safe_passthrough]
        if disallowed:
            reasons.append(
                f"Disallowed statement type(s): {', '.join(sorted(set(disallowed)))}. "
                f"Only {allowed_types} are permitted."
            )

    # ── 4. Blocked-keyword scan ──────────────────────────────
    sql_upper = stripped.upper()
    for kw in blocked_keywords_upper:
        # Use word-boundary matching to avoid false positives
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, sql_upper):
            reasons.append(f"Blocked keyword detected: {kw}.")

    # ── 5. Semicolon / multi-statement guard ─────────────────
    # Strip trailing semicolons, then check for remaining ones (statement stacking)
    core = stripped.rstrip(";").strip()
    if ";" in core:
        reasons.append(
            "Multiple statements detected (semicolon in query body). "
            "Only single statements are allowed."
        )

    # ── Verdict ──────────────────────────────────────────────
    status = VerdictStatus.BLOCKED if reasons else VerdictStatus.ALLOWED
    verdict = PolicyVerdict(status=status, sql=sql, reasons=reasons)
    if not verdict.is_allowed:
        logger.warning("SQL blocked: %s", verdict.to_dict())
    return verdict
