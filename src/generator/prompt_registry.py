"""
Prompt Version Registry — mandatory changelog & metric deltas.

Every prompt version is registered here with:
  - A unique version tag (e.g. ``V1``, ``V2``, ``V3``).
  - A human-readable changelog entry describing exactly what changed.
  - (After evaluation) Metric deltas relative to the previous version.

**No prompt version may be activated in ``prompts.py`` unless it has an
entry in this registry.**  The evaluation harness checks this invariant
at startup.

Usage
-----
    from src.generator.prompt_registry import REGISTRY, record_metric_delta

    # At eval startup — validates current prompts are registered
    from src.generator.prompt_registry import validate_active_versions

    # After an eval run — store the metric delta
    record_metric_delta("bigquery", "V3", {
        "baseline_version": "V2",
        "execution_accuracy_delta_pct": +1.2,
        "valid_sql_delta_pct": +0.0,
        "eval_examples": 50,
        "eval_date": "2026-02-17",
    })
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry data structure
# ---------------------------------------------------------------------------

class PromptVersion:
    """Immutable record of a single prompt version."""

    def __init__(
        self,
        dialect: str,
        version: str,
        date_introduced: str,
        changelog: str,
        *,
        parent_version: Optional[str] = None,
        metric_deltas: Optional[List[Dict[str, Any]]] = None,
    ):
        self.dialect = dialect
        self.version = version
        self.date_introduced = date_introduced
        self.changelog = changelog
        self.parent_version = parent_version
        self.metric_deltas: List[Dict[str, Any]] = metric_deltas or []

    def to_dict(self) -> dict:
        return {
            "dialect": self.dialect,
            "version": self.version,
            "date_introduced": self.date_introduced,
            "changelog": self.changelog,
            "parent_version": self.parent_version,
            "metric_deltas": self.metric_deltas,
        }

    def __repr__(self) -> str:
        return (
            f"PromptVersion({self.dialect}/{self.version}, "
            f"{self.date_introduced}, parent={self.parent_version})"
        )


# ---------------------------------------------------------------------------
# The registry — add ALL prompt versions here
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Dict[str, PromptVersion]] = {
    # ── SQLite ──────────────────────────────────────────────────────
    "sqlite": {
        "V1": PromptVersion(
            dialect="sqlite",
            version="V1",
            date_introduced="2026-02-01",
            changelog=(
                "Baseline zero-shot prompt.  Schema injected as table.column (type).  "
                "No few-shot examples.  Rules: standard SQLite syntax, no backticks."
            ),
        ),
        "V2": PromptVersion(
            dialect="sqlite",
            version="V2",
            date_introduced="2026-02-06",
            changelog=(
                "Quoting fix: explicitly forbid double-quoted identifiers to reduce "
                "quoting_issue errors (27.8 % → 15.0 % of errors).  Added 5 few-shot "
                "examples (singer/concert domain) demonstrating plain column names, "
                "JOINs with table aliases, GROUP BY, DISTINCT, ORDER BY LIMIT."
            ),
            parent_version="V1",
            metric_deltas=[{
                "baseline_version": "V1",
                "execution_accuracy_delta_pct": +4.45,
                "eval_examples": 1034,
                "eval_date": "2026-02-06",
                "notes": "V1=69.05% → V2=73.50%.  Quoting errors halved (89→41).",
            }],
        ),
    },

    # ── BigQuery ────────────────────────────────────────────────────
    "bigquery": {
        "V1": PromptVersion(
            dialect="bigquery",
            version="V1",
            date_introduced="2026-02-01",
            changelog=(
                "Baseline zero-shot BigQuery prompt.  Backtick quoting, dataset-qualified "
                "table names, TIMESTAMP handling rules."
            ),
        ),
        "V2": PromptVersion(
            dialect="bigquery",
            version="V2",
            date_introduced="2026-02-06",
            changelog=(
                "Added mandatory aliasing rules (AS t1, t2 …) and 4 few-shot examples "
                "to fix column ambiguity errors.  ⚠ KNOWN BUG: 'revenue per category' "
                "example joins order_product.product_id → category_tree.category_id, which "
                "is semantically wrong. Also uses search_keywords instead of name for "
                "category display."
            ),
            parent_version="V1",
        ),
        "V3": PromptVersion(
            dialect="bigquery",
            version="V3",
            date_introduced="2026-02-17",
            changelog=(
                "Fixes the incorrect join in V2's 'revenue per category' example.  "
                "Now routes through the product_category junction table: "
                "order_product.product_id → product_category.product_id → "
                "product_category.category_id → category_tree.category_id.  "
                "Uses category_tree.name instead of search_keywords for display.  "
                "Added rule 11: use product_category junction table for product↔category joins."
            ),
            parent_version="V2",
            metric_deltas=[],  # populated after first eval run
        ),
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_registered_version(dialect: str, version: str) -> Optional[PromptVersion]:
    """Look up a registered prompt version.  Returns None if not found."""
    return REGISTRY.get(dialect, {}).get(version)


def validate_active_versions() -> None:
    """Raise ``ValueError`` if the currently active prompt versions in
    ``prompts.py`` are not registered in the registry.

    Call this at evaluation startup to enforce the invariant that every
    active prompt version has a changelog entry.
    """
    from src.generator.prompts import get_active_versions

    active = get_active_versions()
    missing = []
    for dialect, version in active.items():
        pv = get_registered_version(dialect, version)
        if pv is None:
            missing.append(f"{dialect}/{version}")
    if missing:
        raise ValueError(
            f"Active prompt version(s) not registered in prompt_registry: "
            f"{', '.join(missing)}.  Add an entry before activating."
        )
    logger.info(
        "Prompt registry validated: %s",
        ", ".join(f"{d}={v}" for d, v in active.items()),
    )


def record_metric_delta(
    dialect: str,
    version: str,
    delta: Dict[str, Any],
) -> None:
    """Append a metric delta to an existing registry entry.

    Parameters
    ----------
    dialect : str
        ``"sqlite"`` or ``"bigquery"``.
    version : str
        The prompt version tag (e.g. ``"V3"``).
    delta : dict
        Must include at minimum ``baseline_version`` and
        ``execution_accuracy_delta_pct``.
    """
    pv = get_registered_version(dialect, version)
    if pv is None:
        raise KeyError(f"Cannot record delta — {dialect}/{version} not in registry.")
    required_keys = {"baseline_version", "execution_accuracy_delta_pct"}
    missing = required_keys - set(delta.keys())
    if missing:
        raise ValueError(f"Delta dict missing required keys: {missing}")
    delta.setdefault("eval_date", str(date.today()))
    pv.metric_deltas.append(delta)
    logger.info("Recorded metric delta for %s/%s: %s", dialect, version, delta)


def registry_summary() -> str:
    """Return a human-readable summary of the registry for reports."""
    lines = ["# Prompt Version Registry", ""]
    for dialect, versions in sorted(REGISTRY.items()):
        lines.append(f"## {dialect.upper()}")
        lines.append("")
        for ver_id, pv in sorted(versions.items()):
            lines.append(f"### {ver_id} ({pv.date_introduced})")
            lines.append(f"Parent: {pv.parent_version or '(none)'}")
            lines.append(f"Changelog: {pv.changelog}")
            if pv.metric_deltas:
                for md in pv.metric_deltas:
                    lines.append(
                        f"  Metric delta vs {md['baseline_version']}: "
                        f"EX {md['execution_accuracy_delta_pct']:+.2f}%"
                    )
            else:
                lines.append("  Metric delta: (not yet evaluated)")
            lines.append("")
    return "\n".join(lines)


def save_registry_snapshot(output_path: Path) -> None:
    """Persist the current registry state as JSON for audit trails."""
    data = {}
    for dialect, versions in REGISTRY.items():
        data[dialect] = {v: pv.to_dict() for v, pv in versions.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Registry snapshot saved to %s", output_path)
