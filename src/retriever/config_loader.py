"""
Config loader for retrieval settings.

Loads the YAML config from configs/retrieval_config.yaml and provides
a fallback to hardcoded defaults if the file is missing or pyyaml
is not installed.
"""

from pathlib import Path

# Project root: two levels up from src/retriever/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "retrieval_config.yaml"

# Hardcoded fallback (matches the original table_prior.py values)
_FALLBACK_CONFIG = {
    "retrieval": {
        "top_k": 10,
        "semantic_weight": 0.65,
        "bm25_weight": 0.35,
        "table_prior_boost": 0.15,
        "column_group_min_hits": 2,
        "column_group_max_extra": 3,
        "embedding_model": "all-MiniLM-L6-v2",
        "cache_dir": ".cache/schema_embeddings",
    },
    "table_priors": {
        "revenue": ["orders", "order_product"],
        "sales": ["orders", "order_product"],
        "order": ["orders", "order_product"],
        "customer": ["customer"],
        "product": ["product", "order_product"],
    },
    "time_keywords": [
        "last month", "this month", "yesterday", "today", "last week",
        "this week", "last year", "this year", "month", "year", "week",
        "day", "date", "recent", "ago", "since", "between", "before",
        "after", "period", "quarter",
    ],
    "date_column_patterns": [
        "date", "time", "created", "modified", "shipped",
    ],
    "table_anchor_columns": {},
}


def load_retrieval_config(config_path=None):
    """
    Load retrieval configuration from a YAML file.

    Falls back to hardcoded defaults if the file doesn't exist
    or if PyYAML is not installed.

    Args:
        config_path: Path to the YAML config. Defaults to
                     configs/retrieval_config.yaml relative to project root.

    Returns:
        dict with retrieval configuration.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return _FALLBACK_CONFIG.copy()

    try:
        import yaml
    except ImportError:
        return _FALLBACK_CONFIG.copy()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        return _FALLBACK_CONFIG.copy()

    return config
