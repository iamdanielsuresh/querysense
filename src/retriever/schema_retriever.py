"""
Enhanced Schema Retriever

Features:
  1. Hybrid retrieval: semantic (MiniLM-L6-v2) + BM25 keyword matching
  2. Schema embedding caching to disk (saves ~3-5s startup)
  3. Column grouping: pulls related columns from matched tables
  4. Dynamic table priors loaded from YAML config
  5. Column descriptions enrich semantic search
"""

import hashlib
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from src.retriever.config_loader import load_retrieval_config
except ModuleNotFoundError:
    from config_loader import load_retrieval_config


class SchemaRetriever:
    """
    Retrieves relevant schema columns for a natural-language question
    using a hybrid of dense (semantic) and sparse (BM25) scoring,
    with table-prior boosts and column grouping.
    """

    def __init__(self, schema_items, config_path=None):
        # Load config
        self.config = load_retrieval_config(config_path)
        retrieval_cfg = self.config.get("retrieval", {})

        self.semantic_weight = retrieval_cfg.get("semantic_weight", 0.65)
        self.bm25_weight = retrieval_cfg.get("bm25_weight", 0.35)
        self.table_prior_boost = retrieval_cfg.get("table_prior_boost", 0.15)
        self.column_group_min_hits = retrieval_cfg.get("column_group_min_hits", 2)
        self.column_group_max_extra = retrieval_cfg.get("column_group_max_extra", 3)
        model_name = retrieval_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        cache_dir_rel = retrieval_cfg.get("cache_dir", ".cache/schema_embeddings")

        # Table priors & time config
        self.table_priors = self.config.get("table_priors", {})
        self.time_keywords = self.config.get("time_keywords", [])
        self.date_column_patterns = self.config.get("date_column_patterns", [])
        self.table_anchor_columns = self.config.get("table_anchor_columns", {})

        # Build time-strip regex
        if self.time_keywords:
            self._time_strip_re = re.compile(
                r"\b("
                + "|".join(
                    re.escape(kw)
                    for kw in sorted(self.time_keywords, key=len, reverse=True)
                )
                + r")\b",
                re.IGNORECASE,
            )
        else:
            self._time_strip_re = None

        # Store schema
        self.schema_items = schema_items

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Compute or load cached embeddings
        project_root = Path(__file__).resolve().parents[2]
        self._cache_dir = project_root / cache_dir_rel
        self.schema_embeddings = self._load_or_compute_embeddings()

        # Build BM25 index for keyword matching
        self._build_bm25_index()

    # ------------------------------------------------------------------
    # Embedding caching
    # ------------------------------------------------------------------

    def _schema_hash(self):
        """Hash schema texts to detect changes for cache invalidation."""
        content = "|".join(item["text"] for item in self.schema_items)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_or_compute_embeddings(self):
        """Load embeddings from cache or compute and save them."""
        schema_hash = self._schema_hash()
        cache_path = self._cache_dir / f"embeddings_{schema_hash}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.shape[0] == len(self.schema_items):
                    return cached
            except Exception:
                pass  # Re-compute on any cache error

        # Compute fresh embeddings
        texts = [item["text"] for item in self.schema_items]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        # Save to cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)

        return embeddings

    def invalidate_cache(self):
        """Remove all cached embeddings."""
        if self._cache_dir.exists():
            for f in self._cache_dir.glob("embeddings_*.pkl"):
                f.unlink()

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _build_bm25_index(self):
        """Build a BM25 index over schema texts for keyword matching."""
        if BM25Okapi is None:
            self._bm25 = None
            return

        # Tokenize schema texts into words
        self._bm25_corpus = [
            item["text"].lower().split() for item in self.schema_items
        ]
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def _bm25_scores(self, question):
        """Get BM25 scores for a question against all schema items."""
        if self._bm25 is None:
            return np.zeros(len(self.schema_items))

        query_tokens = question.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Normalize to [0, 1] range
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        return scores

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question, top_k=None):
        """
        Given a natural language question, return top_k relevant schema columns
        as a list of (score, item) tuples.

        Scoring = semantic_weight * cosine_sim + bm25_weight * bm25_score + table_prior_boost
        """
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 10)

        question_lower = question.lower()

        # ── 1. Detect preferred tables from priors ──
        preferred_tables = set()
        for keyword, tables in self.table_priors.items():
            if keyword in question_lower:
                preferred_tables.update(tables)

        # ── 2. Strip temporal phrases for embedding ──
        needs_time = any(kw in question_lower for kw in self.time_keywords)
        if needs_time and self._time_strip_re:
            embedding_question = self._time_strip_re.sub("", question).strip()
            if not embedding_question:
                embedding_question = question
        else:
            embedding_question = question

        # ── 3. Semantic similarity ──
        self._last_question_embedding = self.model.encode(
            [embedding_question], normalize_embeddings=True
        )
        q_emb = self._last_question_embedding
        self._embedding_question = embedding_question

        semantic_scores = cosine_similarity(q_emb, self.schema_embeddings)[0]

        # ── 4. BM25 keyword scores (using original question for keywords) ──
        bm25_scores = self._bm25_scores(question)

        # ── 5. Hybrid scoring ──
        scored_items = []
        for idx, item in enumerate(self.schema_items):
            score = (
                self.semantic_weight * semantic_scores[idx]
                + self.bm25_weight * bm25_scores[idx]
            )

            # Table prior boost
            if item["table"] in preferred_tables:
                score += self.table_prior_boost

            scored_items.append((score, item))

        # Sort by score descending
        scored_items.sort(key=lambda x: x[0], reverse=True)

        top_results = [(score, item) for score, item in scored_items[:top_k]]

        # ── 6. Temporal column injection ──
        if needs_time and preferred_tables:
            already_included = {
                (item["table"], item["column"]) for _, item in top_results
            }
            for score, item in scored_items:
                col_lower = item["column"].lower()
                if (
                    item["table"] in preferred_tables
                    and any(p in col_lower for p in self.date_column_patterns)
                    and (item["table"], item["column"]) not in already_included
                ):
                    top_results.append((score, item))
                    already_included.add((item["table"], item["column"]))

        # ── 7. Column grouping ──
        top_results = self._apply_column_grouping(top_results)

        return top_results

    # ------------------------------------------------------------------
    # Column grouping
    # ------------------------------------------------------------------

    def _apply_column_grouping(self, top_results):
        """
        If a table has >= column_group_min_hits in the results,
        pull in anchor columns (e.g. id, date_created) that aren't
        already present, up to column_group_max_extra per table.
        """
        if not self.table_anchor_columns:
            return top_results

        # Count hits per table
        table_hits = {}
        for _, item in top_results:
            table_hits[item["table"]] = table_hits.get(item["table"], 0) + 1

        already_included = {
            (item["table"], item["column"]) for _, item in top_results
        }

        # Build a lookup for quick schema item access
        schema_lookup = {}
        for item in self.schema_items:
            schema_lookup.setdefault(item["table"], {})[item["column"]] = item

        extras = []
        for table, hits in table_hits.items():
            if hits < self.column_group_min_hits:
                continue
            anchors = self.table_anchor_columns.get(table, [])
            added = 0
            for col in anchors:
                if added >= self.column_group_max_extra:
                    break
                if (table, col) not in already_included and col in schema_lookup.get(
                    table, {}
                ):
                    item = schema_lookup[table][col]
                    extras.append((0.0, item))  # Score 0 — these are grouped additions
                    already_included.add((table, col))
                    added += 1

        top_results.extend(extras)
        return top_results