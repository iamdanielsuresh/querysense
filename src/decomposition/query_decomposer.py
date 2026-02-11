"""
Query Decomposition Module

This module handles multi-step query decomposition for complex questions:

1. **Complexity Detection** — Classifies questions as SIMPLE or COMPLEX
   using heuristic signals (keywords, clauses, length) plus an optional
   LLM-based classifier for ambiguous cases.

2. **Question Decomposition** — Breaks complex questions into ordered
   sub-questions that can each be answered with a simpler SQL query.

3. **Chain-of-Thought Reasoning** — Generates an intermediate reasoning
   plan (table selection rationale, JOIN strategy, aggregation approach)
   that is injected into the SQL generation prompt for better results.

Architecture:
    question → classify_complexity()
        → SIMPLE: pass through to generate_sql() with CoT hint
        → COMPLEX: decompose() → build CoT plan → generate_sql() with plan

Usage:
    from src.decomposition.query_decomposer import QueryDecomposer

    decomposer = QueryDecomposer()
    result = decomposer.process(question, schema_items, dialect="sqlite")
    # result["sql"] contains the final SQL
    # result["reasoning"] contains the CoT plan
    # result["complexity"] is "simple" or "complex"
    # result["sub_questions"] is [] for simple, [sub_q, ...] for complex
"""

import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Reuse the same Azure OpenAI client as the generator
_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


# ═══════════════════════════════════════════════════════════════════════
# 1. COMPLEXITY DETECTION
# ═══════════════════════════════════════════════════════════════════════

# Heuristic signals that indicate a complex query
_COMPLEXITY_SIGNALS = {
    # Multi-part questions
    "conjunctions": [
        " and also ", " as well as ", " along with ", " together with ",
        " in addition to ", " besides ",
    ],
    # Comparison / ranking across groups
    "comparison": [
        "compare", "difference between", "more than", "less than",
        "higher than", "lower than", "versus", " vs ",
        "relative to", "compared to", "ratio of",
    ],
    # Nested aggregation
    "nested_agg": [
        "average of the total", "sum of the average", "maximum of the count",
        "minimum of the sum", "count of the distinct",
        "for each", "per each", "among those", "out of those",
        "of those who", "of the ones",
    ],
    # Conditional / temporal complexity
    "conditional": [
        "but not", "except", "excluding", "only if", "unless",
        "however", "on the other hand",
    ],
    # Subquery indicators
    "subquery": [
        "who also", "that have", "that are", "that were",
        "where the", "among", "within the",
        "having more than", "having less than", "having at least",
    ],
    # Multi-step reasoning
    "multi_step": [
        "then find", "then show", "then list", "then calculate",
        "after that", "based on that", "use that to", "first find",
        "and then", "followed by",
    ],
}

# Flatten for quick membership tests
_ALL_COMPLEX_PHRASES = []
for _phrases in _COMPLEXITY_SIGNALS.values():
    _ALL_COMPLEX_PHRASES.extend(_phrases)


def classify_complexity(question: str) -> dict:
    """
    Classify a question's complexity using heuristic signals.

    Returns:
        {
            "complexity": "simple" | "complex",
            "score": float (0-1),
            "signals": list of matched signal categories,
            "matched_phrases": list of matched phrases
        }
    """
    q_lower = question.lower()
    word_count = len(question.split())

    signals_hit = set()
    matched_phrases = []

    for category, phrases in _COMPLEXITY_SIGNALS.items():
        for phrase in phrases:
            if phrase in q_lower:
                signals_hit.add(category)
                matched_phrases.append(phrase)

    # Scoring heuristic
    score = 0.0

    # Signal-based score
    score += len(signals_hit) * 0.2
    score += len(matched_phrases) * 0.05

    # Length-based score (longer questions tend to be more complex)
    if word_count > 20:
        score += 0.1
    if word_count > 35:
        score += 0.15

    # Multiple question marks
    if question.count("?") > 1:
        score += 0.2

    # Contains "and" connecting two verb phrases (rough heuristic)
    and_count = q_lower.count(" and ")
    if and_count >= 2:
        score += 0.15

    # Multiple aggregation keywords
    agg_keywords = ["count", "sum", "average", "avg", "max", "min", "total", "number of"]
    agg_count = sum(1 for kw in agg_keywords if kw in q_lower)
    if agg_count >= 2:
        score += 0.15

    # Multiple clauses needing different logic (list, show + condition)
    action_verbs = ["find", "show", "list", "get", "count", "calculate", "display"]
    action_count = sum(1 for v in action_verbs if v in q_lower)
    if action_count >= 2 and len(signals_hit) >= 1:
        score += 0.1

    # "difference between" or "but not" with decent length are usually complex
    if ("difference between" in q_lower or "but not" in q_lower) and word_count > 12:
        score += 0.1

    # Cap at 1.0
    score = min(score, 1.0)

    # Threshold
    complexity = "complex" if score >= 0.3 else "simple"

    return {
        "complexity": complexity,
        "score": round(score, 3),
        "signals": list(signals_hit),
        "matched_phrases": matched_phrases,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. QUESTION DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════

_DECOMPOSE_PROMPT = """You are an expert at breaking down complex database questions into simpler sub-questions.

Given a complex question about a database, decompose it into 2-4 simpler sub-questions that, when answered in order, would answer the original question. Each sub-question should be answerable with a single simple SQL query.

Question: {question}

Available tables and columns:
{schema_context}

Rules:
1. Output ONLY the sub-questions, numbered 1-4
2. Each sub-question should be self-contained and reference specific tables/columns
3. Order them logically (earlier sub-questions provide context for later ones)
4. Use simple language, as if asking a human analyst
5. Do NOT output any SQL

Sub-questions:"""


def _call_llm(system_msg: str, user_msg: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Helper to call Azure OpenAI."""
    response = _client.chat.completions.create(
        model=_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    return response.choices[0].message.content.strip()


def decompose_question(question: str, schema_context: str) -> List[str]:
    """
    Break a complex question into simpler sub-questions using LLM.

    Args:
        question: The original complex question.
        schema_context: Formatted schema string for context.

    Returns:
        List of 2-4 sub-questions in logical order.
    """
    prompt = _DECOMPOSE_PROMPT.format(
        question=question,
        schema_context=schema_context,
    )

    raw = _call_llm(
        system_msg="You decompose complex database questions into simpler sub-questions.",
        user_msg=prompt,
        temperature=0.0,
    )

    # Parse numbered sub-questions
    sub_questions = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Match lines like "1. ...", "1) ...", "- ..."
        match = re.match(r"^(?:\d+[\.\)]\s*|-\s*)(.*)", line)
        if match:
            sq = match.group(1).strip()
            if sq:
                sub_questions.append(sq)

    # Fallback: if parsing failed, return the original question
    if not sub_questions:
        sub_questions = [question]

    return sub_questions


# ═══════════════════════════════════════════════════════════════════════
# 3. CHAIN-OF-THOUGHT REASONING
# ═══════════════════════════════════════════════════════════════════════

_COT_SIMPLE_PROMPT = """Analyze this database question and provide a brief reasoning plan.

Question: {question}

Schema:
{schema_context}

Provide a concise 2-3 line reasoning plan covering:
1. Which table(s) to use and why
2. What columns to SELECT, any JOINs needed, and any WHERE/GROUP BY/ORDER BY logic

Reasoning:"""

_COT_COMPLEX_PROMPT = """Analyze this complex database question and provide a detailed reasoning plan.

Original question: {question}

This has been decomposed into these sub-questions:
{sub_questions_text}

Schema:
{schema_context}

Provide a detailed reasoning plan covering:
1. Which tables are needed and how they connect (JOINs)
2. For each sub-question, what SQL pattern is needed
3. How to combine the sub-queries into one final SQL (subqueries, CTEs, or JOINs)
4. Any aggregations, filters, or ordering required

Reasoning:"""


def generate_reasoning(
    question: str,
    schema_context: str,
    complexity: str,
    sub_questions: Optional[List[str]] = None,
) -> str:
    """
    Generate a chain-of-thought reasoning plan.

    For simple queries: brief 2-3 line plan.
    For complex queries: detailed plan incorporating sub-questions.
    """
    if complexity == "complex" and sub_questions and len(sub_questions) > 1:
        sub_q_text = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_questions))
        prompt = _COT_COMPLEX_PROMPT.format(
            question=question,
            sub_questions_text=sub_q_text,
            schema_context=schema_context,
        )
        system_msg = "You are an expert SQL analyst who creates detailed query plans."
    else:
        prompt = _COT_SIMPLE_PROMPT.format(
            question=question,
            schema_context=schema_context,
        )
        system_msg = "You are an expert SQL analyst who creates concise query plans."

    reasoning = _call_llm(
        system_msg=system_msg,
        user_msg=prompt,
        temperature=0.0,
        max_tokens=400,
    )

    return reasoning


# ═══════════════════════════════════════════════════════════════════════
# 4. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class QueryDecomposer:
    """
    Orchestrates query decomposition and chain-of-thought reasoning.

    Usage:
        decomposer = QueryDecomposer()
        result = decomposer.process(question, schema_items, dialect="sqlite")
    """

    def __init__(self, cot_enabled: bool = True, decompose_enabled: bool = True):
        """
        Args:
            cot_enabled: Whether to generate chain-of-thought reasoning.
            decompose_enabled: Whether to decompose complex queries.
                If False, all queries are treated as simple.
        """
        self.cot_enabled = cot_enabled
        self.decompose_enabled = decompose_enabled

    def process(
        self,
        question: str,
        schema_items: List[Dict],
        dialect: str = "sqlite",
    ) -> Dict:
        """
        Process a question through the decomposition pipeline.

        Args:
            question: Natural language question.
            schema_items: List of schema column dicts from retriever.
            dialect: "sqlite" or "bigquery".

        Returns:
            {
                "complexity": "simple" | "complex",
                "complexity_score": float,
                "complexity_signals": list,
                "sub_questions": list of str (empty for simple),
                "reasoning": str (CoT plan),
                "enhanced_question": str (question + reasoning for SQL gen),
            }
        """
        # Build schema context string
        schema_context = "\n".join(
            f"{s['table']}.{s['column']} ({s['type']})" for s in schema_items
        )

        # Step 1: Classify complexity
        cx = classify_complexity(question)
        complexity = cx["complexity"]
        sub_questions = []

        # Step 2: Decompose if complex
        if complexity == "complex" and self.decompose_enabled:
            sub_questions = decompose_question(question, schema_context)

        # Step 3: Chain-of-thought reasoning
        reasoning = ""
        if self.cot_enabled:
            reasoning = generate_reasoning(
                question=question,
                schema_context=schema_context,
                complexity=complexity,
                sub_questions=sub_questions if sub_questions else None,
            )

        # Step 4: Build enhanced question for SQL generation
        enhanced_question = self._build_enhanced_question(
            question, reasoning, sub_questions
        )

        return {
            "complexity": complexity,
            "complexity_score": cx["score"],
            "complexity_signals": cx["signals"],
            "sub_questions": sub_questions,
            "reasoning": reasoning,
            "enhanced_question": enhanced_question,
        }

    def _build_enhanced_question(
        self,
        question: str,
        reasoning: str,
        sub_questions: List[str],
    ) -> str:
        """
        Build the enhanced question string that will be passed to generate_sql().

        This injects the CoT reasoning and decomposition into the question
        so the SQL generator has richer context.
        """
        parts = [question]

        if sub_questions and len(sub_questions) > 1:
            parts.append("\n\nThis question can be broken down into:")
            for i, sq in enumerate(sub_questions, 1):
                parts.append(f"  {i}. {sq}")

        if reasoning:
            parts.append(f"\n\nReasoning plan:\n{reasoning}")

        parts.append(
            "\n\nNow write a SINGLE SQL query that answers the original question."
        )

        return "\n".join(parts)
