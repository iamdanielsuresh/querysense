"""Tests for the error-class retry system."""
import pytest
from src.generator.retry import classify_error, build_retry_prompt, SQLErrorClass


# ── classify_error ──────────────────────────────────────────────────

class TestClassifyError:
    def test_empty_input(self):
        assert classify_error("") == SQLErrorClass.EMPTY_QUERY
        assert classify_error("   ") == SQLErrorClass.EMPTY_QUERY

    def test_no_such_table(self):
        assert classify_error("no such table: orders") == SQLErrorClass.NO_SUCH_TABLE
        assert classify_error(
            "Not found: Table bigproject.dataset.orders was not found"
        ) == SQLErrorClass.NO_SUCH_TABLE

    def test_no_such_column(self):
        assert classify_error("no such column: price") == SQLErrorClass.NO_SUCH_COLUMN
        assert classify_error(
            "Unrecognized name: total_amount"
        ) == SQLErrorClass.NO_SUCH_COLUMN

    def test_ambiguous_column(self):
        assert classify_error(
            "Column reference 'id' is ambiguous"
        ) == SQLErrorClass.AMBIGUOUS_COLUMN

    def test_type_mismatch(self):
        assert classify_error(
            "No matching signature for function TIMESTAMP_ADD"
        ) == SQLErrorClass.TYPE_MISMATCH
        assert classify_error(
            "Cannot cast STRING to INT64"
        ) == SQLErrorClass.TYPE_MISMATCH

    def test_group_by(self):
        assert classify_error(
            "SELECT list expression references name which is neither grouped "
            "nor aggregated at [1:8] — must appear in the GROUP BY clause"
        ) == SQLErrorClass.GROUP_BY_ERROR

    def test_function_error(self):
        assert classify_error(
            "TIMESTAMP_SUB does not support MONTH intervals"
        ) == SQLErrorClass.FUNCTION_ERROR

    def test_syntax_error(self):
        assert classify_error(
            "near \"FORM\": syntax error"
        ) == SQLErrorClass.SYNTAX_ERROR

    def test_timeout(self):
        assert classify_error(
            "Query timed out after 30s"
        ) == SQLErrorClass.TIMEOUT
        assert classify_error(
            "Resources exceeded during query execution"
        ) == SQLErrorClass.TIMEOUT

    def test_permission(self):
        assert classify_error(
            "Access Denied: BigQuery: Permission denied"
        ) == SQLErrorClass.PERMISSION_ERROR

    def test_unknown(self):
        assert classify_error(
            "Something completely unexpected happened xyz123"
        ) == SQLErrorClass.UNKNOWN


# ── build_retry_prompt ──────────────────────────────────────────────

class TestBuildRetryPrompt:
    def test_contains_original_question(self):
        prompt = build_retry_prompt(
            "What is total revenue?",
            "no such column: revenue",
            SQLErrorClass.NO_SUCH_COLUMN,
        )
        assert "What is total revenue?" in prompt

    def test_contains_error_class(self):
        prompt = build_retry_prompt(
            "Count orders", "syntax error", SQLErrorClass.SYNTAX_ERROR,
        )
        assert "SYNTAX_ERROR" in prompt

    def test_contains_guidance(self):
        prompt = build_retry_prompt(
            "Get products", "no such table: items",
            SQLErrorClass.NO_SUCH_TABLE,
        )
        assert "Use ONLY the tables" in prompt

    def test_contains_failed_sql(self):
        prompt = build_retry_prompt(
            "Count orders",
            "syntax error near FROM",
            SQLErrorClass.SYNTAX_ERROR,
            failed_sql="SELECT COUNT(*) FORM orders",
        )
        assert "SELECT COUNT(*) FORM orders" in prompt

    def test_no_domain_specific_hints(self):
        """Ensure no hardcoded column names leak into retry guidance."""
        for ec in SQLErrorClass:
            prompt = build_retry_prompt("q", "err", ec)
            assert "product_id" not in prompt
            assert "category_id" not in prompt
            assert "bigcommerce" not in prompt.lower()


# ── Prompt registry validation ──────────────────────────────────────

class TestPromptRegistry:
    def test_active_versions_are_registered(self):
        from src.generator.prompt_registry import validate_active_versions
        # Should not raise
        validate_active_versions()

    def test_all_prompts_have_changelog(self):
        from src.generator.prompt_registry import REGISTRY
        for dialect, versions in REGISTRY.items():
            for ver_id, pv in versions.items():
                assert pv.changelog, f"{dialect}/{ver_id} has empty changelog"

    def test_v3_bigquery_parent_is_v2(self):
        from src.generator.prompt_registry import get_registered_version
        v3 = get_registered_version("bigquery", "V3")
        assert v3 is not None
        assert v3.parent_version == "V2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
