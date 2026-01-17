"""
Tests for lossless reduction utilities.

These tests verify:
1. Minification preserves data
2. Columnar transformation is lossless
3. Deduplication is lossless
4. Pipeline works correctly

Run: pytest tests/test_reduce.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from reduce_lib import (
    minify_json,
    dedup_payloads,
    restore_dedup,
    columnar_transform,
    restore_columnar,
    reduce_payload,
    restore_payload,
    estimate_tokens,
)


# =============================================================================
# Minification Tests
# =============================================================================

class TestMinifyJson:
    """Tests for minify_json function."""

    def test_minify_removes_whitespace(self):
        """Minification should remove whitespace."""
        obj = {"name": "test", "value": 123}
        result, report = minify_json(obj)

        assert " " not in result
        assert "\n" not in result
        assert report.minified_chars < report.original_chars

    def test_minify_sorts_keys(self):
        """Minification should sort keys by default."""
        obj = {"z": 1, "a": 2, "m": 3}
        result, _ = minify_json(obj, sort_keys=True)

        # Keys should appear in sorted order
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_minify_removes_nulls(self):
        """Minification should remove null fields."""
        obj = {"name": "test", "value": None, "count": 0}
        result, report = minify_json(obj, remove_nulls=True)

        parsed = json.loads(result)
        assert "value" not in parsed
        assert "name" in parsed
        assert "count" in parsed  # 0 is not null
        assert report.null_fields_removed == 1

    def test_minify_removes_empty_strings(self):
        """Minification should remove empty string fields."""
        obj = {"name": "test", "description": "", "value": "x"}
        result, report = minify_json(obj, remove_empty_strings=True)

        parsed = json.loads(result)
        assert "description" not in parsed
        assert "name" in parsed
        assert report.empty_strings_removed == 1

    def test_minify_removes_empty_arrays(self):
        """Minification should remove empty array fields."""
        obj = {"name": "test", "items": [], "tags": ["a", "b"]}
        result, report = minify_json(obj, remove_empty_arrays=True)

        parsed = json.loads(result)
        assert "items" not in parsed
        assert "tags" in parsed
        assert report.empty_arrays_removed == 1

    def test_minify_preserves_data(self):
        """Minification should preserve all non-null/empty data."""
        obj = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "nested": {"a": 1, "b": 2},
            "array": [1, 2, 3],
        }
        result, _ = minify_json(obj, remove_nulls=False)

        parsed = json.loads(result)
        assert parsed == obj

    def test_minify_nested_nulls(self):
        """Minification should remove nulls in nested structures."""
        obj = {
            "outer": {
                "keep": 1,
                "remove": None,
                "nested": {
                    "also_keep": 2,
                    "also_remove": None,
                }
            }
        }
        result, report = minify_json(obj)

        parsed = json.loads(result)
        assert parsed["outer"]["keep"] == 1
        assert "remove" not in parsed["outer"]
        assert parsed["outer"]["nested"]["also_keep"] == 2
        assert "also_remove" not in parsed["outer"]["nested"]
        assert report.null_fields_removed == 2


# =============================================================================
# Columnar Transformation Tests
# =============================================================================

class TestColumnarTransform:
    """Tests for columnar_transform function."""

    def test_columnar_transform_basic(self):
        """Basic columnar transformation with enough items to benefit."""
        # Columnar transform only applies when it reduces size
        # Need enough items with longer keys to overcome __cols__/__rows__ overhead
        obj = {
            "items": [
                {"user_id": 1, "username": "alice", "email_address": "alice@test.com"},
                {"user_id": 2, "username": "bob", "email_address": "bob@test.com"},
                {"user_id": 3, "username": "charlie", "email_address": "charlie@test.com"},
                {"user_id": 4, "username": "diana", "email_address": "diana@test.com"},
                {"user_id": 5, "username": "eve", "email_address": "eve@test.com"},
            ]
        }
        result, report = columnar_transform(obj)

        assert "__cols__" in result["items"]
        assert "__rows__" in result["items"]
        assert set(result["items"]["__cols__"]) == {"user_id", "username", "email_address"}
        assert len(result["items"]["__rows__"]) == 5
        assert report.arrays_transformed == 1

    def test_columnar_skips_small_arrays(self):
        """Small arrays should not be transformed due to min_items."""
        obj = {
            "items": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }
        result, report = columnar_transform(obj, min_items=3)

        # Should remain as original array
        assert isinstance(result["items"], list)
        assert "__cols__" not in result["items"][0]
        assert report.arrays_transformed == 0

    def test_columnar_skips_when_not_beneficial(self):
        """Small arrays with short keys should not be transformed (would increase size)."""
        obj = {
            "items": [
                {"id": 1, "n": "A"},
                {"id": 2, "n": "B"},
                {"id": 3, "n": "C"},
            ]
        }
        result, report = columnar_transform(obj)

        # Should remain as original array because transformation would increase size
        assert isinstance(result["items"], list)
        # Verify it was processed but not transformed
        assert report.arrays_processed >= 1
        assert report.arrays_transformed == 0

    def test_columnar_skips_heterogeneous(self):
        """Heterogeneous arrays should not be transformed."""
        obj = [
            {"id": 1, "name": "Alice"},
            {"x": 2, "y": "Bob"},  # Different keys
            {"id": 3, "name": "Charlie"},
        ]
        result, report = columnar_transform(obj, homogeneity_threshold=0.9)

        # Should remain as original array
        assert isinstance(result, list)
        assert report.arrays_transformed == 0

    def test_columnar_roundtrip_lossless(self):
        """Columnar transformation should be completely reversible."""
        obj = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]
        transformed, _ = columnar_transform(obj)
        restored = restore_columnar(transformed)

        # Sort for comparison (columnar sorts columns)
        assert len(restored) == len(obj)
        for orig, rest in zip(obj, restored):
            assert orig == rest

    def test_columnar_nested_arrays(self):
        """Columnar should work on nested arrays when beneficial."""
        # Use longer keys and more items so columnar is beneficial
        obj = {
            "items": [
                {"user_id": i, "username": f"user{i}", "email_address": f"user{i}@test.com"}
                for i in range(10)
            ],
            "other": "data",
        }
        result, report = columnar_transform(obj)

        assert "__cols__" in result["items"]
        assert result["other"] == "data"
        assert report.arrays_transformed == 1

    def test_columnar_handles_missing_keys(self):
        """Columnar should handle records with some missing keys."""
        obj = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob"},  # missing age
            {"id": 3, "name": "Charlie", "age": 35},
        ]
        result, report = columnar_transform(obj, homogeneity_threshold=0.7)

        if report.arrays_transformed > 0:
            # If transformed, missing values should be None
            restored = restore_columnar(result)
            assert restored[1].get("age") is None


# =============================================================================
# Deduplication Tests
# =============================================================================

class TestDedup:
    """Tests for dedup_payloads function."""

    def test_dedup_removes_duplicates(self):
        """Deduplication should replace repeated blocks with references."""
        block = '{"name":"repeated","value":12345678901234567890}'
        text = f"{block} some text {block} more text {block}"

        result = dedup_payloads(text, min_block_size=10, min_occurrences=2)

        # Should have fewer characters
        assert len(result.text) < len(text)
        assert len(result.report.matches) > 0

    def test_dedup_roundtrip_lossless(self):
        """Deduplication should be completely reversible."""
        block = '{"name":"repeated","value":12345678901234567890}'
        original = f"{block} some text {block} more text {block}"

        result = dedup_payloads(original, min_block_size=10, min_occurrences=2)
        restored = restore_dedup(result.text, result.registry)

        assert restored == original

    def test_dedup_no_matches(self):
        """Deduplication with no repeated blocks."""
        text = '{"a":1} {"b":2} {"c":3}'
        result = dedup_payloads(text, min_block_size=5, min_occurrences=2)

        # No changes
        assert result.text == text
        assert len(result.report.matches) == 0


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipeline:
    """Tests for the full reduction pipeline."""

    def test_pipeline_all_reductions(self):
        """Pipeline should apply all reductions."""
        obj = {
            "users": [
                {"id": 1, "name": "Alice", "deleted": None},
                {"id": 2, "name": "Bob", "deleted": None},
                {"id": 3, "name": "Charlie", "deleted": None},
            ],
            "empty_field": "",
            "null_field": None,
        }

        result, report = reduce_payload(obj, minify=True, columnar=True, dedup=False)

        # Should have fewer tokens
        assert report.reduced_tokens < report.original_tokens

        # Report should have minify and columnar info
        assert report.minify is not None
        assert report.columnar is not None

        # Verify we can parse the result
        parsed = json.loads(result)
        assert "users" in parsed
        assert "__cols__" in parsed["users"]

    def test_pipeline_restore_roundtrip(self):
        """Pipeline should be fully reversible."""
        obj = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Charlie"},
            ],
        }

        result, _ = reduce_payload(obj, minify=True, columnar=True, dedup=False, remove_nulls=False)
        restored = restore_payload(result)

        # Users should be restored
        assert "users" in restored
        assert len(restored["users"]) == 3
        assert restored["users"][0]["name"] == "Alice"

    def test_pipeline_minify_only(self):
        """Pipeline with only minification."""
        obj = {"a": 1, "b": None, "c": ""}
        result, report = reduce_payload(obj, minify=True, columnar=False, dedup=False)

        parsed = json.loads(result)
        assert "a" in parsed
        assert "b" not in parsed  # null removed
        assert "c" not in parsed  # empty string removed
        assert report.columnar is None or report.columnar.arrays_transformed == 0

    def test_pipeline_columnar_only(self):
        """Pipeline with only columnar transformation (when beneficial)."""
        # Need enough items with long keys for columnar to be beneficial
        obj = {
            "data": [
                {"field_one": i, "field_two": f"value_{i}", "field_three": f"extra_{i}"}
                for i in range(10)
            ]
        }
        result, report = reduce_payload(obj, minify=False, columnar=True, dedup=False)

        parsed = json.loads(result)
        assert "__cols__" in parsed["data"]
        assert report.columnar.arrays_transformed == 1


# =============================================================================
# Token Estimation Tests
# =============================================================================

class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_basic(self):
        """Basic token estimation."""
        text = "a" * 100
        tokens = estimate_tokens(text, chars_per_token=4.0)
        assert tokens == 25

    def test_estimate_tokens_empty(self):
        """Empty string should return 0."""
        assert estimate_tokens("") == 0
        assert estimate_tokens("", chars_per_token=4.0) == 0

    def test_estimate_tokens_minimum(self):
        """Non-empty should return at least 1."""
        assert estimate_tokens("a") >= 1


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_object(self):
        """Empty object handling."""
        obj = {}
        result, report = reduce_payload(obj)
        assert json.loads(result) == {}

    def test_empty_array(self):
        """Empty array handling."""
        obj = []
        result, _ = reduce_payload(obj)
        assert json.loads(result) == []

    def test_deeply_nested(self):
        """Deeply nested structure."""
        obj = {"a": {"b": {"c": {"d": {"e": [1, 2, 3]}}}}}
        result, _ = reduce_payload(obj)
        restored = restore_payload(result)
        assert restored["a"]["b"]["c"]["d"]["e"] == [1, 2, 3]

    def test_special_characters(self):
        """Unicode and special characters."""
        obj = {"emoji": "Hello World!", "unicode": "caf\u00e9", "quotes": 'He said "hi"'}
        result, _ = reduce_payload(obj)
        restored = restore_payload(result)
        assert restored["emoji"] == obj["emoji"]
        assert restored["unicode"] == obj["unicode"]
        assert restored["quotes"] == obj["quotes"]


# =============================================================================
# Run Standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
