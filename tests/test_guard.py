"""
Tests for /guard command pipeline.

These tests verify:
1. Decision engine logic
2. Guard command pipeline
3. Integration with reduction and trimming

Run: pytest tests/test_guard.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from decision import make_decision, Decision, DecisionResult
from trim_lib import TrimMode
from guard_cmd import run_guard, detect_forensic_patterns, detect_semantic_mode


# =============================================================================
# Decision Engine Tests
# =============================================================================

class TestDecisionEngine:
    """Tests for the decision engine."""

    def test_allow_under_budget(self):
        """Tokens under budget should ALLOW."""
        result = make_decision(
            tokens=1000,
            budget=3500,
            mode=TrimMode.ANALYSIS,
        )
        assert result.decision == Decision.ALLOW
        assert not result.over_budget

    def test_sample_over_budget_analysis(self):
        """Analysis mode over budget without forensic should SAMPLE."""
        result = make_decision(
            tokens=5000,
            budget=3500,
            mode=TrimMode.ANALYSIS,
            is_forensic=False,
        )
        assert result.decision == Decision.SAMPLE
        assert result.over_budget

    def test_block_forensic_over_budget(self):
        """Forensic detection over budget should BLOCK."""
        result = make_decision(
            tokens=5000,
            budget=3500,
            mode=TrimMode.ANALYSIS,
            is_forensic=True,
            forensic_patterns=["request id=abc123"],
        )
        assert result.decision == Decision.BLOCK
        assert "forensic" in result.reason.lower()

    def test_force_bypasses_block(self):
        """Force mode should bypass all blocks."""
        result = make_decision(
            tokens=5000,
            budget=3500,
            mode=TrimMode.ANALYSIS,
            is_forensic=True,
            force=True,
        )
        assert result.decision == Decision.ALLOW
        assert "force" in result.reason.lower()

    def test_allow_sampling_permits_forensic(self):
        """Allow sampling should permit forensic queries."""
        result = make_decision(
            tokens=5000,
            budget=3500,
            mode=TrimMode.ANALYSIS,
            is_forensic=True,
            allow_sampling=True,
        )
        assert result.decision == Decision.SAMPLE

    def test_forensics_mode_blocks(self):
        """Forensics mode should always block if over budget."""
        result = make_decision(
            tokens=5000,
            budget=3500,
            mode=TrimMode.FORENSICS,
            is_forensic=False,  # Even without detection
        )
        assert result.decision == Decision.BLOCK
        assert "forensics mode" in result.reason.lower()

    def test_summary_mode_samples(self):
        """Summary mode should always allow sampling."""
        result = make_decision(
            tokens=5000,
            budget=3500,
            mode=TrimMode.SUMMARY,
            is_forensic=True,  # Even with forensic
        )
        assert result.decision == Decision.SAMPLE


# =============================================================================
# Forensic Pattern Detection Tests
# =============================================================================

class TestForensicDetection:
    """Tests for forensic pattern detection."""

    def test_detect_request_id(self):
        """Detect request id pattern."""
        is_forensic, hits = detect_forensic_patterns("Why did request id=abc123 fail?")
        assert is_forensic
        assert any("request id" in h.lower() for h in hits)

    def test_detect_uuid(self):
        """Detect UUID pattern."""
        is_forensic, hits = detect_forensic_patterns(
            "Check 550e8400-e29b-41d4-a716-446655440000"
        )
        assert is_forensic
        assert any("550e8400" in h for h in hits)

    def test_no_forensic_for_global(self):
        """Global questions should not trigger forensic."""
        is_forensic, _ = detect_forensic_patterns("What categories exist?")
        assert not is_forensic


# =============================================================================
# Semantic Mode Detection Tests
# =============================================================================

class TestSemanticModeDetection:
    """Tests for semantic mode detection."""

    def test_default_is_analysis(self):
        """Default mode should be analysis."""
        mode = detect_semantic_mode("Some prompt")
        assert mode == TrimMode.ANALYSIS

    def test_detect_summary_marker(self):
        """Detect summary mode marker."""
        mode = detect_semantic_mode("#trimmer:mode=summary\nDescribe the data")
        assert mode == TrimMode.SUMMARY

    def test_detect_forensics_marker(self):
        """Detect forensics mode marker."""
        mode = detect_semantic_mode("#trimmer:mode=forensics\nFind record X")
        assert mode == TrimMode.FORENSICS


# =============================================================================
# Guard Command Pipeline Tests
# =============================================================================

class TestGuardPipeline:
    """Tests for the full guard pipeline."""

    @pytest.fixture
    def small_data(self):
        """Small dataset under budget."""
        return [{"id": i, "name": f"Item {i}"} for i in range(5)]

    @pytest.fixture
    def large_data(self):
        """Large dataset over budget."""
        return [
            {"id": i, "name": f"Item {i}", "description": f"Description for item {i}" * 10}
            for i in range(200)
        ]

    def test_guard_allow_under_budget(self, small_data):
        """Small data should be allowed."""
        input_text = json.dumps(small_data)
        result = run_guard(input_text, budget_tokens=5000)

        assert result.decision.decision == Decision.ALLOW
        assert result.output_json is not None

    def test_guard_sample_over_budget_analysis(self, large_data):
        """Large data in analysis mode should sample."""
        input_text = json.dumps(large_data)
        result = run_guard(
            input_text,
            mode=TrimMode.ANALYSIS,
            budget_tokens=500,
        )

        assert result.decision.decision == Decision.SAMPLE
        assert result.trim_report is not None
        assert result.trimmed_tokens < result.original_tokens

    def test_guard_block_forensic_over_budget(self, large_data):
        """Forensic query over budget should block."""
        input_text = f"Why did request id=abc123 fail?\n{json.dumps(large_data)}"
        result = run_guard(input_text, budget_tokens=500)

        assert result.decision.decision == Decision.BLOCK
        assert len(result.forensic_patterns) > 0

    def test_guard_force_bypasses_block(self, large_data):
        """Force should bypass block."""
        input_text = f"Why did request id=abc123 fail?\n{json.dumps(large_data)}"
        result = run_guard(input_text, budget_tokens=500, force=True)

        assert result.decision.decision == Decision.ALLOW

    def test_guard_allow_sampling_permits_forensic(self, large_data):
        """Allow sampling should permit forensic."""
        input_text = f"Why did request id=abc123 fail?\n{json.dumps(large_data)}"
        result = run_guard(input_text, budget_tokens=500, allow_sampling=True)

        assert result.decision.decision == Decision.SAMPLE

    def test_guard_reduction_before_sampling(self, large_data):
        """Reduction should be applied before sampling decision."""
        input_text = json.dumps(large_data)
        result = run_guard(input_text, budget_tokens=1000, no_reduce=False)

        # Reduction should have been applied
        assert result.reduction_report is not None
        assert result.reduced_tokens <= result.original_tokens

    def test_guard_no_reduce_skips_reduction(self, large_data):
        """No-reduce flag should skip reduction."""
        input_text = json.dumps(large_data)
        result = run_guard(input_text, budget_tokens=1000, no_reduce=True)

        # No reduction should be applied
        assert result.reduction_report is None
        assert result.reduced_tokens == result.original_tokens

    def test_guard_output_format_includes_json(self, small_data):
        """Output should include the processed JSON."""
        input_text = json.dumps(small_data)
        result = run_guard(input_text, budget_tokens=5000)

        assert result.output_json is not None
        # Should be parseable JSON
        parsed = json.loads(result.output_json)
        # After columnar transformation, array becomes dict with __cols__/__rows__
        # or remains a list if not enough items
        assert isinstance(parsed, (list, dict))


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        """Empty input should be handled."""
        result = run_guard("", budget_tokens=3500)
        # Should allow (nothing to block)
        assert result.decision.decision == Decision.ALLOW

    def test_no_json_in_input(self):
        """Text without JSON should be handled."""
        result = run_guard("This is just text without any JSON.", budget_tokens=3500)
        assert result.decision.decision == Decision.ALLOW
        assert result.output_json is None

    def test_malformed_json(self):
        """Malformed JSON should fall back gracefully."""
        result = run_guard('{"broken": json}', budget_tokens=3500)
        # json_repair might fix it or return None - either is acceptable
        assert result.decision.decision in [Decision.ALLOW, Decision.SAMPLE, Decision.BLOCK]

    def test_very_small_budget(self):
        """Very small budget should work."""
        data = [{"id": 1}]
        result = run_guard(json.dumps(data), budget_tokens=10)
        # Either sample or block is acceptable for tiny budget
        assert result.decision.decision in [Decision.SAMPLE, Decision.BLOCK, Decision.ALLOW]


# =============================================================================
# Run Standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
