"""
Tests for semantic mode detection and forensic heuristic.

These tests verify:
1. TrimMode enum and marker detection
2. Forensic question heuristic patterns
3. Fail-safe blocking behavior
4. Explicit marker unlocking

Run: pytest tests/test_semantic_modes.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


# =============================================================================
# UNIT TESTS: Mode Detection
# =============================================================================

class TestModeDetection:
    """Tests for detect_semantic_mode() function."""

    def test_default_is_analysis(self):
        """Default mode should be ANALYSIS."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        assert detect_semantic_mode("Some prompt without marker") == TrimMode.ANALYSIS
        assert detect_semantic_mode("What categories exist?") == TrimMode.ANALYSIS

    def test_explicit_analysis_marker(self):
        """Explicit #trimmer:mode=analysis marker."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        prompt = "#trimmer:mode=analysis\nWhat are the categories?"
        assert detect_semantic_mode(prompt) == TrimMode.ANALYSIS

    def test_summary_marker(self):
        """#trimmer:mode=summary marker."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        prompt = "#trimmer:mode=summary\nDescribe the data structure"
        assert detect_semantic_mode(prompt) == TrimMode.SUMMARY

    def test_forensics_marker(self):
        """#trimmer:mode=forensics marker."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        prompt = "#trimmer:mode=forensics\nWhy did request id=abc123 fail?"
        assert detect_semantic_mode(prompt) == TrimMode.FORENSICS


# =============================================================================
# UNIT TESTS: Forensic Question Detection
# =============================================================================

class TestForensicDetection:
    """Tests for detect_forensic_tripwire() heuristic."""

    def test_request_id_pattern(self):
        """Detect 'request id=X' pattern."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("Why did request id=abc123 fail?")
        assert is_forensic
        assert any("request id=abc123" in hit.lower() for hit in hits)

    def test_user_id_pattern(self):
        """Detect 'user id=X' pattern."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("Show details for user id: xyz789")
        assert is_forensic
        assert any("user id" in hit.lower() for hit in hits)

    def test_why_did_fail_pattern(self):
        """Detect 'why did X fail' pattern."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("Why did the payment fail?")
        assert is_forensic
        assert any("why did" in hit.lower() for hit in hits)

    def test_what_happened_to_pattern(self):
        """Detect 'what happened to X' pattern."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("What happened to order-12345?")
        assert is_forensic
        assert any("what happened to" in hit.lower() for hit in hits)

    def test_no_forensic_for_global_questions(self):
        """Global questions should NOT trigger forensic detection."""
        from trimmer_hook import detect_forensic_tripwire

        global_questions = [
            "What categories exist in this data?",
            "How many records are there?",
            "What is the price range?",
            "Show me the data structure",
            "List all unique values",
        ]

        for question in global_questions:
            is_forensic, _ = detect_forensic_tripwire(question)
            assert not is_forensic, f"Should not be forensic: {question}"

    def test_id_with_long_value(self):
        """Detect ID patterns with 6+ character values."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("Find id=abcdef123")
        assert is_forensic

        # Short IDs should not match the strict id= pattern
        is_forensic_short, _ = detect_forensic_tripwire("Find id=abc")
        # Note: may still match other patterns like "find X with id"

    def test_uuid_pattern(self):
        """Detect UUID patterns (almost always forensic)."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire(
            "Check request 550e8400-e29b-41d4-a716-446655440000"
        )
        assert is_forensic
        assert any("550e8400-e29b-41d4-a716-446655440000" in hit for hit in hits)

    def test_what_went_wrong_pattern(self):
        """Detect 'what went wrong' pattern."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("What went wrong with the deployment?")
        assert is_forensic
        assert any("what went wrong" in hit.lower() for hit in hits)

    def test_this_that_record_pattern(self):
        """Detect 'this/that request/order' patterns."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("Why did this request timeout?")
        assert is_forensic
        assert any("this request" in hit.lower() for hit in hits)

        is_forensic2, hits2 = detect_forensic_tripwire("What happened to that order?")
        assert is_forensic2
        assert any("that order" in hit2.lower() for hit2 in hits2)

    def test_order_with_identifier_pattern(self):
        """Detect 'order ABC123' pattern."""
        from trimmer_hook import detect_forensic_tripwire

        is_forensic, hits = detect_forensic_tripwire("Show details for order ORD-12345")
        assert is_forensic
        assert any("order ORD-12345" in hit for hit in hits)

    def test_multiple_hits_returned(self):
        """Multiple forensic patterns should all be returned."""
        from trimmer_hook import detect_forensic_tripwire

        # This prompt contains multiple forensic signals
        is_forensic, hits = detect_forensic_tripwire(
            "Why did request id=abc123 fail? What went wrong with this transaction?"
        )
        assert is_forensic
        assert len(hits) >= 2, f"Expected multiple hits, got: {hits}"


# =============================================================================
# INTEGRATION TESTS: Hook Behavior
# =============================================================================

def run_hook_with_prompt(prompt: str, data: list | dict = None) -> dict:
    """
    Run the hook with a prompt and optional data.

    Returns dict with:
    - exit_code: int
    - stdout: str
    - stderr: str
    - blocked: bool
    """
    if data is not None:
        full_prompt = f"{prompt}\n\n{json.dumps(data)}"
    else:
        full_prompt = prompt

    hook_input = {"prompt": full_prompt}

    result = subprocess.run(
        [sys.executable, str(scripts_dir / "trimmer_hook.py")],
        input=json.dumps(hook_input),
        capture_output=True,
        text=True,
        env={
            **dict(__import__("os").environ),
            "TOKEN_GUARD_PROMPT_LIMIT": "500",
            "TOKEN_GUARD_MIN_CHARS_BEFORE_COUNT": "100",
        },
    )

    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "blocked": result.returncode == 2,
    }


class TestHookBehavior:
    """Integration tests for hook blocking behavior."""

    @pytest.fixture
    def large_data(self):
        """Create large dataset that exceeds token limit."""
        return [{"id": i, "name": f"Item {i}", "value": i * 100} for i in range(200)]

    def test_forensics_mode_blocks(self, large_data):
        """Forensics mode should BLOCK, not sample."""
        prompt = "#trimmer:mode=forensics\nWhy did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        assert result["blocked"], "Forensics mode should block"
        assert "FORENSICS MODE" in result["stderr"]
        assert "sampling not allowed" in result["stderr"].lower()

    def test_forensic_question_without_mode_blocks(self, large_data):
        """Forensic question without explicit mode should BLOCK (fail-safe)."""
        prompt = "Why did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        assert result["blocked"], "Forensic question should block by default"
        assert "FORENSIC SIGNALS DETECTED" in result["stderr"]
        assert "Sampling may hide" in result["stderr"]

    def test_forensic_question_with_explicit_analysis_allows(self, large_data):
        """Forensic question WITH explicit #trimmer:mode=analysis should allow sampling."""
        prompt = "#trimmer:mode=analysis\nWhy did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        # Should NOT block - explicit mode unlocks
        assert not result["blocked"], "Explicit analysis mode should allow"

    def test_forensic_question_with_force_allows(self, large_data):
        """Forensic question WITH #trimmer:force should allow."""
        prompt = "#trimmer:force\nWhy did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        # Should NOT block - force bypasses
        assert not result["blocked"], "Force marker should bypass"

    def test_global_question_samples_normally(self, large_data):
        """Global questions should sample normally (no blocking)."""
        prompt = "What categories exist in this data?"
        result = run_hook_with_prompt(prompt, large_data)

        # Should NOT block - global question, sampling OK
        assert not result["blocked"], "Global question should allow sampling"

    def test_summary_mode_allows_aggressive_trimming(self, large_data):
        """Summary mode should allow trimming."""
        prompt = "#trimmer:mode=summary\nDescribe the structure"
        result = run_hook_with_prompt(prompt, large_data)

        assert not result["blocked"], "Summary mode should allow trimming"


# =============================================================================
# UNIT TESTS: Behavior Matrix
# =============================================================================

class TestBehaviorMatrix:
    """Test the complete behavior matrix from the plan."""

    @pytest.fixture
    def large_data(self):
        return [{"id": i} for i in range(200)]

    def test_matrix_global_default_samples(self, large_data):
        """Global question + default mode = SAMPLE"""
        prompt = "What categories exist?"
        result = run_hook_with_prompt(prompt, large_data)
        assert not result["blocked"]

    def test_matrix_forensic_default_blocks(self, large_data):
        """Forensic question + default mode = BLOCK"""
        prompt = "Why did request id=req123 fail?"
        result = run_hook_with_prompt(prompt, large_data)
        assert result["blocked"]
        assert "FORENSIC SIGNALS DETECTED" in result["stderr"]

    def test_matrix_forensic_forensics_blocks(self, large_data):
        """Forensic question + forensics mode = BLOCK (too large)"""
        prompt = "#trimmer:mode=forensics\nWhy did request id=req123 fail?"
        result = run_hook_with_prompt(prompt, large_data)
        assert result["blocked"]
        assert "FORENSICS MODE" in result["stderr"]

    def test_matrix_forensic_analysis_samples(self, large_data):
        """Forensic question + explicit analysis mode = SAMPLE"""
        prompt = "#trimmer:mode=analysis\nWhy did request id=req123 fail?"
        result = run_hook_with_prompt(prompt, large_data)
        assert not result["blocked"]


# =============================================================================
# RUN STANDALONE
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
