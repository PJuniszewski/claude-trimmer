#!/usr/bin/env python3
"""Tests for JSON extraction from prompts."""
from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from trim_lib import extract_json_from_prompt


def test_wrapper_extraction():
    """Test extraction using ### PAYLOAD START/END markers."""
    prompt = '''Here is some data:

### PAYLOAD START
{"products": [{"id": 1, "name": "Widget"}], "count": 1}
### PAYLOAD END

Please analyze this.'''

    obj, method = extract_json_from_prompt(prompt)

    assert method == "wrapper"
    assert obj is not None
    assert obj["products"][0]["name"] == "Widget"
    assert obj["count"] == 1


def test_wrapper_extraction_case_insensitive():
    """Test that wrapper markers are case-insensitive."""
    prompt = '''### payload start
{"test": true}
### payload end'''

    obj, method = extract_json_from_prompt(prompt)

    assert method == "wrapper"
    assert obj == {"test": True}


def test_fenced_extraction():
    """Test extraction from ```json fenced blocks."""
    prompt = '''Check this JSON:

```json
{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
```

What do you think?'''

    obj, method = extract_json_from_prompt(prompt)

    assert method == "fenced"
    assert obj is not None
    assert len(obj["users"]) == 2
    assert obj["users"][0]["name"] == "Alice"


def test_raw_json_dict():
    """Test extraction when entire prompt is a JSON object."""
    prompt = '{"key": "value", "nested": {"a": 1}}'

    obj, method = extract_json_from_prompt(prompt)

    assert method == "raw"
    assert obj == {"key": "value", "nested": {"a": 1}}


def test_raw_json_array():
    """Test extraction when entire prompt is a JSON array."""
    prompt = '[1, 2, {"three": 3}]'

    obj, method = extract_json_from_prompt(prompt)

    assert method == "raw"
    assert obj == [1, 2, {"three": 3}]


def test_raw_json_with_whitespace():
    """Test extraction of raw JSON with leading/trailing whitespace."""
    prompt = '''
  {"data": "test"}
  '''

    obj, method = extract_json_from_prompt(prompt)

    assert method == "raw"
    assert obj == {"data": "test"}


def test_no_json():
    """Test that non-JSON prompts return None."""
    prompt = "This is just plain text without any JSON."

    obj, method = extract_json_from_prompt(prompt)

    assert method == "none"
    assert obj is None


def test_invalid_json():
    """Test that completely non-JSON text returns None."""
    prompt = "This has no JSON structure at all, just plain text."

    obj, method = extract_json_from_prompt(prompt)

    assert method == "none"
    assert obj is None


def test_broken_json_gets_repaired():
    """Test that broken JSON gets repaired by json_repair."""
    prompt = '{"broken": "json", missing_quote}'

    obj, method = extract_json_from_prompt(prompt)

    # json_repair can fix this, so we get "repaired"
    assert method == "repaired"
    assert obj == {"broken": "json"}


def test_priority_wrapper_over_fenced():
    """Test that wrapper markers take priority over fenced blocks."""
    prompt = '''### PAYLOAD START
{"source": "wrapper"}
### PAYLOAD END

```json
{"source": "fenced"}
```'''

    obj, method = extract_json_from_prompt(prompt)

    assert method == "wrapper"
    assert obj["source"] == "wrapper"


def test_priority_fenced_over_raw():
    """Test that fenced blocks take priority over raw JSON."""
    prompt = '''{"raw": true}

```json
{"fenced": true}
```'''

    obj, method = extract_json_from_prompt(prompt)

    # Note: wrapper comes first, then fenced, then raw
    # If no wrapper, fenced should be extracted
    assert method == "fenced"
    assert obj["fenced"] is True


if __name__ == "__main__":
    import traceback

    tests = [
        test_wrapper_extraction,
        test_wrapper_extraction_case_insensitive,
        test_fenced_extraction,
        test_raw_json_dict,
        test_raw_json_array,
        test_raw_json_with_whitespace,
        test_no_json,
        test_invalid_json,
        test_broken_json_gets_repaired,
        test_priority_wrapper_over_fenced,
        test_priority_fenced_over_raw,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}")
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
