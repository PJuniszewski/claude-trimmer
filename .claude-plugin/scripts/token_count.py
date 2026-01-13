#!/usr/bin/env python3
"""Token counting module using Anthropic's count_tokens API."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from typing import Optional

API_URL = "https://api.anthropic.com/v1/messages/count_tokens"
API_VERSION = "2023-06-01"
DEFAULT_TIMEOUT = 15


def count_tokens(
    content: str,
    model: str,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> int:
    """
    Count tokens for content using Anthropic's count_tokens API.

    Args:
        content: The text content to count tokens for.
        model: The model name (e.g., "claude-sonnet-4-20250514").
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        timeout: Request timeout in seconds.

    Returns:
        Number of input tokens.

    Raises:
        ValueError: If API key is missing.
        RuntimeError: If API request fails.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": API_VERSION,
        "x-api-key": key,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("input_tokens", 0)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}") from e


def count_tokens_safe(
    content: str,
    model: str,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    default: int = -1,
) -> int:
    """
    Count tokens with error handling. Returns default on failure.

    Args:
        content: The text content to count tokens for.
        model: The model name.
        api_key: Anthropic API key.
        timeout: Request timeout in seconds.
        default: Value to return on error.

    Returns:
        Number of input tokens, or default on error.
    """
    try:
        return count_tokens(content, model, api_key, timeout)
    except Exception as e:
        print(f"[trimmer] token count error: {e}", file=sys.stderr)
        return default


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Count tokens for text content")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--file", help="File to read content from (or use stdin)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    try:
        tokens = count_tokens(content, args.model)
        print(f"Tokens: {tokens}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
