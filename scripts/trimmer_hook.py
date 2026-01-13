#!/usr/bin/env python3
"""
Trimmer Hook for Claude Code.

Catches oversized prompts and produces trimmed payloads to keep context healthy.
Reads hook input from stdin, outputs JSON decision to stdout.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import sibling modules
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import re

from token_count import count_tokens_safe
from trim_lib import (
    TrimPolicy,
    TrimResult,
    extract_json_from_prompt,
    trim_payload,
    # New intelligent trimming imports
    TokenBudget,
    SamplingStrategy,
    MinificationConfig,
    IntelligentTrimReport,
    intelligent_trim,
    TrimMode,
)

# Configuration defaults
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_PROMPT_LIMIT = 3500
DEFAULT_WARN_LIMIT = 2500
DEFAULT_MIN_CHARS = 6000
DEFAULT_MODE = "context"  # context, block, warn, off
DEFAULT_CLIPBOARD = True  # Copy minified to clipboard when blocking
DEFAULT_SAMPLING_STRATEGY = "first_last_even"  # first_n, first_last_even
DEFAULT_REMOVE_NULL = True
DEFAULT_ALIAS_KEYS = False
DEFAULT_MIN_RECORDS = 2

# Escape hatch markers
MARKER_OFF = "#trimmer:off"
MARKER_FORCE = "#trimmer:force"

# Semantic mode markers
MARKER_MODE_ANALYSIS = "#trimmer:mode=analysis"
MARKER_MODE_SUMMARY = "#trimmer:mode=summary"
MARKER_MODE_FORENSICS = "#trimmer:mode=forensics"

# Forensic tripwire patterns (conservative fail-safe heuristic)
# NOTE:
# Forensic intent is evaluated regardless of prompt size.
# Blocking occurs only when BOTH conditions are met:
#   1. Forensic pattern detected (specific record lookup)
#   2. Prompt size exceeds threshold (data loss is possible)
# This prevents silent sampling from hiding the answer user is looking for.
# False negatives are acceptable. Silent data loss is not.
FORENSIC_PATTERNS = [
    # Explicit ID lookups
    r"request\s+id[=:]\s*\S+",                     # "request id=abc123"
    r"user\s+id[=:]\s*\S+",                        # "user id=xyz"
    r"order\s+id[=:]\s*\S+",                       # "order id=123"
    r"transaction\s+id[=:]\s*\S+",                 # "transaction id=TX-123"
    r"\b(id|request_id|user_id|order_id)\b\s*[=:]\s*['\"]?\w{6,}['\"]?",  # "id=abc123"
    # Entity + identifier patterns
    r"\b(order|transaction|request)\s+[\w-]{4,}",  # "order ABC123", "transaction TX-999"
    # Failure investigation
    r"why\s+did\s+.+\s+fail",                      # "why did X fail"
    r"\b(what|why)\s+(went\s+wrong|failed|broke)\b",  # "what went wrong", "why failed"
    r"what\s+happened\s+to\s+\S+",                 # "what happened to request X"
    # Specific record references
    r"\b(this|that)\s+(request|order|transaction|record)\b",  # "this request", "that order"
    r"find\s+.+\s+with\s+id",                      # "find record with id X"
    r"show\s+me\s+.+\s+for\s+id",                  # "show me details for id X"
    r"specific\s+\w+\s+id",                        # "specific request id"
    # UUID pattern (almost always forensic)
    r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",  # UUID
]


def detect_semantic_mode(prompt: str) -> TrimMode:
    """
    Detect semantic mode from prompt markers.

    Returns:
        TrimMode based on explicit marker, defaults to ANALYSIS.
    """
    if MARKER_MODE_FORENSICS in prompt:
        return TrimMode.FORENSICS
    if MARKER_MODE_SUMMARY in prompt:
        return TrimMode.SUMMARY
    if MARKER_MODE_ANALYSIS in prompt:
        return TrimMode.ANALYSIS
    return TrimMode.ANALYSIS  # Default


def detect_forensic_tripwire(prompt: str) -> tuple[bool, list[str]]:
    """
    Detect if prompt contains forensic-style queries (asks about specific records).

    This is a conservative tripwire, NOT a full intent detection system.
    Detection is evaluated regardless of prompt size, but blocking decision
    depends on BOTH forensic intent AND prompt size exceeding threshold.

    Returns:
        Tuple of (is_forensic, list_of_matched_patterns)
    """
    hits = []
    for pattern in FORENSIC_PATTERNS:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            hits.append(match.group(0))
    return bool(hits), hits


# Patterns to extract identifier VALUES from forensic hits
# Note: These are applied with case-SENSITIVE matching to avoid false positives
IDENTIFIER_EXTRACTORS = [
    # UUID pattern - extract full UUID (case-insensitive OK)
    (r"\b([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\b", True),
    # id=value, id:value patterns - extract the value (case-insensitive for "id")
    (r"id[=:]\s*['\"]?([a-zA-Z0-9_-]{3,})['\"]?", True),
    # Hex strings (potential IDs) - at least 8 chars (case-insensitive OK)
    (r"\b([a-f0-9]{8,})\b", True),
    # Alphanumeric IDs with common prefixes - MUST be uppercase (case-sensitive!)
    # This avoids matching words like "request" where REQ matches the prefix
    (r"\b((?:TX|ORD|REQ|USR|ID)[_-][A-Z0-9]{4,})\b", False),
]


def extract_identifier_values(forensic_hits: list[str]) -> list[str]:
    """
    Extract actual identifier values from forensic pattern matches.

    For example:
    - "id=9f3a2c7e-4b51-4c6f-8e32-1d9a8f1b1234" → ["9f3a2c7e-4b51-4c6f-8e32-1d9a8f1b1234"]
    - "request id=abc123" → ["abc123"]
    - "order ORD-99999" → ["ORD-99999"]

    Returns:
        List of extracted identifier values
    """
    identifiers = []
    for hit in forensic_hits:
        for pattern, case_insensitive in IDENTIFIER_EXTRACTORS:
            flags = re.IGNORECASE if case_insensitive else 0
            matches = re.findall(pattern, hit, flags)
            identifiers.extend(matches)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for id_val in identifiers:
        if id_val.lower() not in seen:
            seen.add(id_val.lower())
            unique.append(id_val)
    return unique


def check_identifiers_in_payload(identifiers: list[str], payload: str) -> tuple[list[str], list[str]]:
    """
    Check which identifiers actually exist in the DATA portion of the payload.

    This is the key defense against phantom entity injection attacks.
    If an identifier is NOT found in the payload DATA, the model may hallucinate.

    The check looks for identifiers in JSON data portions, not in the question text.
    If an ID appears only once (in the question), it's considered missing from data.
    If an ID appears multiple times (question + data), it's considered found.

    Args:
        identifiers: List of identifier values to check
        payload: The full prompt/payload to search

    Returns:
        Tuple of (found_identifiers, missing_identifiers)
    """
    # Extract JSON portion from the prompt and convert back to string for searching
    json_data, method = extract_json_from_prompt(payload)
    data_str = json.dumps(json_data) if json_data is not None else ""

    found = []
    missing = []
    for id_val in identifiers:
        # Case-insensitive search for the identifier in the DATA portion
        pattern = re.escape(id_val)
        if data_str and re.search(pattern, data_str, re.IGNORECASE):
            found.append(id_val)
        else:
            # ID is NOT in the data - this is a phantom entity
            missing.append(id_val)
    return found, missing


def debug_log(msg: str) -> None:
    """Write debug message to file (disabled in production)."""
    pass  # Uncomment below for debugging
    # try:
    #     with open("/tmp/trimmer_debug.log", "a") as f:
    #         f.write(f"{datetime.now().isoformat()} - {msg}\n")
    # except:
    #     pass


def get_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    clipboard_env = os.environ.get("TOKEN_GUARD_CLIPBOARD", str(DEFAULT_CLIPBOARD))
    remove_null_env = os.environ.get("TOKEN_GUARD_REMOVE_NULL", str(DEFAULT_REMOVE_NULL))
    alias_keys_env = os.environ.get("TOKEN_GUARD_ALIAS_KEYS", str(DEFAULT_ALIAS_KEYS))

    # Parse sampling strategy
    strategy_str = os.environ.get("TOKEN_GUARD_SAMPLING_STRATEGY", DEFAULT_SAMPLING_STRATEGY).lower()
    try:
        sampling_strategy = SamplingStrategy(strategy_str)
    except ValueError:
        sampling_strategy = SamplingStrategy.FIRST_LAST_EVEN

    return {
        "model": os.environ.get("TOKEN_GUARD_MODEL", DEFAULT_MODEL),
        "prompt_limit": int(os.environ.get("TOKEN_GUARD_PROMPT_LIMIT", DEFAULT_PROMPT_LIMIT)),
        "warn_limit": int(os.environ.get("TOKEN_GUARD_WARN_LIMIT", DEFAULT_WARN_LIMIT)),
        "min_chars": int(os.environ.get("TOKEN_GUARD_MIN_CHARS_BEFORE_COUNT", DEFAULT_MIN_CHARS)),
        "mode": os.environ.get("TOKEN_GUARD_MODE", DEFAULT_MODE).lower(),
        "clipboard": clipboard_env.lower() in ("true", "1", "yes"),
        # Intelligent trimming config
        "sampling_strategy": sampling_strategy,
        "remove_null": remove_null_env.lower() in ("true", "1", "yes"),
        "alias_keys": alias_keys_env.lower() in ("true", "1", "yes"),
        "min_records": int(os.environ.get("TOKEN_GUARD_MIN_RECORDS", DEFAULT_MIN_RECORDS)),
    }


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard. Returns True on success."""
    import platform
    import subprocess

    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
        elif system == "Linux":
            # Try xclip first, then xsel
            for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                    proc.communicate(text.encode("utf-8"))
                    if proc.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            proc = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
            proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
    except Exception:
        pass
    return False


def out(obj: dict, exit_code: int = 0) -> None:
    """Write JSON response to stdout and exit."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")))
    sys.stdout.flush()
    sys.exit(exit_code)


def allow() -> None:
    """Return allow decision."""
    debug_log("Outputting allow decision")
    out({
        "hookSpecificOutput": {"hookEventName": "UserPromptSubmit"},
        "suppressOutput": True,
    })


def block(reason: str) -> None:
    """Block the prompt with exit code 2."""
    debug_log(f"Blocking: {reason[:100]}")
    # For UserPromptSubmit, exit code 2 blocks the prompt
    # stderr is shown to user as the block reason
    print(reason, file=sys.stderr)
    sys.exit(2)


def warn(reason: str) -> None:
    """Return allow decision with suppressOutput (warn mode)."""
    # In warn mode, we allow but print warning to stderr
    print(f"[trimmer] WARNING: {reason}", file=sys.stderr)
    out({
        "hookSpecificOutput": {"hookEventName": "UserPromptSubmit"},
        "suppressOutput": True,
    })


def auto_replace(new_prompt: str, info: str) -> None:
    """Replace prompt with new content and continue (NOT SUPPORTED by Claude Code)."""
    print(info, file=sys.stderr)
    out({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "newPrompt": new_prompt,
        },
        "suppressOutput": True,
    })


def add_context(context: str, info: str) -> None:
    """Add context to the conversation and allow prompt to continue."""
    print(info, file=sys.stderr)
    out({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        },
        "suppressOutput": True,
    })


def show_tui_menu(
    original_tokens: int,
    trimmed_tokens: int,
    limit: int,
    trim_result: TrimResult,
    minified_json: str,
    original_prompt: str,
) -> Optional[str]:
    """
    Show interactive TUI menu for user choice.

    Returns:
        - minified_json: user chose to continue with minified
        - original_prompt: user chose to force original
        - None: user chose to cancel
    """
    reduction = original_tokens - trimmed_tokens
    reduction_pct = (reduction / original_tokens * 100) if original_tokens > 0 else 0
    still_over = trimmed_tokens > limit
    notes_str = "; ".join(trim_result.notes) if trim_result.notes else "trimmed"

    # Build menu
    print("\n" + "─" * 50, file=sys.stderr)
    print("  TRIMMER: Prompt exceeds token limit", file=sys.stderr)
    print("─" * 50, file=sys.stderr)
    print(f"  Original: {original_tokens} tokens (limit: {limit})", file=sys.stderr)
    print(f"  Trimmed:  {trimmed_tokens} tokens (-{reduction}, -{reduction_pct:.0f}%)", file=sys.stderr)

    if still_over:
        print(f"  Status:   Still over by {trimmed_tokens - limit} tokens", file=sys.stderr)
    else:
        print(f"  Status:   Now under limit", file=sys.stderr)

    print(f"  Actions:  {notes_str}", file=sys.stderr)
    print("─" * 50, file=sys.stderr)
    print("", file=sys.stderr)
    print("  [1] Continue with minified JSON", file=sys.stderr)
    print("  [2] Force original (bypass trimmer)", file=sys.stderr)
    print("  [3] Cancel", file=sys.stderr)
    print("", file=sys.stderr)

    # Read choice from /dev/tty (works even when stdin is piped)
    try:
        with open("/dev/tty", "r") as tty:
            print("  Choice [1/2/3]: ", file=sys.stderr, end="", flush=True)
            choice = tty.readline().strip()
    except (OSError, IOError):
        # /dev/tty not available (non-interactive), default to block
        return None

    if choice == "1":
        return minified_json
    elif choice == "2":
        return original_prompt
    else:
        return None


def write_trimmed_file(
    trimmed_obj: Any,
    original_tokens: int,
    limit: int,
    method: str,
    report: IntelligentTrimReport,
) -> str:
    """
    Write trimmed payload to file.

    Returns the file path.
    """
    # Find project root (walk up to find .claude or .git)
    cwd = Path.cwd()
    project_root = cwd
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".claude").exists() or (parent / ".git").exists():
            project_root = parent
            break

    # Create output directory
    output_dir = project_root / ".claude" / "trimmer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"trimmed-{timestamp}"

    # Write pretty JSON
    pretty_path = output_dir / f"{base_name}.json"
    with open(pretty_path, "w", encoding="utf-8") as f:
        json.dump(trimmed_obj, f, indent=2, ensure_ascii=False)

    # Write minified JSON
    min_path = output_dir / f"{base_name}.min.json"
    with open(min_path, "w", encoding="utf-8") as f:
        json.dump(trimmed_obj, f, separators=(",", ":"), ensure_ascii=False)

    # Write metadata
    meta_path = output_dir / f"{base_name}.meta.json"
    meta = {
        "timestamp": datetime.now().isoformat(),
        "original_tokens": original_tokens,
        "trimmed_tokens": report.trimmed_tokens,
        "limit": limit,
        "extraction_method": method,
        "sampling_strategy": report.sampling_strategy.value,
        "arrays_trimmed": [
            {
                "path": arr.path,
                "original_count": arr.original_count,
                "kept_count": arr.kept_count,
                "kept_indices": arr.kept_indices,
            }
            for arr in report.arrays_trimmed
        ],
        "files": {
            "pretty": str(pretty_path),
            "minified": str(min_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(pretty_path)


def format_block_reason(
    tokens: int,
    limit: int,
    method: str,
    file_path: str,
    report: IntelligentTrimReport,
    clipboard_copied: bool,
) -> str:
    """Format block reason with options for user."""
    # Get relative path for cleaner display
    try:
        rel_path = Path(file_path).relative_to(Path.cwd())
    except ValueError:
        rel_path = Path(file_path)

    min_path = str(rel_path).replace(".json", ".min.json")

    # Token comparison
    reduction = tokens - report.trimmed_tokens
    reduction_pct = (reduction / tokens * 100) if tokens > 0 else 0
    still_over = report.trimmed_tokens > limit

    lines = [
        f"Prompt: {tokens} tokens (limit: {limit})",
        f"Trimmed: {report.trimmed_tokens} tokens (-{reduction}, -{reduction_pct:.0f}%)",
    ]

    if still_over:
        lines.append(f"  Still over limit by {report.trimmed_tokens - limit} tokens")
    else:
        lines.append(f"  Now under limit")

    # Show array trimming summary
    if report.arrays_trimmed:
        for arr in report.arrays_trimmed:
            lines.append(f"  {arr.path}: {arr.original_count} -> {arr.kept_count} records ({arr.strategy})")

    lines.append(f"File: {min_path}")
    lines.append("")

    if clipboard_copied:
        lines.append("Minified JSON copied to clipboard - just paste (Cmd+V)")
    else:
        lines.append("Paste content from .min.json or add #trimmer:force to bypass")

    return "\n".join(lines)


def format_extraction_failed_reason(tokens: int, limit: int) -> str:
    """Format reason when JSON extraction fails."""
    return (
        f"Prompt: {tokens} tokens (limit: {limit}). No JSON detected.\n\n"
        "Options:\n"
        "  - Wrap JSON with: ### PAYLOAD START / ### PAYLOAD END\n"
        "  - Force: add #trimmer:force to your prompt\n"
        "  - Or reduce payload manually"
    )


def run_hook(hook_input: dict[str, Any]) -> None:
    """
    Main hook logic. Calls allow/block/warn which exit the process.

    Args:
        hook_input: Parsed JSON from stdin with "prompt" field.
    """
    debug_log("run_hook started")
    config = get_config()
    debug_log(f"config: min_chars={config['min_chars']}, limit={config['prompt_limit']}")

    # Mode check
    if config["mode"] == "off":
        debug_log("Mode is off, allowing")
        allow()

    # Extract prompt from Claude Code's format: {"prompts": [{"content": "..."}]}
    prompts = hook_input.get("prompts", [])
    if prompts:
        # Concatenate all prompt contents
        prompt = "\n".join(p.get("content", "") for p in prompts if p.get("content"))
    else:
        # Fallback for legacy format
        prompt = hook_input.get("prompt", "")

    if not prompt:
        debug_log("No prompt, allowing")
        allow()

    # Escape hatch: #trimmer:off
    if MARKER_OFF in prompt:
        debug_log("Marker off found, allowing")
        allow()

    # Escape hatch: #trimmer:force
    force_mode = MARKER_FORCE in prompt

    # Early exit for small prompts - but check forensic patterns first
    if len(prompt) < config["min_chars"] and not force_mode:
        debug_log(f"Small prompt ({len(prompt)} < {config['min_chars']})")
        # Check for forensic patterns even in small prompts
        is_forensic_early, forensic_hits_early = detect_forensic_tripwire(prompt)
        semantic_mode_early = detect_semantic_mode(prompt)
        if is_forensic_early and semantic_mode_early == TrimMode.ANALYSIS:
            has_explicit_allow = (
                MARKER_MODE_ANALYSIS in prompt or
                MARKER_FORCE in prompt
            )
            if not has_explicit_allow:
                # Extract identifier values and check if they exist in payload
                identifier_values = extract_identifier_values(forensic_hits_early)
                found_ids, missing_ids = check_identifiers_in_payload(identifier_values, prompt)

                hits_display = ", ".join(f'"{hit}"' for hit in forensic_hits_early[:3])

                if missing_ids:
                    # PHANTOM ENTITY DETECTED - ID not in payload!
                    missing_display = ", ".join(f'"{id_val}"' for id_val in missing_ids[:3])
                    warning_context = (
                        "⚠️ PHANTOM ENTITY WARNING:\n"
                        f"- You are asking about identifier(s): {missing_display}\n"
                        f"- These identifiers DO NOT EXIST in the provided data\n"
                        f"- The dataset is complete (no sampling occurred)\n"
                        f"- Any explanation for non-existent entities would be HALLUCINATION\n"
                        f"- If the entity doesn't exist, you MUST say 'This ID does not exist in the data'\n"
                        f"- DO NOT fabricate explanations for phantom entities"
                    )
                    info = "[trimmer] PHANTOM ENTITY: Identifier not found in payload"
                else:
                    # ID exists - mild warning
                    warning_context = (
                        "FORENSIC QUERY WARNING:\n"
                        f"- Detected identifier patterns: {hits_display}\n"
                        "- The dataset is complete (no sampling occurred)\n"
                        "- Verify the ID exists before drawing conclusions"
                    )
                    info = "[trimmer] Forensic pattern detected in small payload"
                add_context(warning_context, info)
        debug_log("Allowing small prompt")
        allow()

    # Count tokens
    tokens = count_tokens_safe(prompt, config["model"])
    if tokens < 0:
        # Token counting failed - use character-based estimate as fallback
        # Rough estimate: ~4 chars per token for English text
        estimated_tokens = len(prompt) // 4
        print(f"[trimmer] Token count API failed, using estimate: ~{estimated_tokens} tokens", file=sys.stderr)

        # If estimated tokens significantly exceed limit, block rather than fail open
        # Use 2x multiplier to be conservative (don't block unnecessarily)
        if estimated_tokens > config["prompt_limit"] * 2:
            block(
                f"TOKEN COUNT FAILED - BLOCKING LARGE PROMPT\n"
                f"Estimated: ~{estimated_tokens} tokens (limit: {config['prompt_limit']})\n"
                f"Token counting API unavailable, but prompt appears too large.\n\n"
                f"Options:\n"
                f"  - Reduce payload size\n"
                f"  - Add #trimmer:force to bypass\n"
                f"  - Set ANTHROPIC_API_KEY for accurate counting"
            )
        # Otherwise fail open for smaller prompts
        tokens = estimated_tokens

    # Check force bypass
    if force_mode:
        if tokens > config["prompt_limit"]:
            print(
                f"[trimmer] Force bypass: {tokens} tokens (limit: {config['prompt_limit']})",
                file=sys.stderr,
            )
        allow()

    # Determine threshold and mode
    mode = config["mode"]
    threshold = config["prompt_limit"]

    if mode == "warn":
        threshold = config["warn_limit"]

    # ==========================================================================
    # SEMANTIC MODE DETECTION
    # NOTE: Forensic intent is evaluated regardless of prompt size.
    # Blocking occurs only when prompt size makes silent data loss possible.
    # Principle: Semantics first, then size. But decision depends on BOTH.
    # ==========================================================================

    # Step 1: Detect semantic mode from markers
    semantic_mode = detect_semantic_mode(prompt)
    debug_log(f"Semantic mode: {semantic_mode.value}")

    # Step 2: Check for forensic tripwire (fail-safe heuristic)
    is_forensic, forensic_hits = detect_forensic_tripwire(prompt)
    debug_log(f"Forensic check: is_forensic={is_forensic}, hits={forensic_hits}")

    # Step 3: Decision based on BOTH intent and size
    # - forensic + small payload → ALLOW (no data loss risk)
    # - forensic + large payload → BLOCK (potential data loss)
    if is_forensic and tokens > threshold and semantic_mode == TrimMode.ANALYSIS:
        # Forensic query with potential data loss - BLOCK
        has_explicit_allow = (
            MARKER_MODE_ANALYSIS in prompt or
            MARKER_FORCE in prompt
        )
        if not has_explicit_allow:
            hits_display = "\n".join(f"  - \"{hit}\"" for hit in forensic_hits)
            block(
                f"FORENSIC SIGNALS DETECTED\n"
                f"Detected patterns:\n{hits_display}\n\n"
                f"Prompt size ({tokens} tokens) exceeds limit ({threshold}).\n"
                f"Sampling would hide data - your answer may be in trimmed records.\n\n"
                f"Options:\n"
                f"  - Add #trimmer:mode=forensics (block if payload too large)\n"
                f"  - Add #trimmer:mode=analysis (explicitly allow sampling)\n"
                f"  - Add #trimmer:force (bypass all checks)\n"
                f"  - Reduce payload size manually"
            )

    # If under threshold, allow (even if forensic - no data loss risk)
    # But add epistemic warning if forensic pattern detected
    if tokens <= threshold:
        debug_log(f"Under threshold ({tokens} <= {threshold})")
        # Check if we should add forensic warning for small payloads
        if is_forensic and semantic_mode == TrimMode.ANALYSIS:
            has_explicit_allow = (
                MARKER_MODE_ANALYSIS in prompt or
                MARKER_FORCE in prompt
            )
            if not has_explicit_allow:
                # Extract identifier values and check if they exist in payload
                identifier_values = extract_identifier_values(forensic_hits)
                found_ids, missing_ids = check_identifiers_in_payload(identifier_values, prompt)

                hits_display = ", ".join(f'"{hit}"' for hit in forensic_hits[:3])

                if missing_ids:
                    # PHANTOM ENTITY DETECTED - ID not in payload!
                    missing_display = ", ".join(f'"{id_val}"' for id_val in missing_ids[:3])
                    warning_context = (
                        "⚠️ PHANTOM ENTITY WARNING:\n"
                        f"- You are asking about identifier(s): {missing_display}\n"
                        f"- These identifiers DO NOT EXIST in the provided data\n"
                        f"- The dataset is complete (no sampling occurred)\n"
                        f"- Any explanation for non-existent entities would be HALLUCINATION\n"
                        f"- If the entity doesn't exist, you MUST say 'This ID does not exist in the data'\n"
                        f"- DO NOT fabricate explanations for phantom entities"
                    )
                    info = "[trimmer] PHANTOM ENTITY: Identifier not found in payload"
                else:
                    # ID exists - mild warning
                    warning_context = (
                        "FORENSIC QUERY WARNING:\n"
                        f"- Detected identifier patterns: {hits_display}\n"
                        "- The dataset is complete (no sampling occurred)\n"
                        "- Verify the ID exists before drawing conclusions"
                    )
                    info = "[trimmer] Forensic pattern detected in small payload"
                add_context(warning_context, info)
        allow()

    # FORENSICS MODE: No sampling allowed - block if over threshold
    if semantic_mode == TrimMode.FORENSICS:
        block(
            f"FORENSICS MODE: {tokens} tokens exceeds {threshold} limit.\n\n"
            f"Forensic analysis requires ALL records - sampling not allowed.\n"
            f"Cannot trim without risking loss of critical data.\n\n"
            f"Options:\n"
            f"  - Increase TOKEN_GUARD_PROMPT_LIMIT\n"
            f"  - Reduce payload size manually\n"
            f"  - Use #trimmer:mode=analysis if sampling is acceptable"
        )

    # ==========================================================================

    # Over threshold - attempt to extract and trim JSON
    extracted, method = extract_json_from_prompt(prompt)

    if extracted is None:
        # Extraction failed - block or warn
        reason = format_extraction_failed_reason(tokens, threshold)
        if mode == "warn":
            warn(reason)
        block(reason)

    # Configure intelligent trimming
    token_budget = TokenBudget(
        total_budget=threshold,
        overhead_tokens=50,
        min_records=config["min_records"],
    )
    minification_config = MinificationConfig(
        remove_null_fields=config["remove_null"],
        alias_keys=config["alias_keys"],
    )

    # Trim the payload using intelligent trimming
    trimmed_data, report = intelligent_trim(
        data=extracted,
        token_budget=token_budget,
        sampling_strategy=config["sampling_strategy"],
        minification=minification_config,
    )

    # Create minified JSON string
    minified_json = json.dumps(trimmed_data, separators=(",", ":"), ensure_ascii=False)

    # Write trimmed files for reference
    file_path = write_trimmed_file(
        trimmed_data,
        tokens,
        threshold,
        method,
        report,
    )

    # Get relative path for display
    try:
        rel_path = Path(file_path).relative_to(Path.cwd())
    except ValueError:
        rel_path = Path(file_path)

    # Handle based on mode
    if mode == "warn":
        # Warn mode: allow but show warning
        summary = report.to_summary_line()
        print(f"\n{'─' * 50}", file=sys.stderr)
        print(f"  TRIMMER WARNING: {tokens} tokens (limit {threshold})", file=sys.stderr)
        print(f"  {summary}", file=sys.stderr)
        print(f"  Trimmed file: {rel_path}", file=sys.stderr)
        print(f"{'─' * 50}\n", file=sys.stderr)
        allow()

    if mode == "context":
        # Context mode: add trimmed JSON as context using intelligent report
        context_parts = [
            report.to_claude_context(),
            "",
            "TRIMMED JSON DATA (use this instead of the large payload above):",
            minified_json,
        ]
        context = "\n".join(context_parts)
        info = f"[trimmer] Added trimmed context: {report.original_tokens} -> {report.trimmed_tokens} tokens"
        add_context(context, info)

    if mode == "auto":
        # Auto mode: automatically replace with minified (NOT SUPPORTED by Claude Code)
        reduction = report.original_tokens - report.trimmed_tokens
        reduction_pct = report.reduction_percentage
        info = (
            f"[trimmer] Auto-trimmed: {report.original_tokens} -> {report.trimmed_tokens} tokens "
            f"(-{reduction}, -{reduction_pct:.0f}%)"
        )
        auto_replace(minified_json, info)

    if mode == "interactive":
        # Interactive mode: show TUI menu (note: doesn't work well in Claude Code)
        # Create a legacy TrimResult for compatibility with show_tui_menu
        legacy_result = TrimResult(
            trimmed=trimmed_data,
            notes=[report.to_summary_line()],
            counters={},
        )
        choice = show_tui_menu(
            original_tokens=report.original_tokens,
            trimmed_tokens=report.trimmed_tokens,
            limit=threshold,
            trim_result=legacy_result,
            minified_json=minified_json,
            original_prompt=prompt,
        )

        if choice is None:
            # User cancelled
            block("Cancelled by user")
        elif choice == prompt:
            # User chose to force original
            print(f"[trimmer] Force bypass: {report.original_tokens} tokens", file=sys.stderr)
            allow()
        else:
            # User chose minified
            reduction = report.original_tokens - report.trimmed_tokens
            print(f"[trimmer] Using minified: {report.trimmed_tokens} tokens (-{reduction})", file=sys.stderr)
            auto_replace(choice, "")

    # Block mode (default): stop and optionally copy to clipboard
    clipboard_copied = False
    if config["clipboard"]:
        clipboard_copied = copy_to_clipboard(minified_json)

    reason = format_block_reason(
        tokens, threshold, method, file_path, report, clipboard_copied
    )
    block(reason)


def main() -> None:
    """Entry point for hook."""
    print("[Context-Guard] Hook active", file=sys.stderr)
    debug_log("Hook started")

    # Support --trim-file mode for direct trimming
    if len(sys.argv) > 1:
        if sys.argv[1] == "--trim-file" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Use intelligent trimming
                config = get_config()
                token_budget = TokenBudget(
                    total_budget=config["prompt_limit"],
                    overhead_tokens=50,
                    min_records=config["min_records"],
                )
                minification_config = MinificationConfig(
                    remove_null_fields=config["remove_null"],
                    alias_keys=config["alias_keys"],
                )

                trimmed_data, report = intelligent_trim(
                    data=data,
                    token_budget=token_budget,
                    sampling_strategy=config["sampling_strategy"],
                    minification=minification_config,
                )

                # Output trimmed JSON
                print(json.dumps(trimmed_data, indent=2, ensure_ascii=False))

                # Output report to stderr
                print(f"\n{report.to_claude_context()}", file=sys.stderr)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            return
        elif sys.argv[1] == "--help":
            print("Usage: trimmer_hook.py [--trim-file <path>]")
            print("  Without args: reads hook input from stdin")
            print("  --trim-file: trim a JSON file directly")
            print("\nEnvironment variables:")
            print("  TOKEN_GUARD_PROMPT_LIMIT: Token budget (default: 3500)")
            print("  TOKEN_GUARD_SAMPLING_STRATEGY: first_n or first_last_even (default: first_last_even)")
            print("  TOKEN_GUARD_REMOVE_NULL: Remove null/empty fields (default: true)")
            print("  TOKEN_GUARD_MIN_RECORDS: Minimum records to keep (default: 2)")
            return

    # Normal hook mode: read from stdin
    try:
        debug_log("Reading stdin")
        raw = sys.stdin.read()
        debug_log(f"Got input: {len(raw)} bytes")
        if not raw.strip():
            debug_log("Empty input, allowing")
            allow()  # Exits process

        hook_input = json.loads(raw)
        debug_log(f"Parsed JSON, prompt length: {len(hook_input.get('prompt', ''))}")

        # Run hook (calls allow/block/warn which exit)
        run_hook(hook_input)
    except Exception as e:
        debug_log(f"Exception: {e}")
        print(f"[trimmer] Error: {e}", file=sys.stderr)
        allow()  # Fail open, exits process


if __name__ == "__main__":
    main()
