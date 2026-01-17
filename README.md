# context-guard

**Context-Guard prevents LLMs from reasoning with unjustified certainty when input data is incomplete.**

LLM Epistemic Safety Layer - context integrity enforcement with sampling-aware guardrails. Automatically detects when trimming would compromise analytical accuracy and blocks silent sampling.

## Recommended Usage

Context Guard provides two layers of protection:

1. **Automatic Hook** - Fast, lightweight safety net (warns about risky prompts)
2. **`/guard` Command** - Full analysis pipeline with lossless reduction

### Quick Start

```bash
# The hook automatically warns about large payloads
# For full control, use the /guard command:

# Analyze a file
/guard my_data.json

# Force through despite warnings
/guard my_data.json --force

# Check if forensic query is safe
/guard logs.json --mode forensics

# Use larger token budget
/guard data.json --budget-tokens 5000
```

## Features

### Layer 1: Lightweight Hook Protection

The hook is a fast safety net with NO API calls:
- **Character-based checks** - No token counting API needed
- **Warns about large payloads** - Suggests `/guard` for analysis
- **Forensic detection** - Warns when specific record queries detected
- **Context flooding protection** - Blocks massive payloads
- **Configurable fail-closed mode** - Block forensic+large if desired

### Layer 2: `/guard` Command

Full analysis pipeline with:
- **Lossless reduction** - Minify, columnar transform, remove nulls
- **Token counting** - API or heuristic fallback
- **Decision engine** - ALLOW / SAMPLE / BLOCK
- **Intelligent trimming** - First + last + evenly-spaced sampling
- **Transparent reporting** - Shows exactly what was modified

## Installation

### Via Juni-Tools Marketplace

```bash
# Add the marketplace (run /plugin, select "Add Marketplace", enter: PJuniszewski/juni-tools-marketplace)

# Install and enable
claude /plugin install juni-tools:context-guard
claude /plugin enable context-guard
```

### Set environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # For /guard token counting
export TOKEN_GUARD_MODEL="claude-sonnet-4-20250514"
```

## Configuration

### Hook Configuration (Lightweight)

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_GUARD_MIN_CHARS` | `6000` | Below = always allow |
| `TOKEN_GUARD_WARN_CHARS` | `15000` | Above = warn |
| `TOKEN_GUARD_HARD_LIMIT_CHARS` | `100000` | Above = hard block |
| `TOKEN_GUARD_MODE` | `warn` | `off` or `warn` |

### /guard Command Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_GUARD_MODEL` | `claude-sonnet-4-20250514` | Model for token counting |
| `TOKEN_GUARD_PROMPT_LIMIT` | `3500` | Default token budget |

## Semantic Modes

Context Guard supports three semantic modes that control trimming behavior:

| Mode | Sampling | Use Case |
|------|----------|----------|
| `analysis` | Allowed | Default. "What categories exist?", "Price range?" |
| `summary` | Aggressive | "Describe the data structure" |
| `forensics` | **BLOCKED** | "Why did request id=X fail?" |

### Mode Markers

Add markers to your prompt to explicitly set the mode:

```
#trimmer:mode=analysis    # Allow sampling (default)
#trimmer:mode=summary     # Allow aggressive trimming
#trimmer:mode=forensics   # Block if data exceeds limit
```

### Forensic Question Detection

Context Guard automatically detects forensic questions and warns/blocks:

| Forensic Pattern | Payload Size | Hook | /guard |
|------------------|--------------|------|--------|
| Detected | Small (< warn) | ALLOW | ALLOW |
| Detected | Large (> warn) | WARN* | BLOCK** |
| Not detected | Any size | ALLOW/WARN | SAMPLE |

\* With `TOKEN_GUARD_FAIL_CLOSED=true`, hook will BLOCK
\** Use `--allow-sampling` to override

```
# These patterns trigger forensic detection:
"Why did request id=abc123 fail?"
"What happened to user id: xyz789?"
"Show details for transaction TX-12345"
```

## Usage

### Hook (Automatic)

The hook runs automatically on every prompt:

```
[trimmer] WARNING: Forensic query detected ("request id=abc123") with large payload (~5000 tokens)
[trimmer] HINT: Use /guard to analyze, or add #trimmer:mode=analysis to allow sampling
```

### /guard Command

```bash
# Basic usage
/guard data.json

# With options
/guard data.json --mode forensics           # Strict mode
/guard data.json --allow-sampling           # Permit sampling for forensic
/guard data.json --force                    # Bypass all blocks
/guard data.json --no-reduce                # Skip lossless reduction
/guard data.json --budget-tokens 5000       # Custom budget
/guard data.json --print-only               # Just output report
```

### Output Format

```
============================================================
CONTEXT GUARD ANALYSIS
============================================================

Decision: [OK] ALLOW | [~] SAMPLE | [X] BLOCK
Mode: analysis | summary | forensics

TOKEN ANALYSIS:
  Original:     5,234 tokens
  After reduce: 4,891 tokens (-343)
  After trim:   3,421 tokens
  Budget:       3,500 tokens

LOSSLESS REDUCTIONS:
  - Removed 47 null/empty fields
  - Columnar: 2 arrays transformed (18% reduction)

SAMPLING APPLIED:
  - products: 500 -> 25 (first + last + every ~20th)

============================================================
READY TO SEND PROMPT
============================================================
```

### Escape Hatches

- `#trimmer:off` - Disable hook for this prompt
- `#trimmer:force` - Bypass all blocking
- `#trimmer:mode=analysis` - Explicitly allow sampling for forensic-looking questions

## Design Principles

1. **Lossless reduction before sampling** - Minify, columnar, dedup FIRST
2. **Epistemic safety** - Never answer forensic questions on sampled data
3. **Deterministic** - No ML, no summarization
4. **Reversible** - All reductions can be undone
5. **Fast hook, powerful command** - Quick safety net + full control when needed

## How It Works

### Layer 1: Hook (Lightweight)

Fast, character-based checks with NO API calls:

1. Check escape hatches (#trimmer:off, #trimmer:force)
2. chars < MIN_CHARS? → ALLOW
3. chars > HARD_LIMIT? → BLOCK (context flooding)
4. No JSON markers? → ALLOW
5. Detect forensic patterns
6. forensic + large + fail_closed? → BLOCK
7. forensic + large? → WARN + hint
8. large only? → WARN + hint
9. ALLOW

### Layer 2: /guard Command (Full Analysis)

1. Read input (file or stdin)
2. Extract JSON from prompt
3. Count tokens (API or heuristic)
4. Apply lossless reductions (minify, columnar)
5. Make decision (ALLOW/SAMPLE/BLOCK)
6. If SAMPLE: apply intelligent trimming
7. Output structured report

### Lossless Reduction Pipeline

```
Original JSON
    ↓
Columnar Transform (TOON format for arrays)
    ↓
Minification (remove nulls, sort keys, compact)
    ↓
Reduced JSON
```

### TOON Columnar Format

```python
# Before: [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
# After:  {"__cols__": ["id", "name"], "__rows__": [[1, "A"], [2, "B"]]}
```

## Epistemic Safety

Context Guard prevents confident hallucinations by:

1. **Explicit warnings** in trim reports about sampling limitations
2. **Forensic mode** that blocks instead of sampling
3. **Heuristic detection** of forensic questions
4. **First-last-even sampling** that preserves data distribution

### What Might Be Hidden

When data is sampled:
- Single-record anomalies in middle positions
- Trend reversals after first 60% of data
- Rare outliers that fall outside sample

For forensic queries requiring ALL records, use `/guard --mode forensics`.

## Tests

```bash
# Unit tests
pytest tests/test_reduce.py -v       # Lossless reduction
pytest tests/test_guard.py -v        # /guard command
pytest tests/test_semantic_modes.py -v

# LLM context tests (requires ANTHROPIC_API_KEY)
pytest tests/test_llm_context.py -v -m llm

# Adversarial tests
pytest tests/test_llm_adversarial.py -v -m llm

# All non-LLM tests
pytest tests/ -v --ignore=tests/test_llm_context.py --ignore=tests/test_llm_adversarial.py
```

## License

MIT
