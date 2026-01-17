---
description: Analyze and prepare prompts with JSON for safe context submission
arguments:
  - name: input
    description: Path to file OR "-" for stdin
    required: true
  - name: --mode
    description: "analysis|summary|forensics (default: auto)"
    required: false
  - name: --allow-sampling
    description: Permit sampling for forensic queries
    required: false
  - name: --force
    description: Bypass blocks, emit warnings only
    required: false
  - name: --no-reduce
    description: Skip lossless reduction
    required: false
  - name: --budget-tokens
    description: Token budget (default from env)
    required: false
  - name: --print-only
    description: Just output, never auto-send
    required: false
---

# Context Guard Command

You are executing the `/guard` command for epistemic safety analysis.

## Overview

The `/guard` command provides full control over the Context Guard pipeline:
1. **Lossless reduction** - Minify, columnar transform, remove nulls
2. **Decision engine** - ALLOW / SAMPLE / BLOCK based on mode and forensics
3. **Intelligent trimming** - First + last + evenly-spaced sampling if needed
4. **Transparent reporting** - Show exactly what was modified

## Arguments

- **Input**: $ARGUMENTS.input (file path or "-" for stdin)
- **Mode**: $ARGUMENTS.--mode (optional: analysis, summary, forensics)
- **Allow Sampling**: $ARGUMENTS.--allow-sampling (permit sampling for forensic queries)
- **Force**: $ARGUMENTS.--force (bypass blocks)
- **No Reduce**: $ARGUMENTS.--no-reduce (skip lossless reduction)
- **Budget Tokens**: $ARGUMENTS.--budget-tokens (default: 3500 or $TOKEN_GUARD_PROMPT_LIMIT)
- **Print Only**: $ARGUMENTS.--print-only (just show report)

## Instructions

1. Read the input file or stdin content
2. Run the guard command with the provided arguments:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/guard_cmd.py "$ARGUMENTS.input" \
  ${ARGUMENTS.--mode:+--mode $ARGUMENTS.--mode} \
  ${ARGUMENTS.--allow-sampling:+--allow-sampling} \
  ${ARGUMENTS.--force:+--force} \
  ${ARGUMENTS.--no-reduce:+--no-reduce} \
  ${ARGUMENTS.--budget-tokens:+--budget-tokens $ARGUMENTS.--budget-tokens} \
  ${ARGUMENTS.--print-only:+--print-only}
```

3. Present the results to the user:

### If Decision is ALLOW or SAMPLE

Show the report including:
- Token analysis (original -> reduced -> trimmed)
- Lossless reductions applied
- Sampling applied (if any)
- The "READY TO SEND PROMPT" block with the processed JSON

The user can then copy/paste the processed JSON into their next prompt.

### If Decision is BLOCK

Show the report explaining:
- Why the prompt was blocked
- Forensic patterns detected
- Options to proceed:
  1. Add `#trimmer:mode=analysis` to explicitly allow sampling
  2. Add `#trimmer:force` to bypass all checks
  3. Use `--allow-sampling` flag
  4. Reduce the payload size manually

## Decision Matrix

| Condition | Mode | Decision |
|-----------|------|----------|
| tokens <= budget | any | ALLOW |
| tokens > budget + force | any | ALLOW |
| tokens > budget | summary | SAMPLE |
| tokens > budget + allow-sampling | any | SAMPLE |
| tokens > budget + no forensic | analysis | SAMPLE |
| tokens > budget + forensic | analysis | BLOCK |
| tokens > budget | forensics | BLOCK |

## Modes

### analysis (default)
- Standard mode for global questions
- Allows sampling if data exceeds budget
- BLOCKS if forensic patterns detected (fail-safe)

### summary
- Aggressive mode for overview questions
- Always allows sampling
- Use for: "Describe the structure", "What fields exist?"

### forensics
- Strict mode for single-record queries
- NEVER allows sampling
- Use for: "Why did request id=X fail?", "What happened to order Y?"

## Examples

```bash
# Basic analysis
/guard data.json

# Force through despite forensic detection
/guard data.json --force

# Explicitly allow sampling for forensic query
/guard logs.json --mode analysis --allow-sampling

# Check if forensic query is safe with larger budget
/guard data.json --budget-tokens 10000

# Skip lossless reduction
/guard data.json --no-reduce
```

## Output Format

The command outputs a structured report:

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

FORENSIC PATTERNS DETECTED:
  - "request id=abc123"

LOSSLESS REDUCTIONS:
  - Removed 47 null/empty fields
  - Columnar: 2 arrays transformed (18% reduction)

SAMPLING APPLIED:
  - products: 500 -> 25 (first + last + every ~20th)

============================================================
READY TO SEND PROMPT
============================================================

[TRIMMING CONTEXT]
...
TRIMMED JSON DATA:
{...}
```

## Key Concepts

### Lossless Reduction
All reductions are reversible:
- **Minification**: Remove whitespace, sort keys, remove null/empty
- **Columnar (TOON)**: Transform arrays of objects to column format

### Epistemic Safety
The guard protects against:
1. **Silent sampling** - Never hide data without user awareness
2. **Forensic hallucination** - Block when specific record might be trimmed
3. **Context flooding** - Prevent massive payloads from consuming context
