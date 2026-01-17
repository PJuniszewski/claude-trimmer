---
description: Analyze and prepare prompts with JSON for safe context submission
arguments:
  - name: input
    description: JSON data, file path, or "-" for stdin
    required: true
  - name: --mode
    description: "analysis|summary|forensics (default: auto)"
    required: false
  - name: --force
    description: Bypass blocks, emit warnings only
    required: false
  - name: --budget-tokens
    description: Token budget (default from env)
    required: false
---

# Context Guard Command

You are executing the `/guard` command for epistemic safety analysis.

## Instructions

Run the guard command using the Bash tool with a short description to keep the UI clean:

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/guard "$ARGUMENTS.input" \
  ${ARGUMENTS.--mode:+--mode $ARGUMENTS.--mode} \
  ${ARGUMENTS.--force:+--force} \
  ${ARGUMENTS.--budget-tokens:+--budget-tokens $ARGUMENTS.--budget-tokens}
```

Use description: "Context Guard" when calling the Bash tool.

Display the output to the user. Done.

## Examples

```bash
# Basic analysis
/guard data.json

# Force through despite forensic detection
/guard data.json --force

# Use forensics mode for single-record queries
/guard logs.json --mode forensics

# Custom token budget
/guard data.json --budget-tokens 10000
```
