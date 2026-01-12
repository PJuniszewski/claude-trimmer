# claude-trimmer

Token guard hook for Claude Code that blocks oversized prompts and offers automatic JSON trimming.

## Features

- **Token counting** via Anthropic API before prompts are processed
- **Automatic blocking** when prompts exceed configurable limits
- **JSON extraction** from prompts (wrapper markers, fenced blocks, or raw JSON)
- **Smart trimming** of JSON payloads (list limiting, field whitelisting, string truncation)
- **Escape hatches** (`#trimmer:off`, `#trimmer:force`)

## Installation

### 1. Add marketplace and install plugin

```bash
claude /marketplace add github:PJuniszewski/juni-tools-marketplace
claude /plugin install juni-tools:trimmer
claude /plugin enable trimmer
```

### 2. Set environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export TOKEN_GUARD_MODEL="claude-sonnet-4-20250514"
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`) for persistence.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_GUARD_MODEL` | `claude-sonnet-4-20250514` | Model for token counting |
| `TOKEN_GUARD_PROMPT_LIMIT` | `3500` | Block prompts above this token count |
| `TOKEN_GUARD_MIN_CHARS_BEFORE_COUNT` | `6000` | Skip API call for small prompts |
| `TOKEN_GUARD_MODE` | `block` | Mode: `block`, `warn`, or `off` |
| `TOKEN_GUARD_TRIM_MAX_ITEMS` | `5` | Max items per array after trim |
| `TOKEN_GUARD_TRIM_MAX_STRLEN` | `300` | Max string length after trim |

## Usage

### When blocked

```
Prompt: 5000 tokens (limit: 3500). No JSON detected.

Options:
  - Wrap JSON with: ### PAYLOAD START / ### PAYLOAD END
  - Force: add #trimmer:force to your prompt
  - Or reduce payload manually
```

### With JSON payload

Wrap your JSON with markers for automatic trimming:

```
### PAYLOAD START
{"products": [...], "users": [...]}
### PAYLOAD END

Analyze this data
```

Trimmed files are saved to `.claude/trimmer/`:
- `trimmed-TIMESTAMP.json` - Pretty printed
- `trimmed-TIMESTAMP.min.json` - Minified for pasting
- `trimmed-TIMESTAMP.meta.json` - Statistics

### Escape hatches

- `#trimmer:off` - Disable for this prompt
- `#trimmer:force` - Bypass blocking

## How it works

1. Hook intercepts `UserPromptSubmit` event
2. If prompt > `MIN_CHARS`, count tokens via Anthropic API
3. If tokens > `LIMIT`:
   - Extract JSON from prompt (if present)
   - Trim: limit arrays, whitelist fields, truncate strings
   - Save trimmed files
   - Block with exit code 2
4. Otherwise, allow prompt

## Tests

```bash
python3 tests/test_extract_json.py
python3 tests/test_trim.py
```

## License

MIT
