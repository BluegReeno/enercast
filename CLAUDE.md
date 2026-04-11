# WindCast — Project Guidelines for Claude Code

## Language Policy

- **Conversations**: French OK
- **Code, docs, commits**: English only

---

## Project Overview

**WindCast** — Standardized ML framework for wind power forecasting. Turns raw SCADA data into calibrated power forecasts using reproducible pipelines, MLflow experiment tracking, and open datasets.

**Full specifications**: See `.claude/PRD.md`
**Research**: See `docs/research/brainstorming-2026-03-31.md`

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Language** | Python 3.12+ | ML ecosystem |
| **Package manager** | uv | Fast, lockfile-based |
| **Data processing** | Polars (primary) | 10-min SCADA data |
| **ML models** | XGBoost, AutoGluon-Tabular, LightGBM, scikit-learn | 2 backends via Protocol |
| **Experiment tracking** | MLflow 3.x | Model registry, comparison UI |
| **Hyperparameter tuning** | Optuna | Bayesian optimization |
| **Weather data** | Open-Meteo | Free NWP forecasts |
| **Linting** | ruff | Lint + format |
| **Type checking** | pyright | Static analysis |
| **Testing** | pytest | Unit + integration |

---

## Context System (3 tiers)

Context is organized in progressive disclosure — load only what you need:

### Tier 1: Global Rules (this file)

Always loaded. Keep lean (<100 lines of actual rules). If removing a line wouldn't cause mistakes, cut it.

### Tier 2: Path-Scoped Rules (`.claude/rules/`)

Auto-loaded when you work on matching files. Each rule has a `paths:` frontmatter with glob patterns.

Example: `.claude/rules/frontend.md` loads when editing `src/frontend/**/*.tsx`.

### Tier 3: Reference & Deep Docs

| Location | When to read |
|----------|--------------|
| `.claude/reference/` | Manually, for reusable patterns and code examples |

### Always-Available Documents

| Document | When to Read |
|----------|--------------|
| `.claude/PRD.md` | Project scope, features, architecture |

---

## Task Management

**Workflow:**
1. Check `.claude/tasks/` for the active task
2. Read active task file in `.claude/tasks/`
3. Mark `@claude` when starting, `[x] ✓ YYYY-MM-DD` when done
4. Move completed features to `_archive/`

---

## Code Style

### Naming
- Files: `kebab-case.ts` / `snake_case.py`
- Classes: `PascalCase`
- Functions: `camelCase` (JS) / `snake_case` (Python)
- Constants: `UPPER_SNAKE_CASE`

### Imports
- Group: stdlib → third-party → local
- Sort alphabetically within groups

### Formatting
- Use project formatter (Prettier/Black)
- Max line length: 100 chars

---

## Core Principles

- **Fix forward** — no backward compatibility, remove deprecated code immediately
- **Fail fast** — detailed errors over graceful failures
- **KISS / DRY / YAGNI** — simple, no repetition, no overbuilding
- **Clean comments** — describe functionality, not changes (avoid "LEGACY", "REMOVED", "SIMPLIFIED")

---

## Testing

- Test files: `*.test.ts` / `test_*.py`
- Run before commit
- Prefer integration tests over unit tests for APIs

---

## Session Management

- Use `/handoff` before ending long sessions to capture state for the next session
- Use `/commit` with the `Context:` section when AI context files change

---

## Commands

```bash
# Setup
uv sync                                    # Install dependencies

# Development
uv run python scripts/ingest_kelmarsh.py   # Parse & QC Kelmarsh data
uv run python scripts/build_features.py    # Feature engineering
uv run python scripts/train.py             # Train models (logged to MLflow)
uv run python scripts/evaluate.py          # Evaluate + compare in MLflow
mlflow ui                                  # Open MLflow tracking UI

# Validation (run before every commit)
uv run ruff check src/ tests/ scripts/     # Lint
uv run ruff format --check src/ tests/     # Format check
uv run pyright src/                        # Type check
uv run pytest tests/ -v                    # Tests
```

---

## Common Gotchas

### Data
- SCADA signal naming varies even within same OEM — always check signal mapping files
- Status codes differ by supervision platform (Greenbyte vs PI System vs Bazefield)
- Curtailment distorts power-wind relationship — must be filtered before training
- Power curve is flat at rated wind speed — model struggles in high-wind regime
- Timestamps: SCADA may be local time, NWP is UTC — normalize immediately
- Hill of Towie timestamps are **end-of-period** — shift -10 min for NWP alignment
- Hill of Towie AeroUp retrofit = performance discontinuity — use as covariate or split

### ML
- Power curve non-linearity: V³ feature helps but doesn't fully capture wake/turbulence effects
- Persistence is a strong baseline for short horizons (< 2h) — must beat it to be useful
- NWP wind at hub height ≠ SCADA wind at hub height — systematic bias expected
- Train/val/test split must be temporal (no shuffling) — same lesson as WattCast
- Optuna can overfit: use conservative defaults first (WattCast lesson: v6 was worse than v5)

### MLflow
- Use `mlflow.set_tracking_uri("file:./mlruns")` for local tracking
- Log feature sets as artifacts, not just metrics — enables reproducibility
- Tag experiments by dataset name for cross-site comparison

---

## External Resources

- [PRD](.claude/PRD.md) | [README](README.md)
- [Brainstorming](docs/research/brainstorming-2026-03-31.md) | [Methodology](docs/research/methodology-scaling-pipeline.md)
- [Dataset Catalog](docs/research/datasets-catalog-2026-03-31.md)
