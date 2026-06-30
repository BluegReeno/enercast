# EnerCast

[![CI](https://github.com/BluegReeno/enercast/actions/workflows/ci.yml/badge.svg)](https://github.com/BluegReeno/enercast/actions/workflows/ci.yml)

**The plumbing layer for energy forecasting ML.** Handles data ingestion, QC, schemas, experiment tracking, and evaluation — so domain experts spend their time on what actually improves accuracy: feature engineering, model selection, and understanding client data.

Every hour an engineer spends writing QC scripts, debugging deployment artifacts, or rebuilding data pipelines is an hour NOT spent on accuracy. EnerCast solves the plumbing once. Engineers iterate on features, models, and KPIs — not on boilerplate.

## What the Framework Handles (Plumbing)

| Layer | What's Solved | What the Engineer Controls |
|-------|--------------|---------------------------|
| **Data ingestion** | Parsers for SCADA, ENTSO-E, PVDAQ, Open-Meteo NWP | Write a new parser (~100 lines) for a new dataset |
| **Schemas** | Typed canonical schemas per domain (wind, demand, solar) | Extend with domain-specific fields |
| **QC pipeline** | Parameterizable rules (outliers, gaps, frozen sensors, holidays) | Adjust thresholds per client/site |
| **Feature sets** | Named sets (baseline / enriched / full) per domain | Modify features, create custom sets |
| **Training** | Temporal splits, MLflow logging, multi-horizon training | Choose model (XGBoost, LightGBM, any sklearn-compatible), tune hyperparameters |
| **Evaluation** | MAE, RMSE, MAPE, skill scores, regime analysis | Add custom KPIs (e.g., "accuracy when spot price > X") |
| **Tracking** | MLflow logs everything — features, params, metrics, artifacts | Compare experiments across domains in one UI |

## Quick Start

```bash
# Install dependencies
uv sync

# Wind pipeline (Kelmarsh dataset — default 5/1 train/val split)
uv run python scripts/ingest_kelmarsh.py        # Parse → QC → Parquet
uv run python scripts/build_features.py          # Feature engineering
uv run python scripts/train.py                   # Train XGBoost → MLflow
uv run python scripts/evaluate.py                # Evaluate → MLflow

# Demand pipeline (RTE France — auto 8/2 split from dataset config)
uv run python scripts/ingest_rte_france.py
uv run python scripts/build_features.py --domain demand --dataset rte_france
uv run python scripts/train.py --domain demand --dataset rte_france
uv run python scripts/log_tso_baseline.py        # RTE day-ahead benchmark

# AutoGluon backend (ensemble of CatBoost+LightGBM+XGBoost)
uv run python scripts/train.py --backend autogluon --feature-set wind_full

# Override split config via CLI (useful for experiments)
uv run python scripts/train.py --domain demand --dataset rte_france --train-years 6 --val-years 2

# Compare runs (MAE + Skill bar charts)
uv run python scripts/compare_runs.py --experiment enercast-kelmarsh

# Promote best model to champion (after evaluating in MLflow UI)
uv run python scripts/promote_model.py --experiment enercast-kelmarsh --metric h24_mae

# View results
mlflow ui
```

## Adding a New Client/Dataset

1. Write a parser (~100 lines) that maps raw data to the domain schema
2. Add a `DatasetConfig` in `config.py` (coordinates, capacity, timezone)
3. Run the pipeline: `ingest → build_features → train → evaluate`

Zero changes to the core pipeline. The parser is the only dataset-specific code.

## Domains

| Domain | Dataset | Source | Resolution | What It Demonstrates |
|--------|---------|--------|-----------|---------------------|
| **Wind** | Kelmarsh v4 (6 turbines) | Zenodo | 10 min | SCADA ingestion, power curve modeling, NWP integration |
| **Demand** | RTE France (11y national load) | éCO2mix | 1 hour | Load forecasting, 8-city weighted NWP, TSO benchmark |
| **Demand** | Spain ENTSO-E | Kaggle | 1 hour | 2nd reference implementation, calendar features |
| **Solar** | PVDAQ System 4 | NREL | 15 min | Irradiance-based forecasting, clearsky ratio |

Same pipeline pattern, different parsers and feature configs.

## Pipeline Pattern

```
Raw data (CSV/ZIP/API)
    → [Parser] domain-specific mapping → canonical schema
    → [QC] parameterized rules → flagged data → clean Parquet
    → [Features] domain feature set → feature Parquet
    → [Train] temporal split → model → MLflow run
    → [Evaluate] metrics + skill scores + regime analysis → MLflow artifacts
```

**Framework zones** (grey — solved once): Parser scaffolding, schema validation, QC engine, temporal splits, MLflow logging, evaluation metrics.

**Engineer zones** (blue — where accuracy comes from): Signal mapping, feature design, model choice, QC thresholds, custom KPIs, regime definitions.

## Project Structure

```
src/windcast/
├── config.py                # Pydantic Settings + dataset configs
├── data/                    # Data ingestion & QC
│   ├── schema.py            # Wind SCADA schema (15 cols)
│   ├── demand_schema.py     # Demand schema (12 cols)
│   ├── solar_schema.py      # Solar schema (10 cols)
│   ├── kelmarsh.py          # Wind parser (Kelmarsh v4)
│   ├── spain_demand.py      # Demand parser (ENTSO-E)
│   ├── pvdaq.py             # Solar parser (PVDAQ)
│   ├── qc.py                # Wind QC (9 rules)
│   ├── demand_qc.py         # Demand QC
│   ├── solar_qc.py          # Solar QC
│   └── open_meteo.py        # NWP weather client (Open-Meteo)
├── features/                # Feature engineering
│   ├── registry.py          # Feature set registry (18 sets across 3 domains)
│   ├── wind.py              # Wind-specific features
│   ├── demand.py            # Demand-specific features
│   └── solar.py             # Solar-specific features
├── models/                  # ML models (domain-agnostic)
│   ├── xgboost_model.py     # XGBoost trainer
│   ├── autogluon_model.py   # AutoGluon-Tabular ensemble
│   ├── mlforecast_model.py  # mlforecast (Nixtla) — separate training path
│   ├── persistence.py       # Naive persistence benchmark
│   └── evaluation.py        # Metrics, skill scores, regime analysis
├── training/                # Unified training harness
│   ├── harness.py           # TrainingBackend Protocol + run_training()
│   ├── backends.py          # XGBoostBackend + AutoGluonBackend
│   └── lineage.py           # MLflow lineage tags
└── tracking/
    └── mlflow_utils.py      # MLflow logging utilities
```

## Tech Stack

Python 3.12+ · uv · Polars · XGBoost · AutoGluon-Tabular · LightGBM · mlforecast (Nixtla) · scikit-learn · MLflow · Optuna · Open-Meteo · pytest · ruff · pyright

## Quality

- 322 tests passing
- ruff lint + format clean
- pyright type checking clean
- Strict temporal train/val/test splits (no data leakage)

## Results

| Benchmark | Metric | Score | vs. Baseline |
|-----------|--------|-------|--------------|
| RTE France national load (11 years) | MAE h24 | 1,139 MW | **−5.5% vs. RTE TSO day-ahead** |
| Kelmarsh wind (6 turbines, XGBoost full) | MAE h24 | 231 kW | **skill 0.315 vs. 0.107 XGB baseline** |

## Documentation

- **[PRD](.claude/PRD.md)** — Product requirements and architecture
- **[Research](docs/research/)** — Dataset catalog, methodology, brainstorming

## License

MIT
