"""Annotate existing MLflow runs with lineage tags (generation, data_quality, nwp_source).

Run once after implementing the new tag taxonomy. Classifies runs by creation
date into generations and marks leaked runs.

Usage:
    uv run python scripts/annotate_mlflow_runs.py --dry-run
    uv run python scripts/annotate_mlflow_runs.py
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import mlflow

from windcast.config import get_settings

logger = logging.getLogger(__name__)

# Default generation boundaries (UTC timestamps)
GEN1_END = datetime(2026, 4, 9, 23, 0, 0, tzinfo=UTC)
GEN2_END = datetime(2026, 4, 10, 6, 0, 0, tzinfo=UTC)

GENERATION_RULES: list[dict[str, str | datetime]] = [
    {
        "gen": "gen1",
        "before": GEN1_END,
        "data_quality": "LEAKED",
        "nwp_source": "era5",
        "change_reason": "initial_run_era5_and_ag_leak",
    },
    {
        "gen": "gen2",
        "before": GEN2_END,
        "data_quality": "LEAKED",
        "nwp_source": "era5",
        "change_reason": "fixed_ag_tuning_leak_era5_still_present",
    },
    {
        "gen": "gen3",
        "before": datetime.max.replace(tzinfo=UTC),
        "data_quality": "CLEAN",
        "nwp_source": "forecast",
        "change_reason": "ported_wattcast_forecast_provider",
    },
]


def _classify_run(start_time_ms: int) -> dict[str, str]:
    """Return lineage tags for a run based on its start time."""
    start = datetime.fromtimestamp(start_time_ms / 1000, tz=UTC)
    for rule in GENERATION_RULES:
        if start < rule["before"]:  # type: ignore[operator]
            return {
                "enercast.generation": str(rule["gen"]),
                "enercast.data_quality": str(rule["data_quality"]),
                "enercast.nwp_source": str(rule["nwp_source"]),
                "enercast.change_reason": str(rule["change_reason"]),
            }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MLflow runs with lineage tags")
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment names. Default: all enercast-* experiments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    if args.experiment:
        experiment_names = args.experiment
    else:
        experiments = client.search_experiments()
        experiment_names = [e.name for e in experiments if e.name.startswith("enercast-")]

    if not experiment_names:
        logger.error("No experiments found")
        return

    logger.info("Scanning experiments: %s", experiment_names)

    total_tagged = 0
    total_renamed = 0

    for exp_name in experiment_names:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning("Experiment %s not found, skipping", exp_name)
            continue

        # Get all runs (parents and children)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=5000,
        )

        for run in runs:
            existing_gen = run.data.tags.get("enercast.generation")
            if existing_gen:
                continue  # Already tagged

            tags = _classify_run(run.info.start_time)
            if not tags:
                continue

            run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
            is_leaked = tags["enercast.data_quality"] == "LEAKED"

            if args.dry_run:
                logger.info(
                    "[DRY RUN] %s/%s: %s %s%s",
                    exp_name,
                    run_name,
                    tags["enercast.generation"],
                    tags["enercast.data_quality"],
                    " → rename [LEAKED]" if is_leaked else "",
                )
            else:
                for k, v in tags.items():
                    client.set_tag(run.info.run_id, k, v)

                if is_leaked and not run_name.startswith("[LEAKED]"):
                    client.set_tag(run.info.run_id, "mlflow.runName", f"[LEAKED] {run_name}")
                    total_renamed += 1

                # Add annotation
                note = (
                    f"\n\n---\n**Lineage annotation** (backfilled): "
                    f"{tags['enercast.generation']}, {tags['enercast.data_quality']}, "
                    f"nwp={tags['enercast.nwp_source']}"
                )
                existing_note = run.data.tags.get("mlflow.note.content", "")
                client.set_tag(run.info.run_id, "mlflow.note.content", existing_note + note)

            total_tagged += 1

    action = "Would tag" if args.dry_run else "Tagged"
    logger.info("%s %d runs (%d renamed with [LEAKED] prefix)", action, total_tagged, total_renamed)


if __name__ == "__main__":
    main()
