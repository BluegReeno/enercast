"""Tests for lineage tag logging."""

from __future__ import annotations

import mlflow

from windcast.training.lineage import get_git_info, log_lineage_tags


def test_get_git_info_returns_dict():
    """get_git_info should return git branch and dirty state when run from a git repo."""
    info = get_git_info()
    assert isinstance(info, dict)
    # We're running inside the windcast repo, so these should be present
    assert "mlflow.source.git.branch" in info
    assert "enercast.git.dirty" in info
    assert info["enercast.git.dirty"] in ("true", "false")


def test_log_lineage_tags_sets_tags(tmp_path):
    """log_lineage_tags should set enercast.* and git tags on the active run."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_test.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-lineage")

    with mlflow.start_run() as run:
        log_lineage_tags(
            generation="gen4",
            nwp_source="forecast",
            data_quality="CLEAN",
            change_reason="test_run",
        )

    client = mlflow.MlflowClient()
    run_data = client.get_run(run.info.run_id)
    tags = run_data.data.tags

    assert tags["enercast.generation"] == "gen4"
    assert tags["enercast.nwp_source"] == "forecast"
    assert tags["enercast.data_quality"] == "CLEAN"
    assert tags["enercast.change_reason"] == "test_run"
    assert "mlflow.source.git.branch" in tags
    assert "enercast.git.dirty" in tags


def test_log_lineage_tags_optional_fields(tmp_path):
    """Optional fields (generation, change_reason) should be skipped when None."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_test.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-lineage-optional")

    with mlflow.start_run() as run:
        log_lineage_tags()  # all defaults

    client = mlflow.MlflowClient()
    run_data = client.get_run(run.info.run_id)
    tags = run_data.data.tags

    assert "enercast.generation" not in tags
    assert "enercast.change_reason" not in tags
    assert tags["enercast.nwp_source"] == "forecast"
    assert tags["enercast.data_quality"] == "CLEAN"
