from pathlib import Path
import urllib.parse

import pytest

from mtlearn import data


def test_repo_root_detects_current_source_layout(tmp_path):
    repo = tmp_path / "repo"
    package_dir = repo / "mtlearn" / "python" / "mtlearn"
    package_dir.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'mtlearn'\n")

    nested = package_dir / "layers"
    nested.mkdir()

    assert data.repo_root(nested) == repo


def test_default_data_dir_prefers_environment_variable(monkeypatch, tmp_path):
    custom_data_dir = tmp_path / "custom-data"
    monkeypatch.setenv("MTLEARN_DATA_DIR", str(custom_data_dir))

    assert data.default_data_dir() == custom_data_dir


def test_dataset_path_uses_registered_target_and_explicit_root(tmp_path):
    assert data.dataset_path("misc256", tmp_path) == tmp_path / "misc256"
    assert data.dataset_path("screws_segmentation", tmp_path) == tmp_path / "screws_segmentation"
    assert data.dataset_path("plants_segmentation", tmp_path) == tmp_path / "PlantDataset"


def test_dropbox_download_url_forces_direct_download():
    url = (
        "https://www.dropbox.com/scl/fo/example/token"
        "?rlkey=abc&dl=0&extra=value"
    )

    direct = data.dropbox_download_url(url)
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(direct).query))

    assert query["rlkey"] == "abc"
    assert query["extra"] == "value"
    assert query["dl"] == "1"


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (None, "unknown size"),
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1024 * 1024 * 1024, "1.0 GB"),
    ],
)
def test_format_size(num_bytes, expected):
    assert data.format_size(num_bytes) == expected


def test_extracted_content_root_collapses_single_directory(tmp_path):
    extract_dir = tmp_path / "extract"
    single_root = extract_dir / "dataset"
    single_root.mkdir(parents=True)

    assert data.extracted_content_root(extract_dir) == single_root


def test_extracted_content_root_keeps_mixed_content(tmp_path):
    extract_dir = tmp_path / "extract"
    (extract_dir / "dataset").mkdir(parents=True)
    (extract_dir / "metadata.txt").write_text("metadata")

    assert data.extracted_content_root(extract_dir) == extract_dir


def test_require_local_dataset_prefers_env_var(monkeypatch, tmp_path):
    env_dataset = tmp_path / "env-dataset"
    env_dataset.mkdir()
    (env_dataset / "sample.txt").write_text("ok")
    monkeypatch.setenv("MTLEARN_PRIVATE_DATASET", str(env_dataset))

    assert data.require_local_dataset(
        "missing-under-root",
        tmp_path,
        env_var="MTLEARN_PRIVATE_DATASET",
    ) == env_dataset.resolve()


def test_require_local_dataset_reports_expected_locations(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        data.require_local_dataset(
            "private-set",
            tmp_path,
            env_var="MTLEARN_PRIVATE_DATASET",
            description="private validation set",
        )

    message = str(excinfo.value)
    assert "private validation set is not available" in message
    assert "Set MTLEARN_PRIVATE_DATASET" in message
    assert str(tmp_path / "private-set") in message


def test_restricted_dataset_prefers_authorized_local_copy(monkeypatch, tmp_path):
    env_dataset = tmp_path / "plants-local"
    env_dataset.mkdir()
    (env_dataset / "sample_tfb.png").write_text("ok")
    monkeypatch.setenv("MTLEARN_PLANTS_DATASET", str(env_dataset))

    assert data.ensure_dataset("plants_segmentation", tmp_path) == env_dataset.resolve()


def test_restricted_dataset_requires_local_copy_or_url(monkeypatch, tmp_path):
    monkeypatch.delenv("MTLEARN_PLANTS_DATASET", raising=False)
    monkeypatch.delenv("MTLEARN_PLANTS_SEGMENTATION_URL", raising=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        data.ensure_dataset("plants_segmentation", tmp_path)

    message = str(excinfo.value)
    assert "plants_segmentation is not available" in message
    assert "MTLEARN_PLANTS_DATASET" in message
    assert "MTLEARN_PLANTS_SEGMENTATION_URL" in message
    assert "--url" in message


def test_configured_download_url_prefers_override_then_environment(monkeypatch):
    spec = data.DATASETS["plants_segmentation"]
    monkeypatch.setenv("MTLEARN_PLANTS_SEGMENTATION_URL", "https://example.test/env.zip")

    assert data._configured_download_url(spec) == "https://example.test/env.zip"
    assert data._configured_download_url(spec, "https://example.test/override.zip") == (
        "https://example.test/override.zip"
    )


def test_cli_rejects_url_for_multiple_datasets(capsys):
    result = data.main(["misc256", "screws_segmentation", "--url", "https://example.test/data.zip"])

    captured = capsys.readouterr()
    assert result == 2
    assert "--url can only be used with exactly one selected dataset" in captured.err


def test_cli_reports_restricted_dataset_without_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("MTLEARN_PLANTS_DATASET", raising=False)
    monkeypatch.delenv("MTLEARN_PLANTS_SEGMENTATION_URL", raising=False)

    result = data.main(["plants_segmentation", "--data-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert "plants_segmentation is not available" in captured.err
    assert "MTLEARN_PLANTS_SEGMENTATION_URL" in captured.err
