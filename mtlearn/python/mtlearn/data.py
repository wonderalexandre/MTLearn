"""Dataset download helpers for mtlearn examples and notebooks."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str
    target: tuple[str, ...]
    url: str | None = None
    url_env_var: str | None = None
    local_env_var: str | None = None
    access_note: str | None = None
    download_by_default: bool = True


DATASETS: dict[str, DatasetSpec] = {
    "misc256": DatasetSpec(
        key="misc256",
        description="Small 256x256 sample images used by lightweight examples.",
        target=("misc256",),
        url=(
            "https://www.dropbox.com/scl/fo/ki7t7lkliig7vmzbi3288/"
            "AAsKiUaOsReUehU3VACZYww?rlkey=vm2hln945ytftq5kmk3qmx7mh"
        ),
    ),
    "washer_removal": DatasetSpec(
        key="washer_removal",
        description="Washer-removal image pairs.",
        target=("washer_removal",),
        url=(
            "https://www.dropbox.com/scl/fo/sgo8vztm52flhjcp0jj40/"
            "AFqr_iSdlxpwm2qmI3AV4N8?rlkey=v1e90ktw5q0u98hcw8j00dur9"
        ),
    ),
    "bushing_removal": DatasetSpec(
        key="bushing_removal",
        description="Bushing-removal image pairs.",
        target=("bushing_removal",),
        url=(
            "https://www.dropbox.com/scl/fo/sm16qo4ka55yi2u4vip91/"
            "AMwibDVfy_LW1peEMvfS-7E?rlkey=7wrxpv1858kmpkklvsco7iooo"
        ),
    ),
    "screws_segmentation": DatasetSpec(
        key="screws_segmentation",
        description="Screw-segmentation image pairs.",
        target=("screws_segmentation",),
        url=(
            "https://www.dropbox.com/scl/fo/2owxbpc8oxi7mpegcda3u/"
            "AJJ4GbKk3N2ucOTT6bnOf7g?rlkey=l24jp4tnvzbvcap2wxpc6im95"
        ),
    ),
    "plants_segmentation": DatasetSpec(
        key="plants_segmentation",
        description=(
            "Plant Phenotyping foreground/background segmentation subset used "
            "by the ICPR 2026 notebooks. Requires an authorized local copy or "
            "a review-only download URL."
        ),
        target=("PlantDataset",),
        url_env_var="MTLEARN_PLANTS_SEGMENTATION_URL",
        local_env_var="MTLEARN_PLANTS_DATASET",
        access_note=(
            "This dataset is derived from the Plant Phenotyping Dataset and is "
            "not redistributed by MTLearn. Use only an authorized local copy "
            "or an explicitly authorized review-only package URL."
        ),
        download_by_default=False,
    ),
}


def _looks_like_repo_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "mtlearn" / "python" / "mtlearn").is_dir()
    )


def repo_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _looks_like_repo_root(candidate):
            return candidate
    return None


def default_data_dir(start: Path | None = None) -> Path:
    if data_dir := os.environ.get("MTLEARN_DATA_DIR"):
        return Path(data_dir).expanduser()
    if root := repo_root(start):
        return root / "dat"
    return Path.cwd() / "dat"


def dataset_path(key: str, data_dir: Path | None = None) -> Path:
    spec = DATASETS[key]
    root = (data_dir or default_data_dir()).expanduser()
    return root / Path(*spec.target)


def _local_dataset_path(spec: DatasetSpec) -> Path | None:
    if spec.local_env_var and (env_value := os.environ.get(spec.local_env_var)):
        candidate = Path(env_value).expanduser()
        if has_existing_files(candidate):
            return candidate.resolve()
    return None


def _configured_download_url(
    spec: DatasetSpec,
    url_override: str | None = None,
) -> str | None:
    if url_override:
        return url_override
    if spec.url_env_var and (env_value := os.environ.get(spec.url_env_var)):
        return env_value
    return spec.url


def _missing_download_url_message(spec: DatasetSpec, target_dir: Path) -> str:
    hints = []
    if spec.local_env_var:
        hints.append(f"set {spec.local_env_var} to an authorized local copy")
    if spec.url_env_var:
        hints.append(f"set {spec.url_env_var}")
    hints.append("pass --url to scripts/download_data.py or python -m mtlearn.data")
    hint_text = "; ".join(hints)
    message = (
        f"{spec.key} is not available at {target_dir} and no download URL is "
        f"configured. To use this dataset, {hint_text}."
    )
    if spec.access_note:
        message = f"{message} {spec.access_note}"
    return message


def require_local_dataset(
    name: str,
    data_dir: Path | PathLike[str] | str | None = None,
    *,
    env_var: str | None = None,
    description: str | None = None,
) -> Path:
    """Resolve a local dataset that is not part of the public download registry."""

    candidates: list[Path] = []
    if env_var and (env_value := os.environ.get(env_var)):
        candidates.append(Path(env_value).expanduser())

    root = (
        Path(data_dir).expanduser()
        if data_dir is not None
        else default_data_dir().expanduser()
    )
    candidates.append(root / name)

    for candidate in candidates:
        if has_existing_files(candidate):
            return candidate.resolve()

    label = description or name
    expected_locations = ", ".join(str(candidate) for candidate in candidates)
    env_hint = f"Set {env_var} or " if env_var else ""
    raise FileNotFoundError(
        f"{label} is not available. {env_hint}"
        f"place it at one of: {expected_locations}"
    )


def dropbox_download_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    query["dl"] = "1"
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        dropbox_download_url(url),
        headers={"User-Agent": "mtlearn-dataset-downloader/1.0"},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length else None
        downloaded = 0
        last_reported = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = int(downloaded * 100 / total)
                if percent // 10 != last_reported // 10:
                    print(f"  downloaded {percent:3d}% ({format_size(downloaded)} / {format_size(total)})")
                    last_reported = percent
        print(f"  downloaded {format_size(downloaded)}")


def extracted_content_root(extract_dir: Path) -> Path:
    entries = [entry for entry in extract_dir.iterdir() if entry.name != "__MACOSX"]
    directories = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    if len(directories) == 1 and not files:
        return directories[0]
    return extract_dir


def replace_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(source), str(destination))


def has_existing_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def ensure_dataset(
    key: str,
    data_dir: Path | None = None,
    *,
    force: bool = False,
    keep_archive: bool = False,
    url: str | None = None,
) -> Path:
    spec = DATASETS[key]
    root = (data_dir or default_data_dir()).expanduser().resolve()
    target_dir = root / Path(*spec.target)
    if not force and (local_path := _local_dataset_path(spec)):
        print(f"{spec.key}: using {spec.local_env_var} at {local_path}")
        return local_path
    if has_existing_files(target_dir) and not force:
        print(f"{spec.key}: already available at {target_dir}")
        return target_dir

    download_url = _configured_download_url(spec, url)
    if download_url is None:
        raise FileNotFoundError(_missing_download_url_message(spec, target_dir))

    root.mkdir(parents=True, exist_ok=True)
    print(f"{spec.key}: downloading to {target_dir}")
    with tempfile.TemporaryDirectory(prefix=f"mtlearn-{spec.key}-") as tmp_name:
        tmp_dir = Path(tmp_name)
        archive = tmp_dir / f"{spec.key}.zip"
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()

        download_file(download_url, archive)
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(f"Downloaded file for {spec.key} is not a zip archive")

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)

        content_root = extracted_content_root(extract_dir)
        replace_directory(content_root, target_dir)

        if keep_archive:
            archive_destination = root / f"{spec.key}.zip"
            shutil.copy2(archive, archive_destination)
            print(f"{spec.key}: archive kept at {archive_destination}")

    print(f"{spec.key}: ready at {target_dir}")
    return target_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASETS),
        help="Dataset keys to download. Use --list to see descriptions.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every public registered dataset.",
    )
    parser.add_argument(
        "--include-restricted",
        action="store_true",
        help="Include restricted datasets when using --all. Their URLs must be configured.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered datasets and exit.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Destination data directory. Defaults to MTLEARN_DATA_DIR or ./dat.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and replace existing target directories.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip archive next to the extracted data.",
    )
    parser.add_argument(
        "--url",
        help="Override the download URL for a single selected dataset.",
    )
    return parser.parse_args(argv)


def list_datasets(data_dir: Path | None = None) -> None:
    root = (data_dir or default_data_dir()).expanduser().resolve()
    print(f"Data directory: {root}")
    for key in sorted(DATASETS):
        spec = DATASETS[key]
        target = root / Path(*spec.target)
        local_path = _local_dataset_path(spec)
        status = "present" if local_path or has_existing_files(target) else "missing"
        print(f"{key:22s} {status:8s} {target}")
        print(f"  {spec.description}")
        if local_path:
            print(f"  Local override: {spec.local_env_var}={local_path}")
        if spec.url is None and spec.url_env_var:
            print(f"  Download URL: set {spec.url_env_var} or pass --url.")
        if spec.access_note:
            print(f"  Note: {spec.access_note}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    data_dir = args.data_dir.expanduser().resolve()

    if args.list:
        list_datasets(data_dir)
        return 0

    if args.url and (args.all or len(args.datasets) != 1):
        print("--url can only be used with exactly one selected dataset.", file=sys.stderr)
        return 2

    if args.all:
        selected = sorted(
            key
            for key, spec in DATASETS.items()
            if args.include_restricted or spec.download_by_default
        )
    else:
        selected = args.datasets
    if not selected:
        print("No dataset selected. Use --list, --all, or pass one or more dataset keys.", file=sys.stderr)
        return 2

    for key in selected:
        try:
            ensure_dataset(
                key,
                data_dir,
                force=args.force,
                keep_archive=args.keep_archive,
                url=args.url,
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
