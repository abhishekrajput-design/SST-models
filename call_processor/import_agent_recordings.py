"""
Import agent enrollment recordings from JSON manifests.

Each manifest file in the source directory should be named <agent>.json and
contain an array of objects with an `s3_url` field.

Example:
    python import_agent_recordings.py
    python import_agent_recordings.py --source-dir ..\\Agents-recoding --limit 3
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT.parent / "Agents-recoding"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "agent_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import agent recordings from manifest JSON files."
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing <agent>.json manifest files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where agent_<name>/ audio files will be created.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum recordings to import per agent.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already present on disk.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    urls = []
    for item in payload:
        url = item.get("s3_url")
        if url:
            urls.append(url)
    return urls


def filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Could not derive filename from URL: {url}")
    return name


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        output.write(response.read())


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source_dir.is_dir():
        print(f"ERROR: Source directory not found: {source_dir}")
        return 1

    manifest_paths = sorted(source_dir.glob("*.json"))
    if not manifest_paths:
        print(f"ERROR: No manifest files found in: {source_dir}")
        return 1

    print(f"Source manifests: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(manifest_paths)} manifest files")

    total_downloaded = 0
    total_skipped = 0

    for manifest_path in manifest_paths:
        agent_name = manifest_path.stem.strip().lower()
        target_dir = output_dir / f"agent_{agent_name}"
        urls = load_manifest(manifest_path)
        if args.limit is not None:
            urls = urls[: args.limit]

        print(f"\n[{agent_name}] {len(urls)} recordings")
        target_dir.mkdir(parents=True, exist_ok=True)

        for index, url in enumerate(urls, start=1):
            filename = filename_from_url(url)
            destination = target_dir / filename

            if args.skip_existing and destination.exists():
                total_skipped += 1
                print(f"  {index:02d}. skip {destination.name}")
                continue

            print(f"  {index:02d}. download {destination.name}")
            try:
                download_file(url, destination)
                total_downloaded += 1
            except Exception as exc:
                print(f"      failed: {exc}")

    print("\nImport complete")
    print(f"Downloaded: {total_downloaded}")
    print(f"Skipped: {total_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
