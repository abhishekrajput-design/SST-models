"""
Download agent recordings from S3 URLs in a JSON file.

Usage:
    python download_recordings.py --json ../Agents-recoding/adil.json
    python download_recordings.py --json ../Agents-recoding/adil.json --output data/raw_calls
    python download_recordings.py --json ../Agents-recoding/adil.json --all

The JSON file must be an array of objects with an "s3_url" field, e.g.:
    [{"s3_url": "https://..."}, ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def download_file(url: str, dest_path: str) -> bool:
    """Download a single file from a URL with progress bar. Returns True on success."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if HAS_TQDM and total:
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path), leave=False)
        else:
            bar = None

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if bar:
                        bar.update(len(chunk))

        if bar:
            bar.close()
        return True

    except requests.RequestException as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def url_to_filename(url: str) -> str:
    """Extract filename from URL."""
    path = urlparse(url).path
    return os.path.basename(path)


def load_json_files(json_paths: list) -> list:
    """Load and merge entries from one or more JSON files."""
    all_entries = []
    for path in json_paths:
        if not os.path.exists(path):
            print(f"WARNING: JSON file not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            print(f"WARNING: {path} is not a JSON array, skipping")
            continue
        # Tag each entry with its source agent name
        agent_name = Path(path).stem
        for entry in entries:
            entry["_agent"] = agent_name
        all_entries.extend(entries)
        print(f"  Loaded {len(entries)} URLs from {path}")
    return all_entries


def main():
    agents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Agents-recoding")
    all_json_files = [
        os.path.join(agents_dir, f)
        for f in os.listdir(agents_dir) if f.endswith(".json")
    ] if os.path.isdir(agents_dir) else []

    parser = argparse.ArgumentParser(
        description="Download agent recordings from S3 URLs in a JSON file"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--json", "-j",
        nargs="+",
        metavar="FILE",
        help="Path(s) to agent JSON file(s) containing s3_url entries",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Download all agent JSON files from Agents-recoding/ ({len(all_json_files)} found)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/raw_calls",
        help="Directory to save downloaded files (default: data/raw_calls)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of files to download (useful for testing, default: all)",
    )
    args = parser.parse_args()

    json_files = all_json_files if args.all else args.json
    if not json_files:
        print("ERROR: No JSON files found.")
        sys.exit(1)

    print(f"\nLoading recording URLs...")
    entries = load_json_files(json_files)

    if not entries:
        print("No entries found in the provided JSON files.")
        sys.exit(1)

    if args.limit:
        entries = entries[: args.limit]
        print(f"Limited to {args.limit} files")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDownloading {len(entries)} recording(s) to: {output_dir}")
    print("-" * 60)

    downloaded, skipped, failed = [], [], []

    for i, entry in enumerate(entries, 1):
        url = entry.get("s3_url") or entry.get("url")
        if not url:
            print(f"  [{i}/{len(entries)}] Skipping entry with no s3_url field")
            skipped.append(entry)
            continue

        filename = url_to_filename(url)
        agent = entry.get("_agent", "")
        dest_path = os.path.join(output_dir, filename)

        if os.path.exists(dest_path):
            size_kb = os.path.getsize(dest_path) / 1024
            print(f"  [{i}/{len(entries)}] SKIP  {filename} ({size_kb:.0f}KB already exists)")
            skipped.append(dest_path)
            continue

        print(f"  [{i}/{len(entries)}] DL    {filename}{' (' + agent + ')' if agent else ''}")
        ok = download_file(url, dest_path)
        if ok:
            size_kb = os.path.getsize(dest_path) / 1024
            print(f"         -> {size_kb:.0f}KB saved")
            downloaded.append(dest_path)
        else:
            failed.append(url)

    print(f"\n{'='*60}")
    print(f"  Downloaded : {len(downloaded)}")
    print(f"  Skipped    : {len(skipped)}")
    print(f"  Failed     : {len(failed)}")
    print(f"{'='*60}")

    if downloaded:
        print(f"\nFiles ready for processing in: {output_dir}")
        print("Run pipeline with:")
        for path in downloaded[:3]:
            print(f"  python main.py --input {path}")
        if len(downloaded) > 3:
            print(f"  ... and {len(downloaded) - 3} more")

    if failed:
        print(f"\nFailed URLs:")
        for url in failed:
            print(f"  {url}")


if __name__ == "__main__":
    main()
