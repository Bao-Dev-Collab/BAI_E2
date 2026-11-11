from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}
SPLITS = ("train", "val", "test")
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_cross_split_duplicates(
    data_dir: Path,
) -> tuple[dict[str, list[tuple[str, Path]]], dict[str, int]]:
    files_by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    file_counts: dict[str, int] = {}

    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}. "
                "Expected train, val, and test directories."
            )

        image_paths = sorted(
            path
            for path in split_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No image files found in split: {split_dir}")

        file_counts[split] = len(image_paths)
        for path in image_paths:
            files_by_hash[sha256_file(path)].append((split, path))

    duplicates = {
        digest: entries
        for digest, entries in files_by_hash.items()
        if len({split for split, _ in entries}) > 1
    }
    return duplicates, file_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find byte-identical image files shared across train, val, and test. "
            "Re-encoded or visually similar images are not detected."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root containing train, val, and test directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    duplicates, file_counts = find_cross_split_duplicates(args.data_dir)

    for split in SPLITS:
        print(f"{split}: hashed {file_counts[split]} image files")

    if not duplicates:
        print("No byte-identical images found across splits.")
        return 0

    print(f"Found {len(duplicates)} duplicate content group(s) across splits:")
    for digest, entries in sorted(duplicates.items()):
        print(f"\nSHA-256: {digest}")
        for split, path in sorted(entries, key=lambda entry: (entry[0], str(entry[1]))):
            print(f"  {split}: {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
