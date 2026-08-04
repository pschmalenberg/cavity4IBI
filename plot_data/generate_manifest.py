from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATED_FILES = {"manifest.csv", "checksums.sha256"}


def digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in GENERATED_FILES
        and "__pycache__" not in path.parts
    )
    rows = [
        (path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path))
        for path in files
    ]

    with (ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "bytes", "sha256"))
        writer.writerows(rows)

    with (ROOT / "checksums.sha256").open("w", encoding="ascii", newline="\n") as stream:
        for relative_path, _, sha256 in rows:
            stream.write(f"{sha256}  {relative_path}\n")

    print(f"Hashed {len(rows)} files")


if __name__ == "__main__":
    main()