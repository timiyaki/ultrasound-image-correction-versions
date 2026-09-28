"""Rebuild the source-only v28/v29 ZIPs with UTF-8 filenames."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
SAFE_SUFFIXES = {".py", ".md", ".txt", ".bat", ".json", ".code-workspace", ".patch"}


for version in ("v28", "v29"):
    source = ROOT / "versions" / version
    destination = ROOT / "packages" / f"{version}.zip"
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not files or any(path.suffix.lower() not in SAFE_SUFFIXES for path in files):
        raise RuntimeError(f"Unexpected file in {source}")
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT / "versions").as_posix())
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Invalid ZIP: {destination}")
    print(destination.name, len(files), destination.stat().st_size)
