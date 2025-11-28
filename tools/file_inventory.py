#!/usr/bin/env python3
"""
File Inventory / Codebase Summary Script

Run this from the project root:

    python tools/file_inventory.py
    python tools/file_inventory.py --include-hidden
    python tools/file_inventory.py --no-md --no-json

Outputs:
    - Human-readable summary to stdout
    - (Default) JSON file: data/file_inventory.json
    - (Default) Markdown file: docs/file_inventory.md

This script is meant to complement FILE_DICTIONARY_GUIDE.md and
help agents understand file sizes, line counts, and folder structure.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable


# -----------------------------
# Configuration / Ignore Lists
# -----------------------------

# Directories to skip entirely
DEFAULT_IGNORED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "dist",
    "build",
}

# File extensions to treat as "code" (for per-dir code line counts).
CODE_EXTENSIONS = {".py", ".sh", ".yml", ".yaml", ".toml", ".json", ".md"}


@dataclass
class FileInfo:
    path: str         # relative path
    extension: str
    size_bytes: int
    line_count: int


@dataclass
class DirSummary:
    path: str                      # relative dir path
    total_files: int
    total_size_bytes: int
    total_code_files: int          # with extension in CODE_EXTENSIONS
    total_code_lines: int          # sum of line_count for code files
    total_py_files: int
    total_py_lines: int


# -----------------------------
# Core Functions
# -----------------------------


def iter_files(
    root: Path,
    include_hidden: bool = False,
    ignored_dirs: Iterable[str] = DEFAULT_IGNORED_DIRS,
) -> Iterable[Path]:
    """Yield all file paths under `root`, respecting ignore rules."""
    ignored_dirs = set(ignored_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_path = Path(dirpath)

        # Filter out ignored directories in-place to prevent descending into them
        dirnames[:] = [
            d for d in dirnames
            if (include_hidden or not d.startswith("."))
            and d not in ignored_dirs
        ]

        # Skip ignored dirs at root-level too
        if dirpath_path.name in ignored_dirs:
            continue

        for fname in filenames:
            if not include_hidden and fname.startswith("."):
                continue
            yield dirpath_path / fname


def count_lines(path: Path) -> int:
    """Count lines in a text file; return 0 on binary/unreadable errors."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def collect_file_info(
    root: Path,
    include_hidden: bool = False,
    ignored_dirs: Iterable[str] = DEFAULT_IGNORED_DIRS,
) -> List[FileInfo]:
    """Collect FileInfo for all files under root."""
    infos: List[FileInfo] = []
    for fpath in iter_files(root, include_hidden=include_hidden, ignored_dirs=ignored_dirs):
        rel_path = fpath.relative_to(root).as_posix()
        ext = fpath.suffix.lower()
        size_bytes = fpath.stat().st_size
        line_count = count_lines(fpath) if ext in CODE_EXTENSIONS or ext == ".py" else 0
        infos.append(
            FileInfo(
                path=rel_path,
                extension=ext,
                size_bytes=size_bytes,
                line_count=line_count,
            )
        )
    return infos


def summarize_by_directory(root: Path, files: List[FileInfo]) -> List[DirSummary]:
    """Aggregate FileInfo into per-directory summaries."""
    by_dir: Dict[str, List[FileInfo]] = {}
    for fi in files:
        dir_path = str(Path(fi.path).parent)
        if dir_path == ".":
            dir_path = ""  # root
        by_dir.setdefault(dir_path, []).append(fi)

    summaries: List[DirSummary] = []
    for dir_path, fis in sorted(by_dir.items(), key=lambda kv: kv[0]):
        total_files = len(fis)
        total_size = sum(fi.size_bytes for fi in fis)
        code_files = [fi for fi in fis if fi.extension in CODE_EXTENSIONS]
        py_files = [fi for fi in fis if fi.extension == ".py"]
        total_code_files = len(code_files)
        total_code_lines = sum(fi.line_count for fi in code_files)
        total_py_files = len(py_files)
        total_py_lines = sum(fi.line_count for fi in py_files)

        summaries.append(
            DirSummary(
                path=dir_path or ".",
                total_files=total_files,
                total_size_bytes=total_size,
                total_code_files=total_code_files,
                total_code_lines=total_code_lines,
                total_py_files=total_py_files,
                total_py_lines=total_py_lines,
            )
        )
    return summaries


# -----------------------------
# Output Helpers
# -----------------------------


def print_summary(root: Path, files: List[FileInfo], summaries: List[DirSummary]) -> None:
    """Print a human-readable summary to stdout."""
    total_files = len(files)
    total_size = sum(fi.size_bytes for fi in files)
    total_py_files = sum(1 for fi in files if fi.extension == ".py")
    total_py_lines = sum(fi.line_count for fi in files if fi.extension == ".py")

    print("=" * 70)
    print(f"File Inventory for: {root}")
    print("=" * 70)
    print(f"Total files:        {total_files}")
    print(f"Total size:         {total_size:,} bytes")
    print(f"Total .py files:    {total_py_files}")
    print(f"Total .py lines:    {total_py_lines:,}")
    print()

    print("Per-directory summary (top 25 by Python lines):")
    print("-" * 70)
    # Sort by total_py_lines descending
    for ds in sorted(summaries, key=lambda d: d.total_py_lines, reverse=True)[:25]:
        print(
            f"{ds.path or '.':40} | "
            f"files={ds.total_files:4d} | "
            f"py_files={ds.total_py_files:4d} | "
            f"py_lines={ds.total_py_lines:7d}"
        )
    print()


def write_json(root: Path, files: List[FileInfo], summaries: List[DirSummary], path: Path) -> None:
    """Write file + directory info to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "root": str(root),
        "files": [asdict(fi) for fi in files],
        "directories": [asdict(ds) for ds in summaries],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[JSON] Inventory written to {path}")


def write_markdown(root: Path, files: List[FileInfo], summaries: List[DirSummary], path: Path) -> None:
    """Write a markdown overview (suitable for docs/)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# File Inventory for `{root.name}`\n\n")
        f.write("Generated by `tools/file_inventory.py`.\n\n")

        total_files = len(files)
        total_size = sum(fi.size_bytes for fi in files)
        total_py_files = sum(1 for fi in files if fi.extension == ".py")
        total_py_lines = sum(fi.line_count for fi in files if fi.extension == ".py")

        f.write("## Summary\n\n")
        f.write(f"- Root: `{root}`\n")
        f.write(f"- Total files: `{total_files}`\n")
        f.write(f"- Total size: `{total_size:,}` bytes\n")
        f.write(f"- Python files: `{total_py_files}`\n")
        f.write(f"- Python lines: `{total_py_lines:,}`\n\n")

        f.write("## Per-directory Summary (sorted by Python lines)\n\n")
        f.write("| Directory | Files | Python Files | Python Lines |\n")
        f.write("|-----------|-------|--------------|--------------|\n")
        for ds in sorted(summaries, key=lambda d: d.total_py_lines, reverse=True):
            f.write(
                f"| `{ds.path or '.'}` | {ds.total_files} | "
                f"{ds.total_py_files} | {ds.total_py_lines} |\n"
            )
        f.write("\n")

        f.write("## Largest Python Files (by line count)\n\n")
        py_files = [fi for fi in files if fi.extension == ".py"]
        py_files_sorted = sorted(py_files, key=lambda fi: fi.line_count, reverse=True)
        f.write("| File | Lines | Size (bytes) |\n")
        f.write("|------|-------|--------------|\n")
        for fi in py_files_sorted[:50]:
            f.write(
                f"| `{fi.path}` | {fi.line_count} | {fi.size_bytes} |\n"
            )

    print(f"[MD] Inventory written to {path}")


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate file inventory for the project.")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories (starting with '.').",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write JSON output.",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        help="Do not write Markdown output.",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/file_inventory.json",
        help="Path for JSON output (default: data/file_inventory.json).",
    )
    parser.add_argument(
        "--md-path",
        type=str,
        default="docs/file_inventory.md",
        help="Path for Markdown output (default: docs/file_inventory.md).",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = collect_file_info(root, include_hidden=args.include_hidden)
    summaries = summarize_by_directory(root, files)

    print_summary(root, files, summaries)

    if not args.no_json:
        write_json(root, files, summaries, Path(args.json_path))
    if not args.no_md:
        write_markdown(root, files, summaries, Path(args.md_path))


if __name__ == "__main__":
    main()
