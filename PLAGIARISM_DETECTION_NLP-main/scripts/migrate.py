"""Apply ordered SourceLens SQL migrations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism_detection.storage import PostgresCorpusRepository  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL DSN; defaults to DATABASE_URL",
    )
    parser.add_argument("--migrations", type=Path, default=ROOT / "migrations")
    arguments = parser.parse_args()
    if not arguments.database_url:
        parser.error("--database-url or DATABASE_URL is required")

    repository = PostgresCorpusRepository(arguments.database_url)
    repository.apply_migrations(arguments.migrations)
    print(f"Applied migrations from {arguments.migrations}")


if __name__ == "__main__":
    main()
