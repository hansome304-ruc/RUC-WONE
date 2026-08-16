#!/usr/bin/env python3
"""Launch the private medicine-packaging console."""
from pathlib import Path

from medicine_agentic.packaging_console import main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        main(default_config=ROOT / "configs" / "packaging_console.json")
    )
