#!/usr/bin/env python3
"""Skeleton: hook GraspSkills up to a tool-calling LLM. Fill in your provider.

Tool surface advertised to the LLM:
    - capture_image()
    - find_object(prompt)
    - grasp_object(prompt, arm)
    - open_gripper(arm) / close_gripper(arm)
    - home()
"""
from __future__ import annotations

import argparse
import logging
import sys

from agentic_grasp.skills import GraspSkills


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="Pick up the bottle on the table.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    skills = GraspSkills(connect_arms=not args.dry_run)  # noqa: F841
    raise SystemExit(
        "TODO: paste your Anthropic / OpenAI / Qwen tool-calling loop here. "
        "See README in this repo for a worked Anthropic example."
    )


if __name__ == "__main__":
    sys.exit(main())
