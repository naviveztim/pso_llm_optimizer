#!/usr/bin/env python3
"""Smoke test for family PSO optimizer output."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "data" / "family_pso_plan.json"


def main() -> None:
    if not OUTPUT.exists():
        raise SystemExit(f"Missing output file: {OUTPUT}")

    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    proposals = payload.get("best_proposals_by_persona")

    if not isinstance(metadata, dict):
        raise SystemExit("Invalid output: metadata missing")
    if not isinstance(proposals, dict) or len(proposals) != 5:
        raise SystemExit("Invalid output: expected 5 persona proposals")

    for persona, data in proposals.items():
        if "fitness" not in data or "items" not in data:
            raise SystemExit(f"Invalid proposal shape for {persona}")
        if not isinstance(data["items"], list):
            raise SystemExit(f"Invalid items list for {persona}")

    print("Family PSO smoke test passed")
    print(f"Iterations completed: {metadata.get('iterations_completed')}")
    print(f"Global best fitness: {metadata.get('global_best_fitness')}")


if __name__ == "__main__":
    main()

