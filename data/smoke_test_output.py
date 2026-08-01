#!/usr/bin/env python3
"""Simple validation for the generated Kaufland JSON dataset."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "kaufland_prices_by_category_various_countries.json"


def main() -> None:
    if not OUTPUT.exists():
        raise SystemExit(f"Output not found: {OUTPUT}")

    data = json.loads(OUTPUT.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    categories = data.get("categories", {})

    if not isinstance(categories, dict) or not categories:
        raise SystemExit("Validation failed: categories section is missing or empty")

    total = sum(len(items) for items in categories.values())
    if total <= 0:
        raise SystemExit("Validation failed: no products found")

    print("Validation OK")
    print(f"Products: {total}")
    print(f"Categories: {len(categories)}")
    print(f"Collected at: {metadata.get('collected_at_utc')}")


if __name__ == "__main__":
    main()

