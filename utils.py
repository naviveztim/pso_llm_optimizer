#!/usr/bin/env python3
"""Shared data models and input utilities for the family PSO optimizer."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class OfferItem:
    item_id: int
    product: str
    category: str
    amount: float
    currency: str


@dataclass
class PersonaProfile:
    name: str
    role: str
    keywords: List[str]
    basket_size: int
    min_preferred_items: int


@dataclass
class Particle:
    position: List[int]
    best_selection: List[int] = field(default_factory=list)
    best_fitness: float = float("inf")
    current_fitness: float = float("inf")


@dataclass
class IterationRecord:
    iteration: int
    persona: str
    selection: List[int]
    total_price: float
    fitness: float


def normalize(text: str) -> str:
    """Canonicalize free text for robust keyword matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_offers(dataset_path: Path, country_code: str, max_candidates: int) -> List[OfferItem]:
    """Load valid offers, filter by country, and keep the cheapest candidates."""
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    categories = payload.get("categories") or {}
    offers: List[OfferItem] = []
    current_id = 0

    for category, entries in categories.items():
        for entry in entries:
            price = entry.get("price") or {}
            meta = entry.get("meta") or {}
            countries = meta.get("countries_valid") or []
            country_codes = {str(c.get("country_code")) for c in countries if isinstance(c, dict)}
            if country_code and country_code not in country_codes:
                continue

            raw_amount = price.get("amount")
            if raw_amount is None:
                continue
            try:
                amount = float(str(raw_amount))
            except (TypeError, ValueError):
                continue

            offers.append(
                OfferItem(
                    item_id=current_id,
                    product=str(entry.get("product") or "Unknown"),
                    category=str(category),
                    amount=amount,
                    currency=str(price.get("currency") or "EUR"),
                )
            )
            current_id += 1

    offers.sort(key=lambda x: x.amount)
    trimmed = offers[:max_candidates]
    for idx, item in enumerate(trimmed):
        item.item_id = idx
    return trimmed


def keyword_score(item: OfferItem, keywords: Sequence[str]) -> float:
    """Count how many configured keywords appear in product/category text."""
    text = normalize(f"{item.product} {item.category}")
    score = 0.0
    for keyword in keywords:
        if keyword in text:
            score += 1.0
    return score


def health_score(item: OfferItem) -> float:
    """Return the positive healthy-keyword signal for an item."""
    from config import HEALTHY_KEYWORDS

    return keyword_score(item, HEALTHY_KEYWORDS)


def health_penalty(item: OfferItem) -> float:
    """Return a bounded penalty where healthier keyword matches score lower."""
    from config import HEALTHY_KEYWORDS

    if not HEALTHY_KEYWORDS:
        return 0.0
    return max(0.0, 1.0 - health_score(item) / len(HEALTHY_KEYWORDS))



