#!/usr/bin/env python3
"""Shared data models and input utilities for the family PSO optimizer."""

from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
	profiles: Dict[str, PersonaProfile]
	position: List[int]
	best_position: List[int]
	best_selection: List[int] = field(default_factory=list)
	best_fitness: float = float("inf")


@dataclass
class IterationRecord:
	iteration: int
	persona: str
	selection: List[int]
	total_price: float
	penalty: float
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
			# Filter by country availability and valid numeric price.
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

			# Convert each valid row into a normalized OfferItem.
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

	# Keep the cheapest candidates and reindex to dense ids.
	offers.sort(key=lambda x: x.amount)
	trimmed = offers[:max_candidates]
	for idx, item in enumerate(trimmed):
		item.item_id = idx
	return trimmed

def keyword_score(item: OfferItem, keywords: Sequence[str]) -> float:
	# Count how many profile keywords appear in product/category text.
	text = normalize(f"{item.product} {item.category}")
	score = 0.0
	for kw in keywords:
		if kw in text:
			score += 1.0
	return score


def health_score(item: OfferItem) -> float:
	# Lightweight health proxy from product/category keywords.
  from config import HEALTHY_KEYWORDS

  return keyword_score(item, HEALTHY_KEYWORDS)

def sigmoid(value: float) -> float:
    # Numerically stable sigmoid used for position-to-probability mapping.
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def top_indices(values: Sequence[float], k: int) -> List[int]:
    # Return indices of the largest k values.
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:k]]

def selection_to_vector(selection: Sequence[int], dimension: int) -> List[float]:
    # Convert selected ids to a dense one-hot-like vector in optimizer space.
    vector = [0.0] * dimension
    for idx in selection:
        if 0 <= idx < dimension:
            vector[idx] = 1.0
    return vector

def shortlist_from_position(
    position: Sequence[int],
    profiles: Sequence[PersonaProfile],
    items: Sequence[OfferItem],
    shortlist_size: int,
) -> List[int]:
    # Rank the combined family position by frequency, family preferences, and price.
    key_counts: Dict[Tuple[str, str], int] = {}
    for item in items:
        key = (normalize(item.product), normalize(item.category))
        key_counts[key] = key_counts.get(key, 0) + 1

    scores = []
    for idx, item in enumerate(items):
        key = (normalize(item.product), normalize(item.category))
        duplicate_penalty = 0.02 * max(0, key_counts.get(key, 1) - 1)
        position_frequency = position.count(idx)
        preference_score = sum(keyword_score(item, profile.keywords) for profile in profiles)
        score = 0.8 * position_frequency + 0.4 * preference_score - 0.03 * item.amount - duplicate_penalty
        scores.append(score)
    return top_indices(scores, shortlist_size)
