#!/usr/bin/env python3
"""Collect Kaufland offer prices and store them in categorized JSON."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "kaufland_prices_by_category_various_countries.json"
DEFAULT_LIMIT = 1000


@dataclass
class SourceConfig:
    page_url: str
    country_code: str
    country_name: str
    currency: str


# This list can be extended with additional country pages when needed.
SOURCES: List[SourceConfig] = [
    # SourceConfig(
    #     page_url="https://www.kaufland.ro/oferte/prezentare-generala-oferte.html",
    #     country_code="RO",
    #     country_name="Romania",
    #     currency="RON",
    # ),
    # SourceConfig(
    #     page_url="https://www.kaufland.bg/aktualni-predlozheniya/oferti.html",
    #     country_code="BG",
    #     country_name="Bulgaria",
    #     currency="BGN",
    # ),
    SourceConfig(
        page_url="https://filiale.kaufland.de/angebote/aktuelle-woche.html",
        country_code="DE",
        country_name="Germany",
        currency="EUR",
    ),
]


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


FOOD_CATEGORY_CODES = {"01", "01a", "02", "03", "04", "05", "06", "07", "08"}


def extract_category_code(category: Dict[str, Any]) -> str:
    name = str(category.get("name") or "")
    first_token = name.split("_", 1)[0].strip().lower()
    if first_token:
        return re.sub(r"[^0-9a-z]", "", first_token)

    offer_category_id = str(category.get("offerCategoryId") or "")
    if "_" in offer_category_id:
        token = offer_category_id.split("_")[-1].strip().lower()
        return re.sub(r"[^0-9a-z]", "", token)
    return ""


def is_food_category(category: Dict[str, Any]) -> bool:
    category_name = str(category.get("displayName") or category.get("name") or "")
    normalized = normalize_text(category_name)
    code = extract_category_code(category)
    if code in FOOD_CATEGORY_CODES:
        return True

    keywords = {
        "legume",
        "fructe",
        "carne",
        "mezeluri",
        "peste",
        "lactate",
        "oua",
        "alimente de baza",
        "panificatie",
        "brutarie",
        "paste",
        "orez",
        "faina",
        "conserve",
        "congelate",
        "bauturi",
    }
    return any(keyword in normalized for keyword in keywords)


def extract_ssr_json_objects(html: str) -> Iterable[Dict[str, Any]]:
    pattern = re.compile(
        r"window\.SSR\['[^']+']\s*=\s*({.*?})\s*(?:;|</script>)",
        re.DOTALL,
    )
    for match in pattern.finditer(html):
        blob = match.group(1)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def find_offer_template_payload(html: str) -> Optional[Dict[str, Any]]:
    for obj in extract_ssr_json_objects(html):
        if obj.get("component") == "OfferTemplate":
            return obj
    return None


def fetch_html(url: str) -> Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.7",
        "Referer": "https://filiale.kaufland.de/",
    }
    response = requests.get(url, headers=headers, timeout=35, allow_redirects=True)
    response.raise_for_status()
    return response.text, response.url


def extract_products_from_payload(
    payload: Dict[str, Any],
    source: SourceConfig,
    scraped_at: str,
    max_count: int,
    include_non_food: bool,
    seen_offer_ids: set,
) -> Dict[str, List[Dict[str, Any]]]:
    categories_map: Dict[str, List[Dict[str, Any]]] = {}
    props = payload.get("props") or {}
    offer_data = props.get("offerData") or {}
    cycles = offer_data.get("cycles") or []

    total_added = 0
    for cycle in cycles:
        categories = cycle.get("categories") or []
        for category in categories:
            category_name = category.get("displayName") or category.get("name") or "Unknown"
            category_is_food = is_food_category(category)
            if not category_is_food and not include_non_food:
                continue

            offers = category.get("offers") or []
            if not offers:
                continue

            for offer in offers:
                price = offer.get("price")
                if price is None:
                    continue
                offer_id = str(offer.get("offerId") or "")
                if offer_id and offer_id in seen_offer_ids:
                    continue

                title = str(offer.get("title") or "Unknown product").strip()
                subtitle = str(offer.get("subtitle") or "").strip()
                unit = str(offer.get("unit") or "").strip()
                base_price = str(offer.get("basePrice") or "").strip()

                entry = {
                    "product": title,
                    "price": {
                        "amount": float(price),
                        "currency": source.currency,
                        "unit": unit or None,
                        "base_price_text": base_price or None,
                        "discount_percent": offer.get("discount"),
                    },
                    "meta": {
                        "source": "Kaufland offers page",
                        "source_url": source.page_url,
                        "offer_id": offer.get("offerId"),
                        "collected_at_utc": scraped_at,
                        "valid_from": offer.get("dateFrom"),
                        "valid_to": offer.get("dateTo"),
                        "countries_valid": [
                            {
                                "country_code": source.country_code,
                                "country_name": source.country_name,
                            }
                        ],
                        "category_code": category.get("offerCategoryId"),
                        "food_category_match": category_is_food,
                        "subtitle": subtitle or None,
                        "detail_title": offer.get("detailTitle"),
                    },
                }

                categories_map.setdefault(category_name, []).append(entry)
                total_added += 1
                if offer_id:
                    seen_offer_ids.add(offer_id)
                if total_added >= max_count:
                    return categories_map

    return categories_map


def collect(limit: int) -> Dict[str, Any]:
    scraped_at = datetime.now(timezone.utc).isoformat()
    all_categories: Dict[str, List[Dict[str, Any]]] = {}
    source_reports: List[Dict[str, Any]] = []
    seen_offer_ids: set = set()

    remaining = limit
    for source in SOURCES:
        if remaining <= 0:
            break

        try:
            html, resolved_url = fetch_html(source.page_url)
        except requests.RequestException as exc:
            source_reports.append(
                {
                    "source_url": source.page_url,
                    "country_code": source.country_code,
                    "status": "failed",
                    "reason": f"HTTP error: {exc}",
                }
            )
            continue

        payload = find_offer_template_payload(html)
        if payload is None:
            source_reports.append(
                {
                    "source_url": resolved_url,
                    "country_code": source.country_code,
                    "status": "failed",
                    "reason": "OfferTemplate SSR payload not found",
                }
            )
            continue

        source_count = 0
        food_count = 0
        fallback_count = 0

        extracted_food = extract_products_from_payload(
            payload=payload,
            source=source,
            scraped_at=scraped_at,
            max_count=remaining,
            include_non_food=False,
            seen_offer_ids=seen_offer_ids,
        )
        for category, items in extracted_food.items():
            all_categories.setdefault(category, []).extend(items)
            source_count += len(items)
            food_count += len(items)
        remaining -= food_count

        if remaining > 0:
            extracted_fallback = extract_products_from_payload(
                payload=payload,
                source=source,
                scraped_at=scraped_at,
                max_count=remaining,
                include_non_food=True,
                seen_offer_ids=seen_offer_ids,
            )
            for category, items in extracted_fallback.items():
                all_categories.setdefault(category, []).extend(items)
                source_count += len(items)
                fallback_count += len(items)
            remaining -= fallback_count

        source_reports.append(
            {
                "source_url": resolved_url,
                "country_code": source.country_code,
                "status": "ok",
                "products_collected": source_count,
                "food_products_collected": food_count,
                "fallback_products_collected": fallback_count,
            }
        )

    total_products = sum(len(items) for items in all_categories.values())
    return {
        "metadata": {
            "collector": "custom python scraper",
            "dataset": "Kaufland basic food offers",
            "collected_at_utc": scraped_at,
            "requested_limit": limit,
            "products_collected": total_products,
            "countries_covered": sorted({s.country_code for s in SOURCES}),
            "source_reports": source_reports,
            "notes": [
                "Prices are scraped from publicly available Kaufland offer pages.",
                "Price validity can vary by week and store availability.",
                "Country validity is listed per item in meta.countries_valid.",
                "Collection runs in two passes: food categories first, then additional offers if needed to reach the requested limit.",
            ],
        },
        "categories": all_categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Kaufland food prices into JSON.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum number of products.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON file path.")
    args = parser.parse_args()

    data = collect(limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    total = data["metadata"]["products_collected"]
    print(f"Wrote {total} products to {args.output}")


if __name__ == "__main__":
    main()

