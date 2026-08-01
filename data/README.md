# Kaufland Food Price Collector

This folder contains a Python scraper that collects Kaufland offer prices and writes a categorized JSON dataset.

## What it stores

- Categories of products (food-focused first)
- For each product:
  - product name
  - price amount + currency
  - unit/base price text (when available)
  - metadata (source URL, collection timestamp, validity dates, country validity)

## Files

- `kaufland_scraper.py` - main collector
- `kaufland_prices_by_category_various_countries.json` - generated output
- `smoke_test_output.py` - quick JSON structure validator
- `requirements.txt` - Python dependencies

## Quick run (PowerShell)

```powershell
py -m pip install -r D:\Research\pso_llm_optimizer\data\requirements.txt
py -u D:\Research\pso_llm_optimizer\data\kaufland_scraper.py --limit 1000 --output D:\Research\pso_llm_optimizer\data\kaufland_prices_by_category.json
py -u D:\Research\pso_llm_optimizer\data\smoke_test_output.py
```

## Notes

- Configured source by default:
  - Germany (`DE`) via `https://filiale.kaufland.de/angebote/aktuelle-woche.html`
- Additional countries can be enabled by uncommenting/editing entries in `SOURCES`.
- Collection strategy:
  1. Collect food-category offers first.
  2. If fewer than `--limit`, include additional categories to reach the requested total.
- Prices depend on current offer cycles and may change frequently.
- If you need additional countries, add new entries in `SOURCES` inside `kaufland_scraper.py`.

