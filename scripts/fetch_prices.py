"""
fetch_prices.py

Enrich the catalog with real-world price references by scraping
Google Shopping search results for each garment category.

The script queries e.g. "women's dress price site:nordstrom.com OR site:net-a-porter.com"
for each unique category in the catalog, extracts price mentions from the HTML,
and builds a realistic price distribution that replaces the mock prices.

No API key required — uses the public Google search HTML interface.
Respects rate limits with a randomised delay between requests.

Usage:
  python scripts/fetch_prices.py [--catalog data/processed/catalog.jsonl]
                                 [--delay 2.5]
                                 [--per_category 5]
"""

import argparse
import json
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

PROC_DIR = Path("data/processed")

# Search query template per category
SEARCH_TEMPLATES = {
    "dress":                       "women's dress price USD buy online",
    "shirt, blouse":               "women's blouse shirt price USD buy online",
    "top, t-shirt, sweatshirt":    "women's top t-shirt price USD buy online",
    "sweater":                     "women's sweater price USD buy online",
    "cardigan":                    "cardigan price USD buy online",
    "jacket":                      "women's jacket price USD buy online",
    "coat":                        "women's coat price USD buy online",
    "vest":                        "women's vest price USD buy online",
    "pants":                       "women's pants price USD buy online",
    "shorts":                      "women's shorts price USD buy online",
    "skirt":                       "women's skirt price USD buy online",
    "jumpsuit":                    "women's jumpsuit price USD buy online",
    "cape":                        "women's cape price USD buy online",
    "shoe":                        "women's shoes price USD buy online",
    "bag, wallet":                 "women's handbag wallet price USD buy online",
    "watch":                       "women's watch price USD buy online",
    "belt":                        "women's belt price USD buy online",
    "scarf":                       "women's scarf price USD buy online",
    "glasses":                     "women's sunglasses eyeglasses price USD buy online",
    "hat":                         "women's hat price USD buy online",
    "headband, head covering, hair accessory": "hair accessory headband price USD buy online",
    "glove":                       "women's gloves price USD buy online",
    "tie":                         "fashion tie price USD buy online",
    "tights, stockings":           "tights stockings price USD buy online",
    "sock":                        "fashion socks price USD buy online",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PRICE_RE = re.compile(r"\$\s*(\d{1,4}(?:\.\d{2})?)")


def _search_prices(query: str, n: int = 5) -> list[float]:
    """
    Search Google for the query and extract USD price mentions from the HTML.
    Returns up to n price floats.
    """
    url = "https://www.google.com/search"
    params = {"q": query, "num": 10, "hl": "en", "gl": "us"}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        prices = [float(m) for m in PRICE_RE.findall(resp.text) if 1 < float(m) < 5000]
        # Deduplicate near-identical values
        seen, unique = set(), []
        for p in prices:
            bucket = round(p / 5) * 5
            if bucket not in seen:
                seen.add(bucket)
                unique.append(p)
        return unique[:n]
    except Exception:
        return []


def fetch_category_price_ranges(
    categories: list[str],
    per_category: int = 5,
    delay: float = 2.5,
) -> dict[str, tuple[float, float]]:
    """
    For each category, fetch price samples and return (p10, p90) range.
    Falls back to hardcoded defaults if search fails.
    """
    FALLBACK = {
        "dress":          (40, 500),
        "shirt, blouse":  (25, 300),
        "top, t-shirt, sweatshirt": (15, 150),
        "sweater":        (30, 350),
        "cardigan":       (30, 300),
        "jacket":         (60, 600),
        "coat":           (80, 700),
        "vest":           (25, 200),
        "pants":          (30, 300),
        "shorts":         (20, 150),
        "skirt":          (25, 280),
        "jumpsuit":       (40, 400),
        "cape":           (60, 500),
        "shoe":           (60, 600),
        "bag, wallet":    (50, 800),
        "watch":          (80, 850),
        "belt":           (15, 250),
        "scarf":          (15, 200),
        "glasses":        (20, 400),
        "hat":            (20, 300),
        "headband, head covering, hair accessory": (10, 100),
        "glove":          (15, 150),
        "tie":            (20, 200),
        "tights, stockings": (10, 80),
        "sock":           (8, 40),
    }

    price_ranges: dict[str, tuple[float, float]] = {}

    for cat in tqdm(categories, desc="Fetching prices"):
        query = SEARCH_TEMPLATES.get(cat, f"{cat} price USD buy online")
        prices = _search_prices(query, n=per_category)

        if len(prices) >= 2:
            lo = statistics.quantiles(prices, n=10)[0]   # p10
            hi = statistics.quantiles(prices, n=10)[-1]  # p90
            price_ranges[cat] = (max(5.0, lo), min(2000.0, hi))
            print(f"  {cat}: ${lo:.0f}–${hi:.0f} (from {len(prices)} results)")
        else:
            price_ranges[cat] = FALLBACK.get(cat, (20, 300))
            print(f"  {cat}: using fallback ${price_ranges[cat][0]:.0f}–${price_ranges[cat][1]:.0f}")

        jitter = delay + random.uniform(-0.5, 1.5)
        time.sleep(max(0.5, jitter))

    return price_ranges


def enrich_prices(catalog_path: Path, price_ranges: dict[str, tuple[float, float]]) -> None:
    """Overwrite price field in catalog.jsonl using fetched ranges."""
    tmp = catalog_path.with_suffix(".jsonl.tmp")
    updated = 0
    with open(catalog_path) as fin, open(tmp, "w") as fout:
        for line in fin:
            item = json.loads(line)
            cat = item.get("category", "")
            lo, hi = price_ranges.get(cat, (20.0, 300.0))
            item["price"] = round(random.uniform(lo, hi), 2)
            fout.write(json.dumps(item) + "\n")
            updated += 1
    tmp.replace(catalog_path)
    print(f"Updated prices for {updated} items → {catalog_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(PROC_DIR / "catalog.jsonl"))
    parser.add_argument("--delay", type=float, default=2.5,
                        help="Seconds between search requests")
    parser.add_argument("--per_category", type=int, default=5,
                        help="Price samples to fetch per category")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    with open(catalog_path) as f:
        all_cats = list({json.loads(l)["category"] for l in f})

    print(f"Fetching real price ranges for {len(all_cats)} categories ...")
    ranges = fetch_category_price_ranges(
        all_cats,
        per_category=args.per_category,
        delay=args.delay,
    )
    enrich_prices(catalog_path, ranges)
    print("Done.")
