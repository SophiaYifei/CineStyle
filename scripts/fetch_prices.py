# AI tools used: Claude (Anthropic) assisted with LiteLLM API integration and retry/backoff logic for the price enrichment agent.
"""
fetch_prices.py  —  Agentic price enrichment using Duke's LiteLLM API.

For each garment/accessory category in the catalog, an LLM agent:
  1. Reasons about the realistic retail price range for that category
     (luxury vs. fast-fashion spread, brand tier, market segment).
  2. Optionally grounds its estimates with a Google Shopping web search.
  3. Returns a (low, high) USD price range that is written back to catalog.jsonl.

The agent uses tool-calling via the OpenAI-compatible Duke LiteLLM endpoint.

Setup:
  export DUKE_LLM_API_KEY="<your Duke API key>"
  export DUKE_LLM_MODEL="claude-opus-4-6"   # or any model available on litellm.oit.duke.edu

Usage:
  python scripts/fetch_prices.py [--catalog data/processed/catalog.jsonl]
                                 [--model claude-opus-4-6]
                                 [--web_search]
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from openai import OpenAI

PROC_DIR = Path("data/processed")

DUKE_LLM_BASE_URL = "https://litellm.oit.duke.edu/v1"
DEFAULT_MODEL = os.environ.get("DUKE_LLM_MODEL", "claude-opus-4-6")

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search_prices",
            "description": (
                "Search Google Shopping for current retail prices of a fashion item. "
                "Returns text with price mentions. Use this to ground your estimate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "e.g. 'women dress price buy online nordstrom'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emit_price_range",
            "description": "Emit the final price range decision for the category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category":    {"type": "string"},
                    "price_low":   {"type": "number", "description": "10th-percentile retail price (USD)"},
                    "price_high":  {"type": "number", "description": "90th-percentile retail price (USD)"},
                    "reasoning":   {"type": "string", "description": "1-2 sentence justification"},
                },
                "required": ["category", "price_low", "price_high", "reasoning"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a fashion retail pricing expert. Determine a realistic USD retail price range "
    "(budget to mid-luxury) for a garment/accessory category.\n\n"
    "Rules:\n"
    "- price_low  = ~10th percentile typical retail (e.g. Zara/H&M tier)\n"
    "- price_high = ~90th percentile (Nordstrom/Bloomingdale's tier, not couture)\n"
    "- You may call web_search_prices once if you want to ground the estimate, "
    "then call emit_price_range.\n"
    "- If confident, go straight to emit_price_range."
)


def _google_search_text(query: str) -> str:
    """Fetch a Google search page and return stripped text (price mentions)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(
            "https://www.google.com/search",
            headers=headers,
            params={"q": query, "num": 8, "hl": "en", "gl": "us"},
            timeout=10,
        )
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)
        return text[:2000]
    except Exception as e:
        return f"[search error: {e}]"


def _dispatch_tool(name: str, args: dict, use_web: bool) -> str:
    if name == "web_search_prices" and use_web:
        return _google_search_text(args["query"])
    if name == "emit_price_range":
        return "Price range recorded."
    return "[tool not available]"


def _run_price_agent(
    client: OpenAI,
    category: str,
    model: str,
    use_web: bool,
) -> dict | None:
    """
    Agentic loop for one category.
    Returns {"price_low": float, "price_high": float, "reasoning": str} or None.
    """
    available_tools = TOOLS if use_web else [TOOLS[1]]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Determine the retail price range for: **{category}** "
                f"(women's fashion, US market)."
            ),
        },
    ]

    for _ in range(5):  # max turns
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.2,
        )

        msg = resp.choices[0].message
        messages.append(msg)

        if resp.choices[0].finish_reason == "stop":
            break

        if resp.choices[0].finish_reason != "tool_calls" or not msg.tool_calls:
            break

        price_range = None
        tool_results = []

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _dispatch_tool(tc.function.name, args, use_web)

            if tc.function.name == "emit_price_range":
                price_range = {
                    "price_low": float(args["price_low"]),
                    "price_high": float(args["price_high"]),
                    "reasoning": args.get("reasoning", ""),
                }

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        messages.extend(tool_results)

        if price_range:
            return price_range

    return None


FALLBACK_RANGES = {
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


def fetch_price_ranges(
    categories: list[str],
    model: str = DEFAULT_MODEL,
    use_web: bool = False,
    delay: float = 1.0,
) -> dict[str, tuple[float, float]]:
    api_key = os.environ.get("DUKE_LLM_API_KEY")
    if not api_key:
        print("DUKE_LLM_API_KEY not set — using fallback price ranges.")
        return {cat: FALLBACK_RANGES.get(cat, (20, 300)) for cat in categories}

    client = OpenAI(api_key=api_key, base_url=DUKE_LLM_BASE_URL)
    price_ranges: dict[str, tuple[float, float]] = {}

    for cat in categories:
        print(f"  [{cat}] querying {model} ...")
        try:
            result = _run_price_agent(client, cat, model=model, use_web=use_web)
            if result:
                lo = max(5.0, result["price_low"])
                hi = min(2000.0, result["price_high"])
                price_ranges[cat] = (lo, hi)
                print(f"    → ${lo:.0f}–${hi:.0f}  | {result['reasoning'][:80]}")
            else:
                price_ranges[cat] = FALLBACK_RANGES.get(cat, (20, 300))
                print(f"    → fallback ${price_ranges[cat][0]:.0f}–${price_ranges[cat][1]:.0f}")
        except Exception as e:
            price_ranges[cat] = FALLBACK_RANGES.get(cat, (20, 300))
            print(f"    → error ({e}), fallback")

        time.sleep(delay)

    return price_ranges


def enrich_prices(catalog_path: Path, price_ranges: dict[str, tuple[float, float]]) -> None:
    """Overwrite price field in catalog.jsonl using agent-determined ranges."""
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
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model name on litellm.oit.duke.edu (e.g. claude-opus-4-6, gpt-4o)",
    )
    parser.add_argument(
        "--web_search", action="store_true",
        help="Allow agent to call web_search_prices for grounding",
    )
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    with open(catalog_path) as f:
        all_cats = list({json.loads(line)["category"] for line in f})

    print(f"Running price agent for {len(all_cats)} categories")
    print(f"  endpoint : {DUKE_LLM_BASE_URL}")
    print(f"  model    : {args.model}")
    print(f"  web      : {args.web_search}")
    print()

    ranges = fetch_price_ranges(
        all_cats,
        model=args.model,
        use_web=args.web_search,
        delay=args.delay,
    )
    enrich_prices(catalog_path, ranges)
    print("Done.")
