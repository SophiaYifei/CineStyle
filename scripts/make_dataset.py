# AI tools used: Claude (Anthropic) assisted with HuggingFace datasets APIintegration, synthetic interaction generation logic, and debugging.
"""
make_dataset.py

Uses the detection-datasets/fashionpedia HuggingFace dataset.
Fashionpedia contains editorial fashion images with bounding-box annotations
for 46 garment/accessory categories including dresses, tops, bags, shoes,
hats, watches, belts, etc. — a much richer vocabulary than fashion200k.

Pipeline:
  1. Stream the train + validation splits.
  2. For each annotated object: crop the bounding box from the image.
  3. Filter to the target garment/accessory categories.
  4. Write catalog.jsonl  — one record per crop (id, category, image saved to data/raw/crops/).
  5. Generate synthetic interactions.jsonl for NCF training.

Output:
  data/raw/crops/<id>.jpg            — cropped garment/accessory images
  data/processed/catalog.jsonl       — product metadata
  data/processed/interactions.jsonl  — (user_id, item_id, label) pairs
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CROPS_DIR = RAW_DIR / "crops"
PROC_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Fashionpedia category index → human name
CATEGORY_NAMES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit",
    "cape", "glasses", "hat", "headband, head covering, hair accessory",
    "tie", "glove", "watch", "belt", "leg warmer", "tights, stockings",
    "sock", "shoe", "bag, wallet", "scarf", "umbrella", "hood", "collar",
    "lapel", "epaulette", "sleeve", "pocket", "neckline", "buckle", "zipper",
    "applique", "bead", "bow", "flower", "fringe", "ribbon", "rivet",
    "ruffle", "sequin", "tassel",
]

# Categories we want as top-level garments / accessories (skip component parts)
KEEP_CATEGORIES = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit",
    "cape", "glasses", "hat", "headband, head covering, hair accessory",
    "tie", "glove", "watch", "belt", "tights, stockings", "sock", "shoe",
    "bag, wallet", "scarf",
}

# Coarse aesthetic buckets per category (used as a proxy for "aesthetic" attribute
# when FashionCLIP is not yet run).  Will be refined during build_features step.
AESTHETIC_MAP = {
    "dress": "feminine",
    "skirt": "feminine",
    "jumpsuit": "chic",
    "coat": "outerwear",
    "jacket": "outerwear",
    "cape": "outerwear",
    "cardigan": "casual",
    "sweater": "casual",
    "top, t-shirt, sweatshirt": "casual",
    "shirt, blouse": "smart casual",
    "vest": "smart casual",
    "pants": "casual",
    "shorts": "casual",
    "tights, stockings": "accessories",
    "sock": "accessories",
    "shoe": "accessories",
    "bag, wallet": "accessories",
    "watch": "accessories",
    "glasses": "accessories",
    "hat": "accessories",
    "headband, head covering, hair accessory": "accessories",
    "tie": "formal",
    "glove": "accessories",
    "belt": "accessories",
    "scarf": "accessories",
}

MIN_CROP_PX = 48   # skip tiny crops that will just be noise


def category_name(idx: int) -> str:
    if 0 <= idx < len(CATEGORY_NAMES):
        return CATEGORY_NAMES[idx]
    return "unknown"


def download_fashionpedia(max_items: int = 20_000) -> None:
    """
    Stream Fashionpedia train+val, crop each annotated garment/accessory,
    and write catalog.jsonl.
    """
    catalog_path = PROC_DIR / "catalog.jsonl"
    count = 0

    with open(catalog_path, "w") as f:
        for split in ("train", "val"):
            ds = load_dataset(
                "detection-datasets/fashionpedia",
                split=split,
                streaming=True,
            )
            for sample in tqdm(ds, desc=f"fashionpedia/{split}"):
                if count >= max_items:
                    break

                pil_img: Image.Image = sample["image"]
                W, H = pil_img.width, pil_img.height
                objects = sample["objects"]

                for bbox_id, cat_idx, bbox in zip(
                    objects["bbox_id"],
                    objects["category"],
                    objects["bbox"],
                ):
                    if count >= max_items:
                        break

                    cat = category_name(cat_idx)
                    if cat not in KEEP_CATEGORIES:
                        continue

                    x0, y0, x1, y1 = bbox
                    cw, ch = x1 - x0, y1 - y0
                    if cw < MIN_CROP_PX or ch < MIN_CROP_PX:
                        continue

                    # Clamp to image bounds
                    x0 = max(0, int(x0))
                    y0 = max(0, int(y0))
                    x1 = min(W, int(x1))
                    y1 = min(H, int(y1))

                    crop = pil_img.crop((x0, y0, x1, y1)).convert("RGB")
                    crop_id = str(bbox_id)
                    crop_path = CROPS_DIR / f"{crop_id}.jpg"
                    crop.save(crop_path, "JPEG", quality=92)

                    record = {
                        "id": crop_id,
                        "title": cat.title(),
                        "brand": "",          # Fashionpedia has no brand metadata
                        "price": 0.0,         # No price — filled in by mock or real API
                        "image_url": str(crop_path),
                        "product_url": "",
                        "category": cat,
                        "aesthetic": AESTHETIC_MAP.get(cat, "fashion"),
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1

            if count >= max_items:
                break

    print(f"Saved {count} catalog items → {catalog_path}")


def assign_mock_prices(min_price: float = 15.0, max_price: float = 850.0) -> None:
    """
    Fashionpedia has no price metadata.
    Assign plausible price ranges per category so the price filter in the app works.
    """
    price_ranges = {
        "shoe": (60, 600),
        "bag, wallet": (50, 800),
        "watch": (80, 850),
        "coat": (80, 700),
        "jacket": (60, 600),
        "dress": (40, 500),
        "jumpsuit": (40, 400),
        "glasses": (20, 400),
        "hat": (20, 300),
        "belt": (15, 250),
        "scarf": (15, 200),
        "sweater": (30, 350),
        "cardigan": (30, 300),
        "shirt, blouse": (25, 300),
        "pants": (30, 300),
        "skirt": (25, 280),
        "top, t-shirt, sweatshirt": (15, 150),
        "shorts": (20, 150),
        "tights, stockings": (10, 80),
        "sock": (8, 40),
        "glove": (15, 150),
        "tie": (20, 200),
        "headband, head covering, hair accessory": (10, 100),
        "vest": (25, 200),
        "cape": (60, 500),
    }

    catalog_path = PROC_DIR / "catalog.jsonl"
    tmp_path = PROC_DIR / "catalog.jsonl.tmp"

    with open(catalog_path) as fin, open(tmp_path, "w") as fout:
        for line in fin:
            item = json.loads(line)
            lo, hi = price_ranges.get(item["category"], (min_price, max_price))
            item["price"] = round(random.uniform(lo, hi), 2)
            fout.write(json.dumps(item) + "\n")

    tmp_path.replace(catalog_path)
    print(f"Assigned mock prices → {catalog_path}")


def build_synthetic_interactions(
    n_users: int = 500,
    interactions_per_user: int = 30,
) -> None:
    """
    Generate synthetic implicit feedback for NCF training.
    Heavier users tend to focus on a single category cluster (simulates real taste).
    """
    catalog_path = PROC_DIR / "catalog.jsonl"
    with open(catalog_path) as f:
        catalog = [json.loads(line) for line in f]

    # Group items by category for taste-cohesive sampling
    by_cat: dict[str, list[str]] = {}
    for item in catalog:
        by_cat.setdefault(item["category"], []).append(item["id"])

    all_ids = [item["id"] for item in catalog]
    categories = list(by_cat.keys())

    interactions_path = PROC_DIR / "interactions.jsonl"
    with open(interactions_path, "w") as f:
        for user_id in tqdm(range(n_users), desc="interactions"):
            # Each user has 1-3 "favourite" categories
            fav_cats = random.sample(categories, k=min(random.randint(1, 3), len(categories)))
            fav_pool = []
            for c in fav_cats:
                fav_pool.extend(by_cat[c])

            # 70% from fav categories, 30% random
            n_fav = int(interactions_per_user * 0.7)
            n_rand = interactions_per_user - n_fav

            selected = set()
            for item_id in random.sample(fav_pool, k=min(n_fav, len(fav_pool))):
                selected.add(item_id)
            for item_id in random.sample(all_ids, k=min(n_rand, len(all_ids))):
                selected.add(item_id)

            for item_id in selected:
                record = {"user_id": user_id, "item_id": item_id, "label": 1}
                f.write(json.dumps(record) + "\n")

    print(f"Saved synthetic interactions → {interactions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_items", type=int, default=20_000,
                        help="Max garment crops to extract from Fashionpedia")
    parser.add_argument("--n_users", type=int, default=500)
    parser.add_argument("--interactions_per_user", type=int, default=30)
    args = parser.parse_args()

    download_fashionpedia(max_items=args.max_items)
    assign_mock_prices()
    build_synthetic_interactions(
        n_users=args.n_users,
        interactions_per_user=args.interactions_per_user,
    )
