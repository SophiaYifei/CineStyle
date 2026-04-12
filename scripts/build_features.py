"""
build_features.py

1. Load product catalog (data/processed/catalog.jsonl)
2. Embed each product image with FashionCLIP → 512-dim float32 vector
3. Build FAISS flat-L2 index
4. Save index + metadata lookup

Also exports embed_image() for use by the FastAPI backend.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROC_DIR = Path("data/processed")
INDEX_DIR = Path("models/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "patrickjohncyh/fashion-clip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None


def _load_model() -> tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor
    if _model is None:
        print(f"Loading {MODEL_NAME} on {DEVICE} ...")
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor


def embed_image(image: Image.Image) -> dict:
    """
    Embed a PIL image with FashionCLIP.
    Returns dict with garment_type, color, aesthetic, and 512-dim embedding.
    """
    model, processor = _load_model()

    # --- Attribute classification via text prompts ---
    garment_labels = [
        "dress", "top", "blouse", "jacket", "coat", "pants", "jeans",
        "skirt", "shorts", "suit", "sweater", "hoodie", "shirt",
    ]
    color_labels = [
        "black", "white", "red", "blue", "green", "yellow",
        "pink", "purple", "brown", "grey", "beige", "navy",
    ]
    aesthetic_labels = [
        "casual", "formal", "streetwear", "bohemian", "minimalist",
        "old money", "dark academia", "cottagecore", "y2k", "preppy",
    ]

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        def classify(labels: list[str]) -> str:
            text_inputs = processor(
                text=labels, return_tensors="pt", padding=True
            ).to(DEVICE)
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
            return labels[sims.argmax().item()]

        garment_type = classify(garment_labels)
        color = classify(color_labels)
        aesthetic = classify(aesthetic_labels)
        embedding = image_features.squeeze(0).cpu().float().tolist()

    return {
        "garment_type": garment_type,
        "color": color,
        "aesthetic": aesthetic,
        "embedding": embedding,
    }


def build_index(batch_size: int = 64) -> None:
    """
    Embed all products in catalog.jsonl and build a FAISS index.
    """
    model, processor = _load_model()
    catalog_path = PROC_DIR / "catalog.jsonl"

    with open(catalog_path) as f:
        catalog = [json.loads(line) for line in f]

    dim = 512
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalised vectors == cosine sim
    meta = []

    embeddings_batch = []
    meta_batch = []

    def flush(embeddings_batch, meta_batch):
        arr = np.stack(embeddings_batch).astype(np.float32)
        faiss.normalize_L2(arr)
        index.add(arr)
        meta.extend(meta_batch)

    for item in tqdm(catalog, desc="Embedding catalog"):
        try:
            url = item["image_url"]
            if url.startswith("http://") or url.startswith("https://"):
                import io as _io
                response = requests.get(url, timeout=5)
                img = Image.open(_io.BytesIO(response.content)).convert("RGB")
            else:
                # Local file path (e.g. from Fashionpedia crops)
                img = Image.open(url).convert("RGB")
        except Exception:
            continue

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings_batch.append(feats.squeeze(0).cpu().float().numpy())

        meta_batch.append({
            "id": item["id"],
            "title": item["title"],
            "brand": item["brand"],
            "price": item["price"],
            "image_url": item["image_url"],
            "product_url": item.get("product_url", ""),
        })

        if len(embeddings_batch) >= batch_size:
            flush(embeddings_batch, meta_batch)
            embeddings_batch, meta_batch = [], []

    if embeddings_batch:
        flush(embeddings_batch, meta_batch)

    faiss.write_index(index, str(INDEX_DIR / "products.index"))
    with open(INDEX_DIR / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"Index built: {index.ntotal} vectors → {INDEX_DIR}/products.index")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    build_index(batch_size=args.batch_size)
