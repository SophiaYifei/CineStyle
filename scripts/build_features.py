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

# Dummy image needed so CLIPProcessor can produce both image + text tensors together
_DUMMY_TEXT = ["fashion item"]


def _load_model() -> tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor
    if _model is None:
        print(f"Loading {MODEL_NAME} on {DEVICE} ...")
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor


def _get_image_embeds(
    images: list[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> torch.Tensor:
    """Return L2-normalised image embeddings. Shape: (N, 512)."""
    inputs = processor(
        images=images,
        text=_DUMMY_TEXT * len(images),
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)
    out = model(**inputs)
    feats: torch.Tensor = out.image_embeds          # (N, 512)
    return feats / feats.norm(dim=-1, keepdim=True)


def _get_text_embeds(
    labels: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> torch.Tensor:
    """Return L2-normalised text embeddings. Shape: (N, 512)."""
    dummy_img = Image.new("RGB", (224, 224))
    inputs = processor(
        images=[dummy_img] * len(labels),
        text=labels,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)
    out = model(**inputs)
    feats: torch.Tensor = out.text_embeds           # (N, 512)
    return feats / feats.norm(dim=-1, keepdim=True)


def embed_image(image: Image.Image) -> dict:
    """
    Embed a PIL image with FashionCLIP.
    Returns dict with garment_type, color, aesthetic, and 512-dim embedding.
    """
    model, processor = _load_model()

    garment_labels = [
        "dress", "top", "blouse", "jacket", "coat", "pants", "jeans",
        "skirt", "shorts", "suit", "sweater", "hoodie", "shirt",
        "bag", "shoes", "hat", "scarf", "belt", "watch", "glasses",
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
        img_feat = _get_image_embeds([image], model, processor).squeeze(0)  # (512,)

        def classify(labels: list[str]) -> str:
            txt_feats = _get_text_embeds(labels, model, processor)          # (N, 512)
            sims = (img_feat.unsqueeze(0) @ txt_feats.T).squeeze(0)        # (N,)
            return labels[sims.argmax().item()]

        garment_type = classify(garment_labels)
        color = classify(color_labels)
        aesthetic = classify(aesthetic_labels)

    return {
        "garment_type": garment_type,
        "color": color,
        "aesthetic": aesthetic,
        "embedding": img_feat.cpu().float().tolist(),
    }


def build_index(batch_size: int = 128) -> None:
    """
    Embed all products in catalog.jsonl and build a FAISS index.
    Optimised for GPU (A100): large batch_size, torch.cuda.amp.autocast.
    """
    model, processor = _load_model()
    catalog_path = PROC_DIR / "catalog.jsonl"

    with open(catalog_path) as f:
        catalog = [json.loads(line) for line in f]

    dim = 512
    index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on L2-normalised vecs
    meta: list[dict] = []

    images_batch: list[Image.Image] = []
    meta_batch: list[dict] = []

    def flush() -> None:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            feats = _get_image_embeds(images_batch, model, processor)
        arr = feats.cpu().float().numpy().astype(np.float32)
        faiss.normalize_L2(arr)
        index.add(arr)
        meta.extend(meta_batch)
        images_batch.clear()
        meta_batch.clear()

    for item in tqdm(catalog, desc="Embedding catalog"):
        try:
            url = item["image_url"]
            if url.startswith("http://") or url.startswith("https://"):
                import io as _io
                response = requests.get(url, timeout=8)
                img = Image.open(_io.BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(url).convert("RGB")
        except Exception:
            continue

        images_batch.append(img)
        meta_batch.append({
            "id": item["id"],
            "title": item["title"],
            "brand": item.get("brand", ""),
            "price": item["price"],
            "image_url": item["image_url"],
            "product_url": item.get("product_url", ""),
        })

        if len(images_batch) >= batch_size:
            flush()

    if images_batch:
        flush()

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
