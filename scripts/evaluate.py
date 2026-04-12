"""
evaluate.py

Offline evaluation for CineStyle recommendation models.

Metrics:
  - Precision@K
  - Recall@K
  - NDCG@K
  - MAP@K
  - Visual similarity score (mean cosine distance in embedding space)

Experiment: Frame quality vs. recommendation accuracy
  --experiment flag runs the degradation study (HD / compressed / blurred)

Usage:
  python scripts/evaluate.py --k 5 --k 10
  python scripts/evaluate.py --experiment
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from scripts.build_features import embed_image
from scripts.model import faiss_retrieve, recommend

OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item["id"] in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item["id"] in relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item["id"] in relevant:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    ideal = sorted([1] * min(len(relevant), k), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg_at_k(recommended, relevant, k) / idcg if idcg > 0 else 0.0


def average_precision(recommended: list, relevant: set) -> float:
    hits, ap = 0, 0.0
    for i, item in enumerate(recommended):
        if item["id"] in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant) if relevant else 0.0


def mean_average_precision(all_recs: list[list], all_relevant: list[set]) -> float:
    return np.mean([average_precision(r, rel) for r, rel in zip(all_recs, all_relevant)])


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def evaluate_model(k_values: list[int] = [5, 10]) -> dict:
    """
    Run offline evaluation on held-out interactions.
    Uses last 20% of each user's interactions as ground truth.
    """
    interactions_path = Path("data/processed/interactions.jsonl")
    with open(interactions_path) as f:
        interactions = [json.loads(line) for line in f]

    # Group by user
    user_items: dict[int, list[str]] = {}
    for row in interactions:
        user_items.setdefault(row["user_id"], []).append(row["item_id"])

    # Load FAISS meta for embedding lookup
    from scripts.build_features import INDEX_DIR
    import faiss as _faiss
    index = _faiss.read_index(str(INDEX_DIR / "products.index"))
    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)
    id_to_idx = {item["id"]: i for i, item in enumerate(meta)}

    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    all_recs, all_relevant_sets = [], []

    for user_id, item_ids in list(user_items.items())[:200]:  # sample 200 users
        split = max(1, int(len(item_ids) * 0.8))
        train_ids = item_ids[:split]
        test_ids = set(item_ids[split:])
        if not test_ids:
            continue

        # Use first training item's embedding as query
        q_idx = id_to_idx.get(train_ids[-1])
        if q_idx is None:
            continue
        xb = _faiss.rev_swig_ptr(index.get_xb(), index.ntotal * 512).reshape(index.ntotal, 512)
        query_emb = xb[q_idx]

        recs = faiss_retrieve(query_emb, top_k=max(k_values))
        all_recs.append(recs)
        all_relevant_sets.append(test_ids)

        for k in k_values:
            results[k]["precision"].append(precision_at_k(recs, test_ids, k))
            results[k]["recall"].append(recall_at_k(recs, test_ids, k))
            results[k]["ndcg"].append(ndcg_at_k(recs, test_ids, k))

    summary = {}
    for k in k_values:
        summary[f"Precision@{k}"] = float(np.mean(results[k]["precision"]))
        summary[f"Recall@{k}"] = float(np.mean(results[k]["recall"]))
        summary[f"NDCG@{k}"] = float(np.mean(results[k]["ndcg"]))
    summary[f"MAP@{max(k_values)}"] = float(mean_average_precision(all_recs, all_relevant_sets))

    return summary


# ---------------------------------------------------------------------------
# Experiment: Frame quality degradation
# ---------------------------------------------------------------------------

def degrade_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "hd":
        return image
    elif mode == "compressed":
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=15)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif mode == "blurred":
        return image.filter(ImageFilter.GaussianBlur(radius=4))
    raise ValueError(f"Unknown mode: {mode}")


def run_degradation_experiment(test_images: list[str], k: int = 10) -> dict:
    """
    For each image, embed at HD / compressed / blurred quality.
    Measure cosine similarity between HD embedding and degraded embedding,
    and Precision@K degradation vs HD retrieval as ground truth.
    """
    modes = ["hd", "compressed", "blurred"]
    results: dict[str, list] = {m: [] for m in modes}

    for img_path in test_images:
        img = Image.open(img_path).convert("RGB")
        embeddings = {}
        for mode in modes:
            degraded = degrade_image(img, mode)
            out = embed_image(degraded)
            embeddings[mode] = np.array(out["embedding"])

        # Cosine similarity vs HD
        hd_emb = embeddings["hd"]
        hd_recs = faiss_retrieve(hd_emb, top_k=k)
        hd_ids = {r["id"] for r in hd_recs}

        for mode in modes:
            emb = embeddings[mode]
            cos_sim = float(np.dot(hd_emb, emb) / (np.linalg.norm(hd_emb) * np.linalg.norm(emb) + 1e-8))
            mode_recs = faiss_retrieve(emb, top_k=k)
            p_at_k = precision_at_k(mode_recs, hd_ids, k)
            results[mode].append({"cosine_vs_hd": cos_sim, f"precision@{k}_vs_hd": p_at_k})

    summary = {}
    for mode in modes:
        summary[mode] = {
            "mean_cosine_vs_hd": float(np.mean([r["cosine_vs_hd"] for r in results[mode]])),
            f"mean_precision@{k}_vs_hd": float(np.mean([r[f"precision@{k}_vs_hd"] for r in results[mode]])),
        }
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, action="append", default=[], dest="k_values")
    parser.add_argument("--experiment", action="store_true", help="Run frame quality degradation experiment")
    parser.add_argument("--test_images", nargs="*", default=[], help="Paths to test images for experiment")
    args = parser.parse_args()

    k_values = args.k_values or [5, 10]

    if args.experiment:
        if not args.test_images:
            print("Provide --test_images <path1> <path2> ... for the degradation experiment.")
        else:
            exp_results = run_degradation_experiment(args.test_images, k=max(k_values))
            print("\n=== Frame Quality Degradation Experiment ===")
            for mode, stats in exp_results.items():
                print(f"  {mode}: {stats}")
            out_path = OUTPUT_DIR / "experiment_frame_quality.json"
            with open(out_path, "w") as f:
                json.dump(exp_results, f, indent=2)
            print(f"\nSaved → {out_path}")
    else:
        print("Running offline evaluation ...")
        summary = evaluate_model(k_values=k_values)
        print("\n=== Offline Evaluation ===")
        for metric, value in summary.items():
            print(f"  {metric}: {value:.4f}")
        out_path = OUTPUT_DIR / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved → {out_path}")
