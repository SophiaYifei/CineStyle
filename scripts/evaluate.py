# AI tools used: Claude (Anthropic) assisted with structuring the four-model comparison loop, matplotlib chart generation, and error analysis visualization.
"""
evaluate.py

Offline evaluation for CineStyle recommendation models.

Metrics:
  - Precision@K
  - Recall@K
  - NDCG@K
  - MAP@K
  - Visual similarity score (mean cosine distance in embedding space)

Four-model comparison (default):
  Popularity baseline, FAISS KNN, NeuMF re-ranker, SASRec re-ranker

Experiment: Frame quality vs. recommendation accuracy
  --experiment flag runs the degradation study (HD / compressed / blurred)

Hyperparameter tuning:
  --tune flag runs NCF embed_dim comparison (16, 32, 64, 128, 256)

Error analysis:
  --error_analysis flag finds 5 category-mismatch mispredictions

Usage:
  python scripts/evaluate.py --k 5 --k 10
  python scripts/evaluate.py --experiment
  python scripts/evaluate.py --tune
  python scripts/evaluate.py --error_analysis
"""

import argparse
import json
import math
from collections import Counter
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
# Shared data loading helpers
# ---------------------------------------------------------------------------

def _load_eval_data():
    """Load FAISS index, metadata, interactions, and embeddings for evaluation."""
    import faiss

    from scripts.build_features import INDEX_DIR

    interactions_path = Path("data/processed/interactions.jsonl")
    with open(interactions_path) as f:
        interactions = [json.loads(line) for line in f]

    user_items: dict[int, list[str]] = {}
    for row in interactions:
        user_items.setdefault(row["user_id"], []).append(row["item_id"])

    index = faiss.read_index(str(INDEX_DIR / "products.index"))
    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)

    id_to_idx = {item["id"]: i for i, item in enumerate(meta)}
    DIM = 512
    xb = faiss.rev_swig_ptr(
        index.get_xb(), index.ntotal * DIM
    ).reshape(index.ntotal, DIM).copy()

    embeddings = {meta[i]["id"]: xb[i] for i in range(len(meta))}

    # Popularity counts
    item_counts = Counter(row["item_id"] for row in interactions)
    popular_ranked = sorted(item_counts.keys(),
                            key=lambda x: item_counts[x], reverse=True)

    return user_items, meta, id_to_idx, xb, embeddings, popular_ranked, item_counts


def _popularity_recommend(popular_ranked, id_to_idx, meta, item_counts, top_k=10):
    """Return top-K most popular items as recommendations."""
    results = []
    for item_id in popular_ranked[:top_k]:
        idx = id_to_idx.get(item_id)
        if idx is not None:
            item = dict(meta[idx])
            item["similarity"] = item_counts[item_id]
            results.append(item)
    return results


def _ncf_rerank(ncf_model, candidates, user_id, embeddings, device):
    """Re-rank candidates with NeuMF given a user ID."""
    import torch
    if not candidates:
        return candidates
    ncf_model.eval()
    with torch.no_grad():
        u_tensor = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(device)
        item_embs_t = torch.tensor(
            np.array([embeddings.get(c["id"], np.zeros(512, dtype=np.float32))
                      for c in candidates]),
            dtype=torch.float32,
        ).to(device)
        ncf_scores = ncf_model(u_tensor, item_embs_t).cpu().numpy()
    for c, s in zip(candidates, ncf_scores):
        c["ncf_score"] = float(s)
    return sorted(candidates, key=lambda x: x.get("ncf_score", 0), reverse=True)


def _sasrec_rerank(sasrec_model, candidates, user_history_embs, embeddings, device, max_seq_len=50):
    """Re-rank candidates with SASRec given user interaction history."""
    import torch
    if not user_history_embs or not candidates:
        return candidates
    L = max_seq_len
    history = user_history_embs[-L:]
    seq_np = np.zeros((1, L, 512), dtype=np.float32)
    for j, h in enumerate(history):
        seq_np[0, L - len(history) + j] = h
    seq_tensor = torch.tensor(seq_np, dtype=torch.float32).to(device)
    seq_batch = seq_tensor.expand(len(candidates), -1, -1)
    target_embs = torch.tensor(
        np.array([embeddings.get(c["id"], np.zeros(512, dtype=np.float32))
                  for c in candidates]),
        dtype=torch.float32,
    ).to(device)
    sasrec_model.eval()
    with torch.no_grad():
        scores = sasrec_model(seq_batch, target_embs).cpu().numpy()
    for c, s in zip(candidates, scores):
        c["sasrec_score"] = float(s)
    return sorted(candidates, key=lambda x: x.get("sasrec_score", 0), reverse=True)


# ---------------------------------------------------------------------------
# Held-out evaluation: four-model comparison
# ---------------------------------------------------------------------------

def evaluate_model(k_values: list[int] = [5, 10]) -> dict:
    """
    Run offline evaluation on held-out interactions.
    Compares four models: Popularity, FAISS KNN, NeuMF, SASRec.
    Uses last 20% of each user's interactions as ground truth.
    """
    import torch

    from scripts.model import _load_ncf, _load_sasrec

    user_items, meta, id_to_idx, xb, embeddings, popular_ranked, item_counts = _load_eval_data()

    device = "cpu"
    ncf_model = _load_ncf()
    sasrec_model = _load_sasrec()
    sasrec_seq_len = sasrec_model.max_seq_len if sasrec_model else 50

    K_VALUES = k_values
    N_EVAL = 200

    results_pop = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    results_faiss = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    results_ncf = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    results_sasrec = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    all_recs_p, all_recs_f, all_recs_n, all_recs_s, all_rel = [], [], [], [], []

    for user_id, item_ids in list(user_items.items())[:N_EVAL]:
        split = max(1, int(len(item_ids) * 0.8))
        train_ids = item_ids[:split]
        test_ids = set(item_ids[split:])
        if not test_ids or len(train_ids) < 2:
            continue

        q_idx = id_to_idx.get(train_ids[-1])
        if q_idx is None:
            continue
        query_emb = xb[q_idx]
        history_embs = [xb[id_to_idx[i]] for i in train_ids[:-1] if i in id_to_idx]

        # Popularity baseline (naive)
        recs_p = _popularity_recommend(popular_ranked, id_to_idx, meta, item_counts,
                                       top_k=max(K_VALUES))

        # FAISS KNN (classical ML)
        recs_f = faiss_retrieve(query_emb, top_k=max(K_VALUES))

        # NCF re-rank (feedforward deep learning)
        if ncf_model is not None:
            recs_n = _ncf_rerank(ncf_model, list(recs_f), user_id, embeddings, device)
        else:
            recs_n = list(recs_f)

        # SASRec re-rank (Transformer deep learning)
        if sasrec_model is not None:
            recs_s = _sasrec_rerank(sasrec_model, list(recs_f), history_embs,
                                    embeddings, device, sasrec_seq_len)
        else:
            recs_s = list(recs_f)

        all_recs_p.append(recs_p)
        all_recs_f.append(recs_f)
        all_recs_n.append(recs_n)
        all_recs_s.append(recs_s)
        all_rel.append(test_ids)

        for k in K_VALUES:
            for recs, res in [(recs_p, results_pop), (recs_f, results_faiss),
                              (recs_n, results_ncf), (recs_s, results_sasrec)]:
                res[k]["precision"].append(precision_at_k(recs, test_ids, k))
                res[k]["recall"].append(recall_at_k(recs, test_ids, k))
                res[k]["ndcg"].append(ndcg_at_k(recs, test_ids, k))

    map_pop = float(mean_average_precision(all_recs_p, all_rel))
    map_faiss = float(mean_average_precision(all_recs_f, all_rel))
    map_ncf = float(mean_average_precision(all_recs_n, all_rel))
    map_sasrec = float(mean_average_precision(all_recs_s, all_rel))

    summary = {}
    print(f"\n{'Model':<14} {'Metric':<15} Value")
    print("-" * 45)
    for label, results_k, map_val in [
        ("Popularity", results_pop, map_pop),
        ("FAISS KNN", results_faiss, map_faiss),
        ("NeuMF", results_ncf, map_ncf),
        ("SASRec", results_sasrec, map_sasrec),
    ]:
        for k in K_VALUES:
            for metric in ["precision", "recall", "ndcg"]:
                key = f"{label} {metric.capitalize()}@{k}"
                val = float(np.mean(results_k[k][metric]))
                summary[key] = val
                print(f"{label:<14} {metric.capitalize()+'@'+str(k):<15} {val:.4f}")
        summary[f"{label} MAP@{max(K_VALUES)}"] = map_val
        print(f"{label:<14} {'MAP@'+str(max(K_VALUES)):<15} {map_val:.4f}")

    return summary


def plot_eval_chart(summary: dict, out_path: Path) -> None:
    """Plot horizontal bar chart of all evaluation metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary.keys())
    values = list(summary.values())

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.35)))
    bars = ax.barh(labels, values, color="#d97706", alpha=0.85)
    ax.set_xlim(0, max(values) * 1.3 if values else 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("CineStyle — Offline Retrieval Metrics (4 Models)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved chart → {out_path}")


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


def plot_experiment_chart(summary: dict, k: int, out_path: Path) -> None:
    """Plot degradation experiment bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(summary.keys())
    cos_vals = [summary[m]["mean_cosine_vs_hd"] for m in modes]
    p_vals = [summary[m][f"mean_precision@{k}_vs_hd"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].bar(modes, cos_vals, color="#d97706", alpha=0.85)
    axes[0].set_title("Mean cosine similarity vs HD")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(modes, p_vals, color="#92400e", alpha=0.85)
    axes[1].set_title(f"Mean P@{k} vs HD retrieval")
    axes[1].set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xlabel("Frame quality")

    plt.suptitle("Frame Quality Degradation Experiment", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved chart → {out_path}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning: NCF embed_dim
# ---------------------------------------------------------------------------

def run_hyperparameter_tuning(k_values: list[int] = [5, 10]) -> dict:
    """
    Train NeuMF with varying embed_dim values and evaluate NDCG@10.
    Saves results to data/outputs/hyperparam_tuning.json and .png.
    """
    import torch
    from torch.utils.data import DataLoader

    from scripts.model import NeuMF, BPRDataset, bpr_loss, INDEX_DIR

    user_items, meta, id_to_idx, xb, embeddings, popular_ranked, item_counts = _load_eval_data()

    interactions_path = Path("data/processed/interactions.jsonl")
    with open(interactions_path) as f:
        interactions = [json.loads(line) for line in f]
    n_users = max(row["user_id"] for row in interactions) + 1

    dataset = BPRDataset(interactions_path, embeddings)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    TUNE_DIMS = [16, 32, 64, 128, 256]
    TUNE_EPOCHS = 10
    device = "cpu"
    tune_results = {}

    for dim in TUNE_DIMS:
        print(f"\n{'=' * 50}")
        print(f"Training NeuMF with embed_dim={dim}")
        print(f"{'=' * 50}")

        tune_model = NeuMF(n_users=n_users, embed_dim=dim).to(device)
        tune_opt = torch.optim.Adam(tune_model.parameters(), lr=1e-3)

        avg_loss = 0.0
        for epoch in range(1, TUNE_EPOCHS + 1):
            tune_model.train()
            total = 0.0
            for user_ids, pos_embs, neg_embs in loader:
                user_ids = user_ids.to(device)
                pos_embs = pos_embs.to(device)
                neg_embs = neg_embs.to(device)
                pos_scores = tune_model(user_ids, pos_embs)
                neg_scores = tune_model(user_ids, neg_embs)
                loss = bpr_loss(pos_scores, neg_scores)
                tune_opt.zero_grad()
                loss.backward()
                tune_opt.step()
                total += loss.item()
            avg_loss = total / len(loader)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:02d}  loss={avg_loss:.4f}")

        # Evaluate with NDCG@10
        tune_model.eval()
        ndcg_scores = []
        for uid, iids in list(user_items.items())[:100]:
            split = max(1, int(len(iids) * 0.8))
            train_ids = iids[:split]
            test_ids = set(iids[split:])
            if not test_ids or len(train_ids) < 2:
                continue
            q_idx = id_to_idx.get(train_ids[-1])
            if q_idx is None:
                continue
            cands = faiss_retrieve(xb[q_idx], top_k=10)
            with torch.no_grad():
                u_t = torch.tensor([uid] * len(cands), dtype=torch.long).to(device)
                i_t = torch.tensor(
                    np.array([embeddings.get(c["id"], np.zeros(512, dtype=np.float32))
                              for c in cands]),
                    dtype=torch.float32).to(device)
                scores = tune_model(u_t, i_t).cpu().numpy()
            for c, s in zip(cands, scores):
                c["ncf_score"] = float(s)
            ranked = sorted(cands, key=lambda x: x.get("ncf_score", 0), reverse=True)
            ndcg_scores.append(ndcg_at_k(ranked, test_ids, 10))

        mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        tune_results[dim] = {"final_loss": float(avg_loss), "ndcg@10": mean_ndcg}
        print(f"  → embed_dim={dim}  NDCG@10={mean_ndcg:.4f}  loss={avg_loss:.4f}")

    return tune_results


def plot_tuning_chart(tune_results: dict, out_path: Path) -> None:
    """Plot hyperparameter tuning bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = list(tune_results.keys())
    ndcgs = [tune_results[d]["ndcg@10"] for d in dims]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([str(d) for d in dims], ndcgs, color="#d97706", alpha=0.85)
    ax.set_xlabel("embed_dim")
    ax.set_ylabel("NDCG@10")
    ax.set_title("NCF Hyperparameter Tuning: embed_dim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved chart → {out_path}")


# ---------------------------------------------------------------------------
# Error analysis: category mispredictions
# ---------------------------------------------------------------------------

def run_error_analysis() -> list[dict]:
    """
    Find 5 category-mismatch mispredictions from FAISS retrieval.
    Saves results to data/outputs/error_analysis.json and .png.
    """
    from scripts.build_features import INDEX_DIR, _get_image_embeds, _load_model

    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)

    # Load catalog for category info
    catalog_path = Path("data/processed/catalog.jsonl")
    cat_lookup = {}
    if catalog_path.exists():
        with open(catalog_path) as f:
            for line in f:
                item = json.loads(line)
                cat_lookup[item["id"]] = item.get("category", "")

    model, processor = _load_model()
    error_cases = []

    for item in meta[:300]:
        img_path = item.get("image_url", "")
        if not Path(img_path).exists():
            continue

        probe_cat = cat_lookup.get(item["id"], "")
        if not probe_cat:
            continue

        img = Image.open(img_path).convert("RGB")
        import torch
        with torch.no_grad():
            probe_emb = _get_image_embeds([img], model, processor).squeeze(0).cpu().float().numpy()

        recs = faiss_retrieve(probe_emb, top_k=5)

        for rank, r in enumerate(recs, 1):
            rec_cat = cat_lookup.get(r["id"], "")
            if rec_cat and rec_cat != probe_cat and r["id"] != item["id"]:
                error_cases.append({
                    "rank": rank,
                    "probe_id": item["id"],
                    "probe_category": probe_cat,
                    "probe_image": img_path,
                    "rec_id": r["id"],
                    "rec_category": rec_cat,
                    "rec_image": r.get("image_url", ""),
                    "similarity": r["similarity"],
                })
                break  # one error per probe

        if len(error_cases) >= 5:
            break

    print("=== Error Analysis: 5 Category Mispredictions ===\n")
    for i, e in enumerate(error_cases, 1):
        print(f"{i}. Query: {e['probe_category']} (id={e['probe_id']})")
        print(f"   Recommended at rank {e['rank']}: {e['rec_category']} (id={e['rec_id']})")
        print(f"   Cosine similarity: {e['similarity']:.4f}\n")

    return error_cases


def plot_error_analysis(error_cases: list[dict], out_path: Path) -> None:
    """Visualize the 5 error pairs as a grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(5, len(error_cases))
    if n == 0:
        print("No error cases to plot.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = [axes]

    for row, e in enumerate(error_cases[:n]):
        for col, (key, label) in enumerate([
            ("probe_image", f"Query: {e['probe_category']}"),
            ("rec_image", f"Rec rank {e['rank']}: {e['rec_category']}"),
        ]):
            path = e[key]
            if Path(path).exists():
                axes[row][col].imshow(Image.open(path).convert("RGB"))
            axes[row][col].set_title(label, fontsize=9)
            axes[row][col].axis("off")

    plt.suptitle("Error Analysis: Category Mispredictions", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved chart → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, action="append", default=[], dest="k_values")
    parser.add_argument("--experiment", action="store_true",
                        help="Run frame quality degradation experiment")
    parser.add_argument("--test_images", nargs="*", default=[],
                        help="Paths to test images for experiment")
    parser.add_argument("--tune", action="store_true",
                        help="Run NCF embed_dim hyperparameter tuning")
    parser.add_argument("--error_analysis", action="store_true",
                        help="Run error analysis on category mispredictions")
    args = parser.parse_args()

    k_values = args.k_values or [5, 10]

    if args.tune:
        print("Running hyperparameter tuning (NCF embed_dim) ...")
        tune_results = run_hyperparameter_tuning(k_values=k_values)
        out_json = OUTPUT_DIR / "hyperparam_tuning.json"
        with open(out_json, "w") as f:
            json.dump(tune_results, f, indent=2)
        print(f"\nSaved → {out_json}")
        plot_tuning_chart(tune_results, OUTPUT_DIR / "hyperparam_tuning.png")

    elif args.error_analysis:
        print("Running error analysis ...")
        error_cases = run_error_analysis()
        out_json = OUTPUT_DIR / "error_analysis.json"
        with open(out_json, "w") as f:
            json.dump(error_cases, f, indent=2)
        print(f"Saved → {out_json}")
        plot_error_analysis(error_cases, OUTPUT_DIR / "error_analysis.png")

    elif args.experiment:
        if not args.test_images:
            # Auto-discover test images from catalog
            catalog_path = Path("data/processed/catalog.jsonl")
            if catalog_path.exists():
                with open(catalog_path) as f:
                    items = [json.loads(line) for line in f]
                test_imgs = [i["image_url"] for i in items[:20]
                             if Path(i["image_url"]).exists()]
                if test_imgs:
                    args.test_images = test_imgs
            if not args.test_images:
                print("Provide --test_images <path1> <path2> ... for the degradation experiment.")
                exit(1)

        exp_results = run_degradation_experiment(args.test_images, k=max(k_values))
        print("\n=== Frame Quality Degradation Experiment ===")
        for mode, stats in exp_results.items():
            print(f"  {mode}: {stats}")
        out_path = OUTPUT_DIR / "experiment_frame_quality.json"
        with open(out_path, "w") as f:
            json.dump(exp_results, f, indent=2)
        print(f"\nSaved → {out_path}")
        plot_experiment_chart(exp_results, max(k_values), OUTPUT_DIR / "experiment_chart.png")

    else:
        print("Running four-model offline evaluation ...")
        summary = evaluate_model(k_values=k_values)
        out_path = OUTPUT_DIR / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved → {out_path}")
        plot_eval_chart(summary, OUTPUT_DIR / "eval_chart.png")
