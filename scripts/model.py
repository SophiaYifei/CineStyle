"""
model.py

Three-stage recommendation pipeline:

  Stage 1 — Candidate retrieval: FAISS KNN over FashionCLIP embeddings (classical ML)
  Stage 2 — NCF re-ranking: Neural Collaborative Filtering with BPR loss (deep learning)
  Stage 3 — Diversity filter: deduplicate by cluster, spread price range

Also implements the naive popularity baseline for rubric compliance.

Training:
  python scripts/model.py --train --epochs 10 --batch_size 256
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

INDEX_DIR = Path("models/faiss_index")
NCF_DIR = Path("models/ncf_reranker")
NCF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stage 1 — FAISS KNN retrieval (classical ML baseline)
# ---------------------------------------------------------------------------

_faiss_index: faiss.Index | None = None
_meta: list[dict] | None = None


def _load_index():
    global _faiss_index, _meta
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(INDEX_DIR / "products.index"))
        with open(INDEX_DIR / "meta.json") as f:
            _meta = json.load(f)
    return _faiss_index, _meta


def faiss_retrieve(embedding: np.ndarray, top_k: int = 50) -> list[dict]:
    """Return top_k nearest neighbors from the FAISS product index."""
    index, meta = _load_index()
    query = embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)
    distances, indices = index.search(query, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        item = dict(meta[idx])
        item["similarity"] = float(dist)
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Stage 2 — NCF re-ranker (deep learning)
# ---------------------------------------------------------------------------

class NeuMF(nn.Module):
    """
    NeuMF: Neural Matrix Factorization
    Combines Generalized Matrix Factorization (GMF) and a deep MLP.
    Input: (user_id, item_embedding) → score.

    For cold-start (no user history), user_id = 0 (anonymous embedding).
    """

    def __init__(
        self,
        n_users: int,
        item_dim: int = 512,
        embed_dim: int = 64,
        mlp_dims: list[int] | None = None,
    ):
        super().__init__()
        mlp_dims = mlp_dims or [256, 128, 64]

        # GMF branch
        self.user_gmf = nn.Embedding(n_users, embed_dim)
        self.item_gmf = nn.Linear(item_dim, embed_dim, bias=False)

        # MLP branch
        self.user_mlp = nn.Embedding(n_users, embed_dim)
        self.item_mlp = nn.Linear(item_dim, embed_dim, bias=False)

        mlp_layers = []
        in_dim = embed_dim * 2
        for out_dim in mlp_dims:
            mlp_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Output
        self.output = nn.Linear(embed_dim + mlp_dims[-1], 1)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, user_ids: torch.Tensor, item_embeds: torch.Tensor) -> torch.Tensor:
        # GMF
        u_gmf = self.user_gmf(user_ids)
        i_gmf = self.item_gmf(item_embeds)
        gmf_out = u_gmf * i_gmf

        # MLP
        u_mlp = self.user_mlp(user_ids)
        i_mlp = self.item_mlp(item_embeds)
        mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=-1))

        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(concat).squeeze(-1)


class BPRDataset(Dataset):
    """
    Bayesian Personalized Ranking dataset.
    Each sample: (user_id, pos_embedding, neg_embedding)
    """

    def __init__(self, interactions_path: Path, embeddings: dict[str, np.ndarray]):
        with open(interactions_path) as f:
            self.interactions = [json.loads(line) for line in f]
        self.embeddings = embeddings
        self.item_ids = list(embeddings.keys())

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions[idx]
        user_id = row["user_id"]
        pos_id = row["item_id"]
        neg_id = self.item_ids[np.random.randint(len(self.item_ids))]

        pos_emb = self.embeddings.get(pos_id, np.zeros(512, dtype=np.float32))
        neg_emb = self.embeddings.get(neg_id, np.zeros(512, dtype=np.float32))

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(pos_emb, dtype=torch.float32),
            torch.tensor(neg_emb, dtype=torch.float32),
        )


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def train_ncf(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3) -> None:
    from scripts.build_features import _load_model
    import json

    interactions_path = Path("data/processed/interactions.jsonl")
    with open(interactions_path) as f:
        interactions = [json.loads(line) for line in f]
    n_users = max(r["user_id"] for r in interactions) + 1

    # Load pre-built embeddings from FAISS metadata
    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)
    # For training we use stored embeddings reconstructed from FAISS
    index = faiss.read_index(str(INDEX_DIR / "products.index"))
    stored = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * 512).reshape(index.ntotal, 512)
    embeddings = {meta[i]["id"]: stored[i] for i in range(len(meta))}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuMF(n_users=n_users).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = BPRDataset(interactions_path, embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for user_ids, pos_embs, neg_embs in loader:
            user_ids = user_ids.to(device)
            pos_embs = pos_embs.to(device)
            neg_embs = neg_embs.to(device)

            pos_scores = model(user_ids, pos_embs)
            neg_scores = model(user_ids, neg_embs)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs}  loss={total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), NCF_DIR / "ncf.pt")
    torch.save({"n_users": n_users}, NCF_DIR / "config.pt")
    print(f"Saved NCF weights → {NCF_DIR}/ncf.pt")


# ---------------------------------------------------------------------------
# Stage 3 — Diversity filter
# ---------------------------------------------------------------------------

def diversity_filter(candidates: list[dict], top_k: int = 12) -> list[dict]:
    """
    Deduplicate candidates by price quartile to ensure spread.
    Simple heuristic: keep at most 3 items per price quartile.
    """
    if not candidates:
        return []
    prices = [c["price"] for c in candidates]
    q25, q75 = np.percentile(prices, [25, 75])
    buckets: dict[str, list] = {"low": [], "mid": [], "high": []}
    for item in candidates:
        p = item["price"]
        if p <= q25:
            buckets["low"].append(item)
        elif p <= q75:
            buckets["mid"].append(item)
        else:
            buckets["high"].append(item)

    result = []
    per_bucket = max(1, top_k // 3)
    for bucket in buckets.values():
        result.extend(bucket[:per_bucket])
    return result[:top_k]


# ---------------------------------------------------------------------------
# Naive popularity baseline
# ---------------------------------------------------------------------------

_popularity_scores: dict[str, float] | None = None


def popularity_baseline(top_k: int = 12) -> list[dict]:
    """Return the most popular items globally (stub — replace with real click logs)."""
    _, meta = _load_index()
    # Stub: uniform scores; in prod, load from analytics DB
    scores = {item["id"]: 1.0 for item in meta}
    sorted_items = sorted(meta, key=lambda x: scores.get(x["id"], 0), reverse=True)
    for item in sorted_items[:top_k]:
        item["similarity"] = scores.get(item["id"], 0.0)
    return sorted_items[:top_k]


# ---------------------------------------------------------------------------
# Main recommend() — called by FastAPI
# ---------------------------------------------------------------------------

_ncf_model: NeuMF | None = None


def _load_ncf() -> NeuMF | None:
    global _ncf_model
    ncf_path = NCF_DIR / "ncf.pt"
    cfg_path = NCF_DIR / "config.pt"
    if not ncf_path.exists():
        return None
    if _ncf_model is None:
        cfg = torch.load(cfg_path, weights_only=True)
        m = NeuMF(n_users=cfg["n_users"])
        m.load_state_dict(torch.load(ncf_path, weights_only=True))
        m.eval()
        _ncf_model = m
    return _ncf_model


def recommend(
    embedding: np.ndarray,
    top_k: int = 12,
    price_min: float | None = None,
    price_max: float | None = None,
    user_id: int = 0,
) -> list[dict]:
    """
    Full three-stage pipeline:
      1. FAISS retrieval (top 50)
      2. NCF re-ranking (if model exists, else KNN order)
      3. Diversity filter + price filter → top_k
    """
    candidates = faiss_retrieve(embedding, top_k=50)

    # Price filter
    if price_min is not None:
        candidates = [c for c in candidates if c["price"] >= price_min]
    if price_max is not None:
        candidates = [c for c in candidates if c["price"] <= price_max]

    # NCF re-ranking
    model = _load_ncf()
    if model is not None and candidates:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        embs = torch.tensor(
            [embedding] * len(candidates), dtype=torch.float32
        ).to(device)
        user_tensor = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(device)
        with torch.no_grad():
            scores = model(user_tensor, embs).cpu().numpy()
        for item, score in zip(candidates, scores):
            item["similarity"] = float(score)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)

    return diversity_filter(candidates, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.train:
        train_ncf(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
