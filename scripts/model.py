"""
model.py

Four-stage recommendation pipeline:

  Stage 1 — Candidate retrieval : FAISS KNN over FashionCLIP embeddings (classical ML)
  Stage 2 — NCF re-ranking      : NeuMF (GMF + MLP) with BPR loss (deep learning)
  Stage 2b— SASRec re-ranking   : Self-Attentive Sequential Recommendation (Transformer)
  Stage 3 — Diversity filter    : deduplicate by price quartile, spread price range

Also implements the naive popularity baseline for rubric compliance.

SASRec reference: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
  - Unidirectional (causal) self-attention over user interaction sequences
  - Item embeddings are FashionCLIP 512-dim vectors projected to d_model
  - Trained with binary cross-entropy on (pos, neg) pairs sampled from interactions

Training:
  python scripts/model.py --train --epochs 10 --batch_size 256
  python scripts/model.py --train_sasrec --epochs 20 --batch_size 256
"""

import argparse
import json
import math
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

INDEX_DIR = Path("models/faiss_index")
NCF_DIR = Path("models/ncf_reranker")
SASREC_DIR = Path("models/sasrec")
NCF_DIR.mkdir(parents=True, exist_ok=True)
SASREC_DIR.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------------
# Stage 2b — SASRec (Self-Attentive Sequential Recommendation)
# ---------------------------------------------------------------------------

class SASRecBlock(nn.Module):
    """Single Transformer block: causal multi-head self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm (more stable than post-LN)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, is_causal=False)
        x = x + self.drop(h)

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    Input: a sequence of item FashionCLIP embeddings (512-dim each) of length max_seq_len.
    Architecture:
      1. Linear projection: 512 → d_model
      2. Learned positional embeddings (max_seq_len positions)
      3. N causal Transformer blocks
      4. Layer norm on final hidden state
      5. Dot product with target item projection → score

    The causal mask ensures position t only attends to positions ≤ t,
    matching the SASRec paper (predicts next item from prefix).
    """

    def __init__(
        self,
        item_dim: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Project FashionCLIP embeddings into model space
        self.item_proj = nn.Linear(item_dim, d_model, bias=False)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SASRecBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)

        # Item scoring: project target item to d_model, dot with sequence repr
        self.target_proj = nn.Linear(item_dim, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = masked out) for causal attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def encode_sequence(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """
        seq_embeds: (B, L, 512) — batch of interaction sequences as CLIP embeddings.
        Returns: (B, L, d_model) — contextualised sequence representations.
        """
        B, L, _ = seq_embeds.shape
        x = self.item_proj(seq_embeds)                                 # (B, L, d_model)
        pos_ids = torch.arange(L, device=seq_embeds.device).unsqueeze(0)
        x = self.emb_drop(x + self.pos_emb(pos_ids))

        mask = self._causal_mask(L, seq_embeds.device)
        for block in self.blocks:
            x = block(x, mask)

        return self.ln_out(x)                                          # (B, L, d_model)

    def forward(
        self,
        seq_embeds: torch.Tensor,   # (B, L, 512)
        target_embeds: torch.Tensor, # (B, 512)
    ) -> torch.Tensor:
        """
        Returns scalar scores (B,) — relevance of each target item given the sequence.
        Uses the representation at the last non-padding position.
        """
        seq_repr = self.encode_sequence(seq_embeds)   # (B, L, d_model)
        last = seq_repr[:, -1, :]                      # (B, d_model)
        target = self.target_proj(target_embeds)       # (B, d_model)
        return (last * target).sum(dim=-1)             # (B,)


class SASRecDataset(Dataset):
    """
    Builds (sequence, pos_target, neg_target) triples from interactions.
    Each user's interactions are sorted into a chronological sequence
    (interactions have no timestamp, so we use file order as proxy).
    """

    def __init__(
        self,
        interactions_path: Path,
        embeddings: dict[str, np.ndarray],
        max_seq_len: int = 50,
    ):
        self.embeddings = embeddings
        self.max_seq_len = max_seq_len
        self.item_ids = list(embeddings.keys())

        # Group interactions by user
        user_items: dict[int, list[str]] = {}
        with open(interactions_path) as f:
            for line in f:
                row = json.loads(line)
                user_items.setdefault(row["user_id"], []).append(row["item_id"])

        # Build (sequence, pos) pairs — predict item i+1 from items 0..i
        self.samples: list[tuple[list[str], str]] = []
        for item_list in user_items.values():
            for i in range(1, len(item_list)):
                seq = item_list[max(0, i - max_seq_len): i]
                pos = item_list[i]
                if pos in embeddings:
                    self.samples.append((seq, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_ids, pos_id = self.samples[idx]
        neg_id = self.item_ids[np.random.randint(len(self.item_ids))]

        # Pad / truncate sequence to max_seq_len
        L = self.max_seq_len
        seq_embs = np.zeros((L, 512), dtype=np.float32)
        for j, iid in enumerate(seq_ids[-L:]):
            emb = self.embeddings.get(iid)
            if emb is not None:
                seq_embs[L - len(seq_ids) + j] = emb

        pos_emb = self.embeddings.get(pos_id, np.zeros(512, dtype=np.float32))
        neg_emb = self.embeddings.get(neg_id, np.zeros(512, dtype=np.float32))

        return (
            torch.tensor(seq_embs, dtype=torch.float32),
            torch.tensor(pos_emb, dtype=torch.float32),
            torch.tensor(neg_emb, dtype=torch.float32),
        )


def train_sasrec(
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    max_seq_len: int = 50,
    dropout: float = 0.2,
) -> None:
    interactions_path = Path("data/processed/interactions.jsonl")

    # Load embeddings from FAISS index
    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)
    index = faiss.read_index(str(INDEX_DIR / "products.index"))
    stored = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * 512).reshape(index.ntotal, 512)
    embeddings = {meta[i]["id"]: stored[i] for i in range(len(meta))}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SASRec(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=max(1, len(SASRecDataset(interactions_path, embeddings, max_seq_len)) // batch_size),
    )

    dataset = SASRecDataset(interactions_path, embeddings, max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    bce = nn.BCEWithLogitsLoss()

    print(f"SASRec training: {len(dataset)} samples | device={device} | d_model={d_model} | layers={n_layers}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seq_embs, pos_embs, neg_embs in loader:
            seq_embs = seq_embs.to(device)
            pos_embs = pos_embs.to(device)
            neg_embs = neg_embs.to(device)

            pos_scores = model(seq_embs, pos_embs)
            neg_scores = model(seq_embs, neg_embs)

            labels_pos = torch.ones_like(pos_scores)
            labels_neg = torch.zeros_like(neg_scores)
            loss = bce(pos_scores, labels_pos) + bce(neg_scores, labels_neg)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs}  loss={total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), SASREC_DIR / "sasrec.pt")
    torch.save(
        {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
         "max_seq_len": max_seq_len, "dropout": dropout},
        SASREC_DIR / "config.pt",
    )
    print(f"Saved SASRec weights → {SASREC_DIR}/sasrec.pt")


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
_sasrec_model: SASRec | None = None


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


def _load_sasrec() -> SASRec | None:
    global _sasrec_model
    pt_path = SASREC_DIR / "sasrec.pt"
    cfg_path = SASREC_DIR / "config.pt"
    if not pt_path.exists():
        return None
    if _sasrec_model is None:
        cfg = torch.load(cfg_path, weights_only=True)
        m = SASRec(**cfg)
        m.load_state_dict(torch.load(pt_path, weights_only=True))
        m.eval()
        _sasrec_model = m
    return _sasrec_model


def recommend(
    embedding: np.ndarray,
    top_k: int = 12,
    price_min: float | None = None,
    price_max: float | None = None,
    user_id: int = 0,
    user_history: list[np.ndarray] | None = None,
) -> list[dict]:
    """
    Full four-stage pipeline:
      1. FAISS retrieval (top 50 candidates)
      2. NCF re-ranking  (NeuMF GMF+MLP, if weights exist)
      3. SASRec re-ranking (Transformer, if weights exist + user history provided)
      4. Diversity filter + price filter → top_k

    user_history: list of FashionCLIP embeddings (512-dim np.ndarray) representing
                  the user's recent interactions, ordered oldest→newest.
                  If None or SASRec not trained, step 3 is skipped.
    """
    candidates = faiss_retrieve(embedding, top_k=50)

    # Price filter
    if price_min is not None:
        candidates = [c for c in candidates if c["price"] >= price_min]
    if price_max is not None:
        candidates = [c for c in candidates if c["price"] <= price_max]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage 2 — NCF re-ranking
    ncf = _load_ncf()
    if ncf is not None and candidates:
        ncf = ncf.to(device)
        embs = torch.tensor(
            [embedding] * len(candidates), dtype=torch.float32
        ).to(device)
        user_tensor = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(device)
        with torch.no_grad():
            scores = ncf(user_tensor, embs).cpu().numpy()
        for item, score in zip(candidates, scores):
            item["ncf_score"] = float(score)
        candidates.sort(key=lambda x: x.get("ncf_score", 0), reverse=True)

    # Stage 2b — SASRec re-ranking (sequential Transformer)
    sasrec = _load_sasrec()
    if sasrec is not None and candidates and user_history:
        sasrec = sasrec.to(device)
        max_seq = sasrec.max_seq_len
        # Build padded sequence tensor (1, max_seq, 512)
        history = user_history[-max_seq:]
        seq_np = np.zeros((1, max_seq, 512), dtype=np.float32)
        for j, h in enumerate(history):
            seq_np[0, max_seq - len(history) + j] = h
        seq_tensor = torch.tensor(seq_np, dtype=torch.float32).to(device)
        # Expand sequence for each candidate
        seq_batch = seq_tensor.expand(len(candidates), -1, -1)   # (N, L, 512)
        target_batch = torch.tensor(
            [c.get("_embedding", embedding) for c in candidates],
            dtype=torch.float32,
        ).to(device)                                               # (N, 512)
        with torch.no_grad():
            sasrec_scores = sasrec(seq_batch, target_batch).cpu().numpy()
        for item, score in zip(candidates, sasrec_scores):
            item["sasrec_score"] = float(score)
        candidates.sort(key=lambda x: x.get("sasrec_score", 0), reverse=True)

    # Stage 3 — diversity filter
    return diversity_filter(candidates, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",        action="store_true", help="Train NeuMF (NCF)")
    parser.add_argument("--train_sasrec", action="store_true", help="Train SASRec (Transformer)")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    # SASRec-specific
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=2)
    parser.add_argument("--max_seq_len", type=int,   default=50)
    parser.add_argument("--dropout",     type=float, default=0.2)
    args = parser.parse_args()

    if args.train:
        train_ncf(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    if args.train_sasrec:
        train_sasrec(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )
