# CineStyle

> *You're watching a show. Someone walks in wearing the perfect outfit. You want it. Now you can find it.*

**AIPI 540 · Module Project · Duke University**

---

## What It Does

CineStyle is a real-time fashion identification and recommendation system for film and TV viewers. While watching a scene, a user can capture a frame, select a character, and ask: *"What is she wearing?"* — and get both an identification of the garment and style-matched recommendations to buy or recreate it.

```
User captures a frame from a show
              ↓
  Character + garment region selected
              ↓
  Computer vision pipeline
  → Garment type / color / style attributes extracted
              ↓
  Recommendation engine
  → Similar items ranked by visual + style similarity
              ↓
  Product cards with links to purchase
```

The key contribution is the **end-to-end pipeline from passive viewing to active discovery** — combining vision-based garment parsing with a hybrid recommendation system that uses both visual embeddings and structured style attributes.

---

## Rubric Alignment

### Problem & Motivation

Viewers regularly notice clothing in shows and films but have no frictionless way to identify or shop it. Existing tools (Google Lens, ShopLook) require manual image search with no scene context. CineStyle closes this gap by making fashion discovery a native part of the viewing experience.

### Three Required Modeling Approaches

| Approach | Model | Role |
|---|---|---|
| Naive baseline | Popularity-based recommender (most-clicked items in same genre) | Baseline |
| Classical ML | KNN with CLIP visual embeddings + cosine similarity | Item retrieval |
| Deep learning | Fine-tuned CLIP or FashionCLIP + NCF-style re-ranker | Final model |

All three are implemented, documented, and benchmarked. The deployed app uses the deep learning model.

### Evaluation Strategy

**Offline:**
- Precision@K, Recall@K, NDCG@K (K = 5, 10)
- Mean Average Precision (MAP@K)
- Visual similarity score (cosine distance in embedding space)

**Online (in-app):**
- Click-through rate on recommendations
- "Save to wishlist" rate
- Time-to-first-click

### Experiment

**Experiment: Frame quality vs. recommendation accuracy**

We vary the input frame quality (full HD, compressed, blurred) and measure how retrieval precision degrades. Motivation: real users will capture frames via screenshot on various devices. This tests robustness of the vision pipeline to real-world degradation.

---

## Technical Architecture

### Vision Pipeline (Garment Identification)

```
Input: video frame (image)
  ↓
Person detection: YOLOv8 or GroundingDINO
  ↓
Garment region crop
  ↓
Attribute extraction: FashionCLIP or fine-tuned CLIP
  → garment type, color, texture, aesthetic label
  ↓
Embedding vector (512-dim)
```

**Models considered:**
- `patrickjohncyh/fashion-clip` — CLIP fine-tuned on fashion data (Farfetch dataset, 700K items)
- `SCHP` (Self-Correction for Human Parsing) — semantic segmentation for garment region isolation
- YOLOv8-pose — person bounding box + keypoint-guided crop

### Recommendation Engine

Three-stage pipeline:

**Stage 1 — Candidate Retrieval**
- KNN search over FAISS index of product embeddings
- Returns top-50 visually similar items
- Baseline: popularity rank within detected aesthetic category

**Stage 2 — Re-ranking (Deep Learning)**
- NCF-style model trained on implicit feedback (clicks, saves, purchases)
- Input: user context (session history) + item visual embedding + item metadata
- Loss: BPR (Bayesian Personalized Ranking)

**Stage 3 — Diversity Filter**
- Deduplicate by color/silhouette cluster
- Ensure price range spread

### Data Sources

| Source | Purpose |
|---|---|
| Polyvore dataset | Outfit compatibility training |
| DeepFashion2 | Garment detection + attribute labels |
| Farfetch (via FashionCLIP) | Product embeddings |
| iMaterialist (FGVC) | Fine-grained garment segmentation |
| TMDB stills API | Frame sourcing for demo content |
| Amazon Product API / Nordstrom API | Live product cards |

### Handling Recommendation Challenges

**Sparsity:** New users have no history → cold-start handled by content-based fallback (visual similarity only, no collaborative signal)

**Popularity bias:** Genre-based popularity baseline is our naive model; the neural re-ranker penalizes over-recommended items using a long-tail boost term

**Position effects:** Recommendations are randomized in A/B within price tier to avoid position bias in evaluation

---

## Project Structure

```
cinestyle/
├── README.md
├── requirements.txt
├── setup.py
├── main.py                          # FastAPI backend entry point
├── scripts/
│   ├── make_dataset.py              # Download + preprocess DeepFashion2, Polyvore
│   ├── build_features.py            # Extract CLIP embeddings, build FAISS index
│   ├── model.py                     # Train NCF re-ranker
│   └── evaluate.py                  # Offline eval: Precision@K, NDCG@K, MAP@K
├── models/
│   ├── faiss_index/                 # Product embedding index
│   ├── ncf_reranker/                # Trained NCF weights
│   └── baselines/                   # KNN and popularity models
├── data/
│   ├── raw/                         # Raw dataset downloads
│   ├── processed/                   # Embeddings, interaction logs
│   └── outputs/                     # Eval results, experiment logs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_embedding_exploration.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_experiment_frame_quality.ipynb
├── frontend/                        # Next.js UI
│   ├── app/
│   │   ├── page.tsx                 # Main viewer interface
│   │   ├── components/
│   │   │   ├── FrameCapture.tsx     # Video frame selection
│   │   │   ├── GarmentHighlight.tsx # Click-to-select overlay
│   │   │   └── RecommendationCard.tsx
│   └── ...
└── .gitignore
```

---

## Application Design (UX)

The app is not a Streamlit demo. It is a full editorial-style web experience.

**Flow:**
1. User pastes a show/episode link OR uploads a still image
2. Frame is rendered with a "tap to identify" overlay
3. User clicks on any garment region
4. A panel slides in with:
   - Identified item (type, color, aesthetic label, similar celebrity looks)
   - 6–12 shoppable recommendations in a scroll rail (price range, brand, link)
   - "See more like this" → deeper search
5. User can save items to a wishlist, share a look, or filter by price

**Design language:** Dark cinema aesthetic, warm amber accents, editorial typography — feels like a luxury fashion app, not a class project.

**Tech stack:**

| Layer | Tool |
|---|---|
| Frontend | Next.js + Tailwind CSS |
| Backend | FastAPI + uvicorn |
| Vision inference | Hugging Face Transformers (FashionCLIP) |
| Vector search | FAISS |
| Deployment | Modal or Railway (backend) + Vercel (frontend) |
| Compute | GPU inference via Modal serverless |

---

## Running Locally

**Backend:**
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

**Build product index:**
```bash
python scripts/make_dataset.py
python scripts/build_features.py   # builds FAISS index
```

**Train re-ranker:**
```bash
python scripts/model.py --epochs 10 --batch_size 256
```

**Evaluate:**
```bash
python scripts/evaluate.py --k 5 --k 10
```

---

## Ethics Statement

- Fashion data encodes narrow body and aesthetic standards — outputs may skew toward slim, Western, Eurocentric styles
- Affiliate/purchase links create a commercial incentive that can conflict with genuine user interest
- Character identification (linking outfits to actors) risks misidentification across demographic groups
- Visual similarity search does not account for accessibility, sizing, or sustainability — future versions should surface these filters prominently

---

## Commercial Viability

Strong. The "shop the look" market is validated (LTK, ShopLook, Amazon's "Find on Amazon" feature). CineStyle's differentiation is:

1. **Context awareness** — recommendation is anchored to a specific scene, not a generic style board
2. **Passive discovery** — no manual search required; the viewer's natural viewing behavior triggers the pipeline
3. **Extensibility** — the same architecture applies to sports (what gear is that athlete wearing?) or home décor (what lamp is that?)

Monetization path: affiliate revenue on purchases, white-label API licensing to streaming platforms.

---

## Related Work

- He et al. (2017) — Neural Collaborative Filtering (NCF)
- Guo et al. (2019) — FashionBERT: cross-modal fashion retrieval
- Patashnik et al. (2021) — StyleCLIP
- Han et al. (2017) — Learning Fashion Compatibility with Bidirectional LSTMs (Polyvore)
- FashionCLIP (Chia et al., 2022) — CLIP fine-tuned on Farfetch fashion catalog
- Wu et al. (2022) — Graph Neural Networks in Recommender Systems: A Survey
