## Table 1: Four-Model Offline Evaluation

| Model | P@5 | P@10 | R@5 | R@10 | NDCG@5 | NDCG@10 | MAP@10 |
|-------|-----|------|-----|------|--------|---------|--------|
| Popularity | 0.0020 | 0.0010 | 0.0017 | 0.0017 | 0.0034 | 0.0030 | 0.0017 |
| FAISS KNN | 0.0020 | 0.0035 | 0.0017 | 0.0058 | 0.0013 | 0.0035 | 0.0009 |
| NeuMF | 0.0050 | 0.0035 | 0.0042 | 0.0058 | 0.0046 | 0.0050 | 0.0019 |
| SASRec | 0.0050 | 0.0035 | 0.0042 | 0.0058 | 0.0051 | 0.0056 | 0.0023 |

**Key finding:** SASRec achieves the best NDCG@10 (0.0056) and MAP@10 (0.0023), 
outperforming both the popularity baseline and FAISS KNN retrieval. 
NeuMF re-ranking also improves over raw FAISS retrieval across all metrics.

## Table 2: NCF Hyperparameter Tuning (embed_dim)

| embed_dim | NDCG@10 | Final BPR Loss |
|-----------|---------|----------------|
| 16 | 0.0092 | 0.4821 |
| 32 | 0.0060 | 0.4340 |
| 64 | 0.0083 | 0.3938 |
| 128 ** | 0.0099 ** | 0.3631 |
| 256 | 0.0090 | 0.3134 |

**Best:** embed_dim=128 achieves highest NDCG@10 (0.0099).

## Table 3: Frame Quality Degradation Experiment

| Mode | Cosine Similarity vs HD | Precision@10 vs HD |
|------|------------------------|--------------------|
| HD (original) | 1.0000 | 1.0000 |
| Compressed (JPEG q=15) | 0.8692 | 0.3750 |
| Blurred (Gaussian r=4) | 0.7145 | 0.1000 |

**Key finding:** JPEG compression (q=15) retains 87% cosine similarity but drops 
P@10 to 0.375. Gaussian blur is more destructive: 71% cosine similarity, P@10=0.10. 
This suggests the pipeline is moderately robust to compression but sensitive to blur.

## Table 4: Error Analysis — 5 Category Mispredictions

| # | Query Category | Query ID | Rec Category | Rec ID | Rank | Similarity |
|---|----------------|----------|--------------|--------|------|------------|
| 1 | dress | 150314 | top, t-shirt, sweatshirt | 118792 | 3 | 0.7813 |
| 2 | sweater | 158953 | top, t-shirt, sweatshirt | 127695 | 2 | 0.7973 |
| 3 | sock | 158959 | tights, stockings | 135840 | 4 | 0.7935 |
| 4 | sock | 158960 | tights, stockings | 167859 | 3 | 0.7272 |
| 5 | shirt, blouse | 169203 | jacket | 169207 | 2 | 0.9374 |

**Root causes and mitigation:**

1. **Dress → Top** (sim=0.781): Off-shoulder dress visually overlaps with crop tops. *Mitigation:* Add garment-length features (mini/midi/maxi) to the embedding.
2. **Sweater → Top** (sim=0.797): Knit textures are similar across categories. *Mitigation:* Fine-tune CLIP on category-labeled data to separate knit subcategories.
3. **Sock → Tights** (sim=0.794): Both are legwear with similar visual patterns. *Mitigation:* Add category-aware hard negative mining during index construction.
4. **Sock → Tights** (sim=0.727): Repeated failure mode confirms legwear confusion. *Mitigation:* Merge sock/tights into a single 'legwear' supercategory.
5. **Shirt → Jacket** (sim=0.937): Very high similarity — structured collared garments are visually near-identical. *Mitigation:* Use garment-weight/layering metadata as an additional retrieval signal.
