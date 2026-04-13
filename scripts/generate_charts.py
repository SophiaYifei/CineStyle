# AI tools used: Claude (Anthropic) assisted with matplotlib styling and layout.
"""
generate_charts.py — Generate publication-quality charts and tables from eval results.

Reads JSON outputs from data/outputs/ and produces:
  - eval_chart.png         (grouped bar chart: 4 models × metrics)
  - hyperparam_tuning.png  (bar + line chart: embed_dim sweep)
  - experiment_chart.png   (degradation experiment)
  - error_analysis.png     (misprediction image pairs)
  - tables.md              (markdown tables for the report)

Usage:
  python scripts/generate_charts.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

OUT = Path("data/outputs")

# Consistent styling
COLORS = {
    "Popularity": "#6b7280",   # gray
    "FAISS KNN":  "#3b82f6",   # blue
    "NeuMF":      "#d97706",   # amber
    "SASRec":     "#10b981",   # green
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def generate_eval_chart():
    """Grouped bar chart comparing 4 models across metrics."""
    with open(OUT / "eval_results.json") as f:
        data = json.load(f)

    models = ["Popularity", "FAISS KNN", "NeuMF", "SASRec"]
    metrics = ["Precision@5", "Precision@10", "Recall@5", "Recall@10",
               "Ndcg@5", "Ndcg@10", "MAP@10"]
    display_labels = ["P@5", "P@10", "R@5", "R@10",
                      "NDCG@5", "NDCG@10", "MAP@10"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    width = 0.18

    for i, model in enumerate(models):
        vals = []
        for metric in metrics:
            key = f"{model} {metric}"
            vals.append(data.get(key, 0))
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=COLORS[model], alpha=0.9, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.0005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("CineStyle — Four-Model Offline Evaluation", fontweight="bold", pad=15)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(data.values()) * 1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "eval_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved eval_chart.png")


def generate_tuning_chart():
    """Side-by-side bar charts for hyperparameter tuning."""
    with open(OUT / "hyperparam_tuning.json") as f:
        data = json.load(f)

    dims = [int(d) for d in data.keys()]
    dim_labels = [str(d) for d in dims]
    ndcgs = [data[str(d)]["ndcg@10"] for d in dims]
    losses = [data[str(d)]["final_loss"] for d in dims]

    best_idx = ndcgs.index(max(ndcgs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: NDCG@10
    ndcg_colors = ["#3b82f6"] * len(dims)
    ndcg_colors[best_idx] = "#10b981"  # highlight best
    bars1 = ax1.bar(dim_labels, ndcgs, color=ndcg_colors, alpha=0.9,
                    edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("embed_dim", fontsize=12)
    ax1.set_ylabel("NDCG@10", fontsize=12)
    ax1.set_title("NDCG@10 by embed_dim", fontweight="bold", fontsize=13)
    for bar, val in zip(bars1, ndcgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(ndcgs) * 1.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # Right: BPR Loss
    bars2 = ax2.bar(dim_labels, losses, color="#ef4444", alpha=0.8,
                    edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("embed_dim", fontsize=12)
    ax2.set_ylabel("Final BPR Loss", fontsize=12)
    ax2.set_title("Training Loss by embed_dim", fontweight="bold", fontsize=13)
    for bar, val in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(losses) * 1.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("NCF Hyperparameter Tuning: embed_dim", fontweight="bold",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "hyperparam_tuning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved hyperparam_tuning.png")


def generate_experiment_chart():
    """Degradation experiment bar charts."""
    with open(OUT / "experiment_frame_quality.json") as f:
        data = json.load(f)

    modes = ["hd", "compressed", "blurred"]
    mode_labels = ["HD (original)", "Compressed\n(JPEG q=15)", "Blurred\n(Gaussian r=4)"]
    cos_vals = [data[m]["mean_cosine_vs_hd"] for m in modes]
    p_vals = [data[m]["mean_P@10_vs_hd"] for m in modes]
    bar_colors = ["#10b981", "#d97706", "#ef4444"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = axes[0].bar(mode_labels, cos_vals, color=bar_colors, alpha=0.85,
                        edgecolor="white", linewidth=0.5)
    axes[0].set_title("Cosine Similarity vs HD", fontweight="bold")
    axes[0].set_ylim(0, 1.15)
    for bar, val in zip(bars1, cos_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = axes[1].bar(mode_labels, p_vals, color=bar_colors, alpha=0.85,
                        edgecolor="white", linewidth=0.5)
    axes[1].set_title("Precision@10 vs HD Retrieval", fontweight="bold")
    axes[1].set_ylim(0, 1.15)
    for bar, val in zip(bars2, p_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Frame Quality Degradation Experiment", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "experiment_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved experiment_chart.png")


def generate_error_analysis_chart():
    """Misprediction image pairs grid."""
    with open(OUT / "error_analysis.json") as f:
        errors = json.load(f)

    n = min(5, len(errors))
    if n == 0:
        print("No error cases to plot.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(7, 3.5 * n))
    if n == 1:
        axes = [axes]

    for row, e in enumerate(errors[:n]):
        for col, (key, label) in enumerate([
            ("probe_image", f"Query: {e['probe_category']}"),
            ("rec_image", f"Rank {e['rank']}: {e['rec_category']}\n(sim={e['similarity']:.3f})"),
        ]):
            path = e[key]
            ax = axes[row][col]
            if Path(path).exists():
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(label, fontsize=10, pad=4)
            ax.axis("off")

        # Add row number
        axes[row][0].text(-0.15, 0.5, f"#{row + 1}", transform=axes[row][0].transAxes,
                          fontsize=14, fontweight="bold", va="center", ha="center",
                          color="#d97706")

    fig.suptitle("Error Analysis: Category Mispredictions", fontweight="bold",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error_analysis.png")


def generate_tables():
    """Generate markdown tables for the report."""
    with open(OUT / "eval_results.json") as f:
        eval_data = json.load(f)
    with open(OUT / "hyperparam_tuning.json") as f:
        tune_data = json.load(f)
    with open(OUT / "experiment_frame_quality.json") as f:
        exp_data = json.load(f)
    with open(OUT / "error_analysis.json") as f:
        errors = json.load(f)

    lines = []

    # Table 1: Four-model comparison
    lines.append("## Table 1: Four-Model Offline Evaluation\n")
    lines.append("| Model | P@5 | P@10 | R@5 | R@10 | NDCG@5 | NDCG@10 | MAP@10 |")
    lines.append("|-------|-----|------|-----|------|--------|---------|--------|")

    for model in ["Popularity", "FAISS KNN", "NeuMF", "SASRec"]:
        row = [model]
        for metric in ["Precision@5", "Precision@10", "Recall@5", "Recall@10",
                       "Ndcg@5", "Ndcg@10", "MAP@10"]:
            key = f"{model} {metric}"
            val = eval_data.get(key, 0)
            row.append(f"{val:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("**Key finding:** SASRec achieves the best NDCG@10 (0.0056) and MAP@10 (0.0023), ")
    lines.append("outperforming both the popularity baseline and FAISS KNN retrieval. ")
    lines.append("NeuMF re-ranking also improves over raw FAISS retrieval across all metrics.")
    lines.append("")

    # Table 2: Hyperparameter tuning
    lines.append("## Table 2: NCF Hyperparameter Tuning (embed_dim)\n")
    lines.append("| embed_dim | NDCG@10 | Final BPR Loss |")
    lines.append("|-----------|---------|----------------|")

    best_dim = max(tune_data.keys(), key=lambda d: tune_data[d]["ndcg@10"])
    for dim in ["16", "32", "64", "128", "256"]:
        ndcg = tune_data[dim]["ndcg@10"]
        loss = tune_data[dim]["final_loss"]
        marker = " **" if dim == best_dim else ""
        lines.append(f"| {dim}{marker} | {ndcg:.4f}{marker} | {loss:.4f} |")

    lines.append("")
    lines.append(f"**Best:** embed_dim={best_dim} achieves highest NDCG@10 ({tune_data[best_dim]['ndcg@10']:.4f}).")
    lines.append("")

    # Table 3: Frame quality degradation
    lines.append("## Table 3: Frame Quality Degradation Experiment\n")
    lines.append("| Mode | Cosine Similarity vs HD | Precision@10 vs HD |")
    lines.append("|------|------------------------|--------------------|")

    for mode, label in [("hd", "HD (original)"), ("compressed", "Compressed (JPEG q=15)"),
                        ("blurred", "Blurred (Gaussian r=4)")]:
        cos = exp_data[mode]["mean_cosine_vs_hd"]
        p10 = exp_data[mode]["mean_P@10_vs_hd"]
        lines.append(f"| {label} | {cos:.4f} | {p10:.4f} |")

    lines.append("")
    lines.append("**Key finding:** JPEG compression (q=15) retains 87% cosine similarity but drops ")
    lines.append("P@10 to 0.375. Gaussian blur is more destructive: 71% cosine similarity, P@10=0.10. ")
    lines.append("This suggests the pipeline is moderately robust to compression but sensitive to blur.")
    lines.append("")

    # Table 4: Error analysis
    lines.append("## Table 4: Error Analysis — 5 Category Mispredictions\n")
    lines.append("| # | Query Category | Query ID | Rec Category | Rec ID | Rank | Similarity |")
    lines.append("|---|----------------|----------|--------------|--------|------|------------|")

    for i, e in enumerate(errors, 1):
        lines.append(f"| {i} | {e['probe_category']} | {e['probe_id']} | "
                     f"{e['rec_category']} | {e['rec_id']} | {e['rank']} | {e['similarity']:.4f} |")

    lines.append("")
    lines.append("**Root causes and mitigation:**")
    lines.append("")
    lines.append("1. **Dress → Top** (sim=0.781): Off-shoulder dress visually overlaps with crop tops. "
                 "*Mitigation:* Add garment-length features (mini/midi/maxi) to the embedding.")
    lines.append("2. **Sweater → Top** (sim=0.797): Knit textures are similar across categories. "
                 "*Mitigation:* Fine-tune CLIP on category-labeled data to separate knit subcategories.")
    lines.append("3. **Sock → Tights** (sim=0.794): Both are legwear with similar visual patterns. "
                 "*Mitigation:* Add category-aware hard negative mining during index construction.")
    lines.append("4. **Sock → Tights** (sim=0.727): Repeated failure mode confirms legwear confusion. "
                 "*Mitigation:* Merge sock/tights into a single 'legwear' supercategory.")
    lines.append("5. **Shirt → Jacket** (sim=0.937): Very high similarity — structured collared garments "
                 "are visually near-identical. *Mitigation:* Use garment-weight/layering metadata as an "
                 "additional retrieval signal.")
    lines.append("")

    md = "\n".join(lines)
    with open(OUT / "tables.md", "w") as f:
        f.write(md)
    print(f"Saved tables.md")


if __name__ == "__main__":
    generate_eval_chart()
    generate_tuning_chart()
    generate_experiment_chart()
    generate_error_analysis_chart()
    generate_tables()
    print("\nAll charts and tables generated.")
