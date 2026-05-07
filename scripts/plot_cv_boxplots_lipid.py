"""Boxplot figures from lipid_cv_baselines_metrics.csv.

Two-panel figure (R2 left, MAE right) showing the test-set distribution of
each baseline model across 5 CV folds. Saves PNG (raster, for previews) and
PDF (vector, for the manuscript revision) to figures/.

The two horizontal reference lines on the R2 panel mark the manuscript's
fine-tuned ChemBERTa numbers from the original single-split protocol --
not directly comparable on the y-axis (different protocol, different
data partition), but useful as anchors when reading the figure.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = "lipid_cv_baselines_metrics.csv"
OUT_DIR = "figures"

# Display order (worst to best on R2 in the single split)
MODEL_ORDER = [
    "RDKit-only (Ridge)",
    "RDKit-only (GradientBoosting)",
    "ChemBERTa-frozen-RT (CLS only)",
    "ChemBERTa-Concat (frozen)",
    "RDKit-only (RandomForest)",
]

PAPER_REFS = {
    # Median across 15 epochs on the single manuscript split (Section 3.1)
    "ChemBERTa-RT (paper, fine-tuned)": 0.820,
    "ChemBERTa+RDKit MTL (paper, fine-tuned)": 0.842,
}


def main():
    df = pd.read_csv(INPUT_CSV)
    test = df[df["split"] == "Test"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, metric, ylabel in [
        (axes[0], "R2", r"Test $R^2$"),
        (axes[1], "MAE", "Test MAE"),
    ]:
        sub = test[test["metric"] == metric]
        data = [sub[sub["model"] == m]["value"].values for m in MODEL_ORDER]
        bp = ax.boxplot(
            data,
            tick_labels=MODEL_ORDER,
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#9ec5fe")
            patch.set_edgecolor("#1f4e8c")
        for whisker in bp["whiskers"]:
            whisker.set_color("#1f4e8c")
        for cap in bp["caps"]:
            cap.set_color("#1f4e8c")

        # Overlay individual fold points
        for i, vals in enumerate(data, start=1):
            jitter = (np.random.RandomState(i).uniform(-0.08, 0.08, size=len(vals)))
            ax.scatter(
                np.full_like(vals, i, dtype=float) + jitter,
                vals,
                color="#1f4e8c",
                s=18,
                alpha=0.7,
                zorder=3,
            )

        if metric == "R2":
            for label, y in PAPER_REFS.items():
                ax.axhline(y, linestyle="--", linewidth=1.0, color="#cc4444", alpha=0.85)
                ax.text(
                    len(MODEL_ORDER) + 0.05,
                    y,
                    f"  {label}: {y:.3f}",
                    fontsize=8,
                    color="#cc4444",
                    va="center",
                    ha="left",
                )

        ax.set_ylabel(ylabel)
        ax.set_title(f"5-fold CV — lipid RT, test {metric}")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Lipid RT baselines under 5-fold CV (pooled manuscript train+test, n=70,961)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "lipid_cv_boxplots.png")
    pdf_path = os.path.join(OUT_DIR, "lipid_cv_boxplots.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
