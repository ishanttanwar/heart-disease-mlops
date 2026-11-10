"""Exploratory data analysis: saves report-ready plots to screenshots/eda/.

Usage, from the repository root:

    python -m src.eda

Plots produced:
    class_balance.png            target distribution
    missing_values.png           missing values per feature (if any)
    numeric_distributions.png    histograms + density, split by diagnosis
    numeric_boxplots.png         boxplots of numeric features by diagnosis
    categorical_counts.png       category counts, split by diagnosis
    correlation_heatmap.png      Spearman correlations of all features
    target_correlations.png      each feature's correlation with the target
plus eda_summary.txt with the key numbers for the written report.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from src import config
from src.data_loader import load_clean

logger = logging.getLogger(__name__)

DIAGNOSIS = "diagnosis"
HUE_ORDER = [config.CLASS_LABELS[0], config.CLASS_LABELS[1]]
PALETTE = {HUE_ORDER[0]: "#4C78A8", HUE_ORDER[1]: "#E45756"}

NUMERIC_LABELS = {
    "age": "Age (years)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)",
    "oldpeak": "ST depression (exercise vs rest)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable diagnosis column for plot legends."""
    out = df.copy()
    out[DIAGNOSIS] = out[config.TARGET].map(config.CLASS_LABELS)
    return out


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def _drop_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _figure_legend(fig) -> None:
    """One shared legend above the plots instead of one covering the bars."""
    handles = [Patch(facecolor=PALETTE[lbl], label=lbl) for lbl in HUE_ORDER]
    fig.legend(
        handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.995),
        ncol=2, frameon=False,
    )


def _correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = config.FEATURES + [config.TARGET]
    return df[cols].astype(float).corr(method="spearman")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df[DIAGNOSIS].value_counts().reindex(HUE_ORDER)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(
        counts.index, counts.values, color=[PALETTE[k] for k in counts.index]
    )
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.annotate(
            f"{value} ({value / total:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, out_dir, "class_balance.png")


def plot_missing_values(df: pd.DataFrame, out_dir: Path):
    missing = df[config.FEATURES].isna().sum()
    missing = missing[missing > 0].sort_values()
    if missing.empty:
        logger.info("No missing values; skipping missing_values.png")
        return None

    fig, ax = plt.subplots(figsize=(6, 1.2 + 0.6 * len(missing)))
    bars = ax.barh(missing.index, missing.values, color="#F58518")
    for bar, value in zip(bars, missing.values):
        ax.annotate(
            f"{value} ({value / len(df):.1%})",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=10,
        )
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values per feature")
    ax.set_xlim(0, missing.max() * 1.3)
    return _save(fig, out_dir, "missing_values.png")


def plot_numeric_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for i, col in enumerate(config.NUMERIC_FEATURES):
        ax = axes[i]
        sns.histplot(
            data=df, x=col, hue=DIAGNOSIS, hue_order=HUE_ORDER,
            palette=PALETTE, bins=20, kde=True, element="step",
            common_norm=False, ax=ax,
        )
        ax.set_xlabel(NUMERIC_LABELS[col])
        ax.set_title(col)
        _drop_legend(ax)
    for ax in axes[len(config.NUMERIC_FEATURES):]:
        ax.set_visible(False)
    fig.suptitle("Distributions of numeric features by diagnosis", y=1.02)
    _figure_legend(fig)
    return _save(fig, out_dir, "numeric_distributions.png")


def plot_numeric_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    n = len(config.NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.5))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        # Plain matplotlib: seaborn 0.13 triggers a deprecation warning
        # inside matplotlib >= 3.11 for boxplots.
        groups = [
            df.loc[df[DIAGNOSIS] == label, col].dropna()
            for label in HUE_ORDER
        ]
        parts = ax.boxplot(
            groups, patch_artist=True, widths=0.6,
            medianprops={"color": "black"},
        )
        for box, label in zip(parts["boxes"], HUE_ORDER):
            box.set_facecolor(PALETTE[label])
            box.set_alpha(0.85)
        ax.set_xticks([1, 2], HUE_ORDER)
        ax.set_xlabel("")
        ax.set_ylabel(NUMERIC_LABELS[col])
        ax.set_title(col)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Numeric features by diagnosis", y=1.02)
    return _save(fig, out_dir, "numeric_boxplots.png")


def plot_categorical_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    axes = axes.ravel()
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[[col, DIAGNOSIS]].dropna()
        data[col] = data[col].astype(int).astype(str)
        order = sorted(data[col].unique(), key=int)
        sns.countplot(
            data=data, x=col, hue=DIAGNOSIS, order=order,
            hue_order=HUE_ORDER, palette=PALETTE, ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Patients")
        _drop_legend(ax)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Categorical features by diagnosis", y=1.02)
    _figure_legend(fig)
    return _save(fig, out_dir, "categorical_counts.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = _correlations(df)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Spearman correlation matrix (features and target)")
    return _save(fig, out_dir, "correlation_heatmap.png")


def plot_target_correlations(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = _correlations(df)[config.TARGET].drop(config.TARGET)
    corr = corr.sort_values()
    colors = [PALETTE[HUE_ORDER[1]] if v > 0 else PALETTE[HUE_ORDER[0]]
              for v in corr.values]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman correlation with heart disease")
    ax.set_title("Feature correlation with the target")
    return _save(fig, out_dir, "target_correlations.png")


def save_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = df[DIAGNOSIS].value_counts().reindex(HUE_ORDER)
    missing = df[config.FEATURES].isna().sum()
    lines = [
        f"Rows: {len(df)}",
        f"Features: {len(config.FEATURES)} "
        f"({len(config.NUMERIC_FEATURES)} numeric, "
        f"{len(config.CATEGORICAL_FEATURES)} categorical)",
        "",
        "Class balance:",
        counts.to_string(),
        "",
        "Missing values per feature:",
        missing[missing > 0].to_string() if missing.any() else "none",
        "",
        "Numeric feature summary:",
        df[config.NUMERIC_FEATURES].describe().round(2).to_string(),
    ]
    path = out_dir / "eda_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_eda(df=None, out_dir=None) -> list:
    """Generate every EDA plot and the text summary; return saved paths."""
    plt.switch_backend("Agg")  # headless-safe (works in CI and Docker)
    sns.set_theme(style="whitegrid", context="notebook")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    out_dir = Path(out_dir) if out_dir else config.EDA_DIR
    data = _prepare(load_clean() if df is None else df)

    steps = [
        plot_class_balance, plot_missing_values, plot_numeric_distributions,
        plot_numeric_boxplots, plot_categorical_counts,
        plot_correlation_heatmap, plot_target_correlations, save_summary,
    ]
    saved = [step(data, out_dir) for step in steps]
    return [p for p in saved if p is not None]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    saved = run_eda()
    logger.info("EDA complete: %d files in %s", len(saved), config.EDA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
