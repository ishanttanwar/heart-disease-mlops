"""Draw the end-to-end MLOps architecture diagram used in the report.

    python scripts/make_architecture_diagram.py

Writes screenshots/architecture.png (also handy for the README).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

NAVY = "#1F3864"
BLUE = "#2E75B6"
GREY = "#595959"
LIGHT = "#F2F2F2"

LANES = [
    ("1. Development", 8.55, "#FAFAFA"),
    ("2. Continuous integration", 6.05, "#F4F7FB"),
    ("3. Serving", 3.55, "#FAFAFA"),
    ("4. Observability", 1.25, "#F4F7FB"),
]

BOXES = [
    # (x, y, w, h, title, subtitle)
    (0.6, 8.7, 2.4, 1.0, "UCI Cleveland", "data_loader.py\n303 patients"),
    (3.4, 8.7, 2.4, 1.0, "EDA", "eda.py\n7 plots"),
    (6.2, 8.7, 2.4, 1.0, "Preprocessing", "ColumnTransformer\nimpute + scale + OHE"),
    (9.0, 8.7, 2.8, 1.0, "Training", "train.py\n3 models x 4 variants"),

    (0.6, 6.2, 2.4, 1.0, "MLflow", "15 runs\nparams + metrics"),
    (3.4, 6.2, 2.4, 1.0, "Model registry", "heart-disease-\nclassifier v2"),
    (6.2, 6.2, 2.4, 1.0, "GitHub Actions", "lint -> test ->\ntrain"),
    (9.0, 6.2, 2.8, 1.0, "Artifacts", "model.joblib\nmetadata.json"),

    (0.6, 3.7, 2.4, 1.0, "Docker image", "python:3.12-slim\nnon-root"),
    (3.4, 3.7, 2.4, 1.0, "FastAPI", "/predict /health\n/metrics"),
    (6.2, 3.7, 2.4, 1.0, "Kubernetes", "2 replicas\nprobes + limits"),
    (9.0, 3.7, 2.8, 1.0, "Client", "JSON in,\nprediction out"),

    (2.0, 1.4, 2.6, 0.9, "JSON request logs", "one line per request"),
    (5.2, 1.4, 2.6, 0.9, "Prometheus", "scrape /metrics 5s"),
    (8.4, 1.4, 2.6, 0.9, "Grafana", "9-panel dashboard"),
]

ARROWS = [
    ((3.0, 9.2), (3.4, 9.2)),
    ((5.8, 9.2), (6.2, 9.2)),
    ((8.6, 9.2), (9.0, 9.2)),
    ((10.4, 8.7), (10.4, 7.2)),      # training -> artifacts
    ((9.0, 6.7), (8.6, 6.7)),        # artifacts <- actions
    ((6.2, 6.7), (5.8, 6.7)),        # actions <- registry
    ((3.4, 6.7), (3.0, 6.7)),        # registry <- mlflow
    ((1.8, 6.2), (1.8, 4.7)),        # registered model -> docker image
    ((3.0, 4.2), (3.4, 4.2)),        # docker -> fastapi
    ((5.8, 4.2), (6.2, 4.2)),        # fastapi -> k8s
    ((8.6, 4.2), (9.0, 4.2)),        # k8s -> client
    ((4.0, 3.7), (3.3, 2.3)),        # fastapi -> logs
    ((4.9, 3.7), (6.0, 2.3)),        # fastapi -> prometheus
    ((7.8, 1.85), (8.4, 1.85)),      # prometheus -> grafana
]


def box(ax, x, y, w, h, title, subtitle):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.3, edgecolor=NAVY, facecolor="white", zorder=3,
    ))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=NAVY, zorder=4)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
            fontsize=7.5, color=GREY, linespacing=1.35, zorder=4)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color=BLUE, shrinkA=2, shrinkB=2, zorder=2,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 7.4))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0.6, 10.4)
    ax.axis("off")

    for label, y, colour in LANES:
        ax.add_patch(FancyBboxPatch(
            (0.25, y), 11.9, 1.55,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=0, facecolor=colour, zorder=0,
        ))
        ax.text(0.35, y + 1.42, label, ha="left", va="center",
                fontsize=8.5, fontweight="bold", color=BLUE, zorder=1)

    for args in BOXES:
        box(ax, *args)
    for start, end in ARROWS:
        arrow(ax, start, end)

    ax.text(0.25, 10.2, "Heart Disease Risk Prediction - MLOps Architecture",
            ha="left", va="center", fontsize=13, fontweight="bold",
            color=NAVY)

    out = config.SCREENSHOTS_DIR / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
