"""Data acquisition and cleaning for the Heart Disease UCI (Cleveland) dataset.

Usage, from the repository root:

    python -m src.data_loader            # download if needed, clean, save
    python -m src.data_loader --force    # re-download the raw file first

Pipeline:  download_raw() -> load_raw() -> clean() -> save_clean()

Design note: missing values (marked "?" in the raw file) are kept as NaN in
the cleaned CSV. They are imputed later *inside* the sklearn pipeline, fitted
on training data only, so no information leaks from the test folds.
"""
import argparse
import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _parse_raw(source) -> pd.DataFrame:
    """Parse the headerless raw file and validate it has the expected shape."""
    try:
        df = pd.read_csv(
            source, header=None, na_values=config.MISSING_VALUE_MARKER
        )
    except Exception as exc:  # e.g. an HTML error page instead of data
        raise ValueError(f"Could not parse raw data: {exc}") from exc

    if df.shape[1] != len(config.RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(config.RAW_COLUMNS)} columns, got {df.shape[1]}"
        )
    df.columns = config.RAW_COLUMNS
    return df


def _fetch_url(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "heart-disease-mlops/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _fetch_with_ucimlrepo() -> pd.DataFrame:
    """Fallback: fetch the dataset through the `ucimlrepo` package."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Automatic download failed and the `ucimlrepo` fallback is not "
            "installed. Either run `pip install ucimlrepo` and retry, or "
            f"download {config.DATA_URL} manually and save it as "
            f"{config.RAW_DATA_PATH}"
        ) from exc

    dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
    df = pd.concat(
        [dataset.data.features, dataset.data.targets], axis=1
    )
    return df[config.RAW_COLUMNS]


def download_raw(force: bool = False) -> Path:
    """Download the raw dataset to data/raw/ (skipped if already present)."""
    path = config.RAW_DATA_PATH
    if path.exists() and not force:
        logger.info(
            "Raw data already at %s (use --force to re-download)", path
        )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = _fetch_url(config.DATA_URL, config.DOWNLOAD_TIMEOUT_SECONDS)
        df = _parse_raw(io.BytesIO(content))
        logger.info("Downloaded raw data from %s", config.DATA_URL)
    except (OSError, ValueError) as exc:  # URLError and timeouts are OSError
        logger.warning(
            "Direct download failed (%s); trying ucimlrepo fallback", exc
        )
        df = _fetch_with_ucimlrepo()

    df.to_csv(
        path, header=False, index=False, na_rep=config.MISSING_VALUE_MARKER
    )
    logger.info("Saved raw data to %s (%d rows)", path, len(df))
    return path


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw file with proper column names and NaN for missing."""
    path = Path(path) if path else config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_loader` first."
        )
    return _parse_raw(path)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: binary target, correct dtypes, no duplicates.

    Steps:
      1. keep the 14 documented columns and validate they exist
      2. drop rows with no label (a label cannot be imputed)
      3. drop exact duplicate rows
      4. binarize the target: num 0 -> 0 (no disease), 1-4 -> 1 (disease)
      5. cast numeric features to float and categorical codes to nullable
         integers, so the saved CSV shows 0/1/2 rather than 0.0/1.0/2.0
    Missing feature values are deliberately left as NaN (see module note).
    """
    missing_cols = set(config.RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    out = df[config.RAW_COLUMNS].copy()

    n_start = len(out)
    out = out.dropna(subset=[config.RAW_TARGET])
    n_unlabelled = n_start - len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    n_duplicates = n_start - n_unlabelled - len(out)
    if n_unlabelled or n_duplicates:
        logger.info(
            "Dropped %d unlabelled and %d duplicate rows",
            n_unlabelled, n_duplicates,
        )

    out[config.TARGET] = (out[config.RAW_TARGET] > 0).astype(int)
    out = out.drop(columns=config.RAW_TARGET)

    out[config.NUMERIC_FEATURES] = out[config.NUMERIC_FEATURES].astype(float)
    for col in config.CATEGORICAL_FEATURES:
        out[col] = out[col].astype("Int64")

    return out[config.FEATURES + [config.TARGET]]


def save_clean(df: pd.DataFrame, path=None) -> Path:
    path = Path(path) if path else config.CLEAN_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s", path)
    return path


def load_clean(path=None) -> pd.DataFrame:
    """Load the cleaned CSV, building it first if it does not exist yet.

    All feature columns come back as float64 (NaN for missing) because
    scikit-learn imputers and encoders handle that more reliably than pandas'
    nullable integer types.
    """
    path = Path(path) if path else config.CLEAN_DATA_PATH
    if not path.exists():
        logger.info("%s not found; building it now", path)
        build_clean_dataset()

    df = pd.read_csv(path)
    df[config.FEATURES] = df[config.FEATURES].astype(float)
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def build_clean_dataset(force_download: bool = False) -> pd.DataFrame:
    """Run the full acquisition pipeline and return the cleaned data."""
    raw_path = download_raw(force=force_download)
    cleaned = clean(load_raw(raw_path))
    save_clean(cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _log_summary(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    logger.info("Rows: %d | Features: %d", len(df), len(config.FEATURES))
    logger.info(
        "Class balance: %s",
        ", ".join(
            f"{config.CLASS_LABELS[k]}={v} ({v / len(df):.1%})"
            for k, v in counts.items()
        ),
    )
    missing = df[config.FEATURES].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("Missing values: none")
    else:
        logger.info(
            "Missing values: %s",
            ", ".join(f"{col}={n}" for col, n in missing.items()),
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--force", action="store_true", help="re-download the raw file"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        cleaned = build_clean_dataset(force_download=args.force)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    _log_summary(cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
