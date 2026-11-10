"""Unit tests for data acquisition and cleaning."""
import io

import pandas as pd
import pytest

from src import config, data_loader


def test_clean_binarizes_target(raw_df):
    cleaned = data_loader.clean(raw_df)
    assert set(cleaned[config.TARGET].unique()) <= {0, 1}
    assert config.RAW_TARGET not in cleaned.columns
    expected = (raw_df[config.RAW_TARGET] > 0).astype(int)
    assert (cleaned[config.TARGET].to_numpy() == expected.to_numpy()).all()


def test_clean_column_order(raw_df):
    cleaned = data_loader.clean(raw_df)
    assert list(cleaned.columns) == config.FEATURES + [config.TARGET]


def test_clean_keeps_missing_values_as_nan(raw_df):
    cleaned = data_loader.clean(raw_df)
    assert cleaned["ca"].isna().sum() == 3
    assert cleaned["thal"].isna().sum() == 2


def test_clean_drops_exact_duplicates(raw_df):
    with_duplicate = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    assert len(data_loader.clean(with_duplicate)) == len(
        data_loader.clean(raw_df)
    )


def test_clean_drops_rows_without_label(raw_df):
    raw_df.loc[5, config.RAW_TARGET] = None
    assert len(data_loader.clean(raw_df)) == len(raw_df) - 1


def test_clean_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Missing expected columns"):
        data_loader.clean(raw_df.drop(columns=["chol"]))


def test_parse_raw_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Expected 14 columns"):
        data_loader._parse_raw(io.StringIO("1,2,3\n4,5,6\n"))


def test_parse_raw_treats_question_mark_as_nan():
    row = "63,1,1,145,233,1,2,150,0,2.3,3,?,6,0\n"
    parsed = data_loader._parse_raw(io.StringIO(row))
    assert list(parsed.columns) == config.RAW_COLUMNS
    assert pd.isna(parsed.loc[0, "ca"])


def test_save_and_load_round_trip(raw_df, tmp_path):
    path = tmp_path / "clean.csv"
    cleaned = data_loader.clean(raw_df)
    data_loader.save_clean(cleaned, path)
    loaded = data_loader.load_clean(path)

    assert loaded.shape == cleaned.shape
    assert all(str(loaded[c].dtype) == "float64" for c in config.FEATURES)
    assert loaded[config.TARGET].dtype.kind == "i"
    assert loaded[config.FEATURES].isna().sum().sum() == 5
