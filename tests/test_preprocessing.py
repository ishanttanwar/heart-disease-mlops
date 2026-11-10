"""Unit tests for the preprocessing pipeline."""
import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src import config, preprocessing


def test_output_has_no_missing_values(model_df):
    X, _ = preprocessing.split_xy(model_df)
    assert X.isna().sum().sum() > 0  # the fixture really has gaps
    out = preprocessing.build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_feature_names_match_output_width(model_df):
    X, _ = preprocessing.split_xy(model_df)
    pre = preprocessing.build_preprocessor().fit(X)
    names = preprocessing.get_feature_names(pre)
    assert len(names) == pre.transform(X).shape[1]
    assert len(set(names)) == len(names)  # no collisions
    assert "age" in names


def test_unseen_category_is_ignored_not_fatal(model_df):
    X, _ = preprocessing.split_xy(model_df)
    pre = preprocessing.build_preprocessor().fit(X)
    unseen = X.iloc[[0]].copy()
    unseen["thal"] = 99.0
    assert pre.transform(unseen).shape[0] == 1


def test_engineered_features_added(model_df):
    X, _ = preprocessing.split_xy(model_df)
    pre = preprocessing.build_preprocessor(add_engineered=True).fit(X)
    names = preprocessing.get_feature_names(pre)
    assert "hr_reserve_pct" in names
    assert "oldpeak_log" in names
    assert np.isfinite(pre.transform(X)).all()


def test_hr_reserve_formula():
    import pandas as pd
    df = pd.DataFrame({"age": [40.0], "thalach": [180.0], "oldpeak": [0.0]})
    out = preprocessing.ClinicalFeatures().fit(df).transform(df)
    assert out.loc[0, "hr_reserve_pct"] == pytest.approx(1.0)  # 180 / 180
    assert out.loc[0, "oldpeak_log"] == pytest.approx(0.0)


def test_imputation_statistics_come_from_training_data_only(model_df):
    X_train, X_test, _, _ = preprocessing.train_test_split_stratified(model_df)
    pre = preprocessing.build_preprocessor().fit(X_train)
    num_pipe = pre.named_steps["transform"].named_transformers_["num"]
    fitted_medians = num_pipe.named_steps["impute"].statistics_
    expected = X_train[config.NUMERIC_FEATURES].median().to_numpy()
    assert np.allclose(fitted_medians, expected)


def test_unknown_scaler_rejected():
    with pytest.raises(ValueError, match="Unknown scaler"):
        preprocessing.build_preprocessor(scaler="minmax")


def test_robust_scaler_option_builds(model_df):
    X, _ = preprocessing.split_xy(model_df)
    out = preprocessing.build_preprocessor(scaler="robust").fit_transform(X)
    assert out.shape[0] == len(X)


def test_stratified_split_preserves_class_ratio(model_df):
    X_tr, X_te, y_tr, y_te = preprocessing.train_test_split_stratified(
        model_df
    )
    assert len(X_tr) + len(X_te) == len(model_df)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05


def test_split_is_reproducible(model_df):
    a = preprocessing.train_test_split_stratified(model_df)[0]
    b = preprocessing.train_test_split_stratified(model_df)[0]
    assert a.index.equals(b.index)


@pytest.mark.parametrize("engineered", [False, True])
def test_saved_pipeline_gives_identical_predictions(
    model_df, tmp_path, engineered
):
    X_tr, X_te, y_tr, _ = preprocessing.train_test_split_stratified(model_df)
    model = Pipeline([
        ("pre", preprocessing.build_preprocessor(add_engineered=engineered)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X_tr, y_tr)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    assert np.allclose(
        model.predict_proba(X_te), reloaded.predict_proba(X_te)
    )


def test_pipeline_works_inside_cross_validation(model_df):
    X, y = preprocessing.split_xy(model_df)
    model = Pipeline([
        ("pre", preprocessing.build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    scores = cross_val_score(clone(model), X, y, cv=3, scoring="roc_auc")
    assert len(scores) == 3
    assert np.isfinite(scores).all()
