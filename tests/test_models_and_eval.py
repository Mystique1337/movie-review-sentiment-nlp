"""Tests for the model zoo and the evaluation helpers.

These run on a tiny synthetic corpus so the suite stays fast enough to execute
on every commit. They check wiring and contracts, not accuracy; accuracy is the
job of the experiment scripts.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import compute_metrics, mcnemar_test
from src.models_classical import (
    MODEL_BUILDERS,
    NBTransformer,
    build_model,
    decision_scores,
    top_features,
)

POSITIVE = [
    "wonderful brilliant moving film",
    "superb acting beautiful story",
    "a masterpiece truly excellent",
    "delightful charming and warm",
    "outstanding direction great pace",
    "lovely film wonderful cast",
]
NEGATIVE = [
    "terrible boring awful film",
    "dreadful acting weak story",
    "a disaster truly dreadful",
    "tedious dull and lifeless",
    "appalling direction bad pace",
    "awful film terrible cast",
]
TEXTS = POSITIVE + NEGATIVE
LABELS = [1] * len(POSITIVE) + [0] * len(NEGATIVE)


@pytest.fixture(params=sorted(MODEL_BUILDERS))
def fitted(request):
    model = build_model(request.param, min_df=1, max_features=None)
    model.fit(TEXTS, LABELS)
    return request.param, model


class TestModelZoo:
    def test_unknown_model_rejected(self):
        with pytest.raises(KeyError):
            build_model("transformer_xl")

    def test_every_model_fits_and_separates_easy_data(self, fitted):
        _, model = fitted
        assert model.score(TEXTS, LABELS) > 0.9

    def test_scores_are_bounded(self, fitted):
        _, model = fitted
        scores = decision_scores(model, TEXTS)
        assert scores.shape == (len(TEXTS),)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_scores_rank_classes_correctly(self, fitted):
        _, model = fitted
        scores = decision_scores(model, TEXTS)
        assert scores[: len(POSITIVE)].mean() > scores[len(POSITIVE) :].mean()

    def test_predictions_are_binary(self, fitted):
        _, model = fitted
        assert set(np.unique(model.predict(TEXTS))) <= {0, 1}

    def test_top_features_are_signed_correctly(self, fitted):
        _, model = fitted
        positive, negative = top_features(model, n=3)
        assert all(w > 0 for _, w in positive)
        assert all(w < 0 for _, w in negative)

    def test_vectoriser_is_fitted_within_the_pipeline(self, fitted):
        """Guards against vocabulary leakage from test data."""
        _, model = fitted
        vocab = model.named_steps["tfidf"].vocabulary_
        assert "unseenwordxyz" not in vocab


class TestNBTransformer:
    def test_log_count_ratio_has_expected_sign(self):
        from sklearn.feature_extraction.text import CountVectorizer

        X = CountVectorizer(min_df=1).fit_transform(TEXTS)
        vocab = CountVectorizer(min_df=1).fit(TEXTS).get_feature_names_out()
        r = NBTransformer().fit(X, LABELS).r_
        assert r[list(vocab).index("wonderful")] > 0
        assert r[list(vocab).index("terrible")] < 0

    def test_transform_preserves_shape(self):
        from sklearn.feature_extraction.text import CountVectorizer

        X = CountVectorizer(min_df=1).fit_transform(TEXTS)
        assert NBTransformer().fit(X, LABELS).transform(X).shape == X.shape


class TestMetrics:
    def test_perfect_prediction(self):
        m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], n_boot=50)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0

    def test_inverted_prediction(self):
        m = compute_metrics([0, 1, 0, 1], [1, 0, 1, 0], n_boot=50)
        assert m["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        m = compute_metrics([0, 1, 0, 1], [0, 1, 1, 1], n_boot=50)
        assert np.array(m["confusion_matrix"]).shape == (2, 2)

    def test_auc_present_only_with_scores(self):
        without = compute_metrics([0, 1], [0, 1], n_boot=20)
        with_scores = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], n_boot=20)
        assert "roc_auc" not in without
        assert with_scores["roc_auc"] == 1.0

    def test_bootstrap_interval_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        p = np.where(rng.random(500) < 0.85, y, 1 - y)
        m = compute_metrics(y, p, n_boot=300)
        assert m["accuracy_ci95"][0] <= m["accuracy"] <= m["accuracy_ci95"][1]


class TestMcNemar:
    def test_identical_classifiers_are_not_significant(self):
        y = [0, 1] * 20
        result = mcnemar_test(y, y, y)
        assert result["n01"] == result["n10"] == 0
        assert result["p_value"] == 1.0
        assert result["significant_at_05"] is False

    def test_clearly_better_classifier_is_significant(self):
        y = np.array([0, 1] * 50)
        good = y.copy()
        bad = y.copy()
        bad[:30] = 1 - bad[:30]
        result = mcnemar_test(y, good, bad)
        assert result["n01"] == 30 and result["n10"] == 0
        assert result["significant_at_05"] is True

    def test_is_symmetric_in_its_counts(self):
        y = np.array([0, 1] * 30)
        a, b = y.copy(), y.copy()
        a[:5] = 1 - a[:5]
        b[10:18] = 1 - b[10:18]
        forward = mcnemar_test(y, a, b)
        backward = mcnemar_test(y, b, a)
        assert forward["n01"] == backward["n10"]
        assert forward["p_value"] == pytest.approx(backward["p_value"])
