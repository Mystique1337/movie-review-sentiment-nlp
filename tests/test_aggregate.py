"""Tests for the film-level aggregation layer."""

from __future__ import annotations

import numpy as np
import pytest

from src.aggregate import (
    AGGREGATIONS,
    aggregate_film_sentiment,
    compare_aggregations,
    group_scores_by_film,
)


class TestVerdictBasics:
    def test_unanimously_positive_reviews(self):
        v = aggregate_film_sentiment([0.9, 0.95, 0.88, 0.92, 0.97], "Good Film")
        assert v.sentiment == "positive"
        assert v.score > 0.8
        assert v.share_positive == 1.0

    def test_unanimously_negative_reviews(self):
        v = aggregate_film_sentiment([0.05, 0.1, 0.02, 0.08, 0.12], "Bad Film")
        assert v.sentiment == "negative"
        assert v.score < 0.2

    def test_single_review_has_degenerate_interval(self):
        v = aggregate_film_sentiment([0.8], "One Review", method="mean_probability")
        assert v.n_reviews == 1
        assert v.ci95[0] == v.ci95[1] == pytest.approx(0.8)

    def test_default_rule_is_majority(self):
        """The default was chosen from measured film-level accuracy, not intuition."""
        assert aggregate_film_sentiment([0.9, 0.8, 0.2], "Default").method == "majority"

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            aggregate_film_sentiment([], "Nothing")

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            aggregate_film_sentiment([0.6, 0.7], method="vibes")

    def test_all_methods_run(self):
        verdicts = compare_aggregations([0.2, 0.4, 0.6, 0.8, 0.9], "Mixed")
        assert set(verdicts) == set(AGGREGATIONS)


class TestDecisiveness:
    def test_clear_case_is_decisive(self):
        v = aggregate_film_sentiment([0.95] * 20, "Clear")
        assert v.decisive is True

    def test_evenly_split_case_is_not_decisive(self):
        scores = [0.05, 0.95] * 10
        v = aggregate_film_sentiment(scores, "Split")
        assert v.decisive is False
        assert v.polarisation > 0.4

    def test_polarisation_is_zero_for_identical_scores(self):
        assert aggregate_film_sentiment([0.7] * 8, "Uniform").polarisation == pytest.approx(0.0)

    def test_confidence_interval_brackets_the_point_estimate(self):
        v = aggregate_film_sentiment(np.linspace(0.1, 0.9, 25), "Spread")
        assert v.ci95[0] <= v.score <= v.ci95[1]


class TestTrimmedMeanRobustness:
    def test_trimming_resists_outliers(self):
        """Two troll reviews should not flip a clearly positive film."""
        scores = [0.85] * 18 + [0.001, 0.001]
        trimmed = aggregate_film_sentiment(scores, method="trimmed_mean").score
        plain = aggregate_film_sentiment(scores, method="mean_probability").score
        assert trimmed > plain

    def test_trimming_falls_back_on_tiny_sets(self):
        scores = [0.9, 0.1, 0.8]
        trimmed = aggregate_film_sentiment(scores, method="trimmed_mean").score
        plain = aggregate_film_sentiment(scores, method="mean_probability").score
        assert trimmed == pytest.approx(plain)

    def test_majority_ignores_confidence(self):
        confident = aggregate_film_sentiment([0.99, 0.99, 0.01], method="majority").score
        marginal = aggregate_film_sentiment([0.51, 0.51, 0.49], method="majority").score
        assert confident == marginal == pytest.approx(2 / 3)


class TestDeterminism:
    def test_same_input_gives_same_interval(self):
        scores = [0.3, 0.6, 0.7, 0.2, 0.9, 0.55]
        a = aggregate_film_sentiment(scores, seed=7)
        b = aggregate_film_sentiment(scores, seed=7)
        assert a.ci95 == b.ci95


class TestGrouping:
    def test_groups_scores_by_film_id(self):
        grouped = group_scores_by_film([0.1, 0.9, 0.5], ["tt1", "tt2", "tt1"])
        assert grouped == {"tt1": [0.1, 0.5], "tt2": [0.9]}

    def test_verdict_serialises(self):
        payload = aggregate_film_sentiment([0.6, 0.7, 0.8], "Serialise Me").to_dict()
        assert payload["title"] == "Serialise Me"
        assert "decisive" in payload
