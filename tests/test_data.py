"""Tests for corpus loading.

Anything that needs the downloaded corpora is skipped when they are absent, so
the suite still runs on a clean checkout and in CI. The film-identifier test is
the one that matters most: the whole film-level evaluation rests on mapping each
review to the right IMDb title, and an off-by-one there would silently corrupt
every aggregation result rather than raising an error.
"""

from __future__ import annotations

import pytest

from src.config import ACL_IMDB_DIR, RT_POLARITY_DIR
from src.data import CORPUS_LOADERS, load_corpus

imdb_available = pytest.mark.skipif(
    not (ACL_IMDB_DIR / "test").is_dir(),
    reason="aclImdb not downloaded; run scripts/00_download_data.py",
)
rt_available = pytest.mark.skipif(
    not RT_POLARITY_DIR.is_dir(),
    reason="rt-polaritydata not downloaded; run scripts/00_download_data.py",
)


class TestLoaderRegistry:
    def test_unknown_corpus_rejected(self):
        with pytest.raises(KeyError):
            load_corpus("yelp")

    def test_registry_names(self):
        assert set(CORPUS_LOADERS) == {
            "pang_lee", "imdb_train", "imdb_test", "rt_polarity",
        }


@imdb_available
class TestIMDb:
    @pytest.fixture(scope="class")
    def test_split(self):
        return load_corpus("imdb_test")

    def test_split_size_and_balance(self, test_split):
        assert len(test_split) == 25_000
        assert test_split["label"].mean() == pytest.approx(0.5)

    def test_required_columns(self, test_split):
        assert {"text", "label", "rating", "film_id"} <= set(test_split.columns)

    def test_labels_are_binary(self, test_split):
        assert set(test_split["label"].unique()) == {0, 1}

    def test_ratings_match_labels(self, test_split):
        """aclImdb excludes 5 and 6 stars; <=4 is negative, >=7 positive."""
        assert test_split.loc[test_split["label"] == 0, "rating"].max() <= 4
        assert test_split.loc[test_split["label"] == 1, "rating"].min() >= 7

    def test_every_review_has_a_film(self, test_split):
        assert test_split["film_id"].isna().sum() == 0

    def test_film_ids_look_like_imdb_titles(self, test_split):
        assert test_split["film_id"].str.match(r"^tt\d+$").all()

    def test_no_film_exceeds_the_documented_cap(self, test_split):
        """The dataset authors cap each film at 30 reviews per polarity.

        This is the load-bearing check on the review-id to URL-line mapping: a
        misaligned mapping would spread reviews across neighbouring films and
        almost certainly push some group above the cap.
        """
        counts = test_split.groupby(["film_id", "label"]).size()
        assert counts.max() <= 30

    def test_corpus_contains_many_distinct_films(self, test_split):
        assert test_split["film_id"].nunique() > 3_000

    def test_reviews_are_non_empty(self, test_split):
        assert (test_split["text"].str.strip().str.len() > 0).all()

    def test_invalid_split_rejected(self):
        from src.data import load_imdb

        with pytest.raises(ValueError):
            load_imdb("validation")


@rt_available
class TestRottenTomatoes:
    def test_size_and_balance(self):
        df = load_corpus("rt_polarity")
        assert len(df) == 10_662
        assert df["label"].mean() == pytest.approx(0.5)

    def test_sentences_are_short(self):
        df = load_corpus("rt_polarity")
        assert df["text"].str.split().str.len().median() < 40


class TestPangLee:
    def test_size_and_balance(self):
        df = load_corpus("pang_lee")
        assert len(df) == 2_000
        assert df["label"].mean() == pytest.approx(0.5)

    def test_reviews_are_long(self):
        """Pang & Lee reviews are full-length, unlike the RT snippets."""
        df = load_corpus("pang_lee")
        assert df["text"].str.split().str.len().median() > 300
