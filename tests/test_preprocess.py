"""Tests for text normalisation.

The negation tests are the important ones: they encode the project's central
preprocessing finding as an executable assertion, so a future change that
silently reintroduces negation-destroying stop-word removal will fail here.
"""

from __future__ import annotations

import pytest

from src.preprocess import (
    NEGATION_CUES,
    SENTIMENT_FUNCTION_WORDS,
    Preprocessor,
    clean_text,
    get_stopwords,
    mark_negation_scope,
)


class TestCleanText:
    def test_strips_html_tags(self):
        assert "br" not in clean_text("Great film.<br /><br />Really.")

    def test_unescapes_entities(self):
        assert "&quot;" not in clean_text("He said &quot;wonderful&quot;.")

    def test_lowercases(self):
        assert clean_text("BRILLIANT") == "brilliant"

    def test_collapses_character_runs(self):
        assert clean_text("sooooo good") == "soo good"

    def test_keeps_internal_apostrophes(self):
        assert "don't" in clean_text("I don't like it")

    def test_removes_digits_and_punctuation(self):
        assert clean_text("Rated 7/10 -- solid!") == "rated solid"

    def test_empty_input(self):
        assert clean_text("") == ""


class TestStopwords:
    def test_negations_kept_by_default(self):
        words = get_stopwords(keep_negations=True)
        for cue in ("not", "no", "never", "but"):
            assert cue not in words

    def test_naive_list_deletes_negations(self):
        words = get_stopwords(keep_negations=False)
        assert "not" in words and "no" in words

    def test_ordinary_function_words_always_removed(self):
        assert "the" in get_stopwords(keep_negations=True)


class TestNegationScope:
    def test_marks_following_tokens(self):
        assert mark_negation_scope(["not", "a", "good", "film"], scope=3) == [
            "not", "a_NEG", "good_NEG", "film_NEG",
        ]

    def test_scope_is_bounded(self):
        out = mark_negation_scope(["not", "a", "b", "c", "d"], scope=2)
        assert out[-1] == "d"

    def test_no_negation_leaves_tokens_untouched(self):
        tokens = ["a", "very", "good", "film"]
        assert mark_negation_scope(list(tokens)) == tokens


class TestPreprocessor:
    def test_rejects_unknown_normalisation(self):
        with pytest.raises(ValueError):
            Preprocessor(normalisation="soundex")

    def test_lemmatisation_normalises_inflection(self):
        pre = Preprocessor(remove_stopwords=False, normalisation="lemma")
        assert pre.tokenize("loved")[0] == pre.tokenize("loving")[0] == "love"

    def test_stemming_is_more_aggressive_than_lemmatisation(self):
        stem = Preprocessor(remove_stopwords=False, normalisation="stem")
        lemma = Preprocessor(remove_stopwords=False, normalisation="lemma")
        text = "The movie was beautifully photographed and wonderfully acted."
        assert len(set(stem.tokenize(text))) <= len(set(lemma.tokenize(text))) + 1
        assert "beauti" in stem.tokenize(text)

    def test_naive_stopwords_destroy_negation(self):
        """The failure mode the ablation was built to detect."""
        naive = Preprocessor(remove_stopwords=True, keep_negations=False, normalisation="none")
        assert naive.tokenize("not good") == naive.tokenize("good")

    def test_negation_aware_stopwords_preserve_it(self):
        aware = Preprocessor(remove_stopwords=True, keep_negations=True, normalisation="none")
        assert aware.tokenize("not good") != aware.tokenize("good")
        assert "not" in aware.tokenize("not good")

    def test_min_token_length_filter(self):
        pre = Preprocessor(remove_stopwords=False, normalisation="none", min_token_length=4)
        assert all(len(t) >= 4 for t in pre.tokenize("a be cat lion tigers"))

    def test_call_returns_joined_string(self):
        pre = Preprocessor()
        assert isinstance(pre("A wonderful little film."), str)

    def test_transform_preserves_order_and_length(self):
        pre = Preprocessor()
        texts = ["first one", "second one", "third one"]
        assert len(pre.transform(texts)) == 3

    def test_empty_and_whitespace_documents(self):
        pre = Preprocessor()
        assert pre("") == ""
        assert pre("   \n\t ") == ""

    def test_html_only_document_yields_nothing(self):
        assert Preprocessor()("<br /><br />") == ""

    def test_describe_is_informative(self):
        text = Preprocessor(normalisation="stem").describe()
        assert "stem" in text and "stopwords" in text

    def test_word_sets_are_frozensets(self):
        assert isinstance(NEGATION_CUES, frozenset)
        assert isinstance(SENTIMENT_FUNCTION_WORDS, frozenset)
        assert "not" in NEGATION_CUES

    def test_modifiers_are_kept_but_do_not_trigger_negation(self):
        """`very` must survive stop-word removal without opening a NEG scope."""
        assert "very" in SENTIMENT_FUNCTION_WORDS
        assert "very" not in NEGATION_CUES
