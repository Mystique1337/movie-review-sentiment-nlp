"""Text normalisation for the classical (bag-of-words) arm of the pipeline.

The task prescribes stop-word removal plus stemming or lemmatisation. Rather
than assuming those steps help, this module exposes each one as a switch so the
ablation in ``scripts/02_preprocessing_ablation.py`` can measure their effect.

Two design decisions are worth flagging because they are easy to get wrong:

1.  **Negation handling.** The default English stop-word list of NLTK contains
    ``not``, ``no``, ``nor`` and every ``n't`` contraction. Deleting those words
    destroys exactly the signal a sentiment classifier needs, because "not
    good" collapses to "good". A reduced stop-word list is therefore offered,
    and the ablation quantifies the cost of using the naive list.
2.  **HTML.** aclImdb reviews were scraped from a web page and still contain
    ``<br />`` tags. Left in place, these become a frequent and completely
    uninformative token.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# --------------------------------------------------------------------------- #
# Token-level resources (loaded once)
# --------------------------------------------------------------------------- #
_TOKENIZER = TreebankWordTokenizer()
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

# Two distinct roles are kept apart deliberately.
#
# NEGATION_CUES *flip* polarity, so they are the only words allowed to open a
# negation scope in `mark_negation_scope`.
NEGATION_CUES = frozenset(
    {
        "no", "not", "nor", "never", "none", "nothing", "neither", "nowhere",
        "cannot", "n't", "without", "hardly", "scarcely", "barely",
        "don", "didn", "doesn", "isn", "wasn", "aren", "weren", "won", "wouldn",
        "couldn", "shouldn", "ain", "haven", "hasn", "hadn", "mustn",
    }
)

# Intensifiers and contrastive discourse markers do not flip polarity, but they
# modulate or redirect it ("but the ending ruins it"), so they must survive
# stop-word removal even though they are not negation triggers.
MODIFIER_WORDS = frozenset(
    {
        "but", "however", "although", "though", "yet", "despite",
        "very", "too", "only", "just", "more", "most", "less", "least",
        "so", "such", "quite", "almost", "even",
    }
)

# The full set of function words that stop-word removal must not delete.
SENTIMENT_FUNCTION_WORDS = NEGATION_CUES | MODIFIER_WORDS

HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_LETTER_RE = re.compile(r"[^a-z\s']")
MULTISPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


@lru_cache(maxsize=4)
def get_stopwords(keep_negations: bool = True) -> frozenset[str]:
    """Return the English stop-word list, optionally keeping polarity words."""
    try:
        words = set(stopwords.words("english"))
    except LookupError:  # pragma: no cover - first run on a fresh machine
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english"))
    if keep_negations:
        words -= SENTIMENT_FUNCTION_WORDS
    return frozenset(words)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Strip markup and noise while preserving apostrophes inside words."""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    # "sooooo good" -> "soo good": collapse runs of 3+ identical characters.
    text = REPEATED_CHAR_RE.sub(r"\1\1", text)
    text = NON_LETTER_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def mark_negation_scope(tokens: list[str], scope: int = 3) -> list[str]:
    """Append ``_NEG`` to the ``scope`` tokens following a negation cue.

    This is the standard trick from Das & Chen (2001), popularised for movie
    reviews by Pang, Lee & Vaithyanathan (2002): it lets a unigram model
    distinguish "good" from "not good" without moving to bigrams.
    """
    out: list[str] = []
    remaining = 0
    for tok in tokens:
        if remaining > 0:
            out.append(f"{tok}_NEG")
            remaining -= 1
        else:
            out.append(tok)
        if tok in NEGATION_CUES or tok.endswith("n't"):
            remaining = scope
    return out


class Preprocessor:
    """Configurable text-normalisation pipeline.

    Parameters
    ----------
    remove_stopwords:
        Drop English function words.
    keep_negations:
        When removing stop words, retain negation and intensity cues.
    normalisation:
        ``"none"``, ``"stem"`` (Porter) or ``"lemma"`` (WordNet).
    negation_marking:
        Apply ``mark_negation_scope`` after tokenisation.
    min_token_length:
        Discard tokens shorter than this (1 removes stray letters).
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        keep_negations: bool = True,
        normalisation: str = "lemma",
        negation_marking: bool = False,
        min_token_length: int = 2,
    ) -> None:
        if normalisation not in {"none", "stem", "lemma"}:
            raise ValueError("normalisation must be 'none', 'stem' or 'lemma'")
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        self.normalisation = normalisation
        self.negation_marking = negation_marking
        self.min_token_length = min_token_length
        self._stopwords = (
            get_stopwords(keep_negations) if remove_stopwords else frozenset()
        )
        self._norm_cache: dict[str, str] = {}

    # -- internals ---------------------------------------------------------- #
    def _normalise(self, token: str) -> str:
        if self.normalisation == "none":
            return token
        cached = self._norm_cache.get(token)
        if cached is not None:
            return cached
        if self.normalisation == "stem":
            value = _STEMMER.stem(token)
        else:
            # Lemmatise as verb then as noun; WordNet's default POS is noun, and
            # verbs ("loved" -> "love") are where most of the gain is.
            value = _LEMMATIZER.lemmatize(_LEMMATIZER.lemmatize(token, pos="v"))
        self._norm_cache[token] = value
        return value

    # -- public API --------------------------------------------------------- #
    def tokenize(self, text: str) -> list[str]:
        tokens = _TOKENIZER.tokenize(clean_text(text))
        if self.negation_marking:
            tokens = mark_negation_scope(tokens)
        out = []
        for tok in tokens:
            bare = tok[:-4] if tok.endswith("_NEG") else tok
            if len(bare) < self.min_token_length:
                continue
            if self.remove_stopwords and bare in self._stopwords:
                continue
            normed = self._normalise(bare)
            out.append(f"{normed}_NEG" if tok.endswith("_NEG") else normed)
        return out

    def __call__(self, text: str) -> str:
        """Return the normalised text as a whitespace-joined string."""
        return " ".join(self.tokenize(text))

    def transform(self, texts) -> list[str]:
        return [self(t) for t in texts]

    def describe(self) -> str:
        parts = [
            "stopwords=" + ("on" if self.remove_stopwords else "off"),
            "negations=" + ("kept" if self.keep_negations else "dropped"),
            f"norm={self.normalisation}",
            "negmark=" + ("on" if self.negation_marking else "off"),
        ]
        return ", ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Preprocessor({self.describe()})"


# --------------------------------------------------------------------------- #
# Selected configurations
# --------------------------------------------------------------------------- #
# Chosen by the validation ablation in scripts/02_preprocessing_ablation.py.
# Stop-word removal is deliberately *off*: with TF-IDF weighting, the inverse
# document frequency term already suppresses function words continuously, and
# deleting them outright only discards the negation and intensity cues that
# sentiment depends on. Lemmatisation is retained because it shrinks the feature
# space by roughly 25% at no measurable cost in accuracy.
DEFAULT_PREPROCESSOR_KWARGS = {
    "remove_stopwords": False,
    "keep_negations": True,
    "normalisation": "lemma",
    "negation_marking": False,
    "min_token_length": 2,
}

# The configuration the task brief prescribes literally (remove stop words, then
# stem or lemmatise). Reported alongside the selected one so the cost of the
# prescribed recipe is visible rather than assumed.
PRESCRIBED_PREPROCESSOR_KWARGS = {
    "remove_stopwords": True,
    "keep_negations": False,
    "normalisation": "stem",
    "negation_marking": False,
    "min_token_length": 2,
}
