"""Classical bag-of-words classifiers.

Each model is returned as a scikit-learn ``Pipeline`` so that vectorisation and
classification are fitted together. That matters for methodological hygiene: the
vocabulary and the IDF weights are learned from training folds only, never from
the data a model is scored on.

Four learners are compared:

* **MultinomialNB** - the generative baseline named in the task. Fast, but its
  conditional-independence assumption is badly violated by natural text.
* **LinearSVC** - the discriminative counterpart; the strongest classical
  reported baseline on aclImdb for several years.
* **LogisticRegression** - comparable accuracy to LinearSVC, but emits
  calibrated probabilities, which the film-level aggregation layer needs.
* **NB-SVM** - the interpolation of a Naive Bayes log-count ratio with a linear
  classifier proposed by Wang & Manning (2012), implemented here as a custom
  transformer. It is consistently the best non-neural model on this corpus.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import SEED


# --------------------------------------------------------------------------- #
# NB-SVM feature weighting (Wang & Manning, 2012)
# --------------------------------------------------------------------------- #
class NBTransformer(BaseEstimator, TransformerMixin):
    """Scale each feature by its Naive Bayes log-count ratio.

    For feature *i*, r_i = log( (p_i / ||p||_1) / (q_i / ||q||_1) ) where p and
    q are smoothed per-class count vectors. Multiplying the design matrix by r
    injects the generative prior into an otherwise purely discriminative model.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        X = sparse.csr_matrix(X)
        p = np.asarray(X[y == 1].sum(axis=0)).ravel() + self.alpha
        q = np.asarray(X[y == 0].sum(axis=0)).ravel() + self.alpha
        self.r_ = np.log((p / p.sum()) / (q / q.sum()))
        return self

    def transform(self, X):
        return sparse.csr_matrix(X).multiply(self.r_).tocsr()


# --------------------------------------------------------------------------- #
# Vectoriser
# --------------------------------------------------------------------------- #
def build_vectorizer(
    ngram_max: int = 2,
    min_df: int = 2,
    max_features: int | None = 200_000,
    sublinear_tf: bool = True,
    binary: bool = False,
    use_idf: bool = True,
) -> TfidfVectorizer:
    """TF-IDF over word n-grams.

    ``sublinear_tf`` replaces raw counts with 1 + log(tf), which dampens the
    effect of a reviewer repeating the same adjective eight times. Tokenisation
    is a plain whitespace split because the text has already been normalised by
    :class:`src.preprocess.Preprocessor`.

    ``use_idf`` is switched off for NB-SVM: that model derives its own feature
    weights from the Naive Bayes log-count ratio, and multiplying those by IDF
    as well weights every feature twice, which measurably degrades it.
    """
    return TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        binary=binary,
        use_idf=use_idf,
        token_pattern=r"\S+",
        lowercase=False,
        strip_accents=None,
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def make_naive_bayes(**vec_kwargs) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer(**vec_kwargs)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]
    )


def make_linear_svm(C: float = 0.5, **vec_kwargs) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer(**vec_kwargs)),
            ("clf", LinearSVC(C=C, random_state=SEED, dual="auto", max_iter=5_000)),
        ]
    )


def make_logistic_regression(C: float = 4.0, **vec_kwargs) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer(**vec_kwargs)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    random_state=SEED,
                    max_iter=2_000,
                ),
            ),
        ]
    )


def make_nbsvm(C: float = 1.0, **vec_kwargs) -> Pipeline:
    # Wang & Manning binarise the document-term matrix and let the NB log-count
    # ratio supply the weighting, so IDF is disabled here.
    vec_kwargs.setdefault("binary", True)
    vec_kwargs.setdefault("sublinear_tf", False)
    vec_kwargs.setdefault("use_idf", False)
    return Pipeline(
        [
            ("tfidf", build_vectorizer(**vec_kwargs)),
            ("nb", NBTransformer(alpha=1.0)),
            (
                "clf",
                LogisticRegression(
                    C=C, solver="liblinear", random_state=SEED, max_iter=2_000
                ),
            ),
        ]
    )


MODEL_BUILDERS = {
    "naive_bayes": make_naive_bayes,
    "linear_svm": make_linear_svm,
    "logistic_regression": make_logistic_regression,
    "nbsvm": make_nbsvm,
}

MODEL_DISPLAY_NAMES = {
    "naive_bayes": "Multinomial Naive Bayes",
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "nbsvm": "NB-SVM",
    "distilbert": "DistilBERT (fine-tuned)",
    "vader": "VADER (lexicon, unsupervised)",
}


def build_model(name: str, **kwargs) -> Pipeline:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](**kwargs)


def decision_scores(pipeline: Pipeline, X) -> np.ndarray:
    """Return a positive-class score in [0, 1] for any of the pipelines above.

    ``LinearSVC`` has no ``predict_proba``; its signed margin is squashed with a
    logistic function so that all models expose a comparable score. The result
    is a ranking score, not a calibrated probability, for the SVM.
    """
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    margins = pipeline.decision_function(X)
    return 1.0 / (1.0 + np.exp(-margins))


def top_features(pipeline: Pipeline, n: int = 20) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Most positive and most negative weighted n-grams of a linear model."""
    vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        weights = np.asarray(clf.coef_).ravel()
    else:  # MultinomialNB: use the log-probability ratio between classes
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(weights)
    negative = [(vocab[i], float(weights[i])) for i in order[:n]]
    positive = [(vocab[i], float(weights[i])) for i in order[-n:][::-1]]
    return positive, negative
