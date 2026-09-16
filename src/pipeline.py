"""The end-to-end system: preprocess -> vectorise -> classify -> aggregate.

``SentimentSystem`` is the object a user of this project actually interacts
with. It wraps whichever backend was trained (a scikit-learn pipeline or the
fine-tuned transformer) behind one interface, so the film-level aggregation and
the command line do not need to know which model is underneath.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from src.aggregate import FilmVerdict, aggregate_film_sentiment
from src.config import MODELS_DIR, LABEL_NAMES
from src.models_classical import decision_scores
from src.preprocess import DEFAULT_PREPROCESSOR_KWARGS, Preprocessor

log = logging.getLogger(__name__)

CLASSICAL_PATH = MODELS_DIR / "best_classical.joblib"
TRANSFORMER_PATH = MODELS_DIR / "distilbert-imdb"


class SentimentSystem:
    """Binary sentiment for single reviews and for whole films."""

    def __init__(self, backend: str = "classical", model_path: Path | None = None) -> None:
        if backend not in {"classical", "distilbert"}:
            raise ValueError("backend must be 'classical' or 'distilbert'")
        self.backend = backend
        self._transformer_cfg = None

        if backend == "classical":
            path = Path(model_path or CLASSICAL_PATH)
            if not path.exists():
                raise FileNotFoundError(
                    f"No trained model at {path}. Run `python scripts/03_train_classical.py` first."
                )
            bundle = joblib.load(path)
            self.model = bundle["pipeline"]
            self.preprocessor = Preprocessor(
                **bundle.get("preprocessor_kwargs", DEFAULT_PREPROCESSOR_KWARGS)
            )
            self.model_name = bundle.get("model_name", "classical")
            self.metadata = bundle.get("metadata", {})
        else:
            from src.models_transformer import TrainingConfig, load_fine_tuned

            self.model, self.tokenizer = load_fine_tuned(model_path or TRANSFORMER_PATH)
            self.preprocessor = None  # transformers consume raw text
            self.model_name = "distilbert"
            self._transformer_cfg = TrainingConfig()
            meta_file = Path(model_path or TRANSFORMER_PATH) / "run_metadata.json"
            self.metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    # -- document level ----------------------------------------------------- #
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Positive-class score in [0, 1], one per input review."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts:
            return np.array([])
        if self.backend == "classical":
            return decision_scores(self.model, self.preprocessor.transform(texts))
        from src.models_transformer import predict_proba

        return predict_proba(self.model, self.tokenizer, texts, self._transformer_cfg)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return (self.score(texts) >= 0.5).astype(int)

    def predict_labels(self, texts: Sequence[str]) -> list[str]:
        return [LABEL_NAMES[i] for i in self.predict(texts)]

    def explain(self, text: str) -> dict:
        """Score one review and, for linear models, name the decisive tokens."""
        score = float(self.score([text])[0])
        out = {
            "text": text[:300] + ("..." if len(text) > 300 else ""),
            "score": score,
            "sentiment": LABEL_NAMES[int(score >= 0.5)],
            "confidence": abs(score - 0.5) * 2,
        }
        if self.backend == "classical" and hasattr(self.model.named_steps["clf"], "coef_"):
            vec = self.model.named_steps["tfidf"]
            clf = self.model.named_steps["clf"]
            processed = self.preprocessor(text)
            x = vec.transform([processed])
            coefs = np.asarray(clf.coef_).ravel()
            contributions = x.multiply(coefs).tocoo()
            vocab = vec.get_feature_names_out()
            pairs = sorted(
                ((vocab[j], float(v)) for j, v in zip(contributions.col, contributions.data)),
                key=lambda kv: kv[1],
            )
            # Split by sign rather than taking head and tail slices: a short
            # review may contain fewer than sixteen n-grams, and slicing would
            # then list the same token as both supporting and opposing evidence.
            out["top_positive_tokens"] = [p for p in reversed(pairs) if p[1] > 0][:8]
            out["top_negative_tokens"] = [p for p in pairs if p[1] < 0][:8]
        return out

    # -- film level --------------------------------------------------------- #
    def judge_film(
        self, reviews: Sequence[str], title: str = "unnamed film",
        method: str = "majority",
    ) -> FilmVerdict:
        """Aggregate a set of reviews into one verdict about the film."""
        return aggregate_film_sentiment(self.score(reviews), title=title, method=method)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SentimentSystem(backend={self.backend!r}, model={self.model_name!r})"


def save_classical(pipeline, model_name: str, preprocessor_kwargs: dict,
                   metadata: dict | None = None, path: Path = CLASSICAL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": model_name,
            "preprocessor_kwargs": preprocessor_kwargs,
            "metadata": metadata or {},
        },
        path,
        compress=3,
    )
    log.info("Saved %s to %s (%.1f MB)", model_name, path, path.stat().st_size / 1e6)
    return path
