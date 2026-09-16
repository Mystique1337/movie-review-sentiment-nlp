"""Film-level sentiment aggregation.

The task asks for more than a per-document classifier: the finished system must
"analyse movie reviews and determine the overall sentiment towards the movie".
A film is therefore represented by a *set* of reviews, and this module turns
that set into one binary verdict plus an honest statement of uncertainty.

Three aggregation rules are implemented and compared:

``majority``
    Each review casts one vote. It discards confidence entirely: a review the
    model scores at 0.51 counts as much as one scored at 0.99.
``mean_probability``
    Average the positive-class scores, then threshold at 0.5. Sensitive to how
    confident the model is, but a handful of extreme scores can dominate a small
    review set.
``trimmed_mean``
    Average after discarding the most extreme 10% at each end, which is the
    textbook remedy for the previous rule's outlier sensitivity.

``majority`` is the default, and that is an empirical decision rather than an
obvious one. The trimmed mean was the expected winner, on the reasoning that
film review sets contain occasional outliers and that discarding confidence
throws information away. Measured on 658 films whose reviews genuinely disagree,
majority voting reached 0.9255 against 0.9088 for the trimmed mean
(``results/tables/t17_film_aggregation.csv``). The explanation is visible in the
calibration analysis: the deployed classifier pushes roughly a quarter of its
predictions past 0.99 confidence, so averaging those saturated scores reproduces
a vote count anyway, while leaving the result exposed to any single review the
model got confidently wrong. Majority voting caps each review's influence at one
vote, which is exactly the robustness the trimmed mean was supposed to supply.

Whichever rule is used, a percentile bootstrap over reviews gives a confidence
interval on the film score. That interval is what makes the output actionable:
"positive, 0.82 [0.74, 0.89], n = 40" tells a studio something that a bare
"positive" does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

from src.config import SEED

AGGREGATIONS = ("majority", "mean_probability", "trimmed_mean")


@dataclass
class FilmVerdict:
    """The system's overall judgement about one film."""

    title: str
    n_reviews: int
    sentiment: str                      # "positive" | "negative"
    score: float                        # aggregated positive-class score in [0, 1]
    ci95: tuple[float, float]
    method: str
    share_positive: float               # fraction of individual reviews judged positive
    polarisation: float                 # std. dev. of per-review scores
    decisive: bool = field(init=False)  # CI excludes 0.5

    def __post_init__(self) -> None:
        self.decisive = not (self.ci95[0] <= 0.5 <= self.ci95[1])

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        verdict = self.sentiment.upper()
        hedge = "" if self.decisive else "  (not statistically decisive)"
        return (
            f"{self.title}: {verdict}  "
            f"score={self.score:.3f} CI95=[{self.ci95[0]:.3f}, {self.ci95[1]:.3f}]  "
            f"n={self.n_reviews}  positive_share={self.share_positive:.1%}  "
            f"polarisation={self.polarisation:.3f}{hedge}"
        )


def _aggregate_scores(scores: np.ndarray, method: str, trim: float = 0.1) -> float:
    if method == "majority":
        return float((scores >= 0.5).mean())
    if method == "mean_probability":
        return float(scores.mean())
    if method == "trimmed_mean":
        if len(scores) < 5:  # trimming a tiny set throws away too much
            return float(scores.mean())
        lo, hi = np.quantile(scores, [trim, 1 - trim])
        kept = scores[(scores >= lo) & (scores <= hi)]
        return float(kept.mean()) if len(kept) else float(scores.mean())
    raise ValueError(f"Unknown aggregation '{method}'. Choose from {AGGREGATIONS}")


def aggregate_film_sentiment(
    scores: Sequence[float],
    title: str = "unnamed film",
    method: str = "majority",
    n_boot: int = 2_000,
    seed: int = SEED,
) -> FilmVerdict:
    """Collapse per-review positive-class scores into one verdict for a film."""
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot judge a film with no reviews")

    point = _aggregate_scores(scores, method)

    rng = np.random.default_rng(seed)
    if scores.size == 1:
        lo = hi = point
    else:
        idx = rng.integers(0, scores.size, size=(n_boot, scores.size))
        boot = np.array([_aggregate_scores(scores[row], method) for row in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])

    return FilmVerdict(
        title=title,
        n_reviews=int(scores.size),
        sentiment="positive" if point >= 0.5 else "negative",
        score=float(point),
        ci95=(float(lo), float(hi)),
        method=method,
        share_positive=float((scores >= 0.5).mean()),
        polarisation=float(scores.std()),
    )


def compare_aggregations(
    scores: Sequence[float], title: str = "unnamed film"
) -> dict[str, FilmVerdict]:
    """Run every aggregation rule on the same review set."""
    return {m: aggregate_film_sentiment(scores, title, method=m) for m in AGGREGATIONS}


def group_scores_by_film(
    scores: Iterable[float], film_ids: Iterable[str]
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for score, film in zip(scores, film_ids):
        grouped.setdefault(film, []).append(float(score))
    return grouped
