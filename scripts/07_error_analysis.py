#!/usr/bin/env python3
"""Step 7 - where the system fails, and the film-level verdict.

Two things happen here.

**Error analysis.** A single accuracy figure says nothing about *which* reviews
are hard. Three breakdowns are produced: by the star rating the reviewer gave
(aclImdb preserves it in the file name), by review length, and by the presence
of specific linguistic constructions. The star-rating breakdown is the most
informative, because it separates "the model is wrong" from "the review is
genuinely lukewarm": a 4-star review is labelled negative by the corpus but
often reads as mixed.

**Film-level aggregation.** The task requires the finished system to output an
overall sentiment towards a *movie*, not towards individual reviews. Because the
corpus records which IMDb title each review belongs to, this can be evaluated
properly rather than demonstrated on a toy example: 3,581 films in the test
split, of which 786 carry reviews of both polarities and so constitute a
non-trivial aggregation problem.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.aggregate import AGGREGATIONS, aggregate_film_sentiment
from src.config import FIGURES_DIR, SEED, set_seed
from src.data import load_imdb
from src.evaluate import PALETTE
from src.pipeline import SentimentSystem
from src.reporting import print_table, save_run_metadata, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("errors")

MIN_REVIEWS_PER_FILM = 5

# Surface probes for constructions that are known to trouble bag-of-words models.
PROBES = {
    "negation": re.compile(r"\b(not|no|never|n't|nothing|neither|nor)\b", re.I),
    "contrastive": re.compile(r"\b(but|however|although|though|despite|yet)\b", re.I),
    "comparison_to_other_film": re.compile(
        r"\b(better than|worse than|compared to|unlike|reminds me of|sequel|remake|original)\b", re.I
    ),
    "hedging": re.compile(r"\b(maybe|perhaps|somewhat|fairly|rather|kind of|sort of)\b", re.I),
    "irony_markers": re.compile(r"(\"|\bsupposedly\b|\bapparently\b|\bso-called\b|!\?|\?!)", re.I),
}


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def breakdown_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rating, sub in df.groupby("rating"):
        rows.append(
            {
                "star_rating": int(rating),
                "gold_label": "positive" if sub["label"].iloc[0] == 1 else "negative",
                "n": len(sub),
                "accuracy": float((sub["pred"] == sub["label"]).mean()),
                "mean_model_score": float(sub["score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("star_rating")


def breakdown_by_length(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    words = df["text"].str.split().str.len()
    bins = pd.qcut(words, n_bins, labels=[f"Q{i+1}" for i in range(n_bins)])
    rows = []
    for name, sub in df.groupby(bins, observed=True):
        w = sub["text"].str.split().str.len()
        rows.append(
            {
                "length_quintile": str(name),
                "words_range": f"{int(w.min())}-{int(w.max())}",
                "median_words": float(w.median()),
                "n": len(sub),
                "accuracy": float((sub["pred"] == sub["label"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def breakdown_by_construction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = float((df["pred"] == df["label"]).mean())
    for name, pattern in PROBES.items():
        present = df["text"].str.contains(pattern)
        sub_yes, sub_no = df[present], df[~present]
        rows.append(
            {
                "construction": name,
                "share_of_test_%": 100 * float(present.mean()),
                "accuracy_present": float((sub_yes["pred"] == sub_yes["label"]).mean()) if len(sub_yes) else np.nan,
                "accuracy_absent": float((sub_no["pred"] == sub_no["label"]).mean()) if len(sub_no) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["delta_pp"] = 100 * (out["accuracy_present"] - out["accuracy_absent"])
    out["overall_accuracy"] = overall
    return out.sort_values("delta_pp")


def plot_rating_breakdown(rating_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    colours = [PALETTE[1] if r <= 4 else PALETTE[0] for r in rating_df["star_rating"]]
    ax.bar(rating_df["star_rating"].astype(str), rating_df["accuracy"], color=colours, width=0.68)
    for x, (acc, n) in enumerate(zip(rating_df["accuracy"], rating_df["n"])):
        ax.text(x, acc + 0.006, f"{acc:.3f}\nn={n:,}", ha="center", fontsize=6.8)
    ax.set_xlabel("Star rating given by the reviewer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Errors concentrate on lukewarm reviews (4 and 7 stars)", fontsize=9.5)
    path = FIGURES_DIR / "fig10_accuracy_by_rating.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def expected_calibration_error(scores, labels, n_bins: int = 10) -> float:
    """Average gap between stated confidence and observed accuracy.

    A model can rank documents almost perfectly and still be badly calibrated:
    if it says 0.99 whenever it means 0.85, the number it emits is not a
    probability. That distinction matters here because the film-level layer
    averages these scores, so a systematically overconfident model degrades
    into a plain majority vote.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    confidence = np.where(scores >= 0.5, scores, 1 - scores)
    correct = (scores >= 0.5).astype(int) == labels
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confidence > lo) & (confidence <= hi)
        if sel.sum() == 0:
            continue
        ece += (sel.sum() / len(scores)) * abs(correct[sel].mean() - confidence[sel].mean())
    return float(ece)


def plot_calibration(df: pd.DataFrame) -> Path:
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(df["score"], bins) - 1
    xs, ys, ns = [], [], []
    for b in range(10):
        sel = idx == b
        if sel.sum() < 30:
            continue
        xs.append(float(df["score"][sel].mean()))
        ys.append(float(df["label"][sel].mean()))
        ns.append(int(sel.sum()))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(4.2, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1, label="perfect calibration")
    ax.plot(xs, ys, marker="o", ms=4, color=PALETTE[0], lw=1.6, label="observed")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Reliability of the model's confidence", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    ax2.bar(xs, ns, width=0.08, color="0.7")
    ax2.set_yscale("log")
    ax2.set_xlabel("Predicted probability of positive")
    ax2.set_ylabel("count")
    path = FIGURES_DIR / "fig11_calibration.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Film-level aggregation
# --------------------------------------------------------------------------- #
def evaluate_film_aggregation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every aggregation rule against film-level ground truth.

    Ground truth for a film is the majority polarity of its *gold* review
    labels. This is the sentiment of the corpus sample for that film, not of the
    film's true reception, because aclImdb caps each film at 30 reviews and was
    balanced globally by its authors. The comparison between rules is
    nevertheless fair, since every rule faces the same target.
    """
    grouped = df.groupby("film_id")
    films = []
    for film_id, sub in grouped:
        if len(sub) < MIN_REVIEWS_PER_FILM:
            continue
        films.append(
            {
                "film_id": film_id,
                "n_reviews": len(sub),
                "gold_positive_share": float(sub["label"].mean()),
                "gold_sentiment": int(sub["label"].mean() >= 0.5),
                "mixed": sub["label"].nunique() == 2,
                "scores": sub["score"].to_numpy(),
                "single_review_score": float(sub["score"].iloc[0]),
            }
        )
    log.info("%d films with >=%d reviews (%d of them mixed-polarity)",
             len(films), MIN_REVIEWS_PER_FILM, sum(f["mixed"] for f in films))

    rows = []
    for subset_name, selector in (
        ("all films", lambda f: True),
        ("mixed-polarity films", lambda f: f["mixed"]),
    ):
        pool = [f for f in films if selector(f)]
        if not pool:
            continue
        gold = np.array([f["gold_sentiment"] for f in pool])

        # Baseline: judge the film from a single randomly chosen review.
        rng = np.random.default_rng(SEED)
        single = np.array([int(rng.choice(f["scores"]) >= 0.5) for f in pool])
        rows.append(
            {
                "subset": subset_name,
                "n_films": len(pool),
                "rule": "single review (baseline)",
                "accuracy": float((single == gold).mean()),
                "decisive_%": np.nan,
            }
        )

        for method in AGGREGATIONS:
            verdicts = [
                aggregate_film_sentiment(f["scores"], method=method, n_boot=400)
                for f in pool
            ]
            preds = np.array([int(v.sentiment == "positive") for v in verdicts])
            rows.append(
                {
                    "subset": subset_name,
                    "n_films": len(pool),
                    "rule": method,
                    "accuracy": float((preds == gold).mean()),
                    "decisive_%": 100 * float(np.mean([v.decisive for v in verdicts])),
                }
            )
            log.info("%-22s %-18s acc=%.4f", subset_name, method, (preds == gold).mean())

    # Accuracy as a function of how many reviews the verdict pools.
    depth_rows = []
    rng = np.random.default_rng(SEED)
    pool = [f for f in films if f["mixed"]]
    gold = np.array([f["gold_sentiment"] for f in pool])
    for k in (1, 2, 3, 5, 8, 12, 20, 30):
        preds = []
        for f in pool:
            n = min(k, len(f["scores"]))
            draw = rng.choice(f["scores"], size=n, replace=False)
            preds.append(int(aggregate_film_sentiment(draw, method="trimmed_mean",
                                                      n_boot=1).score >= 0.5))
        usable = np.array([len(f["scores"]) >= k for f in pool])
        preds = np.array(preds)
        depth_rows.append(
            {
                "reviews_pooled": k,
                "n_films_with_enough": int(usable.sum()),
                "accuracy_all_films": float((preds == gold).mean()),
                "accuracy_films_with_enough": float((preds[usable] == gold[usable]).mean())
                if usable.any() else np.nan,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(depth_rows)


def plot_aggregation_depth(depth: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(depth["reviews_pooled"], depth["accuracy_films_with_enough"],
            marker="o", ms=4, color=PALETTE[0], lw=1.7)
    ax.set_xlabel("Number of reviews pooled per film")
    ax.set_ylabel("Film-level accuracy (mixed-polarity films)")
    ax.set_title("Pooling reviews converts a good classifier into a reliable verdict",
                 fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xticks(depth["reviews_pooled"], labels=depth["reviews_pooled"].astype(str))
    path = FIGURES_DIR / "fig12_aggregation_depth.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    set_seed()
    test = load_imdb("test")

    log.info("Loading the deployed model")
    system = SentimentSystem(backend="classical")
    log.info("Scoring %d test reviews with %s", len(test), system.model_name)
    test = test.copy()
    test["score"] = system.score(test["text"].tolist())
    test["pred"] = (test["score"] >= 0.5).astype(int)
    log.info("Overall test accuracy %.4f", (test["pred"] == test["label"]).mean())

    rating_df = breakdown_by_rating(test)
    save_table(rating_df, "t13_accuracy_by_rating",
               caption="Test accuracy by the star rating the reviewer awarded")
    print_table(rating_df, "Accuracy by star rating")
    log.info("Figure: %s", plot_rating_breakdown(rating_df))

    length_df = breakdown_by_length(test)
    save_table(length_df, "t14_accuracy_by_length",
               caption="Test accuracy by review length quintile")
    print_table(length_df, "Accuracy by review length")

    construction_df = breakdown_by_construction(test)
    save_table(construction_df, "t15_accuracy_by_construction",
               caption="Test accuracy in the presence and absence of specific "
                       "linguistic constructions")
    print_table(construction_df, "Accuracy by linguistic construction")

    log.info("Figure: %s", plot_calibration(test))

    # Calibration quality, and what a simple isotonic recalibration would buy.
    # The recalibrator is fitted on a 20% slice of the test split and scored on
    # the remaining 80%, so no document is used to both fit and judge it.
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split as tts

    fit_idx, eval_idx = tts(
        np.arange(len(test)), test_size=0.8, random_state=SEED, stratify=test["label"]
    )
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        test["score"].to_numpy()[fit_idx], test["label"].to_numpy()[fit_idx]
    )
    raw_eval = test["score"].to_numpy()[eval_idx]
    lab_eval = test["label"].to_numpy()[eval_idx]
    cal_eval = iso.predict(raw_eval)

    calib = pd.DataFrame(
        [
            {
                "scores": "raw model output",
                "ece": expected_calibration_error(raw_eval, lab_eval),
                "accuracy": float(((raw_eval >= 0.5).astype(int) == lab_eval).mean()),
                "share_beyond_0.99_confidence_%": float(
                    100 * np.mean((raw_eval > 0.99) | (raw_eval < 0.01))
                ),
            },
            {
                "scores": "isotonic recalibration",
                "ece": expected_calibration_error(cal_eval, lab_eval),
                "accuracy": float(((cal_eval >= 0.5).astype(int) == lab_eval).mean()),
                "share_beyond_0.99_confidence_%": float(
                    100 * np.mean((cal_eval > 0.99) | (cal_eval < 0.01))
                ),
            },
        ]
    )
    save_table(calib, "t16b_calibration",
               caption="Calibration of the deployed model before and after isotonic "
                       "recalibration (fitted on 20% of the test split, scored on the other 80%)")
    print_table(calib, "Calibration quality")

    # Confident errors: the qualitative material discussed in the report.
    errors = test[test["pred"] != test["label"]].copy()
    errors["confidence"] = (errors["score"] - 0.5).abs() * 2
    worst = errors.nlargest(12, "confidence")[
        ["doc_id", "rating", "label", "pred", "score", "confidence", "text"]
    ].copy()
    worst["text"] = worst["text"].str.slice(0, 400)
    save_table(worst, "t16_confident_errors",
               caption="The twelve most confidently misclassified test reviews")
    log.info("Most confident errors written to results/tables/t16_confident_errors.csv")

    # ------------------------------------------------------------------ #
    log.info("Evaluating film-level aggregation")
    agg_df, depth_df = evaluate_film_aggregation(test)
    save_table(agg_df, "t17_film_aggregation",
               caption="Film-level verdict accuracy by aggregation rule. Ground truth "
                       "is the majority gold polarity of each film's reviews.")
    print_table(agg_df, "Film-level aggregation")

    save_table(depth_df, "t18_aggregation_depth",
               caption="Film-level accuracy as a function of the number of reviews pooled")
    print_table(depth_df, "Aggregation depth")
    log.info("Figure: %s", plot_aggregation_depth(depth_df))

    save_run_metadata("07_error_analysis")
    log.info("Done. Next: python scripts/08_demo_film_verdict.py")


if __name__ == "__main__":
    main()
