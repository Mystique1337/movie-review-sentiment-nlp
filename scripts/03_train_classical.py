#!/usr/bin/env python3
"""Step 3 - train and compare the classical supervised models.

Four supervised learners are trained on the full 25,000-document aclImdb train
split and scored once on the official 25,000-document test split. A lexicon
baseline (VADER) is included without any training at all, to establish how much
of the final accuracy is actually attributable to supervised learning rather
than to the obvious polarity of the vocabulary.

Model selection (the regularisation strength of each learner) is done by
cross-validation inside the training split. The test split is used exactly once
per model, at the end.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import SEED, set_seed
from src.data import load_imdb
from src.evaluate import (
    compute_metrics,
    mcnemar_test,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_top_features,
    save_metrics,
    text_report,
)
from src.models_classical import (
    MODEL_DISPLAY_NAMES,
    build_model,
    decision_scores,
    top_features,
)
from src.pipeline import save_classical
from src.preprocess import (
    DEFAULT_PREPROCESSOR_KWARGS,
    PRESCRIBED_PREPROCESSOR_KWARGS,
    Preprocessor,
)
from src.reporting import print_table, save_run_metadata, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("classical")

# Search grids kept deliberately small: each extra point costs a full 5-fold fit
# on 25,000 documents, and the resources available for this project are a single
# laptop rather than a cluster.
PARAM_GRIDS = {
    "naive_bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
    "linear_svm": {"clf__C": [0.1, 0.5, 1.0]},
    "logistic_regression": {"clf__C": [1.0, 4.0, 16.0]},
    # Widened after an initial run selected the grid's upper boundary, which is
    # the standard signal that the search range was too narrow. Selection is
    # still made by cross-validation on the training split only.
    "nbsvm": {"clf__C": [1.0, 10.0, 30.0, 100.0]},
}


def vader_baseline(texts, labels) -> dict:
    """Unsupervised rule-based sentiment, for reference only."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:  # pragma: no cover
        nltk.download("vader_lexicon", quiet=True)
        analyzer = SentimentIntensityAnalyzer()

    t0 = time.time()
    compound = np.array([analyzer.polarity_scores(t)["compound"] for t in texts])
    preds = (compound >= 0).astype(int)
    scores = (compound + 1) / 2  # map [-1, 1] to [0, 1]
    metrics = compute_metrics(labels, preds, scores, label="vader")
    metrics["train_seconds"] = 0.0
    metrics["predict_seconds"] = time.time() - t0
    return metrics, preds, scores


def main() -> None:
    set_seed()
    train, test = load_imdb("train"), load_imdb("test")
    log.info("Train %d documents, test %d documents", len(train), len(test))

    pre = Preprocessor(**DEFAULT_PREPROCESSOR_KWARGS)
    log.info("Preprocessing (%s)", pre.describe())
    t0 = time.time()
    X_train = pre.transform(train["text"])
    X_test = pre.transform(test["text"])
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    log.info("Preprocessed 50,000 documents in %.1fs", time.time() - t0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, predictions, roc_inputs = [], {}, {}

    for name in ("naive_bayes", "linear_svm", "logistic_regression", "nbsvm"):
        log.info("=== %s ===", MODEL_DISPLAY_NAMES[name])
        search = GridSearchCV(
            build_model(name), PARAM_GRIDS[name], cv=cv,
            scoring="accuracy", n_jobs=-1, refit=True, verbose=0,
        )
        t0 = time.time()
        search.fit(X_train, y_train)
        train_seconds = time.time() - t0
        model = search.best_estimator_

        t0 = time.time()
        preds = model.predict(X_test)
        scores = decision_scores(model, X_test)
        predict_seconds = time.time() - t0

        metrics = compute_metrics(y_test, preds, scores, label=name)
        metrics.update(
            best_params={k: float(v) for k, v in search.best_params_.items()},
            cv_accuracy=float(search.best_score_),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            n_features=int(len(model.named_steps["tfidf"].get_feature_names_out())),
        )
        save_metrics(metrics, f"03_{name}")
        predictions[name] = preds
        roc_inputs[MODEL_DISPLAY_NAMES[name]] = (y_test, scores)

        log.info("test accuracy %.4f  (CV %.4f, fit %.0fs)",
                 metrics["accuracy"], metrics["cv_accuracy"], train_seconds)
        print(text_report(y_test, preds))

        rows.append(
            {
                "model": MODEL_DISPLAY_NAMES[name],
                "cv_accuracy": metrics["cv_accuracy"],
                "test_accuracy": metrics["accuracy"],
                "ci95_low": metrics["accuracy_ci95"][0],
                "ci95_high": metrics["accuracy_ci95"][1],
                "f1_macro": metrics["f1_macro"],
                "roc_auc": metrics["roc_auc"],
                "n_features": metrics["n_features"],
                "train_s": train_seconds,
                "predict_s": predict_seconds,
            }
        )

        plot_confusion_matrix(
            y_test, preds, MODEL_DISPLAY_NAMES[name],
            f"fig04_cm_{name}.png",
        )
        if name in {"linear_svm", "nbsvm", "logistic_regression"}:
            pos, neg = top_features(model, n=18)
            plot_top_features(
                pos, neg,
                f"Highest-weighted n-grams, {MODEL_DISPLAY_NAMES[name]}",
                f"fig05_features_{name}.png",
            )

    # Untrained reference point.
    log.info("=== VADER lexicon baseline (no training) ===")
    vader_metrics, vader_preds, vader_scores = vader_baseline(test["text"], y_test)
    save_metrics(vader_metrics, "03_vader")
    predictions["vader"] = vader_preds
    roc_inputs[MODEL_DISPLAY_NAMES["vader"]] = (y_test, vader_scores)
    log.info("test accuracy %.4f", vader_metrics["accuracy"])
    rows.append(
        {
            "model": MODEL_DISPLAY_NAMES["vader"],
            "cv_accuracy": float("nan"),
            "test_accuracy": vader_metrics["accuracy"],
            "ci95_low": vader_metrics["accuracy_ci95"][0],
            "ci95_high": vader_metrics["accuracy_ci95"][1],
            "f1_macro": vader_metrics["f1_macro"],
            "roc_auc": vader_metrics["roc_auc"],
            "n_features": 7_500,  # size of the VADER lexicon
            "train_s": 0.0,
            "predict_s": vader_metrics["predict_seconds"],
        }
    )

    results = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    save_table(results, "t05_classical_model_comparison",
               caption="Classical model comparison on the official aclImdb test split "
                       "(25,000 documents). CV accuracy is 5-fold on the training split.")
    print_table(results, "Classical models on aclImdb test")

    plot_roc_curves(roc_inputs, "fig06_roc_classical.png")

    # Paired significance tests against the strongest model.
    best_name = max(
        ("naive_bayes", "linear_svm", "logistic_regression", "nbsvm"),
        key=lambda n: (predictions[n] == y_test).mean(),
    )
    sig_rows = []
    for name, preds in predictions.items():
        if name == best_name:
            continue
        test_result = mcnemar_test(y_test, predictions[best_name], preds)
        sig_rows.append(
            {
                "comparison": f"{MODEL_DISPLAY_NAMES[best_name]} vs {MODEL_DISPLAY_NAMES[name]}",
                "best_only_correct": test_result["n01"],
                "other_only_correct": test_result["n10"],
                "p_value": test_result["p_value"],
                "significant_p<.05": test_result["significant_at_05"],
            }
        )
    sig = pd.DataFrame(sig_rows)
    save_table(sig, "t06_mcnemar_classical",
               caption=f"Exact McNemar tests against {MODEL_DISPLAY_NAMES[best_name]} "
                       "on the aclImdb test split")
    print_table(sig, "Paired significance tests")

    # -- cost of the prescribed preprocessing recipe ------------------------ #
    # The ablation in step 2 chose its configuration on validation data. This
    # confirms the decision on the held-out test split, so the report can state
    # the cost of the textbook recipe rather than infer it.
    log.info("=== %s under the prescribed preprocessing recipe ===", MODEL_DISPLAY_NAMES[best_name])
    pre_strict = Preprocessor(**PRESCRIBED_PREPROCESSOR_KWARGS)
    Xs_train = pre_strict.transform(train["text"])
    Xs_test = pre_strict.transform(test["text"])
    strict = GridSearchCV(build_model(best_name), PARAM_GRIDS[best_name], cv=cv,
                          scoring="accuracy", n_jobs=-1).fit(Xs_train, y_train)
    strict_preds = strict.predict(Xs_test)
    strict_metrics = compute_metrics(y_test, strict_preds, label=f"{best_name}_prescribed")
    save_metrics(strict_metrics, f"03_{best_name}_prescribed")

    best_row = next(r for r in rows if r["model"] == MODEL_DISPLAY_NAMES[best_name])
    prep_cmp = pd.DataFrame(
        [
            {
                "preprocessing": "selected (lemmatise, keep function words)",
                "description": pre.describe(),
                "test_accuracy": float((predictions[best_name] == y_test).mean()),
                "n_features": best_row["n_features"],
            },
            {
                "preprocessing": "prescribed (stop-word removal + stemming)",
                "description": pre_strict.describe(),
                "test_accuracy": strict_metrics["accuracy"],
                "n_features": int(len(strict.best_estimator_.named_steps["tfidf"].get_feature_names_out())),
            },
        ]
    )
    mc = mcnemar_test(y_test, predictions[best_name], strict_preds)
    prep_cmp["mcnemar_p"] = [float("nan"), mc["p_value"]]
    save_table(prep_cmp, "t06b_preprocessing_on_test",
               caption=f"Effect of the preprocessing recipe on the test split "
                       f"({MODEL_DISPLAY_NAMES[best_name]}, 25,000 test documents)")
    print_table(prep_cmp.drop(columns=["description"]), "Preprocessing recipe, test split")

    # Persist the best model for the CLI and for the film-level demonstration.
    log.info("Refitting %s on the full training split for deployment", best_name)
    final = GridSearchCV(build_model(best_name), PARAM_GRIDS[best_name], cv=cv,
                         scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
    save_classical(
        final.best_estimator_,
        model_name=best_name,
        preprocessor_kwargs=DEFAULT_PREPROCESSOR_KWARGS,
        metadata={
            "test_accuracy": float((predictions[best_name] == y_test).mean()),
            "trained_on": "aclImdb train (25,000 documents)",
            "best_params": {k: float(v) for k, v in final.best_params_.items()},
        },
    )

    save_run_metadata("03_classical", {"best_model": best_name})
    log.info("Done. Next: python scripts/04_learning_curve.py")


if __name__ == "__main__":
    main()
