#!/usr/bin/env python3
"""Step 4 - the scenario ladder: how little labelled data is enough?

The task asks for the system to be evaluated "starting from a small dataset of
movie reviews; then progressively test it on larger datasets". This script
implements that literally: the identical pipeline is retrained on stratified
subsamples of 200, 500, 1,000, 2,500, 5,000, 10,000 and 25,000 documents and
each fit is scored on the *same* full 25,000-document test split, so the only
thing that varies is the amount of supervision.

Small subsamples are noisy, so every size below 25,000 is repeated over several
random draws and the mean plus the range across draws is reported.

Each rung retunes its own regularisation strength by cross-validation *inside*
that subsample. Holding a single hyper-parameter fixed across the ladder would
confound the effect of data volume with the effect of a setting that happens to
suit one end of it: the optimal penalty for 200 documents is not the optimal
penalty for 25,000.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from src.config import SEED, TRAIN_SIZE_LADDER, set_seed
from src.data import load_imdb
from src.evaluate import plot_learning_curves
from src.models_classical import MODEL_DISPLAY_NAMES, build_model
from src.preprocess import DEFAULT_PREPROCESSOR_KWARGS, Preprocessor
from src.reporting import print_table, save_run_metadata, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("curve")

MODELS = ("naive_bayes", "linear_svm", "logistic_regression", "nbsvm")

# Same search ranges as step 3, so the top rung of the ladder reproduces the
# headline table rather than diverging from it.
PARAM_GRIDS = {
    "naive_bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
    "linear_svm": {"clf__C": [0.1, 0.5, 1.0]},
    "logistic_regression": {"clf__C": [1.0, 4.0, 16.0]},
    "nbsvm": {"clf__C": [1.0, 10.0, 30.0, 100.0]},
}


def repeats_for(size: int, total: int) -> int:
    """More random draws where the subsample is small and therefore noisy."""
    if size >= total:
        return 1
    if size <= 2_500:
        return 5
    return 3


def fit_tuned(model_name: str, texts, labels):
    """Fit with the regularisation strength chosen by CV within this subsample."""
    # 3 folds rather than 5: at 200 documents a 5-fold split leaves 40 per fold,
    # and the extra folds cost more than the variance reduction is worth.
    n_splits = 3 if len(labels) < 1_000 else 5
    search = GridSearchCV(
        build_model(model_name), PARAM_GRIDS[model_name],
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=-1, refit=True,
    )
    search.fit(texts, labels)
    return search.best_estimator_, search.best_params_


def main() -> None:
    set_seed()
    train, test = load_imdb("train"), load_imdb("test")

    pre = Preprocessor(**DEFAULT_PREPROCESSOR_KWARGS)
    log.info("Preprocessing 50,000 documents once and reusing for every fit")
    t0 = time.time()
    X_train_all = np.array(pre.transform(train["text"]), dtype=object)
    X_test = pre.transform(test["text"])
    y_train_all = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    log.info("Preprocessing took %.1fs", time.time() - t0)

    total = len(X_train_all)
    rows = []

    for size in TRAIN_SIZE_LADDER:
        n_rep = repeats_for(size, total)
        if size >= total:
            draws = [np.arange(total)]
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=n_rep, train_size=size, random_state=SEED
            )
            draws = [idx for idx, _ in splitter.split(X_train_all, y_train_all)]

        for model_name in MODELS:
            accs, times, feats, chosen = [], [], [], []
            for rep, idx in enumerate(draws):
                t0 = time.time()
                model, params = fit_tuned(
                    model_name, X_train_all[idx].tolist(), y_train_all[idx]
                )
                fit_s = time.time() - t0
                acc = float(model.score(X_test, y_test))
                accs.append(acc)
                times.append(fit_s)
                feats.append(len(model.named_steps["tfidf"].get_feature_names_out()))
                chosen.append(next(iter(params.values())))

            rows.append(
                {
                    "train_size": size,
                    "model": MODEL_DISPLAY_NAMES[model_name],
                    "model_key": model_name,
                    "repeats": len(draws),
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_min": float(np.min(accs)),
                    "accuracy_max": float(np.max(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "n_features_mean": float(np.mean(feats)),
                    "fit_seconds_mean": float(np.mean(times)),
                    "median_hyperparameter": float(np.median(chosen)),
                }
            )
            log.info("n=%6d  %-24s acc=%.4f +/- %.4f  (%d draws, %.1fs/fit, median param %.3g)",
                     size, MODEL_DISPLAY_NAMES[model_name], np.mean(accs),
                     np.std(accs), len(draws), np.mean(times), np.median(chosen))

    df = pd.DataFrame(rows)
    save_table(df.drop(columns=["model_key"]), "t07_learning_curve",
               caption="Test accuracy as a function of the number of labelled training "
                       "documents. Every row is scored on the same full 25,000-document "
                       "aclImdb test split.")

    pivot = df.pivot(index="train_size", columns="model", values="accuracy_mean").reset_index()
    save_table(pivot, "t08_learning_curve_pivot",
               caption="Scenario ladder: mean test accuracy by training-set size")
    print_table(pivot, "Scenario ladder (mean test accuracy)")

    curves = {
        MODEL_DISPLAY_NAMES[key]: {
            "sizes": sub["train_size"].tolist(),
            "accuracy": sub["accuracy_mean"].tolist(),
            "ci_low": sub["accuracy_min"].tolist(),
            "ci_high": sub["accuracy_max"].tolist(),
        }
        for key in MODELS
        for sub in [df[df["model_key"] == key].sort_values("train_size")]
    }
    log.info("Figure: %s", plot_learning_curves(curves, "fig07_learning_curve.png"))

    # How much of the full-data accuracy is recovered at each rung of the ladder?
    best_key = df[df["train_size"] == max(TRAIN_SIZE_LADDER)].nlargest(1, "accuracy_mean")["model_key"].iloc[0]
    best = df[df["model_key"] == best_key].sort_values("train_size")
    ceiling = best["accuracy_mean"].iloc[-1]
    efficiency = pd.DataFrame(
        {
            "train_size": best["train_size"].to_numpy(),
            "accuracy": best["accuracy_mean"].to_numpy(),
            "share_of_full_data_accuracy_%": 100 * best["accuracy_mean"].to_numpy() / ceiling,
            "labelling_cost_relative_%": 100 * best["train_size"].to_numpy() / max(TRAIN_SIZE_LADDER),
        }
    )
    save_table(efficiency, "t09_label_efficiency",
               caption=f"Label efficiency of {MODEL_DISPLAY_NAMES[best_key]}: accuracy "
                       "recovered per unit of annotation effort")
    print_table(efficiency, f"Label efficiency ({MODEL_DISPLAY_NAMES[best_key]})")

    save_run_metadata("04_learning_curve", {"ladder": list(TRAIN_SIZE_LADDER)})
    log.info("Done. Next: python scripts/05_train_transformer.py")


if __name__ == "__main__":
    main()
