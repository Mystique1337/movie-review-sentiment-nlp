#!/usr/bin/env python3
"""Step 2 - measure what each prescribed preprocessing step actually buys.

The task specifies stop-word removal and stemming or lemmatisation. Rather than
applying them on faith, this script runs a factorial ablation and lets the
validation data decide. The headline question is whether the textbook recipe
(naive stop-word list plus Porter stemming) is in fact the best configuration
for sentiment, where function words such as "not" carry the signal.

All scoring happens on a validation split carved out of aclImdb *train*. The
official test split is not touched here.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import METRICS_DIR, SEED, VALIDATION_FRACTION, set_seed
from src.data import load_imdb
from src.evaluate import plot_bar_comparison
from src.models_classical import build_model
from src.preprocess import Preprocessor
from src.reporting import print_table, save_run_metadata, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("ablation")

# (short name, description for the report, Preprocessor kwargs)
CONFIGS = [
    ("raw", "no stop-word removal, no normalisation",
     dict(remove_stopwords=False, normalisation="none")),
    ("sw-naive", "naive NLTK stop-word list (removes negations)",
     dict(remove_stopwords=True, keep_negations=False, normalisation="none")),
    ("sw-keepneg", "stop-word list with negation cues retained",
     dict(remove_stopwords=True, keep_negations=True, normalisation="none")),
    ("sw-keepneg+stem", "negation-aware stop words + Porter stemming",
     dict(remove_stopwords=True, keep_negations=True, normalisation="stem")),
    ("sw-keepneg+lemma", "negation-aware stop words + WordNet lemmatisation",
     dict(remove_stopwords=True, keep_negations=True, normalisation="lemma")),
    ("sw-naive+stem", "textbook recipe: naive stop words + stemming",
     dict(remove_stopwords=True, keep_negations=False, normalisation="stem")),
    ("lemma-only", "lemmatisation, no stop-word removal",
     dict(remove_stopwords=False, normalisation="lemma")),
    ("sw-keepneg+lemma+negmark", "negation-aware stop words + lemma + NEG scope marking",
     dict(remove_stopwords=True, keep_negations=True, normalisation="lemma",
          negation_marking=True)),
]

NGRAM_SETTINGS = [(1, "unigrams"), (2, "unigrams + bigrams")]


def main() -> None:
    set_seed()
    train = load_imdb("train")
    tr, va = train_test_split(
        train, test_size=VALIDATION_FRACTION, random_state=SEED, stratify=train["label"]
    )
    log.info("Ablation on %d train / %d validation documents", len(tr), len(va))

    rows = []
    for short, description, kwargs in CONFIGS:
        pre = Preprocessor(**kwargs)

        t0 = time.time()
        tr_text = pre.transform(tr["text"])
        va_text = pre.transform(va["text"])
        prep_seconds = time.time() - t0

        vocab_tokens = sum(len(t.split()) for t in tr_text)
        log.info("%-26s preprocessed in %5.1fs (%d tokens kept)",
                 short, prep_seconds, vocab_tokens)

        for ngram_max, ngram_label in NGRAM_SETTINGS:
            t0 = time.time()
            model = build_model("logistic_regression", ngram_max=ngram_max)
            model.fit(tr_text, tr["label"])
            acc = float(model.score(va_text, va["label"]))
            fit_seconds = time.time() - t0
            n_features = len(model.named_steps["tfidf"].get_feature_names_out())

            rows.append(
                {
                    "config": short,
                    "description": description,
                    "features": ngram_label,
                    "val_accuracy": acc,
                    "n_features": n_features,
                    "tokens_retained_m": vocab_tokens / 1e6,
                    "preprocess_s": prep_seconds,
                    "fit_s": fit_seconds,
                }
            )
            log.info("  %-20s acc=%.4f  (%d features, fit %.1fs)",
                     ngram_label, acc, n_features, fit_seconds)

    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    save_table(df, "t04_preprocessing_ablation",
               caption="Effect of each preprocessing choice on validation accuracy "
                       "(logistic regression on aclImdb, 20,000 train / 5,000 validation)")
    print_table(df.drop(columns=["description"]), "Preprocessing ablation")

    # Figure: unigram+bigram row per configuration, in the order defined above.
    best_per_config = (
        df[df["features"] == "unigrams + bigrams"]
        .set_index("config")
        .reindex([c[0] for c in CONFIGS])
    )
    plot_bar_comparison(
        labels=list(best_per_config.index),
        values=best_per_config["val_accuracy"].tolist(),
        title="Preprocessing ablation (logistic regression, unigrams + bigrams)",
        ylabel="Validation accuracy",
        filename="fig03_preprocessing_ablation.png",
        ylim=(0.86, 0.91),
    )

    best = df.iloc[0]
    log.info("Best configuration: %s / %s at %.4f", best["config"], best["features"], best["val_accuracy"])
    (METRICS_DIR / "02_best_preprocessing.json").write_text(
        json.dumps(
            {
                "config": best["config"],
                "features": best["features"],
                "val_accuracy": float(best["val_accuracy"]),
                "kwargs": dict(next(c[2] for c in CONFIGS if c[0] == best["config"])),
            },
            indent=2,
        )
    )
    save_run_metadata("02_ablation")
    log.info("Done. Next: python scripts/03_train_classical.py")


if __name__ == "__main__":
    main()
