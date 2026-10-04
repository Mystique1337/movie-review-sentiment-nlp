#!/usr/bin/env python3
"""Step 6 - does the system generalise beyond the corpus it was trained on?

Accuracy on a held-out split of the *same* corpus overstates what a deployed
system would achieve, because train and test share vocabulary, era, review
length and annotation conventions. Blitzer et al. (2007, pp. 440, 442) measure
exactly this loss for sentiment classifiers moved between review domains, and
step 1 already showed the warning sign: aclImdb and the Rotten Tomatoes snippets share only about a third of their
frequent vocabulary.

This script therefore transfers models across corpora in both directions and
contrasts each transfer result with a within-corpus reference obtained by
cross-validating the same model on the target corpus itself. The gap between the
two is the honest estimate of the cost of domain shift.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.config import MODELS_DIR, SEED, set_seed
from src.data import load_corpus
from src.evaluate import PALETTE, compute_metrics, save_metrics
from src.models_classical import build_model, decision_scores
from src.preprocess import DEFAULT_PREPROCESSOR_KWARGS, Preprocessor
from src.reporting import print_table, save_run_metadata, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("crossdomain")

BEST_CLASSICAL = "nbsvm"

# Identical to the grid in step 3. Every fit in this script is tuned, otherwise
# the in-domain row here would disagree with the headline table purely because
# one used a default penalty and the other a cross-validated one.
PARAM_GRID = {"clf__C": [1.0, 10.0, 30.0, 100.0]}


def fit_tuned(texts, labels, model_name: str = BEST_CLASSICAL):
    search = GridSearchCV(
        build_model(model_name), PARAM_GRID,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=-1, refit=True,
    )
    search.fit(texts, labels)
    return search.best_estimator_


def within_corpus_reference(texts, labels, model_name: str = BEST_CLASSICAL) -> float:
    """Cross-validated accuracy of the same model trained on the target corpus.

    This is the reference point a transfer result should be read against: it is
    what the pipeline achieves when it is allowed to learn the target domain's
    own vocabulary. It is not an absolute ceiling, and a stronger model can
    exceed it, which is exactly what the transformer does on the RT sentences.
    """
    inner = GridSearchCV(
        build_model(model_name), PARAM_GRID,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(inner, texts, labels, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def transformer_scores(texts) -> np.ndarray | None:
    """Score texts with the fine-tuned checkpoint, if step 5 has been run."""
    path = MODELS_DIR / "distilbert-imdb"
    if not path.exists():
        log.warning("No fine-tuned DistilBERT at %s - skipping the transformer arm", path)
        return None
    from src.models_transformer import TrainingConfig, load_fine_tuned, predict_proba

    model, tokenizer = load_fine_tuned(path)
    return predict_proba(model, tokenizer, list(texts), TrainingConfig())


def main() -> None:
    set_seed()
    pre = Preprocessor(**DEFAULT_PREPROCESSOR_KWARGS)

    log.info("Loading corpora")
    corpora = {
        "aclImdb train": load_corpus("imdb_train"),
        "aclImdb test": load_corpus("imdb_test"),
        "Pang & Lee v2.0": load_corpus("pang_lee"),
        "RT sentences": load_corpus("rt_polarity"),
    }

    log.info("Preprocessing")
    processed = {name: pre.transform(df["text"]) for name, df in corpora.items()}
    labels = {name: df["label"].to_numpy() for name, df in corpora.items()}

    # ---------------------------------------------------------------------- #
    # Within-corpus references
    # ---------------------------------------------------------------------- #
    log.info("Within-corpus references (cross-validated on the target corpus)")
    ceilings = {}
    for name in ("Pang & Lee v2.0", "RT sentences"):
        ceilings[name] = within_corpus_reference(processed[name], labels[name])
        log.info("  %-18s %.4f", name, ceilings[name])
    # For aclImdb the reference is the honest train->test result, not CV.
    imdb_model = fit_tuned(processed["aclImdb train"], labels["aclImdb train"])
    ceilings["aclImdb test"] = float(
        imdb_model.score(processed["aclImdb test"], labels["aclImdb test"])
    )
    log.info("  %-18s %.4f", "aclImdb test", ceilings["aclImdb test"])

    # ---------------------------------------------------------------------- #
    # Transfer: trained on aclImdb, applied elsewhere
    # ---------------------------------------------------------------------- #
    rows = []
    for target in ("aclImdb test", "Pang & Lee v2.0", "RT sentences"):
        preds = imdb_model.predict(processed[target])
        scores = decision_scores(imdb_model, processed[target])
        m = compute_metrics(labels[target], preds, scores,
                            label=f"imdb->{target}")
        save_metrics(m, f"06_classical_imdb_to_{target.split()[0].lower()}")
        rows.append(
            {
                "model": "NB-SVM",
                "trained_on": "aclImdb train (25,000)",
                "evaluated_on": target,
                "n_test": m["n"],
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "roc_auc": m["roc_auc"],
                "within_corpus_reference": ceilings[target],
                "gap_vs_reference_pp": 100 * (ceilings[target] - m["accuracy"]),
            }
        )
        log.info("NB-SVM  aclImdb -> %-18s acc=%.4f (reference %.4f)",
                 target, m["accuracy"], ceilings[target])

    # ---------------------------------------------------------------------- #
    # Reverse transfer: trained on the small 2,000-document corpus
    # ---------------------------------------------------------------------- #
    small_model = fit_tuned(processed["Pang & Lee v2.0"], labels["Pang & Lee v2.0"])
    for target in ("aclImdb test", "RT sentences"):
        preds = small_model.predict(processed[target])
        scores = decision_scores(small_model, processed[target])
        m = compute_metrics(labels[target], preds, scores, label=f"pangLee->{target}")
        rows.append(
            {
                "model": "NB-SVM",
                "trained_on": "Pang & Lee (2,000)",
                "evaluated_on": target,
                "n_test": m["n"],
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "roc_auc": m["roc_auc"],
                "within_corpus_reference": ceilings[target],
                "gap_vs_reference_pp": 100 * (ceilings[target] - m["accuracy"]),
            }
        )
        log.info("NB-SVM  Pang&Lee -> %-18s acc=%.4f (reference %.4f)",
                 target, m["accuracy"], ceilings[target])

    # ---------------------------------------------------------------------- #
    # Transformer transfer
    # ---------------------------------------------------------------------- #
    for target in ("aclImdb test", "Pang & Lee v2.0", "RT sentences"):
        scores = transformer_scores(corpora[target]["text"])
        if scores is None:
            break
        preds = (scores >= 0.5).astype(int)
        m = compute_metrics(labels[target], preds, scores, label=f"distilbert->{target}")
        save_metrics(m, f"06_distilbert_to_{target.split()[0].lower()}")
        rows.append(
            {
                "model": "DistilBERT",
                "trained_on": "aclImdb train (25,000)",
                "evaluated_on": target,
                "n_test": m["n"],
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "roc_auc": m["roc_auc"],
                "within_corpus_reference": ceilings[target],
                "gap_vs_reference_pp": 100 * (ceilings[target] - m["accuracy"]),
            }
        )
        log.info("DistilBERT  aclImdb -> %-18s acc=%.4f", target, m["accuracy"])

    df = pd.DataFrame(rows)
    save_table(df, "t12_cross_domain",
               caption="Cross-corpus transfer. The within-corpus reference is the same "
                       "model cross-validated on the target corpus (train/test for "
                       "aclImdb). A positive gap is the cost of domain shift; a negative "
                       "gap means the transferred model beat that reference.")
    print_table(df, "Cross-domain transfer")

    # Figure: in-domain vs out-of-domain, grouped by model.
    import matplotlib.pyplot as plt

    imdb_trained = df[df["trained_on"].str.startswith("aclImdb")]
    targets = ["aclImdb test", "Pang & Lee v2.0", "RT sentences"]
    models = imdb_trained["model"].unique()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    width = 0.8 / (len(models) + 1)
    xs = np.arange(len(targets))
    for i, model in enumerate(models):
        sub = imdb_trained[imdb_trained["model"] == model].set_index("evaluated_on").reindex(targets)
        ax.bar(xs + i * width, sub["accuracy"], width=width,
               label=model, color=PALETTE[i % len(PALETTE)])
        for x, v in zip(xs + i * width, sub["accuracy"]):
            if not np.isnan(v):
                ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ceiling_vals = [ceilings[t] for t in targets]
    ax.bar(xs + len(models) * width, ceiling_vals, width=width,
           label="within-corpus reference", color="0.65")
    for x, v in zip(xs + len(models) * width, ceiling_vals):
        ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs + width * len(models) / 2, labels=targets, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_title("Models trained on aclImdb lose most of their edge off-domain", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    path = Path(__file__).resolve().parents[1] / "results/figures/fig09_cross_domain.png"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure: %s", path)

    save_run_metadata("06_cross_domain")
    log.info("Done. Next: python scripts/07_error_analysis.py")


if __name__ == "__main__":
    main()
