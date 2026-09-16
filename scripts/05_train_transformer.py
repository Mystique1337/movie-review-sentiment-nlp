#!/usr/bin/env python3
"""Step 5 - the deep-learning arm: fine-tune DistilBERT.

Two questions are asked of the transformer:

1.  Does contextual pretraining beat a well-tuned bag of words on the full
    25,000-document training split, and by how much relative to its cost?
2.  Does it beat the classical models when labelled data is scarce? Pretraining
    supplies linguistic knowledge that the classical arm has to learn from the
    labels alone, so the gap should be widest at the bottom of the ladder. The
    same three rungs used in step 4 are repeated here so the two arms are
    directly comparable.

Everything runs on the laptop's integrated GPU through PyTorch's Metal
(MPS) backend. Wall-clock time is recorded because "which model is best" and
"which model is worth it" are different questions, and the second one matters
for a project constrained to consumer hardware.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from src.config import MODELS_DIR, SEED, VALIDATION_FRACTION, set_seed
from src.data import load_imdb
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics, text_report
from src.models_transformer import (
    MODEL_NAME,
    TrainingConfig,
    fine_tune_distilbert,
    get_device,
    predict_proba,
)
from src.reporting import print_table, save_run_metadata, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("transformer")

# Matches the rungs of the classical ladder that matter for the comparison:
# a tiny set, a small set, and the full corpus.
LADDER = (500, 2_500, 25_000)


def token_length_report(texts, tokenizer, max_length: int) -> pd.DataFrame:
    """How much of a review does a 256-subword window actually cover?

    Whitespace word counts understate the problem: WordPiece splits rare words
    and every punctuation mark into separate tokens, so a 256-token window holds
    noticeably fewer than 256 words.
    """
    sample = list(texts[:3_000])
    lengths = np.array([len(tokenizer(t, truncation=False)["input_ids"]) for t in sample])
    return pd.DataFrame(
        [
            {
                "n_sampled": len(lengths),
                "median_subword_tokens": float(np.median(lengths)),
                "mean_subword_tokens": float(lengths.mean()),
                "p90_subword_tokens": float(np.percentile(lengths, 90)),
                "max_subword_tokens": int(lengths.max()),
                f"share_truncated_at_{max_length}_%": float(100 * (lengths > max_length).mean()),
                f"mean_retained_at_{max_length}_%": float(
                    100 * np.minimum(lengths, max_length).sum() / lengths.sum()
                ),
            }
        ]
    )


def main() -> None:
    set_seed()
    device = get_device()
    log.info("Device: %s", device)

    train, test = load_imdb("train"), load_imdb("test")
    y_test = test["label"].to_numpy()
    test_texts = test["text"].tolist()

    cfg = TrainingConfig()

    # -- input budget ------------------------------------------------------- #
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tl = token_length_report(train["text"].tolist(), tokenizer, cfg.max_length)
    save_table(tl, "t10_subword_truncation",
               caption=f"Subword-token budget for {MODEL_NAME} at max_length={cfg.max_length}")
    print_table(tl, "Subword token lengths (aclImdb train)")

    rows = []
    for size in LADDER:
        log.info("=" * 70)
        log.info("Fine-tuning on %d labelled documents", size)
        log.info("=" * 70)

        if size >= len(train):
            subset = train
        else:
            idx, _ = next(
                StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=SEED)
                .split(train["text"], train["label"])
            )
            subset = train.iloc[idx]

        tr, va = train_test_split(
            subset, test_size=VALIDATION_FRACTION, random_state=SEED,
            stratify=subset["label"],
        )

        # A small training set needs more passes to converge; a large one overfits.
        run_cfg = TrainingConfig(epochs=3 if size < 5_000 else 2)

        t0 = time.time()
        model, tok, history = fine_tune_distilbert(
            tr["text"].tolist(), tr["label"].tolist(),
            va["text"].tolist(), va["label"].tolist(),
            config=run_cfg,
            output_dir=MODELS_DIR / "distilbert-imdb" if size >= len(train) else None,
        )
        train_seconds = time.time() - t0

        t0 = time.time()
        scores = predict_proba(model, tok, test_texts, run_cfg, device)
        predict_seconds = time.time() - t0
        preds = (scores >= 0.5).astype(int)

        metrics = compute_metrics(y_test, preds, scores, label=f"distilbert_{size}")
        metrics.update(
            train_size=size,
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            history=history,
            n_parameters=int(sum(p.numel() for p in model.parameters())),
        )
        save_metrics(metrics, f"05_distilbert_{size}")

        log.info("n=%d  test accuracy %.4f  (train %.0fs, inference %.0fs)",
                 size, metrics["accuracy"], train_seconds, predict_seconds)
        print(text_report(y_test, preds))

        rows.append(
            {
                "train_size": size,
                "model": "DistilBERT (fine-tuned)",
                "epochs": run_cfg.epochs,
                "test_accuracy": metrics["accuracy"],
                "ci95_low": metrics["accuracy_ci95"][0],
                "ci95_high": metrics["accuracy_ci95"][1],
                "f1_macro": metrics["f1_macro"],
                "roc_auc": metrics["roc_auc"],
                "train_s": train_seconds,
                "predict_s": predict_seconds,
            }
        )

        if size >= len(train):
            plot_confusion_matrix(y_test, preds, "DistilBERT (fine-tuned)",
                                  "fig08_cm_distilbert.png")
            np.save(MODELS_DIR / "distilbert_test_scores.npy", scores)
            (MODELS_DIR / "distilbert-imdb" / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "base_model": MODEL_NAME,
                        "train_size": size,
                        "test_accuracy": metrics["accuracy"],
                        "config": run_cfg.to_dict(),
                        "device": str(device),
                        "train_seconds": train_seconds,
                    },
                    indent=2,
                )
            )

        del model
        try:
            import torch

            if device.type == "mps":
                torch.mps.empty_cache()
        except Exception:  # pragma: no cover
            pass

    df = pd.DataFrame(rows)
    save_table(df, "t11_transformer_results",
               caption="DistilBERT fine-tuning across the scenario ladder, scored on the "
                       "full 25,000-document aclImdb test split")
    print_table(df, "DistilBERT on aclImdb test")

    save_run_metadata("05_transformer", {"device": str(device), "ladder": list(LADDER)})
    log.info("Done. Next: python scripts/06_cross_domain.py")


if __name__ == "__main__":
    main()
