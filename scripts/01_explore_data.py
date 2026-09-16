#!/usr/bin/env python3
"""Step 1 - characterise the three corpora before any modelling.

Produces the corpus-statistics table and two figures used in the report. The
point of this step is to make the later design decisions defensible: the review
length distribution is what justifies the 256-token truncation limit for
DistilBERT, and the vocabulary overlap is what predicts the cross-domain drop
observed in step 6.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR
from src.data import load_corpus
from src.evaluate import PALETTE
from src.preprocess import Preprocessor, clean_text
from src.reporting import print_table, save_run_metadata, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("explore")

CORPORA = {
    "Pang & Lee v2.0": "pang_lee",
    "aclImdb train": "imdb_train",
    "aclImdb test": "imdb_test",
    "RT sentence polarity": "rt_polarity",
}


def describe(df: pd.DataFrame, name: str) -> dict:
    words = df["text"].str.split().str.len()
    chars = df["text"].str.len()
    vocab = Counter()
    for text in df["text"].sample(min(2_000, len(df)), random_state=0):
        vocab.update(clean_text(text).split())
    return {
        "corpus": name,
        "documents": len(df),
        "positive_%": 100 * df["label"].mean(),
        "median_words": float(words.median()),
        "mean_words": float(words.mean()),
        "p90_words": float(words.quantile(0.90)),
        "max_words": int(words.max()),
        "median_chars": float(chars.median()),
        "vocab_2k_sample": len(vocab),
        "hapax_%": 100 * sum(1 for c in vocab.values() if c == 1) / max(len(vocab), 1),
    }


def plot_length_distributions(frames: dict[str, pd.DataFrame]) -> Path:
    """Length histograms normalised so corpora of different spread stay comparable.

    Matplotlib's ``density=True`` divides by bin width, which on a logarithmic
    axis makes a narrow corpus (RT, 10-50 words) tower over a broad one
    (Pang & Lee, 300-2000 words) purely as an artefact of binning. Each corpus
    is instead weighted to sum to one, so the heights mean "share of this
    corpus" and can be read against each other.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    bins = np.logspace(0.5, 3.6, 46)
    for i, (name, df) in enumerate(frames.items()):
        lengths = df["text"].str.split().str.len().to_numpy()
        ax.hist(lengths, bins=bins, histtype="step", lw=1.8,
                weights=np.full(len(lengths), 1.0 / len(lengths)),
                color=PALETTE[i % len(PALETTE)], label=name)
    ax.axvline(256, ls="--", c="0.3", lw=1.1)
    ax.annotate("DistilBERT\ninput limit", xy=(256, ax.get_ylim()[1] * 0.92),
                xytext=(300, ax.get_ylim()[1] * 0.92), fontsize=7.5, va="top", color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("Review length (whitespace tokens, log scale)")
    ax.set_ylabel("Share of corpus")
    ax.set_title("Review length varies by an order of magnitude across corpora",
                 fontsize=9.5, pad=8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    path = FIGURES_DIR / "fig01_length_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_class_discriminative_words(df: pd.DataFrame) -> Path:
    """Words whose frequency differs most between the two classes (IMDb train)."""
    pre = Preprocessor(remove_stopwords=True, keep_negations=True, normalisation="lemma")
    sample = df.sample(4_000, random_state=0)
    pos, neg = Counter(), Counter()
    for text, label in zip(sample["text"], sample["label"]):
        (pos if label == 1 else neg).update(pre.tokenize(text))

    total_pos, total_neg = sum(pos.values()), sum(neg.values())
    rows = []
    for word in set(pos) | set(neg):
        p, n = pos.get(word, 0), neg.get(word, 0)
        if p + n < 40:  # ignore rare words, whose ratios are pure noise
            continue
        rows.append((word, np.log(((p + 1) / total_pos) / ((n + 1) / total_neg)), p + n))
    ranked = sorted(rows, key=lambda r: r[1])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    for ax, items, colour, side in (
        (axes[0], ranked[:18], PALETTE[1], "Negative reviews"),
        (axes[1], ranked[-18:], PALETTE[0], "Positive reviews"),
    ):
        items = items[::-1] if side.startswith("Positive") else items
        names = [r[0] for r in items][::-1]
        vals = [r[1] for r in items][::-1]
        ax.barh(np.arange(len(names)), vals, color=colour, height=0.72)
        ax.set_yticks(np.arange(len(names)), labels=names, fontsize=7.5)
        ax.set_xlabel("log frequency ratio")
        ax.set_title(side, fontsize=9)
    fig.suptitle("Most class-discriminative lemmas in aclImdb train (4,000-review sample)",
                 fontsize=9.5)
    # Widen the gutter so the right panel's tick labels do not sit on the left
    # panel's bars.
    fig.subplots_adjust(wspace=0.42)
    path = FIGURES_DIR / "fig02_discriminative_words.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def vocabulary_overlap(a: pd.DataFrame, b: pd.DataFrame, n: int = 5_000) -> float:
    """Jaccard overlap of the top-``n`` vocabularies of two corpora."""
    def top_vocab(df):
        c = Counter()
        for text in df["text"].sample(min(2_000, len(df)), random_state=1):
            c.update(clean_text(text).split())
        return {w for w, _ in c.most_common(n)}

    va, vb = top_vocab(a), top_vocab(b)
    return len(va & vb) / len(va | vb)


def main() -> None:
    frames = {name: load_corpus(key) for name, key in CORPORA.items()}

    stats = pd.DataFrame([describe(df, name) for name, df in frames.items()])
    save_table(stats, "t01_corpus_statistics",
               caption="Descriptive statistics of the three movie-review corpora")
    print_table(stats, "Corpus statistics")

    log.info("Figure: %s", plot_length_distributions(frames))
    log.info("Figure: %s", plot_class_discriminative_words(frames["aclImdb train"]))

    overlap = pd.DataFrame(
        [
            {
                "pair": "aclImdb train vs aclImdb test",
                "jaccard_top5k": vocabulary_overlap(frames["aclImdb train"], frames["aclImdb test"]),
            },
            {
                "pair": "aclImdb train vs Pang & Lee",
                "jaccard_top5k": vocabulary_overlap(frames["aclImdb train"], frames["Pang & Lee v2.0"]),
            },
            {
                "pair": "aclImdb train vs RT sentences",
                "jaccard_top5k": vocabulary_overlap(frames["aclImdb train"], frames["RT sentence polarity"]),
            },
        ]
    )
    save_table(overlap, "t02_vocabulary_overlap",
               caption="Lexical overlap between corpora (Jaccard index of the 5,000 most frequent types)")
    print_table(overlap, "Vocabulary overlap")

    # Truncation cost: what fraction of each corpus exceeds the transformer limit?
    trunc = pd.DataFrame(
        [
            {
                "corpus": name,
                "share_over_256_words_%": 100 * (df["text"].str.split().str.len() > 256).mean(),
                "share_over_512_words_%": 100 * (df["text"].str.split().str.len() > 512).mean(),
            }
            for name, df in frames.items()
        ]
    )
    save_table(trunc, "t03_truncation_cost",
               caption="Share of documents affected by transformer input truncation")
    print_table(trunc, "Truncation exposure")

    save_run_metadata("01_explore")
    log.info("Done. Next: python scripts/02_preprocessing_ablation.py")


if __name__ == "__main__":
    main()
