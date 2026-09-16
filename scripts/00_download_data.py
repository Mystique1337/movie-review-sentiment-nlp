#!/usr/bin/env python3
"""Step 0 - acquire every corpus and NLTK resource the project needs.

Run this once. Everything downstream assumes ``data/raw`` is populated.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nltk

from src.data import download_all, load_imdb, load_pang_lee, load_rt_polarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("download")

NLTK_PACKAGES = [
    "stopwords",
    "wordnet",
    "omw-1.4",
    "punkt",
    "punkt_tab",
    "movie_reviews",
    "vader_lexicon",
]


def main() -> None:
    log.info("Fetching NLTK resources")
    for pkg in NLTK_PACKAGES:
        ok = nltk.download(pkg, quiet=True)
        log.info("  %-28s %s", pkg, "ok" if ok else "FAILED")

    log.info("Fetching movie-review corpora")
    download_all()

    log.info("Materialising cached DataFrames")
    for name, loader in (
        ("aclImdb train", lambda: load_imdb("train")),
        ("aclImdb test", lambda: load_imdb("test")),
        ("Pang & Lee polarity v2.0", load_pang_lee),
        ("RT sentence polarity", load_rt_polarity),
    ):
        df = loader()
        log.info("  %-28s %6d documents  (%.0f%% positive)",
                 name, len(df), 100 * df["label"].mean())

    log.info("Done. Next: python scripts/01_explore_data.py")


if __name__ == "__main__":
    main()
