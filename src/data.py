"""Corpus acquisition and loading.

Three publicly available movie-review corpora are used, deliberately chosen to
differ in size and in document granularity:

===================  =========  ===============  ==================================
Corpus               Documents  Granularity      Role in this project
===================  =========  ===============  ==================================
Pang & Lee polarity      2,000  full review      small-data scenario
Stanford aclImdb        50,000  full review      main train/test corpus
RT sentence polarity    10,662  single sentence  out-of-domain generalisation test
===================  =========  ===============  ==================================

Every loader returns a ``pandas.DataFrame`` with the columns ``text`` (str) and
``label`` (int, 0 = negative, 1 = positive) so that downstream code never needs
to know which corpus it is looking at.
"""

from __future__ import annotations

import logging
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    ACL_IMDB_DIR,
    INTERIM_DIR,
    RAW_DIR,
    RT_POLARITY_DIR,
    SEED,
)

log = logging.getLogger(__name__)

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
RT_URL = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
# Parsing 50,000 individual files takes a few seconds, so each corpus is cached
# as Parquet after the first read. Parquet needs pyarrow or fastparquet, which
# are not strictly required to use this package, so a missing engine degrades to
# "no caching" rather than raising.
def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except ImportError:
        log.debug("No Parquet engine available; re-reading %s from source", path.stem)
        return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        log.debug("No Parquet engine available; skipping cache for %s", path.stem)


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _download_and_extract(url: str, archive_name: str, marker: Path) -> None:
    """Fetch ``url`` into ``data/raw`` and untar it, unless ``marker`` exists."""
    if marker.exists():
        log.info("%s already present, skipping download", marker.name)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / archive_name
    if not archive.exists():
        log.info("Downloading %s -> %s", url, archive)
        urllib.request.urlretrieve(url, archive)

    log.info("Extracting %s", archive)
    with tarfile.open(archive, "r:gz") as tar:
        # Guard against path traversal in archive members (CVE-2007-4559 class).
        for member in tar.getmembers():
            target = (RAW_DIR / member.name).resolve()
            if not str(target).startswith(str(RAW_DIR.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(RAW_DIR)


def download_all() -> None:
    """Idempotently make every raw corpus available on disk."""
    _download_and_extract(IMDB_URL, "aclImdb_v1.tar.gz", ACL_IMDB_DIR)
    _download_and_extract(RT_URL, "rt-polaritydata.tar.gz", RT_POLARITY_DIR)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
FILM_ID_RE = re.compile(r"/title/(tt\d+)/")


def _read_film_ids(split: str, label_name: str) -> list[str | None]:
    """Map each review id to the IMDb title it discusses.

    The corpus ships ``urls_pos.txt`` / ``urls_neg.txt``; per the dataset README,
    "a review with unique id 200 will have its URL on line 200 of this file",
    and review ids are the zero-based prefix of the review file names. Recovering
    this mapping is what makes a *film-level* verdict possible rather than only a
    document-level one, since the reviews of a single film can then be pooled.
    """
    url_file = ACL_IMDB_DIR / split / f"urls_{label_name}.txt"
    if not url_file.exists():
        return []
    ids = []
    for line in url_file.read_text(encoding="utf-8").splitlines():
        m = FILM_ID_RE.search(line)
        ids.append(m.group(1) if m else None)
    return ids


def _read_imdb_split(split: str) -> pd.DataFrame:
    """Read one official aclImdb split ("train" or "test") into a DataFrame."""
    rows: list[dict[str, object]] = []
    for label_name, label in (("neg", 0), ("pos", 1)):
        folder = ACL_IMDB_DIR / split / label_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"{folder} not found. Run `python scripts/00_download_data.py` first."
            )
        film_ids = _read_film_ids(split, label_name)
        for path in sorted(folder.glob("*.txt")):
            # aclImdb encodes id and star rating in the file name: <id>_<rating>.txt
            review_id_str, rating_str = path.stem.split("_")
            review_id = int(review_id_str)
            rows.append(
                {
                    "text": path.read_text(encoding="utf-8"),
                    "label": label,
                    "rating": int(rating_str),
                    "doc_id": f"{split}/{label_name}/{path.stem}",
                    "film_id": film_ids[review_id] if review_id < len(film_ids) else None,
                }
            )
    return pd.DataFrame(rows)


def load_imdb(split: str = "train", cache: bool = True) -> pd.DataFrame:
    """Load the Stanford Large Movie Review Dataset (Maas et al., 2011).

    50,000 polarised reviews, split 25k/25k by the dataset authors into train
    and test. Reviews with 5 or 6 stars were excluded by the authors, and no
    more than 30 reviews per film are included, which limits the risk that a
    model simply memorises film-specific vocabulary.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    cache_path = INTERIM_DIR / f"imdb_{split}.parquet"
    if cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    df = _read_imdb_split(split)
    if cache:
        _write_cache(df, cache_path)
    return df


def load_pang_lee(cache: bool = True) -> pd.DataFrame:
    """Load the Pang & Lee (2004) polarity dataset v2.0 via NLTK.

    2,000 full-length reviews (1,000 per class) drawn from the IMDb archive of
    rec.arts.movies.reviews. This is the "small dataset" of the scenario ladder
    and predates the Stanford corpus by seven years.
    """
    cache_path = INTERIM_DIR / "pang_lee.parquet"
    if cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    import nltk
    from nltk.corpus import movie_reviews

    try:
        movie_reviews.fileids()
    except LookupError:
        nltk.download("movie_reviews", quiet=True)

    rows = []
    for fid in movie_reviews.fileids():
        category = fid.split("/")[0]  # "pos" or "neg"
        rows.append(
            {
                "text": movie_reviews.raw(fid),
                "label": 1 if category == "pos" else 0,
                "doc_id": fid,
            }
        )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if cache:
        _write_cache(df, cache_path)
    return df


def load_rt_polarity(cache: bool = True) -> pd.DataFrame:
    """Load the Rotten Tomatoes sentence polarity corpus (Pang & Lee, 2005).

    10,662 single sentences taken from critic snippets. Because the unit is a
    sentence rather than a full review, this corpus is used only as an
    out-of-domain test set: it measures whether a model trained on long IMDb
    reviews transfers to short, journalistic text.
    """
    cache_path = INTERIM_DIR / "rt_polarity.parquet"
    if cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    rows = []
    for fname, label in (("rt-polarity.neg", 0), ("rt-polarity.pos", 1)):
        path = RT_POLARITY_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/00_download_data.py` first."
            )
        # The files are distributed in Latin-1, not UTF-8.
        with path.open(encoding="latin-1") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    rows.append(
                        {"text": line, "label": label, "doc_id": f"{fname}:{i}"}
                    )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if cache:
        _write_cache(df, cache_path)
    return df


CORPUS_LOADERS = {
    "pang_lee": load_pang_lee,
    "imdb_train": lambda: load_imdb("train"),
    "imdb_test": lambda: load_imdb("test"),
    "rt_polarity": load_rt_polarity,
}


def load_corpus(name: str) -> pd.DataFrame:
    """Load a corpus by its short name."""
    if name not in CORPUS_LOADERS:
        raise KeyError(f"Unknown corpus '{name}'. Choose from {sorted(CORPUS_LOADERS)}")
    return CORPUS_LOADERS[name]()
