"""Central configuration: paths, random seeds and shared constants.

Keeping every path and magic number in one module means an experiment can be
reproduced by changing a single file rather than hunting through scripts.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"

ACL_IMDB_DIR = RAW_DIR / "aclImdb"
RT_POLARITY_DIR = RAW_DIR / "rt-polaritydata"

for _d in (INTERIM_DIR, TABLES_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# The scenario ladder required by the task: the same pipeline is evaluated on a
# progressively larger number of labelled training documents.
TRAIN_SIZE_LADDER = (200, 500, 1_000, 2_500, 5_000, 10_000, 25_000)

# Fraction of the training split held out for model selection. The official IMDb
# test split is touched exactly once per model, after all tuning is finished.
VALIDATION_FRACTION = 0.2

LABEL_NAMES = ("negative", "positive")


def set_seed(seed: int = SEED) -> None:
    """Seed every random number generator the project can reach."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is only needed for the transformer arm
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is an optional dependency
        pass
