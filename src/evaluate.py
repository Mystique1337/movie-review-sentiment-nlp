"""Evaluation metrics, statistical tests and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, LABEL_NAMES, METRICS_DIR, SEED

# A colour-blind-safe qualitative palette (Okabe & Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true, y_pred, y_score=None, *, label: str = "", n_boot: int = 1_000
) -> dict:
    """Accuracy, macro P/R/F1, ROC-AUC and a bootstrap CI on accuracy.

    A 95% percentile bootstrap interval is reported because a single accuracy
    figure on a 25,000-document test set still carries roughly +/-0.4 points of
    sampling noise, and several models in this study differ by less than that.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "label": label,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_negative": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    boot = (y_true[idx] == y_pred[idx]).mean(axis=1)
    out["accuracy_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return out


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """Exact McNemar test for two classifiers on the same test set.

    Comparing two accuracies with an unpaired test throws away the fact that
    both models saw identical documents. McNemar conditions on the discordant
    pairs, which is the correct paired test for classification decisions
    (Dietterich, 1998).
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    n01 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    n10 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(n01, n, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p_value": float(p), "significant_at_05": bool(p < 0.05)}


def save_metrics(metrics: dict, name: str) -> Path:
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def text_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=list(LABEL_NAMES), digits=4)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_pred, title: str, filename: str) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], labels=list(LABEL_NAMES))
    ax.set_yticks([0, 1], labels=list(LABEL_NAMES))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}\n{cm_norm[i, j]:.1%}",
                ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_learning_curves(curves: dict[str, dict[str, list]], filename: str) -> Path:
    """One line per model: test accuracy against number of training documents.

    The y-axis is cropped to the region the curves actually occupy. Anchoring it
    at the 0.5 chance line would leave two thirds of the panel empty and hide
    the differences between models, which are what the figure exists to show.
    Line styles alternate because the two strongest models overlap almost
    exactly and a colour-only encoding would render one of them invisible.
    """
    styles = ["-", "--", "-", "--", "-", "--"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    lo, hi = 1.0, 0.0
    for i, (model, data) in enumerate(curves.items()):
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(data["sizes"], data["accuracy"], marker=markers[i % len(markers)],
                ms=4.2, color=colour, label=model, lw=1.6,
                ls=styles[i % len(styles)], alpha=0.9)
        if "ci_low" in data:
            ax.fill_between(data["sizes"], data["ci_low"], data["ci_high"],
                            color=colour, alpha=0.13, lw=0)
        lo = min(lo, min(data.get("ci_low", data["accuracy"])))
        hi = max(hi, max(data.get("ci_high", data["accuracy"])))

    ax.set_xscale("log")
    ax.set_xlabel("Labelled training documents (log scale)")
    ax.set_ylabel("Accuracy on the aclImdb test split")
    ax.set_title("More data helps every model, but not equally", fontsize=10)
    ax.set_ylim(max(0.0, lo - 0.02), min(1.0, hi + 0.015))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc_curves(results: dict[str, tuple], filename: str) -> Path:
    """``results`` maps a display name to ``(y_true, y_score)``."""
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for i, (name, (y_true, y_score)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.5,
                label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on the IMDb test split", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bar_comparison(
    labels: list[str], values: list[float], errors=None, *,
    title: str, ylabel: str, filename: str, ylim=None,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = np.arange(len(labels))
    ax.bar(xs, values, yerr=errors, capsize=3, width=0.6,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))])
    ax.set_xticks(xs, labels=labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    for x, v in zip(xs, values):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_top_features(positive, negative, title: str, filename: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.0), sharex=False)
    for ax, feats, colour, side in (
        (axes[0], negative, PALETTE[1], "Negative"),
        (axes[1], positive, PALETTE[0], "Positive"),
    ):
        names = [f[0] for f in feats][::-1]
        vals = [f[1] for f in feats][::-1]
        ax.barh(np.arange(len(names)), vals, color=colour, height=0.7)
        ax.set_yticks(np.arange(len(names)), labels=names, fontsize=7.5)
        ax.set_title(f"{side} evidence", fontsize=9)
        ax.set_xlabel("model weight")
    fig.suptitle(title, fontsize=10)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path
