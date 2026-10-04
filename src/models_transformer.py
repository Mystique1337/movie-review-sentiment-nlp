"""Deep-learning arm: fine-tuning DistilBERT for binary sentiment.

DistilBERT (Sanh et al., 2019, pp. 1, 3) is a 66M-parameter distillation of
BERT-base that is 40% smaller while retaining roughly 97% of its
language-understanding ability.
That trade-off is what makes this arm feasible on the hardware available for the
project (an Apple M4 laptop with 16 GB of unified memory) rather than on a
rented GPU, which is itself a finding worth reporting.

The training loop is written directly in PyTorch instead of using the
``Trainer`` API. It is only about sixty lines, and it keeps the exact optimiser,
schedule and evaluation cadence visible in the report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import MODELS_DIR, SEED, set_seed

log = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"


def get_device() -> torch.device:
    """Prefer Apple Metal, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    max_length: int = 256          # covers ~85% of IMDb reviews without truncation
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5    # the standard BERT fine-tuning rate
    weight_decay: float = 0.01
    epochs: int = 2                # IMDb saturates fast; 3+ overfits
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = SEED

    def to_dict(self) -> dict:
        return asdict(self)


class ReviewDataset(Dataset):
    """Tokenised movie reviews, padded per batch rather than to ``max_length``."""

    def __init__(self, texts, labels, tokenizer, max_length: int) -> None:
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def _make_loader(dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    from transformers import DataCollatorWithPadding

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        num_workers=0,  # MPS + multiprocessing workers is a known source of hangs
    )


def fine_tune_distilbert(
    train_texts,
    train_labels,
    val_texts=None,
    val_labels=None,
    config: TrainingConfig | None = None,
    output_dir: Path | None = None,
    log_every: int = 50,
):
    """Fine-tune DistilBERT and return ``(model, tokenizer, history)``.

    Note that the *raw* review text is passed in, not the stop-word-stripped and
    lemmatised version used by the classical models. Subword tokenisation and
    the pretrained positional encoding expect ordinary English; stripping
    function words before a transformer destroys the syntax it relies on.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    cfg = config or TrainingConfig()
    set_seed(cfg.seed)
    device = get_device()
    log.info("Fine-tuning %s on %s (%d examples)", MODEL_NAME, device, len(train_texts))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    train_loader = _make_loader(
        ReviewDataset(train_texts, train_labels, tokenizer, cfg.max_length),
        tokenizer, cfg.batch_size, shuffle=True,
    )

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    history: dict[str, list] = {"loss": [], "val_accuracy": [], "epoch_seconds": []}
    start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            if step % log_every == 0:
                log.info(
                    "epoch %d  step %d/%d  loss %.4f",
                    epoch + 1, step, len(train_loader), running / step,
                )

        history["loss"].append(running / max(len(train_loader), 1))
        history["epoch_seconds"].append(time.time() - epoch_start)

        if val_texts is not None:
            probs = predict_proba(model, tokenizer, val_texts, cfg, device)
            acc = float(((probs >= 0.5).astype(int) == np.asarray(val_labels)).mean())
            history["val_accuracy"].append(acc)
            log.info("epoch %d  validation accuracy %.4f", epoch + 1, acc)

    history["total_seconds"] = time.time() - start
    history["device"] = str(device)
    history["config"] = cfg.to_dict()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        log.info("Saved fine-tuned model to %s", output_dir)

    return model, tokenizer, history


@torch.no_grad()
def predict_proba(model, tokenizer, texts, config: TrainingConfig | None = None, device=None) -> np.ndarray:
    """Positive-class probabilities for a list of raw review strings."""
    cfg = config or TrainingConfig()
    device = device or get_device()
    model.eval().to(device)

    loader = _make_loader(
        ReviewDataset(texts, None, tokenizer, cfg.max_length),
        tokenizer, cfg.eval_batch_size, shuffle=False,
    )
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def load_fine_tuned(path: Path | str = MODELS_DIR / "distilbert-imdb"):
    """Reload a saved fine-tuned checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No fine-tuned model at {path}. Run `python scripts/05_train_transformer.py`."
        )
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
