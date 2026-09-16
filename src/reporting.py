"""Helpers that turn experiment output into artefacts the report can cite.

Every table is written twice: as CSV (machine-readable, diffable in git) and as
a Markdown fragment that is pasted verbatim into the report. Writing both from
one function is what stops the numbers in the report drifting away from the
numbers the code produced.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import METRICS_DIR, TABLES_DIR


def save_table(df: pd.DataFrame, name: str, *, caption: str = "",
               float_fmt: str = "%.4f", index: bool = False) -> Path:
    """Write ``df`` to ``results/tables/<name>.{csv,md}`` and return the CSV path."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=index, float_format=float_fmt)

    md_lines = []
    if caption:
        md_lines += [f"**{caption}**", ""]
    md_lines.append(df.to_markdown(index=index, floatfmt=".4f"))
    (TABLES_DIR / f"{name}.md").write_text("\n".join(md_lines) + "\n")
    return csv_path


def print_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{title}\n{'-' * len(title)}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unversioned"


def environment_fingerprint() -> dict:
    """Record exactly what produced a result, for reproducibility."""
    import sklearn

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or platform.machine(),
        "scikit_learn": sklearn.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = bool(torch.backends.mps.is_available())
    except ImportError:
        info["torch"] = None
    return info


def save_run_metadata(name: str, extra: dict | None = None) -> Path:
    payload = environment_fingerprint()
    if extra:
        payload.update(extra)
    path = METRICS_DIR / f"{name}_env.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
