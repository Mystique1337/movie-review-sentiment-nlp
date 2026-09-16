#!/usr/bin/env python3
"""Command-line interface to the trained sentiment system.

Examples
--------
Classify one review::

    python predict.py --text "A tedious, self-indulgent mess."

Classify every line of a file and print per-review verdicts::

    python predict.py --file reviews.txt

Aggregate a set of reviews into one verdict about a film::

    python predict.py --file reviews.txt --film "Dune: Part Two" --aggregate

Use the fine-tuned transformer instead of the classical model::

    python predict.py --file reviews.txt --backend distilbert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.aggregate import compare_aggregations
from src.pipeline import SentimentSystem


def read_reviews(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    if args.file:
        path = Path(args.file)
        if not path.exists():
            sys.exit(f"error: no such file: {path}")
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else payload["reviews"]
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not sys.stdin.isatty():
        return [ln.strip() for ln in sys.stdin if ln.strip()]
    sys.exit("error: provide --text, --file, or pipe reviews on stdin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary sentiment analysis for movie reviews.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--text", help="a single review to classify")
    src_group.add_argument("--file", help="text file (one review per line) or .json list")
    parser.add_argument("--backend", choices=["classical", "distilbert"], default="classical")
    parser.add_argument("--model-path", default=None, help="override the model location")
    parser.add_argument("--film", default="unnamed film", help="film title for the aggregate verdict")
    parser.add_argument("--aggregate", action="store_true", help="emit one verdict for the whole set")
    parser.add_argument("--compare-aggregations", action="store_true",
                        help="show all three aggregation rules side by side")
    parser.add_argument("--explain", action="store_true",
                        help="list the tokens that drove each decision (linear models only)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    reviews = read_reviews(args)
    system = SentimentSystem(backend=args.backend, model_path=args.model_path)
    scores = system.score(reviews)

    if args.aggregate or args.compare_aggregations:
        if args.compare_aggregations:
            verdicts = compare_aggregations(scores, title=args.film)
            if args.json:
                print(json.dumps({k: v.to_dict() for k, v in verdicts.items()}, indent=2))
            else:
                print(f"\nFilm-level sentiment for {args.film!r} "
                      f"({len(reviews)} reviews, backend={args.backend})\n")
                for method, verdict in verdicts.items():
                    print(f"  [{method:>16}] {verdict.summary()}")
        else:
            verdict = system.judge_film(reviews, title=args.film)
            if args.json:
                print(json.dumps(verdict.to_dict(), indent=2))
            else:
                print("\n" + verdict.summary() + "\n")
        return

    if args.json:
        payload = [
            {"review": r[:200], "score": float(s), "sentiment": "positive" if s >= 0.5 else "negative"}
            for r, s in zip(reviews, scores)
        ]
        print(json.dumps(payload, indent=2))
        return

    for review, score in zip(reviews, scores):
        label = "positive" if score >= 0.5 else "negative"
        snippet = review[:78].replace("\n", " ")
        print(f"[{label:>8}]  p={score:.3f}  {snippet}{'...' if len(review) > 78 else ''}")
        if args.explain:
            detail = system.explain(review)
            for key in ("top_positive_tokens", "top_negative_tokens"):
                if key in detail:
                    tokens = ", ".join(f"{t} ({w:+.3f})" for t, w in detail[key])
                    print(f"            {key.replace('_', ' ')}: {tokens}")


if __name__ == "__main__":
    main()
