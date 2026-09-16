#!/usr/bin/env python3
"""Step 8 - the finished system applied to realistic scenarios.

The project guidelines ask for typical application scenarios to be simulated and
documented. Three are covered here:

A. **A studio monitoring reception.** Pool every review the corpus holds for one
   film and issue a single verdict with a confidence interval.
B. **A borderline film.** The same, for a title whose reviews genuinely split,
   to show what the system does when there is no clear answer.
C. **Hard single reviews.** Hand-written cases that isolate negation, contrast,
   irony and mixed sentiment, to expose the failure modes qualitatively rather
   than only in aggregate.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.aggregate import compare_aggregations
from src.config import RESULTS_DIR, set_seed
from src.data import load_imdb
from src.pipeline import SentimentSystem
from src.reporting import print_table, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("demo")

# Constructed probes. Each isolates one phenomenon discussed in the report.
HARD_CASES = [
    ("simple positive", "An absolute triumph. Beautifully acted and genuinely moving."),
    ("simple negative", "Dull, overlong and painfully predictable from the first scene."),
    ("negation", "This is not a good film. I would not recommend it to anyone."),
    ("negated negative", "I expected a disaster, but it is not bad at all."),
    ("contrast", "The cinematography is stunning and the score is lovely, but the "
                 "script is a mess and the ending ruins everything that came before."),
    ("contrast reversed", "The script is a mess and the pacing drags, but the lead "
                          "performance is so extraordinary that the whole thing works."),
    ("irony", "Oh wonderful, another two hours of explosions pretending to be a plot. "
              "Truly a masterpiece of modern cinema."),
    ("comparison", "Nowhere near as good as the original, though it is better than the "
                   "third one."),
    ("mixed, mild", "It is fine. Some parts work, some do not. You will forget it by "
                    "the weekend."),
    ("plot description", "A man loses his wife in a car accident and travels across the "
                         "country to scatter her ashes."),
]


def scenario_film(system: SentimentSystem, test: pd.DataFrame, film_id: str,
                  description: str) -> dict:
    sub = test[test["film_id"] == film_id]
    scores = system.score(sub["text"].tolist())
    verdicts = compare_aggregations(scores, title=film_id)

    gold_share = float(sub["label"].mean())
    print(f"\n{'=' * 78}\n{description}\n  IMDb title: {film_id}   reviews: {len(sub)}"
          f"   gold positive share: {gold_share:.1%}\n{'=' * 78}")
    for method, verdict in verdicts.items():
        print(f"  [{method:>16}] {verdict.summary()}")

    return {
        "scenario": description,
        "film_id": film_id,
        "n_reviews": len(sub),
        "gold_positive_share": gold_share,
        "gold_sentiment": "positive" if gold_share >= 0.5 else "negative",
        **{
            f"{m}_sentiment": v.sentiment for m, v in verdicts.items()
        },
        **{
            f"{m}_score": round(v.score, 4) for m, v in verdicts.items()
        },
        "trimmed_mean_ci_low": round(verdicts["trimmed_mean"].ci95[0], 4),
        "trimmed_mean_ci_high": round(verdicts["trimmed_mean"].ci95[1], 4),
        "trimmed_mean_decisive": verdicts["trimmed_mean"].decisive,
    }


def main() -> None:
    set_seed()
    test = load_imdb("test")
    system = SentimentSystem(backend="classical")
    log.info("Loaded %s", system)

    # Pick the scenario films from the data rather than hard-coding titles:
    # one clearly-received film and one that genuinely divides its reviewers.
    per_film = test.groupby("film_id")["label"].agg(["size", "mean"])
    eligible = per_film[per_film["size"] >= 12]
    clear_id = eligible[eligible["mean"] >= 0.9].sort_values("size", ascending=False).index[0]
    divided = eligible.assign(balance=(eligible["mean"] - 0.5).abs())
    divided_id = divided.sort_values(["balance", "size"], ascending=[True, False]).index[0]

    rows = [
        scenario_film(system, test, clear_id,
                      "Scenario A - a film with broadly consistent reception"),
        scenario_film(system, test, divided_id,
                      "Scenario B - a film whose audience is genuinely split"),
    ]
    scenarios = pd.DataFrame(rows)
    save_table(scenarios, "t19_application_scenarios",
               caption="The deployed system applied to two films from the test split")

    # -- Scenario C: hard single reviews ----------------------------------- #
    print(f"\n{'=' * 78}\nScenario C - diagnostic single reviews\n{'=' * 78}")
    case_rows = []
    for phenomenon, text in HARD_CASES:
        detail = system.explain(text)
        case_rows.append(
            {
                "phenomenon": phenomenon,
                "review": text,
                "predicted": detail["sentiment"],
                "score": round(detail["score"], 4),
                "confidence": round(detail["confidence"], 4),
                "decisive_tokens": ", ".join(
                    t for t, _ in (detail.get("top_positive_tokens") or [])[:4]
                ),
            }
        )
        print(f"  [{detail['sentiment']:>8}] p={detail['score']:.3f}  ({phenomenon})")
        print(f"            {text[:96]}{'...' if len(text) > 96 else ''}")

    cases = pd.DataFrame(case_rows)
    save_table(cases.drop(columns=["review"]), "t20_diagnostic_cases",
               caption="System output on hand-constructed reviews isolating specific "
                       "linguistic phenomena")
    (RESULTS_DIR / "tables" / "t20_diagnostic_cases_full.csv").write_text(
        cases.to_csv(index=False)
    )
    print_table(cases.drop(columns=["review", "decisive_tokens"]), "Diagnostic cases")

    # A machine-readable record of the whole demonstration.
    (RESULTS_DIR / "metrics" / "08_demo.json").write_text(
        json.dumps({"scenarios": rows, "diagnostic_cases": case_rows}, indent=2, default=str)
    )
    log.info("Done. All experiment artefacts are in results/.")


if __name__ == "__main__":
    main()
