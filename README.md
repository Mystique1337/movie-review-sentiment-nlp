# Movie Review Sentiment Analysis

A reproducible NLP pipeline that classifies movie reviews as positive or negative, and then pools the reviews of a single film into one overall verdict with a confidence interval.

The project is built around a question that a single accuracy number cannot answer: **how much of a sentiment system's reported performance survives contact with data it was not trained on?** Every design decision here is settled by measurement rather than convention, and several of the measurements contradict the textbook advice.

[![tests](https://github.com/Mystique1337/movie-review-sentiment-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/Mystique1337/movie-review-sentiment-nlp/actions/workflows/ci.yml)

---

## Headline results

| System | aclImdb test accuracy | Training time |
|---|---|---|
| **DistilBERT (fine-tuned)** | **0.9086** | 33 min (Apple M4 GPU) |
| **NB-SVM** (deployed) | **0.9063** | 36 s (CPU) |
| Logistic Regression | 0.9062 | 27 s (CPU) |
| Linear SVM | 0.9055 | 24 s (CPU) |
| Multinomial Naive Bayes | 0.8677 | 26 s (CPU) |
| VADER lexicon (no training) | 0.6983 | 0 s |

Scored once on the official 25,000-document test split, which no model saw during training or tuning. NB-SVM was deployed as the highest test-split scorer; it is statistically indistinguishable from logistic regression and the linear SVM (exact McNemar p ≥ 0.64), and cross-validation alone would have picked the linear SVM.

Pooling reviews per film lifts a 90.6% document classifier to a **96.7%** film-level verdict across 1,611 films.

---

## Four findings worth the reader's time

**1. The standard preprocessing recipe makes this task harder.**
Removing stop words with the default NLTK list and applying Porter stemming scored **0.8946**. Keeping function words and only lemmatising scored **0.9063**. The 1.17-point gap is significant under an exact McNemar test (p < 0.0001, n = 25,000). The cause is simple: the NLTK English stop-word list contains `not`, `no`, `nor` and every `n't` contraction, so `not good` collapses into `good`. TF-IDF already suppresses uninformative function words continuously through the IDF term, which makes deleting them outright redundant as well as harmful.

**2. Fine-tuning a transformer bought 0.23 points for 56 times the training cost.**
DistilBERT reached 0.9086 against NB-SVM's 0.9063. On the same laptop, that gain cost 33 minutes of GPU fine-tuning instead of 36 seconds of CPU fitting. The transformer earns its keep somewhere else entirely, in cross-domain transfer and in the mid-data regime, not on the headline number.

**3. Transformers are not automatically better when labels are scarce.**
The common expectation is that pretraining wins hardest with little data. Measured on the same subsamples, the ordering reverses below roughly a thousand documents:

| Training documents | Linear SVM | NB-SVM | DistilBERT |
|---:|---:|---:|---:|
| 500 | **0.8152** | 0.8000 | 0.7232 |
| 2,500 | 0.8685 | 0.8639 | **0.8773** |
| 25,000 | 0.9055 | 0.9063 | **0.9086** |

At 500 documents a linear SVM beats a fine-tuned DistilBERT by 9.2 points. A short fine-tuning schedule on 400 training examples is simply not enough signal to adapt 66 million parameters.

**4. Which model generalises better depends on document length, and the effect is large in both directions.**
Both models were trained on IMDb and applied unchanged to two other corpora:

| Evaluated on | Median length | NB-SVM | DistilBERT |
|---|---:|---:|---:|
| aclImdb test (in-domain) | 172 words | 0.9063 | **0.9086** |
| Pang & Lee reviews | 697 words | **0.9000** | 0.8190 |
| RT critic sentences | 20 words | 0.7375 | **0.8155** |

DistilBERT is 7.8 points better on short sentences and 8.1 points worse on long reviews. The mechanism is its fixed 256-subword window: at least 98% of Pang & Lee reviews exceed it, so the transformer decides from under a third of each document while the bag-of-words model reads all of it. On RT sentences nothing is truncated at all, and the pretrained representation wins.

The practical reading is that neither architecture is simply better. Choosing between them means knowing how long the target documents are, and in-domain accuracy on IMDb would have told you nothing about either result.

---

## What the system does

```
raw review text
      |
      v
  normalisation        HTML stripping, lowercasing, lemmatisation
      |                (function words deliberately retained)
      v
  vectorisation        binary unigrams + bigrams, weighted by
      |                the Naive Bayes log-count ratio
      v
  classification       logistic regression (NB-SVM)  ->  P(positive) per review
      |
      v
  aggregation          majority vote over a film's reviews
      |                + bootstrap confidence interval
      v
  film verdict         "positive, 0.82 [0.74, 0.89], n = 40"
```

The aggregation layer is the part that answers the actual business question. A studio does not want 40 separate verdicts about 40 reviews; it wants one statement about the film, and an honest signal when the reviews are too divided to support one.

---

## Quickstart

```bash
git clone https://github.com/Mystique1337/movie-review-sentiment-nlp.git
cd movie-review-sentiment-nlp

pip install -r requirements.txt
python scripts/00_download_data.py      # ~95 MB, three corpora
python scripts/03_train_classical.py    # ~5 min on a laptop CPU
```

Classify a single review:

```bash
python predict.py --text "A tedious, self-indulgent mess that wastes a talented cast."
```

```
[negative]  p=0.000  A tedious, self-indulgent mess that wastes a talented cast.
```

Judge a whole film from a file of reviews, one per line:

```bash
python predict.py --file reviews.txt --film "tt0015648" --aggregate
```

```
tt0015648: POSITIVE  score=0.900 CI95=[0.767, 1.000]  n=30
positive_share=90.0%  polarisation=0.239
```

That verdict is the system running on the 30 test-split reviews of one film whose true positive share is 96.7%.

See which n-grams drove a decision:

```bash
python predict.py --text "Not the disaster I expected." --explain
```

```
[negative]  p=0.377  Not the disaster I expected.
            top positive tokens: not (+1.720), disaster (+0.753)
            top negative tokens: expect (-1.659), not the (-0.064), the (-0.012)
```

Worth dwelling on, because it is wrong. A reader understands the sentence as mild praise. The model reads `expect` as strong negative evidence and never recovers, which is exactly the compositional failure the error analysis quantifies in `results/tables/t15_accuracy_by_construction.csv`.

Other useful flags: `--backend distilbert`, `--compare-aggregations`, `--json`.

---

## Reproducing every number

Each script writes its tables to `results/tables/`, its figures to `results/figures/`, and a machine-readable record of the environment that produced them to `results/metrics/`. Those directories are committed, so the numbers quoted above can be checked without rerunning anything.

```bash
make all              # steps 0-8, roughly 90 min including transformer fine-tuning
make classical-only   # skips step 5, roughly 25 min, no GPU needed
```

| Step | Script | What it establishes |
|---|---|---|
| 0 | `00_download_data.py` | Fetches aclImdb, Pang & Lee v2.0 and RT sentence polarity |
| 1 | `01_explore_data.py` | Corpus statistics, length distributions, vocabulary overlap |
| 2 | `02_preprocessing_ablation.py` | Factorial ablation of every preprocessing switch |
| 3 | `03_train_classical.py` | Four supervised models plus a lexicon baseline, McNemar tests |
| 4 | `04_learning_curve.py` | The scenario ladder: 200 to 25,000 training documents |
| 5 | `05_train_transformer.py` | DistilBERT fine-tuning across three rungs of that ladder |
| 6 | `06_cross_domain.py` | Transfer in both directions across three corpora |
| 7 | `07_error_analysis.py` | Errors by star rating, length and construction; calibration; film aggregation |
| 8 | `08_demo_film_verdict.py` | The finished system on three application scenarios |

Every experiment is seeded (`src/config.SEED = 42`). Hyper-parameters are selected by cross-validation inside the training split only; the test split is scored once per model. The deployed model was then chosen among the tuned models by test accuracy, which the report discusses.

---

## Corpora

| Corpus | Documents | Unit | Role |
|---|---:|---|---|
| Stanford aclImdb ([Maas et al., 2011](https://ai.stanford.edu/~amaas/data/sentiment/)) | 50,000 | full review | main train/test corpus |
| Polarity dataset v2.0 (Pang & Lee, 2004) | 2,000 | full review | small-data and transfer corpus |
| Sentence polarity v1.0 (Pang & Lee, 2005) | 10,662 | single sentence | out-of-domain test |

None of these are redistributed here. `scripts/00_download_data.py` fetches them from their original hosts.

aclImdb ships the IMDb title behind every review in `urls_pos.txt` / `urls_neg.txt`. Recovering that mapping is what makes the film-level evaluation real rather than simulated: 3,581 distinct films in the test split, 786 of which carry reviews of both polarities. `tests/test_data.py` checks the mapping against the dataset's documented cap of 30 reviews per film, which an off-by-one error would violate.

---

## Repository layout

```
src/
  config.py              paths, seed, the scenario ladder
  data.py                corpus loading, film-id recovery
  preprocess.py          configurable normalisation pipeline
  models_classical.py    NB, SVM, LogReg, NB-SVM (Wang & Manning, 2012)
  models_transformer.py  DistilBERT fine-tuning loop
  aggregate.py           film-level verdicts and confidence intervals
  evaluate.py            metrics, McNemar, bootstrap, figures
  pipeline.py            the deployed end-to-end system
  reporting.py           table export and environment fingerprinting
scripts/                 00-08, the reproducible experiment ladder
tests/                   94 unit tests
results/                 committed tables, figures and metrics
predict.py               command-line interface
```

---

## Testing

```bash
make test
```

94 tests covering text normalisation, the model zoo, evaluation metrics, the aggregation layer and corpus integrity. Tests that need the downloaded corpora skip themselves when those are absent, so the suite runs on a clean checkout and in CI.

Two of them encode findings rather than behaviour. `test_naive_stopwords_destroy_negation` asserts that the naive stop-word list really does collapse `not good` into `good`, and `test_no_film_exceeds_the_documented_cap` would fail if the review-to-film mapping ever slipped.

---

## Known limitations

- **Truncation.** 43.6% of aclImdb reviews exceed DistilBERT's 256-subword window, and the model sees only 68% of the average review's tokens. A longer window or a chunk-and-pool strategy would likely close part of the remaining gap.
- **Film-level ground truth is the corpus sample, not reality.** aclImdb caps each film at 30 reviews and is globally balanced by construction, so a film's majority polarity here does not represent its true reception. The comparison between aggregation rules remains fair, since every rule faces the same target.
- **Two classes only.** Reviews rated 5 and 6 stars were excluded by the dataset authors. Genuinely neutral text is outside what this system can express, and the error analysis shows accuracy is already lowest on the 4-star and 7-star reviews that sit closest to that boundary.
- **English only.**

---

## References

Key works behind the design decisions. Full bibliographic detail, verified against Crossref and arXiv, is in `report/references.bib`.

- Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. *ACL 2007*, 440–447.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923.
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *ACL-HLT 2011*, 142–150.
- Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. *EMNLP 2002*, 79–86.
- Paramesha, K., Gururaj, H. L., Nayyar, A., & Ravishankar, K. C. (2023). Sentiment analysis on cross-domain textual data using classical and deep learning approaches. *Multimedia Tools and Applications, 82*(20), 30759–30782.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter (arXiv:1910.01108). arXiv. https://doi.org/10.48550/arXiv.1910.01108
- Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *ACL 2012*, 90–94.

---

## License

MIT, see [LICENSE](LICENSE). The corpora remain under the terms set by their respective authors.
