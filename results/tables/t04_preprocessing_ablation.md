**Effect of each preprocessing choice on validation accuracy (logistic regression on aclImdb, 20,000 train / 5,000 validation)**

| config                   | description                                           | features           |   val_accuracy |   n_features |   tokens_retained_m |   preprocess_s |   fit_s |
|:-------------------------|:------------------------------------------------------|:-------------------|---------------:|-------------:|--------------------:|---------------:|--------:|
| lemma-only               | lemmatisation, no stop-word removal                   | unigrams + bigrams |         0.8982 |       200000 |              4.5257 |         6.2337 |  5.0921 |
| sw-keepneg               | stop-word list with negation cues retained            | unigrams + bigrams |         0.8974 |       200000 |              2.6331 |         5.8021 |  4.4818 |
| raw                      | no stop-word removal, no normalisation                | unigrams + bigrams |         0.8970 |       200000 |              4.5257 |         5.7428 |  5.4514 |
| sw-keepneg+lemma+negmark | negation-aware stop words + lemma + NEG scope marking | unigrams + bigrams |         0.8968 |       200000 |              2.6331 |         6.4686 |  4.4679 |
| sw-keepneg+stem          | negation-aware stop words + Porter stemming           | unigrams + bigrams |         0.8964 |       200000 |              2.6331 |         6.2787 |  4.2439 |
| sw-keepneg+lemma         | negation-aware stop words + WordNet lemmatisation     | unigrams + bigrams |         0.8942 |       200000 |              2.6331 |         6.9398 |  4.2212 |
| sw-keepneg+lemma+negmark | negation-aware stop words + lemma + NEG scope marking | unigrams           |         0.8938 |        35708 |              2.6331 |         6.4686 |  1.2040 |
| sw-naive                 | naive NLTK stop-word list (removes negations)         | unigrams + bigrams |         0.8910 |       200000 |              2.4815 |         5.7828 |  4.2179 |
| sw-naive+stem            | textbook recipe: naive stop words + stemming          | unigrams + bigrams |         0.8910 |       200000 |              2.4815 |         6.2900 |  4.4012 |
| lemma-only               | lemmatisation, no stop-word removal                   | unigrams           |         0.8878 |        30995 |              4.5257 |         6.2337 |  1.5574 |
| raw                      | no stop-word removal, no normalisation                | unigrams           |         0.8876 |        41045 |              4.5257 |         5.7428 |  1.6252 |
| sw-keepneg               | stop-word list with negation cues retained            | unigrams           |         0.8864 |        40931 |              2.6331 |         5.8021 |  1.0734 |
| sw-keepneg+lemma         | negation-aware stop words + WordNet lemmatisation     | unigrams           |         0.8856 |        30916 |              2.6331 |         6.9398 |  1.0158 |
| sw-naive                 | naive NLTK stop-word list (removes negations)         | unigrams           |         0.8850 |        40907 |              2.4815 |         5.7828 |  1.1306 |
| sw-keepneg+stem          | negation-aware stop words + Porter stemming           | unigrams           |         0.8848 |        27492 |              2.6331 |         6.2787 |  1.0579 |
| sw-naive+stem            | textbook recipe: naive stop words + stemming          | unigrams           |         0.8822 |        27474 |              2.4815 |         6.2900 |  1.0670 |
