**Exact McNemar tests against NB-SVM on the aclImdb test split**

| comparison                              |   best_only_correct |   other_only_correct |   p_value | significant_p<.05   |
|:----------------------------------------|--------------------:|---------------------:|----------:|:--------------------|
| NB-SVM vs Multinomial Naive Bayes       |                1869 |                  905 |    0.0000 | True                |
| NB-SVM vs Linear SVM                    |                 737 |                  718 |    0.6370 | False               |
| NB-SVM vs Logistic Regression           |                 683 |                  680 |    0.9568 | False               |
| NB-SVM vs VADER (lexicon, unsupervised) |                6218 |                 1019 |    0.0000 | True                |
