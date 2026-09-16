**Cross-corpus transfer. The within-corpus reference is the same model cross-validated on the target corpus (train/test for aclImdb). A positive gap is the cost of domain shift; a negative gap means the transferred model beat that reference.**

| model      | trained_on             | evaluated_on    |   n_test |   accuracy |   f1_macro |   roc_auc |   within_corpus_reference |   gap_vs_reference_pp |
|:-----------|:-----------------------|:----------------|---------:|-----------:|-----------:|----------:|--------------------------:|----------------------:|
| NB-SVM     | aclImdb train (25,000) | aclImdb test    |    25000 |     0.9063 |     0.9063 |    0.9667 |                    0.9063 |                0.0000 |
| NB-SVM     | aclImdb train (25,000) | Pang & Lee v2.0 |     2000 |     0.9000 |     0.9000 |    0.9635 |                    0.8645 |               -3.5500 |
| NB-SVM     | aclImdb train (25,000) | RT sentences    |    10662 |     0.7375 |     0.7356 |    0.8268 |                    0.7775 |                4.0050 |
| NB-SVM     | Pang & Lee (2,000)     | aclImdb test    |    25000 |     0.8558 |     0.8558 |    0.9309 |                    0.9063 |                5.0480 |
| NB-SVM     | Pang & Lee (2,000)     | RT sentences    |    10662 |     0.7033 |     0.6995 |    0.7766 |                    0.7775 |                7.4189 |
| DistilBERT | aclImdb train (25,000) | aclImdb test    |    25000 |     0.9086 |     0.9086 |    0.9701 |                    0.9063 |               -0.2280 |
| DistilBERT | aclImdb train (25,000) | Pang & Lee v2.0 |     2000 |     0.8190 |     0.8189 |    0.9117 |                    0.8645 |                4.5500 |
| DistilBERT | aclImdb train (25,000) | RT sentences    |    10662 |     0.8155 |     0.8146 |    0.9097 |                    0.7775 |               -3.7985 |
