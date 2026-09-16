**Effect of the preprocessing recipe on the test split (NB-SVM, 25,000 test documents)**

| preprocessing                             | description                                             |   test_accuracy |   n_features |   mcnemar_p |
|:------------------------------------------|:--------------------------------------------------------|----------------:|-------------:|------------:|
| selected (lemmatise, keep function words) | stopwords=off, negations=kept, norm=lemma, negmark=off  |          0.9063 |       200000 |    nan      |
| prescribed (stop-word removal + stemming) | stopwords=on, negations=dropped, norm=stem, negmark=off |          0.8946 |       200000 |      0.0000 |
