**DistilBERT fine-tuning across the scenario ladder, scored on the full 25,000-document aclImdb test split**

|   train_size | model                   |   epochs |   test_accuracy |   ci95_low |   ci95_high |   f1_macro |   roc_auc |   train_s |   predict_s |
|-------------:|:------------------------|---------:|----------------:|-----------:|------------:|-----------:|----------:|----------:|------------:|
|          500 | DistilBERT (fine-tuned) |        3 |          0.7232 |     0.7174 |      0.7280 |     0.7228 |    0.7988 |  118.5530 |    266.8851 |
|         2500 | DistilBERT (fine-tuned) |        3 |          0.8773 |     0.8732 |      0.8815 |     0.8773 |    0.9508 |  355.0989 |    299.4184 |
|        25000 | DistilBERT (fine-tuned) |        2 |          0.9086 |     0.9048 |      0.9122 |     0.9086 |    0.9701 | 1995.8066 |    277.3934 |
