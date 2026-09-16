**Film-level verdict accuracy by aggregation rule. Ground truth is the majority gold polarity of each film's reviews.**

| subset               |   n_films | rule                     |   accuracy |   decisive_% |
|:---------------------|----------:|:-------------------------|-----------:|-------------:|
| all films            |      1611 | single review (baseline) |     0.8498 |     nan      |
| all films            |      1611 | majority                 |     0.9671 |      65.7976 |
| all films            |      1611 | mean_probability         |     0.9609 |      75.0466 |
| all films            |      1611 | trimmed_mean             |     0.9628 |      69.5841 |
| mixed-polarity films |       658 | single review (baseline) |     0.7401 |     nan      |
| mixed-polarity films |       658 | majority                 |     0.9255 |      46.0486 |
| mixed-polarity films |       658 | mean_probability         |     0.9058 |      52.4316 |
| mixed-polarity films |       658 | trimmed_mean             |     0.9088 |      46.5046 |
