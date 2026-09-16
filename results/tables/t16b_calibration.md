**Calibration of the deployed model before and after isotonic recalibration (fitted on 20% of the test split, scored on the other 80%)**

| scores                 |    ece |   accuracy |   share_beyond_0.99_confidence_% |
|:-----------------------|-------:|-----------:|---------------------------------:|
| raw model output       | 0.0227 |     0.9065 |                          23.7400 |
| isotonic recalibration | 0.0104 |     0.9061 |                          34.5400 |
