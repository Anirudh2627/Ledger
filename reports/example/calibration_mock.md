# Judge Calibration Report

- Generated: 2026-09-26T22:13:08+00:00
- Judge model: `mock-judge-v1`
- Labels: 20 | judged: 20 | judge failures: 0

## Agreement with human labels (1-5 scale)

| Dimension | n | MAE | RMSE | Bias (J-H) | Pearson r | Spearman rho | Within +-1 |
|---|---|---|---|---|---|---|---|
| correctness | 20 | 0.400 | 0.837 | 0.300 | 0.901 | 0.893 | 0.950 |
| groundedness | 20 | 0.950 | 1.323 | -0.550 | 0.654 | 0.609 | 0.700 |
| instruction_following | 20 | 0.800 | 1.140 | 0.300 | 0.726 | 0.713 | 0.850 |
| relevance | 20 | 0.750 | 1.245 | 0.250 | -0.014 | 0.126 | 0.800 |
| **overall** | 20 | 0.575 | 0.692 | 0.008 | 0.883 | 0.696 | 0.850 |

## Pass-decision agreement

- Agreement rate: 0.800
- Cohen's kappa: 0.600 (substantial)

|  | Human: PASS | Human: FAIL |
|---|---|---|
| Judge: PASS | 10 | 4 |
| Judge: FAIL | 0 | 6 |

## Notes

- Bias > 0 means the judge scores *higher* than humans (lenient); < 0 means harsher.
- Correlations are undefined (shown as `-`) when a column is constant.
- Calibration results describe the judge on THIS label sample; re-run after any change to judge model, prompt or rubric.
