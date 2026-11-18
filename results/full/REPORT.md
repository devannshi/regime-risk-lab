# Executed research results

Generated from actual runs: 100 independent seeds per synthetic scenario; 252 chronological real-data fits. Runtime 43.2 seconds.

## Real-data forecasts

| asset | model | n | mean_log_score | breach_rate | adaptive_breach_rate | q05_loss |
|---|---|---|---|---|---|---|
| EURUSD | gaussian_iid | 5267 | 3.75933 | 0.04690 | 0.04955 | 0.00064 |
| EURUSD | student_iid | 5267 | 3.80821 | 0.05088 | 0.05012 | 0.00064 |
| EURUSD | gaussian_hmm | 5267 | 3.81813 | 0.05411 | 0.05012 | 0.00062 |
| EURUSD | student_hmm | 5267 | 3.83515 | 0.05107 | 0.04993 | 0.00061 |
| EURUSD | rolling_gaussian | 5267 | 3.80809 | 0.05335 | 0.04898 | 0.00061 |
| EURUSD | ewma_gaussian | 5267 | 3.85056 | 0.05639 | 0.04955 | 0.00058 |
| EURUSD | ewma_student | 5267 | 3.86451 | 0.06398 | 0.05031 | 0.00058 |
| GBPUSD | gaussian_iid | 5267 | 3.64610 | 0.04766 | 0.04785 | 0.00072 |
| GBPUSD | student_iid | 5267 | 3.75260 | 0.05506 | 0.04860 | 0.00071 |
| GBPUSD | gaussian_hmm | 5267 | 3.73548 | 0.04993 | 0.04860 | 0.00067 |
| GBPUSD | student_hmm | 5267 | 3.78403 | 0.04955 | 0.04879 | 0.00068 |
| GBPUSD | rolling_gaussian | 5267 | 3.73417 | 0.05278 | 0.04955 | 0.00068 |
| GBPUSD | ewma_gaussian | 5267 | 3.79519 | 0.05810 | 0.05031 | 0.00064 |
| GBPUSD | ewma_student | 5267 | 3.80636 | 0.06645 | 0.05126 | 0.00064 |
| JPYUSD | gaussian_iid | 5267 | 3.61534 | 0.04841 | 0.05202 | 0.00071 |
| JPYUSD | student_iid | 5267 | 3.71019 | 0.05601 | 0.05392 | 0.00071 |
| JPYUSD | gaussian_hmm | 5267 | 3.72911 | 0.04785 | 0.05126 | 0.00067 |
| JPYUSD | student_hmm | 5267 | 3.75513 | 0.04443 | 0.05031 | 0.00066 |
| JPYUSD | rolling_gaussian | 5267 | 3.68478 | 0.04500 | 0.05012 | 0.00067 |
| JPYUSD | ewma_gaussian | 5267 | 3.72405 | 0.05126 | 0.05031 | 0.00065 |
| JPYUSD | ewma_student | 5267 | 3.76920 | 0.05810 | 0.05107 | 0.00065 |

## Known-state simulation

| dgp | model | trials | mean_log_score | breach_rate | state_accuracy | convergence_rate |
|---|---|---|---|---|---|---|
| break | gaussian_hmm | 100 | 2.61456 | 0.08745 | 0.69015 | 1.00000 |
| break | gaussian_iid | 100 | 2.03277 | 0.10408 | nan | 1.00000 |
| break | student_hmm | 100 | 2.67852 | 0.08095 | 0.70347 | 1.00000 |
| break | student_iid | 100 | 2.45852 | 0.13345 | nan | 1.00000 |
| contaminated | gaussian_hmm | 100 | 3.16617 | 0.04435 | 0.89063 | 1.00000 |
| contaminated | gaussian_iid | 100 | 2.82253 | 0.03910 | nan | 1.00000 |
| contaminated | student_hmm | 100 | 3.18302 | 0.04282 | 0.89453 | 1.00000 |
| contaminated | student_iid | 100 | 3.06079 | 0.06482 | nan | 1.00000 |
| gaussian | gaussian_hmm | 100 | 3.31505 | 0.05137 | 0.92545 | 1.00000 |
| gaussian | gaussian_iid | 100 | 3.06698 | 0.04585 | nan | 1.00000 |
| gaussian | student_hmm | 100 | 3.30125 | 0.04488 | 0.91637 | 1.00000 |
| gaussian | student_iid | 100 | 3.18771 | 0.05915 | nan | 1.00000 |
| student | gaussian_hmm | 100 | 3.33405 | 0.05372 | 0.88612 | 1.00000 |
| student | gaussian_iid | 100 | 3.04407 | 0.04335 | nan | 1.00000 |
| student | student_hmm | 100 | 3.35767 | 0.05247 | 0.90283 | 1.00000 |
| student | student_iid | 100 | 3.24072 | 0.06632 | nan | 1.00000 |

## Interpretation rules

Higher log score is better; 5% breaches should be near 0.05; lower quantile loss is better. Confidence intervals compare paired dates using circular 20-observation blocks. They are marginal descriptive intervals, not simultaneous claims across all tested comparisons. State accuracy is available only in simulations and states are aligned by emission variance. Calibration feedback uses past errors only. Neither observed calibration nor an asymptotic p-value establishes future conditional coverage.

FX observations are Federal Reserve noon buying rates, not executable quotes. H.10 publication lag and historical revisions are not reconstructed. This is a retrospective forecasting benchmark, not a real-time trading P&L claim. Missing dates are dropped per asset; a return across a holiday represents an observation interval rather than exactly 24 hours.
