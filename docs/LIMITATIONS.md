# What the evidence supports

This project establishes that the implemented recursions match independent
finite examples and an external Gaussian-HMM oracle, that forecast timing passes
future-perturbation tests, and that the retained experiments produce the reported
scores. It does not establish new universal guarantees or a trading strategy.

1. **No real regime labels.** A higher-variance state is a model component.
   State recovery accuracy is meaningful only in the synthetic experiments.
2. **Retrospective observation-time benchmark.** H.10 rates are released with a
   lag and can be revised. The frozen FRED series is not a real-time historical
   vintage archive. Noon buying rates are not executable fills. No trading P&L,
   Sharpe ratio, funding return or transaction-cost claim is made.
3. **Restricted model class.** Emissions are univariate, degrees of freedom are
   fixed, states have geometric durations, transition probabilities are constant
   within each annual refit segment, and contemporaneous cross-currency dependence
   is not modeled. Quotes across holidays span unequal calendar intervals.
4. **Local optimization.** Multiple starts reduce but do not eliminate local
   optima. Near-absorbing states, weak separation and low occupancy can impair
   estimation. All selected-start convergence flags and restart scores are saved.
5. **Historical score is not future validity.** Bootstrap intervals depend on a
   block-length choice and a local dependence approximation. They are unadjusted
   across comparisons and do not include all forms of model/data uncertainty.
6. **Calibration is not conditional reliability.** The online controller can
   improve average coverage while missing important conditional subgroups or
   clustering errors. Its clipping and fixed learning rate prevent invoking an
   unrestricted asymptotic adaptive-conformal theorem.
7. **Synthetic labels do not validate market interpretation.** Simulation
   parameters create known regimes by construction. A method can recover those
   regimes without demonstrating that the same mechanism governs actual markets.
8. **No outcome-driven winner.** All declared families are reported. Additional
   exploratory variants would require a new protocol and fresh evaluation before
   a claim of predictive improvement.

Useful next experiments: public real-time vintages where available, a
GARCH baseline, explicit-duration state models, joint currency
distributions, training-only hyperparameter selection, and simultaneous confidence
bands for a family of score comparisons. These are future work, not shipped claims.
