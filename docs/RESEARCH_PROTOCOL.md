# Research protocol and contribution boundary

## Question

Do latent regimes improve next-observation currency return distributions after
controlling for heavy tails, and can a past-only calibration layer improve tail
coverage without damaging quantile loss?

The contribution is an original, executable comparison and an audit of causal
inference. HMMs, Student-t mixtures, Baum–Welch, and adaptive calibration are
established methods. This repository makes no first-ever algorithm claim.

## Fixed main experiment

- Sources: Federal Reserve H.10 EUR/USD, GBP/USD, and inverted JPY/USD noon
  buying rates, January 2000 through December 2025. Preserve the downloaded
  snapshot, its SHA-256 and acquisition date. No interpolation of missing days.
- Training: trailing 1,250 observed returns. Refit every 252 observations.
- Models: Gaussian IID, Student-t IID, Gaussian two-state HMM, Student-t two-state
  HMM, 252-observation rolling Gaussian and EWMA Gaussian (lambda 0.94).
- HMM fitting: four seeded starts, 150 maximum EM iterations, fixed Student-t
  degrees of freedom 5, emission-variance state ordering. All convergence
  diagnostics retained. IID models use one start.
- Forecast timing: fit parameters on the past, form state probabilities and
  distribution **before** the next outcome, then update the filter with it.
- Primary metric: out-of-sample log predictive density. Tail diagnostics: 1% and
  5% violations, 5% pinball loss, PIT, Kupiec and Christoffersen tests.
- Four prespecified score contrasts per currency, paired across observation dates,
  with circular moving-block bootstrap (20 observations, 2,000 replicates).
  These intervals are not adjusted for multiplicity; do not interpret 12
  intervals as 12 independent discoveries. No model is chosen for deployment.
- Calibration: one-sided standardized quantile update with step size 0.02,
  initialized at zero, using only realized earlier errors. Report both coverage
  and quantile loss. This controller is not a finite-sample conditional-coverage
  guarantee and is not presented as a new conformal theorem.
- Simulations: 100 independent paired seeds per scenario, 1,000 training and
  400 test observations. Gaussian, Student-t, contaminated, and scale-break
  generators. All four parametric model families are evaluated on every path.

## Interpretation limits

The real series has no observable “true regime.” State accuracy is evaluated
only against simulation labels. A high-variance state is not inherently a bear
market or a recession. Label ordering identifies a statistical convention,
not a causal economic mechanism.

The FX source is released with a publication lag. This experiment orders quotes
by their observation dates and does not reconstruct when a free FRED subscriber
could first receive each value. Historical revisions are not reconstructed.
Consequently it is a retrospective risk-forecast benchmark, not a claim of
implementable trading returns. Observation intervals may cross holidays.

## Research decisions

The main protocol was written before the full experiment. A small quick run is
used to verify code and figure generation. Any later sensitivity analysis is
labeled separately and all tested settings are reported. No result is discarded
because it is unfavorable, and no holdout-driven strategy winner is promoted.
