# References and originality

- Rabiner, L. R. (1989), *A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition*, Proceedings of the IEEE 77(2), 257–286.
  [DOI](https://doi.org/10.1109/5.18626). Foundational HMM algorithms.
- Stamp, M., *A Revealing Introduction to Hidden Markov Models*.
  [Author's tutorial](https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf).
  Background on likelihood, smoothing and state-path distinctions.
- Hamilton, J. D. (1989), *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*, Econometrica 57(2).
  [DOI](https://doi.org/10.2307/1912559). Economic regime-switching motivation.
- Peel, D. and McLachlan, G. J. (2000), *Robust Mixture Modelling Using the t
  Distribution*, Statistics and Computing 10, 339–348.
  [DOI](https://doi.org/10.1023/A:1008981510081). Student-t latent-precision updates.
- Christoffersen, P. F. (1998), *Evaluating Interval Forecasts*, International
  Economic Review 39(4), 841–862. [DOI](https://doi.org/10.2307/2527341).
- Gibbs, I. and Candès, E. (2021), *Adaptive Conformal Inference Under Distribution
  Shift*. [Paper](https://arxiv.org/abs/2106.00170). Related online calibration;
  the simple standardized quantile controller here is not claimed to inherit
  this paper's guarantees.
- [`hmmlearn` tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html).
  External Gaussian-HMM oracle used only in tests; the package's inference
  routines are not called by this project's implementation.

## What is original here

The implementation, experiment orchestration, causal timing tests, paired
component ablation, report generation and visualizations were written for this
repository. Standard algorithms are credited above. Original implementation does
not imply that a standard algorithm or the broad research question is novel.

## Data provenance

Board of Governors of the Federal Reserve System (US), H.10 Foreign Exchange
Rates, via FRED: [DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU),
[DEXUSUK](https://fred.stlouisfed.org/series/DEXUSUK),
[DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS).
The series pages classify the data as public domain with citation requested.
JPY is reciprocated to express all currencies as USD per foreign-currency unit.
The exact frozen file and acquisition metadata are in `data/`.
