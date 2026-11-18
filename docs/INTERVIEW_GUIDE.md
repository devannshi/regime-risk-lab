# Questions this project should let you answer

1. **Why can a regime chart contain look-ahead even if its model was fitted only
   on training data?** Smoothing and Viterbi on the test sequence use later test
   observations. A trading-time decision requires a filtered or predicted belief.
2. **What separates prediction from filtering?** The prediction for interval t
   uses observations through t−1; filtering t incorporates its realized return.
3. **Why compare Student-t IID with Student-t HMM?** It isolates time dependence
   from distributional heavy tails. Comparing only Gaussian IID with Student-t
   HMM confounds two improvements.
4. **What does the negative result against EWMA mean?** State machinery is not
   automatically valuable. A responsive variance estimator can be competitive
   even when its conditional distribution is simpler.
5. **Why can good log score coexist with bad VaR?** Log score assesses the entire
   predictive density. Tail calibration and clustering can fail locally even
   when average density fit improves.
6. **Why does a calibrated violation rate not prove safety?** Errors can cluster,
   concentrate in a particular regime, or become severe when they occur. Average
   frequency is only one diagnostic.
7. **How do you know your forward algorithm is correct?** Exhaustive sums over all
   paths for a small example, a Gaussian-HMM external oracle, and segment-bridge
   equivalence provide independent checks.
8. **How are states identified?** Only up to permutation. Ordering by variance
   provides stable labels, not evidence that state 1 is objectively a recession.
9. **What uncertainty is missing from the intervals?** Model selection, vintage
   revisions, new structural breaks, and uncertainty about the block-length
   approximation. The reported intervals are not simultaneous across all models.
10. **What would you improve before deployment?** Reconstruct actual information
    availability, add strong heavy-tailed volatility baselines, monitor forecast
    calibration, validate stability on untouched periods, and separate risk
    forecasting from execution and funding assumptions.

Be able to derive the filter, explain the Student-t scale/variance distinction,
read the raw forecast audit, reproduce one result, and discuss an unfavorable
finding without hiding it. These are stronger demonstrations than memorizing a
large list of model names.
