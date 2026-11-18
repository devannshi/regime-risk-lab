# Research decisions and evidence history

1. Implemented independent inference and tests; ran a small smoke experiment to
   check execution and plotting. Specified the main protocol before the full run.
2. Ran all four synthetic scenarios and the six-model real-data protocol. Results
   were committed with the original report, audit tables, figures and manifest.
3. The initial results showed that Gaussian EWMA was competitive with the HMM.
   Added a **follow-up** Student-t EWMA benchmark with the same variance recursion
   and fixed df=5. This is a stronger control for whether the HMM adds value beyond
   both adaptive volatility and heavy tails. It is not retrospectively described
   as part of the initial prespecified six-model protocol. No existing model or
   unfavorable outcome was removed. The main simulation design is unchanged.

The additional comparison is exploratory on the same historical evaluation
period. All variants are shown, and there is no model-selection or live-trading
claim. A confirmatory superiority claim would need genuinely untouched data.
