# Mathematical specification

## Hidden state and emissions

Let $z_t\in\{1,\ldots,K\}$ follow a Markov chain with row-stochastic transition
matrix $A$. Given $z_t=k$, the observed return $y_t$ has density $f_k$:

$$y_t\mid z_t=k\sim N(\mu_k,s_k^2)$$

or a Student-t distribution with location $\mu_k$, scale $s_k$ and fixed
$\nu=5$ degrees of freedom. Student-t emission **variance** is
$s_k^2\nu/(\nu-2)$; confusing it with scale squared understates forecast risk.

Both models assume conditional independence of observations given the state
path. Regime changes create serial dependence in the unconditional distribution.
The IID ablations set $K=1$ and use the same emission families.

## Filtering, smoothing and timing

Write $\alpha_{t-1}(i)=P(z_{t-1}=i\mid y_{1:t-1})$. Before observing $y_t$:

$$p_t(j)=\sum_i\alpha_{t-1}(i)A_{ij},\qquad
g_t(y)=\sum_jp_t(j)f_j(y).$$

The scored forecast is $g_t$. After the outcome:

$$\alpha_t(j)=\frac{p_t(j)f_j(y_t)}{\sum_\ell p_t(\ell)f_\ell(y_t)}.$$

The first observation instead uses the initial distribution $\pi$. To bridge a
training and test segment, use the last training posterior multiplied by $A$ as
the first test prior. A test checks equality with uninterrupted filtering.

For numerical stability, the forward routine subtracts the largest log joint
state/emission mass among reachable states, normalizes the probabilities, and adds the removed
offset to the log-likelihood increment. Backward smoothing uses:

$$P(z_t=i\mid z_{t+1}=j,y_{1:t})=
\frac{\alpha_t(i)A_{ij}}{p_{t+1}(j)}.$$

Smoothing and Viterbi use future data and are explicitly offline diagnostics.
Training EM may smooth the **training window**; test forecasts only filter.

## Estimation

The E-step computes smoothed occupancy $\gamma_{tk}$ and pairwise transition
counts $\xi_{tij}$. Gaussian updates are weighted means and variances. The
Student-t representation introduces an auxiliary precision with expectation

$$u_{tk}=\frac{\nu+1}{\nu+(y_t-\mu_k)^2/s_k^2}.$$

Then:

$$\mu_k^{new}=\frac{\sum_t\gamma_{tk}u_{tk}y_t}
{\sum_t\gamma_{tk}u_{tk}},\qquad
(s_k^2)^{new}=\frac{\sum_t\gamma_{tk}u_{tk}(y_t-\mu_k^{new})^2}
{\sum_t\gamma_{tk}}.$$

Transition rows normalize expected transition counts; the initial probabilities
use $\gamma_{1k}$. Four seeded starts are used for two-state fits. The retained
start has the largest training likelihood, not the best test score. Floors
prevent degenerate scales and transitions; a likelihood-decreasing candidate is
rejected and marked in the diagnostics. The maximum iteration count is a
separate stopping condition, not a claim of convergence.

Training-window standardization improves numerical conditioning. When converting
the predicted density back to raw returns, subtract $\log(s_{train})$ from the
standardized log density. Quantiles are transformed by location and scale.

State labels are sorted by emission variance. This resolves a reporting
permutation but does not make an inferred state a factual economic regime.

## Distributional evaluation

The one-step mean is $m_t=\sum_kp_t(k)\mu_k$. Total variance includes both
within-state and between-state terms:

$$v_t=\sum_kp_t(k)(\operatorname{Var}_k+\mu_k^2)-m_t^2.$$

The lower-tail quantile solves $\sum_kp_t(k)F_k(q_t)=\alpha$ by vectorized
bisection, bracketed by the component quantiles. It is not a weighted average
of component quantiles.

- **Log score:** $\log g_t(y_t)$, a proper scoring rule. Higher is better.
- **Pinball loss:** $(\alpha-1\{y_t<q_t\})(y_t-q_t)$. Lower is better.
- **PIT:** $G_t(y_t)$. A histogram assesses marginal calibration; uniform-looking
  histograms alone do not establish temporal independence.
- **QLIKE-style score:** $\log(v_t)+(y_t-m_t)^2/v_t$, reported as a variance-risk
  diagnostic, not an exact measure of latent realized volatility.

Kupiec tests the unconditional breach frequency. Christoffersen's first-order
independence test compares a two-state breach Markov model with an IID Bernoulli
model. Empty transition rows make the independence diagnostic undefined; these
cases return `NaN`, not a fabricated favorable p-value. The tests are asymptotic
and a failure to reject is not evidence that a model is correct.

## Adaptive quantile calibration

The optional controller adjusts a base quantile with a standardized correction:

$$\widetilde q_t=q_t+c_t\sqrt{v_t},\qquad
c_{t+1}=\operatorname{clip}[c_t+\eta(\alpha-1\{y_t<\widetilde q_t\}),-4,4].$$

Here $\eta=0.02$ and $c_0=0$. Too many downside violations lower future
thresholds. The current error changes only the **next** forecast. This is a
pragmatic stochastic quantile controller inspired by online calibration; it is
not an implementation of a claimed new conformal theorem. Report its effect on
both coverage and loss: obtaining the right average breach rate is not enough.

## Uncertainty and failure experiments

Score comparisons use paired dates. A circular moving-block bootstrap resamples
20-observation contiguous blocks, wraps at the end, and computes 2,000 replicate
means. The reported 2.5%/97.5% percentiles account for local serial dependence
under a block-bootstrap approximation. They exclude parameter/vintage uncertainty
and are not simultaneous intervals over the comparisons. The initial protocol
contains 12 contrasts; the labeled Student-t EWMA follow-up adds three.

Synthetic experiments use 100 independent seeds per scenario and common paths
across models. The state path is observed only for evaluation. Four generators
cover a matched Gaussian HMM, a variance-matched Student-t HMM, independent rare
observation contamination, and a persistent scale increase during the test
segment. The last two deliberately violate the fitted model. Standard errors
are across independent simulation seeds, not across correlated observations.
