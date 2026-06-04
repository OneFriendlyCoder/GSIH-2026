# ETF Proxy Models — NAV, Risk and Hedging

A deep, line-by-line, math-rich walk-through of `ETF_Proxy_Models_NAV_Risk_and_Hedging.py` — three coupled estimation/optimization problems (NAV reconstruction, risk attribution, portfolio hedging) solved with regularized linear regression, robust covariance estimation, convex / quasi-convex optimization, and a final "horse-race" ensemble.

---

## Table of Contents

1. [The Problem in One Sentence](#1-the-problem-in-one-sentence)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Model and Return Convention](#3-data-model-and-return-convention)
4. [Helper Layer (Provided Scaffolding)](#4-helper-layer-provided-scaffolding)
5. [stdin Handling and Default Portfolio](#5-stdin-handling-and-default-portfolio)
6. [Workhorse Regressors](#6-workhorse-regressors)
   - 6.1 [Ridge Regression](#61-ridge-regression-fit_ridge_weights)
   - 6.2 [LASSO via Coordinate Descent](#62-lasso-via-coordinate-descent-lasso_cd)
7. [Task 1 — NAV Model](#7-task-1--nav-model)
   - 7.1 [`fit_nav_one_etf` — the multi-stage NAV pipeline](#71-fit_nav_one_etf--the-multi-stage-nav-pipeline)
   - 7.2 [`fit_nav_improved` — TimeSeries-Split Ridge with Auto-α](#72-fit_nav_improved--timeseries-split-ridge-with-auto-α)
8. [Task 2 — Risk Model](#8-task-2--risk-model)
   - 8.1 [Two-stage residual orthogonalization](#81-two-stage-residual-orthogonalization-fit_risk_one_etf)
   - 8.2 [`fit_risk_improved` — variance-gated CV ridge](#82-fit_risk_improved--variance-gated-cv-ridge)
9. [Task 3 — Portfolio Hedging](#9-task-3--portfolio-hedging)
   - 9.1 [P&L per unit notional](#91-pl-per-unit-notional-pnl_per_unit_notional)
   - 9.2 [Min-variance hedge — the analytical optimum](#92-min-variance-hedge--the-analytical-optimum)
   - 9.3 [Cost-aware LASSO hedge](#93-cost-aware-lasso-hedge-build_hedge_costaware)
   - 9.4 [Risk-based / Replication-based hedges](#94-risk-based--replication-based-hedges)
   - 9.5 [Max-effort hedge](#95-max-effort-hedge-build_hedge_max_effort)
   - 9.6 [Hybrid hedge (HER + TE + cost objective)](#96-hybrid-hedge-her--te--cost-objective)
   - 9.7 [Ledoit-Wolf shrinkage hedge](#97-ledoit-wolf-shrinkage-hedge)
   - 9.8 [Regime-blended covariance hedge](#98-regime-blended-covariance-hedge)
   - 9.9 [CVaR (tail-risk) hedge](#99-cvar-tail-risk-hedge)
   - 9.10 [Robust hedge (vol-mismatch penalty)](#910-robust-hedge-build_hedge_robust)
   - 9.11 [Optimized hedge (var ratio + cost)](#911-optimized-hedge-build_hedge_optimized)
10. [The Horse-Race Selector](#10-the-horse-race-selector)
11. [Output Assembly](#11-output-assembly)
12. [Numerical Stability and Engineering Tricks](#12-numerical-stability-and-engineering-tricks)
13. [Why This Architecture Works](#13-why-this-architecture-works)
14. [Possible Improvements](#14-possible-improvements)
15. [Appendix A — Notation Cheat Sheet](#15-appendix-a--notation-cheat-sheet)
16. [Appendix B — Technique Summary Table](#16-appendix-b--technique-summary-table)

---

## 1. The Problem in One Sentence

Given one year of EOD prices for 5 target fixed-income ETFs and 22 liquid proxy instruments (10 ETFs + 4 Treasury futures + 4 OTR Treasuries + 4 CDX indices), produce three deliverables:

1. **NAV model** — a `(5 × 22)` weight matrix that lets daily proxy returns reconstruct each target ETF's NAV path (lowest MAPE wins).
2. **Risk model** — a `(5 × 12)` weight matrix using only the non-ETF subset, so that the implied bucketed DV01 (2y/5y/10y/30y) and total CS01 match a hidden benchmark.
3. **Hedge basket** — given a per-call signed-notional portfolio, output a `(?)` × 1 hedging series that minimizes a weighted combination of (variance, tracking error, transaction cost).

The three tasks are linked: NAV weights tell you replication; risk weights tell you sensitivities; hedging uses **both** plus a direct variance-minimization on realized P&L.

---

## 2. High-Level Architecture

```
EOD_PRICES (252 × 23) ──► RETURNS = pct_change   (constant input)
                              │
                              ├─► fit_nav_improved ──► nav_weights (5×22)  [Task 1]
                              ├─► fit_risk_improved ─► risk_weights (5×12)  [Task 2]
                              │
JSON stdin ─► PORTFOLIO ──┐   │
                          ▼   ▼
                _gather_candidates  ◄── 17 different hedge strategies
                          │
                          ▼
                _hedge_score (HER − cost penalty)  ── horse-race selector
                          │
                          ▼
                hedge_notionals  [Task 3]
                          │
                          ▼
                    submission JSON
```

The whole script runs once at import time — no on-line inference. All three outputs are baked into the final JSON via `df.to_csv()`.

---

## 3. Data Model and Return Convention

### 3.1 Simple returns

The problem statement *mandates* simple (arithmetic) returns:

$$r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1$$

```python
def compute_returns(prices):
    return prices.pct_change().dropna()
```

Log returns are explicitly forbidden because the scoring evaluator uses simple-return composition $1 + r_t$ to reconstruct NAV levels (`reconstruct_nav` chains $\prod (1+r_t)$).

### 3.2 Par-quoted vs ETF instruments

The 12 "rates + credit" instruments (TSY futures, OTR Treasuries, CDX) are quoted **per 100 par notional**, whereas ETFs are quoted **per share / per NAV**. This dichotomy shows up everywhere — most importantly in `pnl_per_unit_notional` and `compute_cost`, where the scaling factor differs by a factor of 100.

```python
PAR_QUOTED = set(TSY_FUTURES + TSY_BONDS + CDX)
```

---

## 4. Helper Layer (Provided Scaffolding)

### 4.1 `predict_returns` & `reconstruct_nav`

```python
def reconstruct_nav(weights, proxy_returns, base_nav):
    pred_returns = predict_returns(weights, proxy_returns)
    return (1 + pred_returns).cumprod() * base_nav
```

This is the **evaluation function** the scorer uses. The model's daily predicted return is

$$\hat r_t = \mathbf{w}^\top \mathbf{r}^{\text{proxy}}_t,$$

then a cumulative product reconstructs the NAV path. Critically: the optimizer is fitting daily *returns*, but the metric is MAPE on *levels* — small daily-return bias compounds.

### 4.2 `compute_mape`

$$\text{MAPE} = \frac{1}{T}\sum_{t=1}^T \left|\frac{\hat N_t - N_t}{N_t}\right| \times 100$$

### 4.3 `compute_pnl_series` — the P&L formula

Given a notionals dict and the price panel, daily P&L in mm USD:

$$\text{PnL}_t = \sum_i s_i \cdot (P_{i,t} - P_{i,t-1})$$

where the scaling factor $s_i$ depends on instrument type:

| Instrument | Scaling |
|---|---|
| ETF | $s_i = N_i / P_{i,0}$ (number of shares = notional / price-at-day-0) |
| Par-quoted | $s_i = N_i / 100$ (par-notional units) |

```python
scaled = pd.Series({c: n * (1.0 / p0[c] if c not in PAR_QUOTED else 1.0 / 100.0)
                    for c, n in notionals.items()})
return prices[scaled.index].diff().iloc[1:].mul(scaled, axis=1).sum(axis=1)
```

This formula is **the** ground truth — every hedge function builds its objective so that it minimizes the *variance* of exactly this P&L expression.

### 4.4 `compute_her`

Hedge Effectiveness Ratio:

$$\text{HER} = 1 - \frac{\text{Var}(\text{PnL}_{\text{port}} + \text{PnL}_{\text{hedge}})}{\text{Var}(\text{PnL}_{\text{port}})}$$

HER = 1 means the hedge perfectly cancels the portfolio's variance; HER = 0 means it does nothing; HER < 0 means the "hedge" *added* risk.

### 4.5 `compute_cost`

One-way transaction cost on |market value| at day 0:

$$\text{TC} = \sum_i \text{bps}_i \times \text{MV}_i \times 10^{-4}$$

where $\text{MV}_i = |N_i|$ for ETFs and $\text{MV}_i = |N_i|/100 \cdot P_{i,0}$ for par-quoted. Cost rates: 0.5 bps for rates, 1 bp for CDX, 2 bps for ETFs.

---

## 5. stdin Handling and Default Portfolio

```python
def _read_stdin_safely(timeout: float = 1.0) -> str:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready: return sys.stdin.read()
    return ""
```

A non-blocking read of stdin with a 1-second timeout. If no test input is piped, fall through to a default portfolio. This makes the script runnable both interactively (returns empty → uses default) and inside the grader (returns the JSON).

The default `_DEFAULT_PORTFOLIO` is the placeholder example from the problem statement — `[150, -120, 180, -100, 130]`.

---

## 6. Workhorse Regressors

### 6.1 Ridge Regression (`fit_ridge_weights`)

The Tikhonov-regularized OLS:

$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}}\, \|\mathbf{y} - X\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$$

Closed-form solution:

$$\hat{\mathbf{w}} = (X^\top X + \lambda I_p)^{-1} X^\top \mathbf{y}$$

```python
XtX = X.T @ X
lam = alpha * (np.trace(XtX) / n_feat if n_feat else 1.0)
return np.linalg.solve(XtX + lam * np.eye(n_feat), X.T @ y)
```

**Subtlety — adaptive λ scaling.** Instead of using `alpha` directly, the code scales by $\text{tr}(X^\top X) / p$ — the average feature variance times $n$. This makes `alpha` **scale-invariant**: doubling all features (or doubling $n$) doesn't change the effective penalty, because $\text{tr}(X^\top X)$ scales with them. Without this, `alpha=1e-5` would mean wildly different things depending on whether features are in % or in bps.

**Why ridge first, not OLS?** With $T \approx 250$ days and $p = 22$ features, OLS is technically well-posed but the design matrix is nearly singular when proxy ETFs co-move strongly with each other (which they do — many track the same underlying index). Ridge:

- Shrinks small eigenvalues away from 0 (better conditioning).
- Smooths the regression coefficients, producing more stable out-of-sample predictions.
- With tiny α (e.g. 1e-5), it's essentially OLS but numerically robust.

### 6.2 LASSO via Coordinate Descent (`lasso_cd`)

The LASSO objective:

$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}}\, \tfrac{1}{2n}\|\mathbf{y} - X\mathbf{w}\|_2^2 + \alpha \sum_j c_j |w_j|$$

where $c_j$ is a per-feature cost weight (used here to **penalize expensive instruments more**).

**Soft-thresholding closed form** for each coordinate update:

$$\rho_j = \mathbf{x}_j^\top (\mathbf{y} - X_{-j}\mathbf{w}_{-j}), \qquad \tau_j = \alpha\, c_j\, n$$

$$w_j^{\text{new}} = \frac{1}{\|\mathbf{x}_j\|_2^2}\, S(\rho_j, \tau_j),\qquad S(\rho, \tau) = \text{sign}(\rho)\cdot \max(|\rho|-\tau, 0)$$

i.e. *if* $|\rho_j| > \tau_j$, take the OLS coefficient *shrunk* toward zero by $\tau_j/\|x_j\|^2$; otherwise set $w_j = 0$.

```python
if rho > thresh:    w[j] = (rho - thresh) / col_sq[j]
elif rho < -thresh: w[j] = (rho + thresh) / col_sq[j]
else:               w[j] = 0.0
```

The residual is maintained incrementally:

```python
if w[j] != 0.0: residual += Xn[:,j] * w[j]   # remove old
... compute rho ...
if w[j] != 0.0: residual -= Xn[:,j] * w[j]   # add new
```

so each pass over all $p$ coordinates costs $O(np)$, not $O(np^2)$.

**Why a hand-rolled LASSO?** sklearn's `Lasso` doesn't directly support **per-feature alpha weights**, which is required here to penalize expensive instruments (Proxy ETFs at 2bp) more heavily than cheap ones (Treasury futures at 0.5bp). The weighted LASSO is exactly the convex relaxation of the cost-aware hedging problem (see §9.3).

**Standardization.** Columns are scaled to unit standard deviation before fitting, then coefficients are un-scaled at the end:

```python
X_std = X.std(axis=0, ddof=0)
Xn = X / X_std
...
return w / X_std
```

This ensures the per-feature penalty $\tau_j$ acts uniformly across instruments regardless of their natural volatility scale.

**Convergence.** The loop terminates when the largest single-coordinate change drops below `tol * max|w|`. CD on LASSO is guaranteed to converge because the objective is convex and the per-coordinate problem has a closed-form minimizer (Friedman et al. 2010).

---

## 7. Task 1 — NAV Model

### 7.1 `fit_nav_one_etf` — the Multi-Stage NAV Pipeline

This is the most carefully engineered estimator in the file. For each target ETF, the pipeline is:

#### Step 1 — Sanity check

```python
if len(aligned) < 25 or len(avail) == 0:
    return full.values
```

Need at least 25 aligned observations to do anything statistical.

#### Step 2 — Variance-floor filter

```python
y_var = float(y.var(ddof=0))
proxy_avg_var = float(X.var(ddof=0).mean())
if proxy_avg_var > 0 and y_var < low_var_ratio * proxy_avg_var:
    return full.values
```

If the target's own daily-return variance is less than 35% of the average proxy variance, the target is essentially a constant (a money-market or ultra-short fund) and a proxy model is meaningless. Return all zeros.

This is why `Target_ETF_1` ends up all-zero in the sample output — its NAV moves so little that no proxy combination is more informative than the identity (no model).

#### Step 3 — Correlation-based pre-screening

```python
corrs = X.corrwith(y).abs().fillna(0.0)
selected = corrs[corrs > corr_thresh].index.tolist()
if len(selected) < 4:
    selected = corrs.nlargest(min(7, len(corrs))).index.tolist()
```

Only consider proxies with $|\rho(X_j, y)| > 0.09$. If fewer than 4 pass, take the top-7 by correlation.

This is **screening before fitting** — it kills noisy / unrelated instruments before they pollute the regression. It also reduces the effective number of features, mitigating multicollinearity among the 22 candidates.

#### Step 4 — Time-decay sample weights

```python
sw = np.exp(np.linspace(-1.8, 0.0, n))
sw = sw / sw.sum() * n
```

Weights are exponentially decaying with the **most recent observations weighted ~e^1.8 ≈ 6× more** than the oldest. The intent: recent regime is more representative of the test period (which immediately follows training). Normalization preserves the effective sample size = $n$.

#### Step 5 — Three-bagged ElasticNetCV

```python
for l1 in (0.4, 0.65, 0.85):
    m = ElasticNetCV(l1_ratio=l1, alphas=np.logspace(-5.5, 0.5, 35), cv=5, ...)
    m.fit(Xsel.values, y.values, sample_weight=sw)
    coefs.append(m.coef_)
coef = np.median(np.stack(coefs, axis=0), axis=0)
```

ElasticNet objective:

$$\min_{\mathbf{w}}\, \tfrac{1}{2n}\|y - X\mathbf{w}\|_2^2 + \alpha\left[\rho \|\mathbf{w}\|_1 + \tfrac{1-\rho}{2}\|\mathbf{w}\|_2^2\right]$$

ρ = `l1_ratio` controls the mix:
- ρ = 0.4: mostly ridge — keeps many small weights, stable.
- ρ = 0.85: mostly LASSO — sparse, picks 2-3 dominant proxies.
- ρ = 0.65: balanced.

**5-fold CV** picks the best α from a log-spaced grid of 35 values. Then the three fitted coefficient vectors are combined via **per-coefficient median**:

$$\hat w_j = \text{median}\bigl(w_j^{(\rho=0.4)},\, w_j^{(\rho=0.65)},\, w_j^{(\rho=0.85)}\bigr)$$

The median is robust — if any one l1_ratio finds an unstable basin (e.g. LASSO swapping between two collinear features), the median ignores it. This is a **bagging** technique applied to the regularization grid rather than to data.

#### Step 6 — Ridge fallback

```python
if coef is None or not np.any(np.abs(coef) > 1e-12):
    coef = fit_ridge_weights(y, Xsel, alpha=3e-5)
```

If ElasticNetCV failed (didn't converge, no scikit-learn) or returned all zeros, fall back to plain ridge.

#### Step 7 — Asymmetric clip (positive bias)

```python
coef = np.where(coef < -0.05 * np.abs(coef), -0.05 * np.abs(coef), coef)
```

Slightly truncates negative coefficients. The intent: fixed-income ETFs should have **positive** exposure to most proxies (long-only NAV replication). Mild positive bias improves out-of-sample robustness when the in-sample regression goes slightly negative due to noise.

### 7.2 `fit_nav_improved` — TimeSeries-Split Ridge with Auto-α

A simpler, but more aggressively tuned, alternative — and what the main script actually calls first.

#### Time-series 80/20 split + 50-point ridge grid

```python
n_train = int(0.8 * len(aligned))
y_tr, y_val = y[:n_train], y[n_train:]
...
for alpha in alpha_range:   # logspace(-6, 1, 50)
    w = np.linalg.solve(XtX_tr + alpha * I_p, Xty_tr)
    val_mse = np.mean((y_val - X_val @ w)**2)
    if val_mse < best_val_mse: best_alpha = alpha
```

Train on the first 80% (≈ 200 days), tune α on the last 20% (≈ 50 days), then refit on the full dataset at the chosen α. **TimeSeries split, not k-fold**, because:

1. Returns may have weak autocorrelation/regime structure.
2. The actual test period is *after* the training data, so we should select α that generalizes forward in time, not into random folds.

#### Magnitude thresholding

```python
w = np.where(np.abs(w) > mag_thresh, w, 0.0)   # mag_thresh = 0.01
```

Coefficients below 1% are zeroed — cleaner submission, fewer instruments charged with transaction costs if these weights are also used by `build_hedge_replication`.

#### L1-budget normalization

```python
abs_sum = float(np.sum(np.abs(w)))
if abs_sum > 0.1:
    w = w / abs_sum
```

If $\sum |w_j| > 0.1$, rescale so $\sum |w_j| = 1$. This enforces a budget constraint: the NAV is a *convex-ish* combination of proxies, which prevents the model from producing pathological coefficients like (+5, −4) that happen to fit in-sample noise but blow up out-of-sample.

**Why 0.1 threshold?** It is just a guard for the all-zero ETF (Target_ETF_1) — if everything is zero, leave it; otherwise normalize.

---

## 8. Task 2 — Risk Model

The risk model is restricted to **non-ETF** proxies only (`RISK_PROXY = TSY_FUTURES + TSY_BONDS + CDX`). The rationale (from the problem statement): proxy ETFs mix rate and credit, so they pollute clean attribution.

### 8.1 Two-stage residual orthogonalization (`fit_risk_one_etf`)

This is a classical **sequential regression** approach to decompose ETF returns into rate-driven and credit-driven components:

#### Stage 1 — Fit rates first

```python
X_fut = aligned[fut_cols].values
m = RidgeCV(alphas=np.logspace(-5, 2, 30), cv=6, fit_intercept=False)
m.fit(X_fut, y)
fut_coef = m.coef_
y_resid = y - X_fut @ fut_coef
```

Regress $y$ on the 4 Treasury futures, get $\hat\beta_{\text{rate}}$. Compute the residual:

$$\tilde y_t = y_t - X^{\text{rate}}_t\, \hat\beta_{\text{rate}}$$

#### Stage 2 — Fit credit on the residual

```python
X_cr = aligned[cr_cols].values
m.fit(X_cr, y_resid)
cr_coef = m.coef_
```

Regress $\tilde y$ on the 4 CDX indices, get $\hat\beta_{\text{credit}}$.

**Why this two-stage decomposition?** Treasury futures are correlated with CDX (especially in stress periods). A joint regression would split the explained variance "ambiguously" between rates and credit. By regressing rates *first*:

1. The rate beta absorbs all explainable rate variation, irrespective of credit correlation.
2. The credit beta then captures only the **excess credit risk** — the part of returns not already explained by rates.

This matches the **risk-attribution** convention used by the scorer: bucketed DV01 should reflect *pure* rate sensitivity, CS01 should reflect *pure* credit sensitivity.

OTR Treasury bonds are *not used* in this routine (`use_cols = fut_cols + cr_cols`). The reason: futures and OTR bonds are nearly perfectly collinear in daily returns (they track the same yield curve), so including both adds no information but blows up the condition number. Futures are chosen because they have richer tenor coverage (TU/FV/TY/US span 2y to 25y).

#### `RidgeCV(cv=6, fit_intercept=False)`

- **6-fold CV** (default 5 in sklearn, here bumped to 6 for slightly better α resolution).
- **`fit_intercept=False`** because daily returns have ~zero mean by construction — no benefit, and including an intercept adds a degree of freedom that the regularization then has to control.

### 8.2 `fit_risk_improved` — Variance-Gated CV Ridge

A reformulation of risk fitting that uses TimeSeries-split ridge similar to `fit_nav_improved`, but operating only on `TSY_FUTURES + CDX`.

```python
if X_avg_var > 0 and y_var < low_var_ratio * X_avg_var:
    continue
```

Same variance gating — skip ETFs with too little variation.

The actual ridge tuning uses `np.logspace(-8, -2, 40)` — a *much tighter* range than NAV's `logspace(-6, 1, 50)`. Reason: risk attribution should be **close to OLS** to faithfully estimate the *true* beta; we don't want heavy shrinkage biasing the betas toward zero. The cost of larger out-of-sample variance is acceptable because risk weights are used downstream by the evaluator to compute analytical DV01/CS01, not to predict returns directly.

---

## 9. Task 3 — Portfolio Hedging

The hedging machinery is, by line count, the bulk of the file. Eleven distinct hedge builders are implemented, all eventually fed into a horse-race selector.

The underlying optimization is always some variant of:

$$\min_{\mathbf{h}} \quad \mathcal{L}\bigl(\mathbf{y} + X\mathbf{h}\bigr)$$

where:
- $\mathbf{y} = \sum_i N_i\, \mathbf{x}^{\text{unit-pnl}}_i$ — portfolio P&L per day
- $X$ — daily unit-notional P&L of each hedging instrument
- $\mathbf{h}$ — signed hedge notionals (the decision variable)
- $\mathcal{L}$ — loss function (variance, HER, CVaR, etc.)

### 9.1 P&L per Unit Notional (`pnl_per_unit_notional`)

The crucial transformation that linearizes everything:

```python
p0 = prices.iloc[0]
scaling = pd.Series({c: (1.0/100.0 if c in PAR_QUOTED else 1.0/p0[c])
                     for c in prices.columns})
return prices.diff().iloc[1:].mul(scaling, axis=1)
```

This produces a $T \times n$ matrix where column $i$ contains the daily P&L (in mm USD) of **1 mm USD long position** in instrument $i$. Then total P&L is linear in notionals:

$$\text{PnL}_t = \sum_i N_i \cdot \text{unit\_pnl}_{i,t}$$

This linearization is what lets every hedge problem reduce to a quadratic / convex optimization with a closed-form or fast iterative solution.

### 9.2 Min-Variance Hedge — the Analytical Optimum

The fundamental hedging problem: minimize the variance of $\mathbf{y} + X\mathbf{h}$:

$$\min_{\mathbf{h}} \, (\mathbf{y} + X\mathbf{h})^\top (\mathbf{y} + X\mathbf{h})$$

Differentiating and setting $\partial/\partial \mathbf{h} = 0$:

$$2 X^\top (\mathbf{y} + X\mathbf{h}) = 0 \;\Longrightarrow\; X^\top X\, \mathbf{h} = -X^\top \mathbf{y}$$

$$\boxed{\,\mathbf{h}^\star = -(X^\top X)^{-1} X^\top \mathbf{y}\,}$$

This is just OLS with a *minus sign*. The intuition: project $-\mathbf{y}$ onto the column space of $X$.

With ridge regularization (numerical safety):

```python
XtX = X.T @ X
lam = 1e-3 * np.trace(XtX) / XtX.shape[0]
h = -np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X.T @ y)
```

This is the baseline that every other hedge function refines.

### 9.3 Cost-Aware LASSO Hedge (`build_hedge_costaware`)

Pure min-variance picks up small positions in every instrument because *some* explained variance > 0. But each instrument carries transaction costs. So **add a cost-weighted L1 penalty**:

$$\min_{\mathbf{h}}\; \tfrac{1}{2n}\|\mathbf{y} + X\mathbf{h}\|_2^2 + \alpha\sum_j c_j\, |h_j|$$

LASSO sparsity automatically zeros out instruments whose marginal variance reduction doesn't justify their cost.

#### Auto-α via knee finding

```python
score = np.abs(Xn.T @ y) / (X.shape[0] * cost_w)
sorted_s = np.sort(score)[::-1]
k = min(6, len(sorted_s) - 1)
alpha = sorted_s[k] if k < len(sorted_s) else 0.0
```

`score[j] = |X_j · y| / (n · c_j)` is exactly the threshold that activates feature $j$ in LASSO's first iteration (the marginal correlation, cost-normalized). Sorting and picking the 6th-largest sets α so that **~6 features enter the model**. Self-tuning sparsity.

The minus sign at the end (`h = -w`) flips the optimization from "predict y" to "hedge y."

### 9.4 Risk-Based / Replication-Based Hedges

#### Risk-based (`build_hedge_from_risk`)

Use the Task-2 weights as a hedge map directly:

$$h_j = -\sum_i N_i\, w_{ij}^{\text{risk}} / s_j$$

where $s_j = P_{j,0}/100$ for par-quoted instruments, 1 otherwise. The minus sign turns "exposure" into "offsetting hedge."

This produces a hedge that is **clean** in risk dimensions (only rate/credit instruments) but may not minimize realized variance (since it doesn't see realized P&L).

#### Replication-based (`build_hedge_replication`)

Same idea but with NAV weights (Task 1) — uses all 22 instruments. Captures **total NAV variance** but may be expensive (uses 2bp Proxy ETFs).

Both functions are essentially **dual-use of Tasks 1 and 2 outputs**, demonstrating that the three subproblems aren't independent — the same betas can serve replication and risk.

### 9.5 Max-Effort Hedge (`build_hedge_max_effort`)

Minimum-variance ratio with **box constraints** $|h_j| \le 10000$ mm and L-BFGS-B refinement from multiple starts:

```python
for x0 in (h_minvar, np.zeros(n_feat), h_minvar*1.3, h_minvar*0.7):
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B", ...)
```

The 4-start multi-start search guards against local minima. Box bounds prevent the optimizer from picking absurd notionals to chase numerical noise.

### 9.6 Hybrid Hedge (HER + TE + Cost Objective)

The first explicit **multi-objective** hedge — directly models the scorer's loss:

$$\mathcal{L}(\mathbf{h}) = -w_{\text{HER}}\cdot \text{HER}(\mathbf{h}) + w_{\text{TE}}\cdot \text{TE}(\mathbf{h}) + w_{\text{cost}}\cdot \text{TC}(\mathbf{h})$$

with chosen weights `(2.2, 9.5, 0.009)`. These are empirically tuned against the scorer's hidden weight scheme. Seeded from `build_hedge_costaware` and refined via L-BFGS-B.

### 9.7 Ledoit-Wolf Shrinkage Hedge

The min-variance hedge formula in covariance language:

$$\mathbf{h}^\star = -\Sigma_{hh}^{-1}\, \Sigma_{hp}$$

where $\Sigma$ is the covariance of $(\text{PnL}_{\text{port}}, \text{PnL}_{\text{hedge}})$. Replace the sample covariance by **Ledoit-Wolf shrunk** estimate:

$$\hat\Sigma_{\text{LW}} = \delta^\star F + (1 - \delta^\star)\hat\Sigma_{\text{sample}}$$

with $F = \mu I$ and $\delta^\star$ the closed-form optimal shrinkage intensity (see the Regime Navigator report for the full derivation).

```python
all_returns = np.column_stack([y.reshape(-1, 1), X])
lw = LedoitWolf().fit(all_returns)
cov = lw.covariance_
sigma_ph = cov[0, 1:]
sigma_hh = cov[1:, 1:]
h = -np.linalg.solve(sigma_hh + 1e-8 * np.eye(len(hedge_cols)), sigma_ph)
```

**Why LW for hedging?** With $T \approx 250$ days and $p+1$ instruments, sample covariance is reasonably well-conditioned but the smallest eigenvalues are still biased toward zero. LW shrinkage stabilizes the inverse, producing a hedge that is **less sensitive to in-sample noise** and generalizes better.

### 9.8 Regime-Blended Covariance Hedge

A novel twist: blend full-sample and recent-window LW covariances:

$$\hat\Sigma_{\text{blend}} = \beta\, \hat\Sigma_{\text{recent}} + (1-\beta)\, \hat\Sigma_{\text{full}}$$

with $\beta = 0.6$ and recent window = 60 days.

The motivation: the most recent 60 days are most representative of the next test period's regime, but they're also noisy (small $T$). The blend balances **regime relevance** vs **estimation noise**. The min-variance hedge is then computed against this blended Σ.

This mirrors the Regime Navigator's EMA stress index — same philosophical pattern of "recent matters more, but recent alone is too noisy."

### 9.9 CVaR (Tail-Risk) Hedge

Instead of minimizing variance (which is symmetric), minimize **Conditional Value-at-Risk** of the net P&L:

$$\text{CVaR}_\alpha(\mathbf{h}) = -E\bigl[\text{PnL} \,\big|\, \text{PnL} \le \text{VaR}_\alpha\bigr]$$

i.e. expected loss in the worst $\alpha$-fraction of days.

```python
n_tail = max(2, int(alpha * T))   # alpha = 0.05 → ~5% worst days
def cvar_obj(h):
    net = y + X @ h
    sorted_net = np.sort(net)
    left = sorted_net[:n_tail]      # tail of losses
    right = sorted_net[-n_tail:]    # tail of gains
    return -float(np.mean(left)) + float(np.mean(right))
```

The objective is the **two-sided** tail expectation — penalizes both extreme losses *and* extreme gains. Effectively a robust loss function.

L-BFGS-B from multiple starts. CVaR is **not differentiable** at the threshold but is *piecewise-linear* in $\mathbf{h}$, so quasi-Newton solvers handle it fine in practice (the subgradient is well-defined almost everywhere).

**Why include CVaR?** Variance-minimizing hedges can leave the portfolio exposed to fat-tail events. A CVaR hedge explicitly trades some "average" hedging efficiency for **better worst-case** protection — which can matter when the test window happens to contain a volatile stretch.

### 9.10 Robust Hedge (`build_hedge_robust`)

Adds a **vol-matching** term:

$$\mathcal{L}(\mathbf{h}) = -3.2 \cdot \text{HER} + 12 \cdot \text{TE} + 0.006 \cdot \text{cost} + 5.5 \cdot \Bigl|\frac{\text{TE}}{\sigma_{\text{port}}} - 1\Bigr|$$

The last term penalizes hedges where the residual volatility is *very different* from the portfolio volatility. The reasoning: an over-hedged portfolio (TE → 0 but at the cost of huge notionals) is fragile — small estimation errors in $X$ become big realized P&L errors. Keeping TE near 1× the original σ ensures hedge "scale" is honest.

### 9.11 Optimized Hedge (`build_hedge_optimized`)

Variance ratio + L1 cost penalty:

$$\mathcal{L}(\mathbf{h}) = \frac{\text{Var}(\mathbf{y} + X\mathbf{h})}{\text{Var}(\mathbf{y})} + w_c \sum_j c_j |h_j|$$

Differentiable everywhere (the cost term is smoothed by L-BFGS-B's bounded-step internally). The variance ratio is **scale-free** so weight choices for cost generalize across portfolio sizes.

---

## 10. The Horse-Race Selector

`_gather_candidates` runs ~17 hedge configurations in parallel:

| # | Strategy | Instrument set | Key knob |
|---|---|---|---|
| 1 | replication (from NAV w) | all | NAV-based |
| 2 | optimized (loose) | all | cost_w=1e-6 |
| 3 | optimized (tight) | all | drop_tol=0.35 |
| 4 | optimized (liquid only) | TSY_F + CDX | — |
| 5 | robust | all | vol-mismatch |
| 6 | LW shrinkage | all | — |
| 7 | LW shrinkage (liquid) | TSY_F + CDX | — |
| 8 | regime-blended (full) | all | β=0.6 |
| 9 | regime-blended (liquid) | TSY_F + CDX | β=0.6 |
| 10 | CVaR (α=0.05) | all | — |
| 11 | CVaR (α=0.05, liquid) | TSY_F + CDX | — |
| 12 | CVaR (α=0.10) | all | wider tail |
| 13 | cost-aware LASSO | all | — |
| 14 | risk-based | TSY_F + CDX | from risk_w |
| 15 | risk-based (liquid) | TSY_F + CDX | from risk_w |
| 16 | max-effort | all | high max_notional |
| 17 | max-effort (liquid) | TSY_F + CDX | high max_notional |

Each candidate is scored:

```python
return float(her) - cost_pen * float(tc)   # cost_pen = 2.0
```

This is the **selector's** proxy for the hidden scorer:

$$\text{score} = \text{HER} - 2\cdot \text{TC}$$

The candidate with the highest in-sample score wins. Fallback chain:

```python
hedge_notionals = best_candidate
if hedge_notionals empty: hedge_notionals = build_hedge_hybrid(...)
if empty: hedge_notionals = build_hedge_costaware(...)
if empty: hedge_notionals = build_hedge(..., method="ridge")
```

### Why a horse race?

- **No single method dominates across portfolios.** A portfolio dominated by `Target_ETF_5` may favor a CDX-heavy hedge; a portfolio of `Target_ETF_1, 2` (pure rates) may favor TSY futures only.
- **Different objective surfaces produce different minima.** The MV optimizer may find a hedge that's strong on training but weak on hold-out; LW shrinkage may produce a sturdier hedge.
- **The hidden scorer's weights are unknown.** By trying many strategies that span the (HER, TE, cost) Pareto front and picking the one that maximizes a *proxy* score, the chance of matching the hidden weights is higher than committing to a single method.

This is the **ensemble / model-averaging** philosophy applied at the strategy level.

---

## 11. Output Assembly

Three submissions are placed in a single JSON:

```python
submission = {
  "nav":   nav_model.to_csv(),
  "risk":  risk_model.to_csv(),
  "hedge": hedging_basket.to_csv()
}
print(json.dumps(submission))
```

The `to_csv()` strings are unparsed CSV bodies. The grader expects exact column orders (`ETF, Proxy_ETF_1, ..., CDX_HY_10Y` for NAV, etc.) — the code uses the constants from the top of the file to guarantee this ordering.

---

## 12. Numerical Stability and Engineering Tricks

| Trick | Where | Purpose |
|---|---|---|
| Adaptive λ = α · tr(XᵀX)/p | `fit_ridge_weights` | Scale-invariant regularization |
| Column standardization in LASSO | `lasso_cd` | Per-feature penalty acts uniformly |
| Incremental residual updates in CD | `lasso_cd` | O(np) per pass instead of O(np²) |
| `fit_intercept=False` everywhere | RidgeCV/ElasticNetCV | Returns are ~zero-mean already |
| Time-decay sample weights | `fit_nav_one_etf` | Recent regime weighted higher |
| Median over l1_ratio bagged fits | `fit_nav_one_etf` | Robust to regularization-path instability |
| 80/20 time-series CV | `fit_nav_improved` | Honors temporal ordering |
| L1-budget rescaling | `fit_nav_improved` | Prevents pathological coefficients |
| Magnitude thresholding | both improved fits | Sparser output, lower hedge cost |
| Two-stage rate-then-credit fit | `fit_risk_one_etf` | Clean risk attribution |
| Variance-floor gating | both improved fits | Skip near-constant targets |
| LW shrinkage | hedge variants | Stable covariance inverse |
| Regime-blended cov | `build_hedge_regime_blend` | Recent regime relevance |
| CVaR loss | `build_hedge_cvar` | Tail-risk awareness |
| Multi-start L-BFGS-B | several hedges | Local-minimum guard |
| Vol-mismatch penalty | `build_hedge_robust` | Avoid over-hedging fragility |
| `_read_stdin_safely` with timeout | top | Works both with and without stdin |
| Horse-race selector | `_gather_candidates` | Strategy-level ensemble |
| Cascading fallbacks | end of hedge block | Always produce *some* output |

---

## 13. Why This Architecture Works

| Scoring component | How the code targets it |
|---|---|
| **NAV MAPE** | ElasticNet CV + time-decay weights + l1_ratio bagging + ridge fallback |
| **Risk L2_DV01 / L2_CS01** | Two-stage rate→credit residualization on TSY_FUTURES + CDX (clean buckets) |
| **HER** | Min-variance / LW shrinkage / regime blend — all directly minimize residual variance |
| **Tracking error** | Same min-variance objective minimizes σ(net P&L) |
| **Transaction cost** | LASSO with per-instrument cost weights + drop-tolerance filters + sparsity |
| **Generalization** | TimeSeries CV + ridge fallbacks + horse race; no single method gets to dominate |

The key design insight is **multi-resolution defense**:

1. At the *coefficient* level: ElasticNet/ridge with CV picks α optimally.
2. At the *regularization-path* level: l1_ratio bagging via median.
3. At the *strategy* level: horse-race selects among 17 hedge formulations.
4. At the *failure* level: cascading fallbacks ensure the script always emits valid CSVs.

---

## 14. Possible Improvements

1. **Joint estimation across the 5 ETFs.** All five share the same proxy basis. A multi-task LASSO or group-LASSO would borrow strength across ETFs and reduce per-ETF coefficient variance.
2. **Bayesian model averaging.** Instead of picking the single best horse-race candidate, weight the top-k by posterior probability and take a convex combination. Could improve robustness.
3. **PCA / factor model on proxies.** Compress the 22-instrument basis into 4-5 orthogonal factors before regression. Eliminates multicollinearity entirely.
4. **Walk-forward CV for hedge α selection.** Currently the horse-race uses in-sample HER for selection; rolling-window backtest of each candidate would better predict out-of-sample performance.
5. **Explicit positivity constraint on NAV weights.** For long-only fixed-income ETFs, weights should be nonneg. NNLS or constrained ridge would respect this rigorously instead of the asymmetric-clip heuristic.
6. **Sector decomposition for credit.** The 4 CDX instruments span IG/HY × 5y/10y. A more granular credit factor model (e.g. separate IG vs HY level/slope) could improve risk attribution.
7. **Round notionals to tradable lots.** The hedge basket emits arbitrary mm-USD numbers. Rounding to (say) integer mm USD would reduce nuisance trades at marginal cost loss.
8. **Drop redundant hedge variants.** With 17 candidates and one winner, marginal candidates burn compute. A pruned horse race (~6 strong candidates) would be just as effective and faster.

---

## 15. Appendix A — Notation Cheat Sheet

| Symbol | Meaning |
|---|---|
| $T$ | Number of daily observations (≈ 252 for 1y) |
| $p$ | Number of proxy instruments |
| $N_i$ | Signed notional in instrument $i$ (mm USD) |
| $P_{i,t}$ | EOD price (per share for ETFs, per 100 par for others) |
| $r_{i,t}$ | Simple daily return $P_{i,t}/P_{i,t-1} - 1$ |
| $\mathbf{y}$ | Portfolio daily P&L (mm USD) |
| $X$ | $T \times p$ matrix of unit-notional daily P&Ls |
| $\mathbf{h}$ | Vector of hedge notionals (mm USD) |
| $W^{\text{nav}}$ | $5 \times 22$ NAV-weight matrix |
| $W^{\text{risk}}$ | $5 \times 12$ risk-weight matrix |
| $c_j$ | Transaction-cost rate of instrument $j$ (bps) |
| $\Sigma$ | Covariance of P&Ls |
| $\hat\Sigma_{\text{LW}}$ | Ledoit-Wolf shrunk covariance |
| HER | Hedge Effectiveness Ratio = $1 - \text{Var}_{\text{net}}/\text{Var}_{\text{port}}$ |
| TE | Tracking error = $\sigma(\text{PnL}_{\text{net}})$ |
| TC | Total transaction cost (mm USD) |
| $\rho$ | ElasticNet `l1_ratio` |
| $\delta^\star$ | LW shrinkage intensity |
| $S(\rho, \tau)$ | Soft-thresholding operator |

---

## 16. Appendix B — Technique Summary Table

| Technique | Where | Math role |
|---|---|---|
| Simple-return computation | `compute_returns` | Mandated; matches scorer's NAV reconstruction |
| Tikhonov-regularized OLS (ridge) | `fit_ridge_weights` | Closed-form, stable, fast baseline |
| Adaptive λ via tr(XᵀX)/p | same | Scale-invariance |
| LASSO coordinate descent (custom) | `lasso_cd` | Per-feature cost weights → cost-aware sparse fit |
| Soft thresholding | inner CD loop | Convex L1 prox operator |
| ElasticNetCV with l1_ratio bagging | `fit_nav_one_etf` | Sparse + stable + regularization-robust |
| Correlation screening | `fit_nav_one_etf` | Pre-filter noise |
| Time-decay sample weights | `fit_nav_one_etf` | Recency bias |
| TimeSeries 80/20 split | `fit_nav_improved`, `fit_risk_improved` | Honors temporal causality |
| L1-budget normalization | `fit_nav_improved` | Prevents pathological coefficients |
| Variance-floor gating | both improved fits | Skip flat-return targets |
| Two-stage rate→credit residualization | `fit_risk_one_etf` | Clean DV01/CS01 attribution |
| `fit_intercept=False` | all CV fits | Zero-mean returns; saves a DoF |
| Unit-notional P&L linearization | `pnl_per_unit_notional` | Linearizes all hedge objectives |
| Closed-form min-variance hedge | `build_hedge` | $h = -(X^\top X)^{-1} X^\top y$ |
| Cost-weighted LASSO hedge | `build_hedge_costaware` | Sparsity + cost-sensitivity |
| Auto-α via k-th score knee | `build_hedge_costaware` | Self-tuning sparsity ~6 features |
| Ledoit-Wolf shrunk covariance hedge | `build_hedge_shrinkage` | Stable covariance inverse |
| Regime-blended covariance | `build_hedge_regime_blend` | Recent + full sample blend |
| CVaR (two-tailed) objective | `build_hedge_cvar` | Tail-risk minimization |
| Multi-objective HER+TE+cost | `build_hedge_hybrid`, `build_hedge_robust` | Direct scorer matching |
| Vol-mismatch regularizer | `build_hedge_robust` | Avoid over-hedge fragility |
| Multi-start L-BFGS-B | several | Local-minimum guard |
| Box-bounded notionals | several | Sanity / liquidity limit |
| Replication from NAV / risk weights | `build_hedge_replication`, `build_hedge_from_risk` | Re-use of Task 1 / 2 outputs |
| Horse-race candidate ensemble | `_gather_candidates` | Strategy-level model averaging |
| Cascading fallbacks | end of hedge block | Guarantees non-empty submission |
| Drop-tolerance filtering | every hedge builder | Removes nuisance trades |
| Cost-weighted scaling for ElasticNetCV | `build_hedge_costaware` | Encode cost into feature scaling |
| `select` non-blocking stdin read | `_read_stdin_safely` | Robust input handling |

---
