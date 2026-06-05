# ETF Proxy Models — NAV, Risk & Hedging: Code Flow Report

This report walks through `ETF_Proxy_Models_NAV_Risk_and_Hedging.py` end-to-end. It explains **what problem the script is solving**, **the overall execution flow**, and then **each function** — what it does, what mathematical/statistical method it uses, and how that method maps onto the problem.

---

## 0. TL;DR — What ML Method Is Actually Used

The whole solution is built on **regularized linear regression** (no trees, no neural nets). The model that actually runs at submission time for each task is:

| Task | ML method that actually runs | One-line description |
|------|------------------------------|----------------------|
| **NAV model** | **Ridge regression** (L2) with a **hold-out validation grid-search for the penalty α** | `fit_nav_improved`: regress each ETF's daily returns on *all* proxy returns; pick α by lowest validation MSE on a chronological 80/20 split; then zero tiny weights and L1-normalize so the weights form a replication portfolio. |
| **Risk model** | **Ridge regression** (L2) with **hold-out validation grid-search for α**, restricted to macro factors (futures + CDX) | `fit_risk_improved`: same ridge-with-validation recipe, but the feature set is only Treasury futures + CDX (no Proxy ETFs), and weights are kept as raw betas (no normalization). |
| **Hedge** | **Minimum-variance / cost-penalized regression, selected from an ensemble bake-off** | ~18 candidate hedges are generated, each scored by `HER − 2·cost`; the single best-scoring one is submitted. In practice the winner is almost always a **min-variance optimizer with a cost penalty** (`build_hedge_optimized`, an L-BFGS-B variance-minimizer). |

**Plain-English version:**

- **NAV → Ridge regression.** "Express each target ETF as a regularized linear blend of proxy returns; choose the regularization strength by validation; normalize to a replication basket."
- **Risk → Ridge regression on macro factors only.** "Same idea, but only rates (Treasury futures) and credit (CDX) are allowed as explanatory factors."
- **Hedge → Minimum-variance hedge with a transaction-cost penalty.** "Pick notionals that minimize the variance of (portfolio PnL + hedge PnL) while penalizing expensive trades; chosen as the best of several optimizers."

> **About the "ensemble":** The ensemble only exists for the **hedge** task — it is a *selection* mechanism (run many optimizers, keep the one with the best measured Hedge Effectiveness Ratio minus cost), **not** a blended model. There is **no ensemble for NAV or Risk** — those each run a single Ridge model (with a plain Ridge fallback if it misbehaves). `fit_nav_one_etf`'s ElasticNet ensemble exists in the file but is **not** on the executed path (`fit_nav_improved` is used instead). So if you only remember three things: **NAV = Ridge, Risk = Ridge, Hedge = cost-penalized min-variance optimizer chosen by a bake-off.**

The remaining sections give the full detail; everything is a variation or fallback around these three core methods.

---

## 1. The Problem Being Solved

We are given **End-Of-Day (EOD) prices** for:

- **5 Target ETFs** (`Target_ETF_1 … Target_ETF_5`) — the instruments we care about.
- A universe of **proxy instruments** we are allowed to trade/model with:
  - 10 **Proxy ETFs** (`Proxy_ETF_1 … Proxy_ETF_10`)
  - 4 **Treasury Futures** (`TU, FV, TY, US`)
  - 4 **Treasury On-the-run Bonds** (`UST_2Y, UST_5Y, UST_10Y, UST_30Y`)
  - 4 **CDS Indices** (`CDX_IG_5Y, CDX_IG_10Y, CDX_HY_5Y, CDX_HY_10Y`)

The script must produce **three deliverables** (three pandas DataFrames serialized to CSV inside a JSON blob):

| Task | Output variable | What it is | Allowed instruments |
|------|----------------|------------|---------------------|
| **1. NAV estimation** | `nav_model` | A weight matrix mapping each target ETF's daily return onto proxy returns, so the ETF's NAV path can be *reconstructed/replicated*. | **ALL** proxies (`ALL_PROXY`) |
| **2. Risk estimation** | `risk_model` | A weight matrix expressing each target ETF's risk in terms of **macro risk factors** only. | **RISK_PROXY** = futures + bonds + CDX (no Proxy ETFs) |
| **3. Hedge creation** | `hedging_basket` | A vector of **notionals** in proxy instruments that, when held against a given `PORTFOLIO` of target ETFs, minimizes residual PnL variance net of transaction costs. | All hedge instruments |

The economic intuition:

- **NAV model** = *replication*: explain the ETF's return as a linear combination of proxy returns (regression of returns on returns). Quality is judged by how closely the reconstructed NAV tracks the true NAV (MAPE).
- **Risk model** = *factor decomposition*: explain the ETF's risk using only tradeable macro hedging factors (rates via futures/bonds, credit via CDX), deliberately excluding Proxy ETFs which would "cheat" by being near-perfect substitutes.
- **Hedge** = *minimum-variance hedge*: choose notionals so portfolio PnL + hedge PnL has the smallest variance, while paying as little transaction cost as possible. Measured by the **Hedge Effectiveness Ratio (HER)**.

---

## 2. Top-Level Execution Flow

The script runs top-to-bottom as a module (there is no `main()`). The flow is:

```
1. Imports + load constants
   └─ EOD_PRICES = read CSV
   └─ Define instrument universes (TARGET_ETFS, PROXY_ETFS, ...) and COST_BPS

2. Define helper functions (provided boilerplate)
   └─ compute_returns, predict_returns, reconstruct_nav, compute_mape,
      compute_pnl_series, compute_cost, compute_her

3. Read PORTFOLIO from stdin (JSON) or fall back to a default portfolio

4. Define all model-fitting + hedge-building functions

5. RETURNS = compute_returns(EOD_PRICES)        ← single source of returns

6. FIT NAV MODEL    (with fallbacks):
   nav_weights = fit_nav_improved(...)
       └─ if too sparse → fit_proxy_model(ridge)
       └─ on exception  → fit_proxy_model(ridge)

7. FIT RISK MODEL   (with fallbacks):
   risk_weights = fit_risk_improved(...)
       └─ if too sparse → fit_proxy_model(ridge, RISK_PROXY)
       └─ on exception  → fit_proxy_model(ridge, RISK_PROXY)

8. BUILD HEDGE (the "ensemble / bake-off"):
   cands = _gather_candidates(...)              ← runs ~18 hedge strategies
   score each candidate with _hedge_score (HER − cost penalty)
   pick the best-scoring candidate
       └─ fallbacks: hybrid → costaware → ridge → liquid-ridge → empty

9. ASSEMBLE OUTPUTS:
   nav_model       = nav_weights
   risk_model      = risk_weights
   hedging_basket  = hedge_notionals.to_frame("Notional")

10. submission = {"nav":..., "risk":..., "hedge":...}  → print(json.dumps(...))
```

The defining design choice of this solution: **for the hedge, it does not commit to one method.** It runs a whole *zoo* of hedging strategies, scores each against the actual objective (HER minus cost), and keeps the winner. The NAV and Risk models, by contrast, use one primary method each with a robust ridge fallback.

---

## 3. Constants & Setup (lines 1–56)

- `EOD_PRICES` — the price panel (dates × instruments), sorted by date.
- Instrument lists define which columns belong to which asset class.
- `ALL_PROXY` (NAV universe) = everything; `RISK_PROXY` = everything **except Proxy ETFs**.
- `PAR_QUOTED` — set of instruments quoted per 100 par notional (futures, bonds, CDX). ETFs are quoted as NAV-per-share. This distinction matters everywhere PnL/notional scaling happens.
- `COST_BPS` — one-way transaction cost in basis points on market value: Proxy ETFs 2.0, futures 0.5, bonds 0.5, CDX 1.0. Cheaper instruments (futures/bonds) are favoured by the cost-aware models.

---

## 4. Helper Functions (provided boilerplate, lines 71–117)

These are the **evaluation primitives** — the same math the grader uses.

### `compute_returns(prices)`
Simple daily returns `prices.pct_change().dropna()`. The single transformation from price levels to return space; every model below is fit in return space.

### `predict_returns(weights, proxy_returns)`
Matrix product `proxy_returns @ weights`. Given a weight vector/matrix, predict the target return as a linear combination of proxy returns. This is the *linear factor model* assumption made throughout.

### `reconstruct_nav(weights, proxy_returns, base_nav)`
Chains predicted returns into a NAV path: `(1 + pred_returns).cumprod() * base_nav`. This is how a return-space model is turned back into a price/NAV level for scoring.

### `compute_mape(predicted_nav, actual_nav)`
Mean Absolute Percentage Error between reconstructed and true NAV. The **NAV task's accuracy metric** — lower is better. Inner-joins on dates and averages `|pred − actual| / actual`.

### `compute_pnl_series(notionals, prices)`
PnL of a basket over time: `PnL_t = Σ_i scaling_i · (P_i,t − P_i,t−1)`. Scaling converts a notional into "units": for ETFs `1/P_0` (notional is market value), for par-quoted instruments `1/100`. This is the engine that turns a notionals dict into a daily PnL series.

### `compute_cost(notionals, prices)`
One-way setup cost of the hedge basket at t=0. For ETFs `|N|·bps/1e4`; for par instruments `(|N|/100)·P_0·bps/1e4`. Used to penalize expensive baskets.

### `compute_her(pnl_port, pnl_hedge)`
**Hedge Effectiveness Ratio** = `1 − Var(port + hedge) / Var(port)`. HER=1 means the hedge perfectly removed all variance; HER=0 means it did nothing; negative means it made things worse. This is the **hedge task's objective**, and the score function maximizes it.

---

## 5. Input Handling (lines 128–158)

### `_read_stdin_safely(timeout=1.0)`
Non-blocking stdin read using `select.select`. Returns `""` if stdin is a TTY or nothing is ready within the timeout. This lets the grader pipe in a JSON test case without the script hanging when run interactively.

### Portfolio resolution
Parse stdin JSON into `TEST_INPUT`. If it contains a non-empty `"portfolio"` dict, use it (coercing to floats); otherwise use `_DEFAULT_PORTFOLIO` (a long/short mix: +150, −120, +180, −100, +130). `PORTFOLIO` is the target book the hedge must neutralize.

---

## 6. Core Estimators (the building blocks)

### `fit_ridge_weights(target_returns, proxy_returns, alpha=1e-5)` — lines 161–168
**Method: Ridge (Tikhonov-regularized) least squares, closed form.**
Aligns y (target) and X (proxies), then solves `(XᵀX + λI) w = Xᵀy` where `λ = alpha · trace(XᵀX)/n_features`. Scaling λ by the average eigenvalue makes the regularization strength **scale-invariant**. Ridge is used because proxy returns are highly collinear (many ETFs/factors move together), which makes plain OLS unstable; the `λI` term keeps `XᵀX` invertible and shrinks weights. This is the **universal fallback** estimator used throughout the file.

### `lasso_cd(X, y, alpha, weights=None, ...)` — lines 171–208
**Method: LASSO via coordinate descent with per-feature (weighted) penalties.**
A hand-rolled L1-regularized regression:
- Standardizes columns (`Xn = X / std`) so the penalty is comparable across instruments.
- Iterates coordinate-by-coordinate, applying the **soft-thresholding** operator: for each feature j it computes `rho = Xnⱼ · residual`, then sets `wⱼ = soft(rho, alpha·weightⱼ·n)/colSq`. Values inside the threshold band are driven exactly to **zero** → sparsity.
- `weights[j]` lets each instrument carry a different penalty (used to penalize expensive instruments more — *cost-aware sparsity*).
- Converges when the max coefficient change falls below `tol`. Returns coefficients **un-standardized** (`w / X_std`).
Why LASSO: it produces *sparse* hedges (few instruments), which is realistic and cheaper to trade.

---

## 7. NAV Model Functions (Task 1)

### `fit_proxy_model(...)` — lines 211–227
**Method: per-ETF ridge regression** assembled into a weight matrix.
Builds a zero matrix `W` (ETFs × proxies), then for each ETF regresses its returns on the available proxy returns via `fit_ridge_weights` and writes the row. Wrapped in try/except per ETF so one bad fit can't kill the whole matrix. This is the **simple, dependable baseline** for both NAV and Risk (used as fallback).

### `fit_nav_one_etf(etf, returns, proxy_list, corr_thresh=0.09, low_var_ratio=0.35)` — lines 282–326
**Method: correlation feature-screening → time-weighted ElasticNet ensemble → ridge fallback, with a long-only-ish sign constraint.** This is the *primary* per-ETF NAV estimator. Step by step:
1. **Guard:** need ≥25 aligned observations.
2. **Low-variance skip:** if the ETF's return variance is tiny vs. the average proxy variance (`y_var < 0.35 · proxy_avg_var`), return zeros — a near-constant ETF shouldn't be force-fit (avoids overfitting noise).
3. **Feature screening:** keep proxies whose absolute correlation with the ETF exceeds `corr_thresh=0.09`; if fewer than 4 survive, take the top-7 by correlation. This reduces dimensionality and collinearity before fitting.
4. **Time weighting:** exponential sample weights `exp(linspace(−1.8, 0))` give **recent observations more influence** (markets are non-stationary).
5. **ElasticNet ensemble:** if scikit-learn is available, fit `ElasticNetCV` at three L1 ratios (0.4, 0.65, 0.85) — i.e. three different blends of L1/L2 — each with 5-fold CV over a log-spaced alpha grid. Take the **median** coefficient across the three fits (an ensemble that's robust to any single hyperparameter choice).
6. **Fallback:** if sklearn is absent or all coefficients vanish, use `fit_ridge_weights`.
7. **Soft sign constraint:** `coef = max(coef, −0.05·|coef|)` clips strongly *negative* weights — replication weights should be mostly positive (an ETF is a positive blend of its constituents), so large shorts are suppressed.

### `fit_nav_model_all(...)` — lines 329–337
Loops `fit_nav_one_etf` over all target ETFs into the `W` matrix (try/except per ETF). The "all-ETF" wrapper for the primary NAV method.

### `fit_nav_improved(target_etfs, proxy_list, returns, alpha_range, mag_thresh=0.01)` — lines 585–632
**Method: ridge with a hold-out validation grid search + magnitude thresholding + L1 normalization.** This is the **actual NAV method chosen at runtime** (tried first). Per ETF:
1. Need ≥30 aligned rows.
2. **Chronological 80/20 split** (train = first 80%, validation = last 20%) — respects time ordering, no leakage.
3. **Alpha selection:** sweep `alpha_range = logspace(−6, 1, 50)`; for each, solve ridge on train and measure **validation MSE**; keep the alpha with lowest val MSE. (Precomputes `XᵀX_tr`, `Xᵀy_tr` once for speed.)
4. **Refit** on the full sample with the best alpha.
5. **Sparsify:** zero out weights with `|w| ≤ mag_thresh = 0.01`.
6. **Normalize:** if the surviving weights sum (in absolute value) to > 0.1, divide by that sum so weights sum to 1 in L1 — turning the fit into a **replication portfolio** (weights that look like portfolio proportions, improving NAV tracking and interpretability).

> NAV selection logic (lines 953–961): run `fit_nav_improved`; if it produced fewer than 5 non-zero weights total (looks broken), fall back to `fit_proxy_model` ridge; any exception → same ridge fallback.

---

## 8. Risk Model Functions (Task 2)

The risk model deliberately uses **only macro factors** (no Proxy ETFs), and separates **rates** (futures/bonds) from **credit** (CDX).

### `fit_risk_one_etf(etf, returns, rate_cols, credit_cols, all_cols)` — lines 340–386
**Method: two-stage orthogonal regression (rates first, then credit on the residual), each via RidgeCV.**
1. Use futures + CDX columns.
2. **Stage 1 — rates:** regress ETF return on **Treasury futures** with `RidgeCV` (CV over `logspace(−5,2,30)`), or ridge fallback. Store futures coefficients. Compute the **residual** `y_resid = y − X_fut·fut_coef`.
3. **Stage 2 — credit:** regress the *residual* on **CDX** with RidgeCV/ridge. Store credit coefficients.
This **sequential / Frisch–Waugh-style** decomposition assigns the rates-explainable variance to rate instruments first, then lets credit explain only what's left — giving a cleaner factor attribution than throwing everything into one regression (where collinearity between rates and credit would muddy the split).

### `fit_risk_model_all(...)` — lines 389–400
Loops `fit_risk_one_etf` into the RISK_PROXY weight matrix.

### `fit_risk_improved(target_etfs, returns, low_var_ratio=0.20)` — lines 635–678
**Method: single ridge with hold-out validation alpha search** over futures + CDX. This is the **risk method chosen at runtime**. Mirrors `fit_nav_improved` but:
- Uses only `TSY_FUTURES + CDX`.
- Low-variance skip at ratio 0.20.
- 80/20 chronological split, alpha swept over `logspace(−8, −2, 40)` (much smaller alphas → closer to OLS, because here we want an accurate factor sensitivity, not a sparse replication).
- No magnitude threshold, no normalization (risk loadings are betas, not portfolio weights).

### `fit_risk_model_simple(target_etfs, returns)` — lines 681–711
**Method: plain RidgeCV (with intercept) over the full RISK_PROXY set.** A simpler one-stage alternative (defined but not the primary path; another ridge fallback flavour).

> Risk selection logic (lines 963–971): `fit_risk_improved`; if < 4 non-zero weights → `fit_proxy_model` ridge on RISK_PROXY; on exception → same.

---

## 8.5. HEDGING — DEEP DIVE (Problem, Solution Method, and the Ensemble)

This section answers four things in detail: **(A) what problem the hedge solves, (B) how it is solved mathematically, (C) how the ensemble works and why it is used, and (D) how the pieces are integrated end-to-end.**

### A. What problem is being solved?

We are given a **portfolio of target ETFs** — the `PORTFOLIO` dict, e.g. `{Target_ETF_1: +150, Target_ETF_2: −120, …}` (notionals in mm USD; positive = long, negative = short). This book has **PnL risk**: every day its value moves up and down as ETF prices change.

The job of the hedge is to choose a **basket of notionals in proxy instruments** (`hedging_basket`) such that, when held *alongside* the portfolio, the **combined day-to-day PnL barely moves**. Formally we want to minimize the variance of the combined PnL stream while not spending too much on transaction costs.

Define:

- `y_port,t` = the portfolio's daily PnL on day *t* (a time series).
- `h` = the vector of hedge notionals we are solving for (one number per hedge instrument).
- `X` = the matrix of **per-unit-notional daily PnL** of each hedge instrument (`pnl_per_unit_notional`), so `X·h` is the hedge's daily PnL series.
- Combined PnL = `y_port + X·h`.

The objective the grader cares about is the **Hedge Effectiveness Ratio**:

```
HER = 1 − Var(y_port + X·h) / Var(y_port)
```

- `HER = 1` → the hedge cancelled *all* portfolio variance (perfect hedge).
- `HER = 0` → the hedge did nothing.
- `HER < 0` → the hedge made risk *worse*.

But a perfect in-sample hedge can be expensive and over-fit, so we also subtract a **transaction-cost penalty**. The real selection objective used in the code (`_hedge_score`) is:

```
score = HER − 2 · transaction_cost
```

So the hedging problem is precisely: **find notionals `h` that maximize variance reduction (HER) per dollar of trading cost.** This is the classic **minimum-variance hedge** problem from portfolio risk management, with a cost regularizer bolted on.

### B. How is it solved? (the core math)

Every individual hedge builder is some way of solving "pick `h` to make `y_port + X·h` small/stable." There are three mathematical families used:

1. **Closed-form minimum-variance regression (the analytic core).**
   Minimizing `Var(y_port + X·h)` over `h` is just a least-squares regression of `−y_port` on `X`. The ridge-regularized solution is:
   ```
   h = −(XᵀX + λI)⁻¹ Xᵀ y_port
   ```
   The `λI` term (Tikhonov regularization) keeps `XᵀX` invertible when hedge instruments are collinear and stops the notionals from blowing up. This is the analytic starting point used inside almost every builder. The minus sign means: take a position *opposite* to the portfolio's replicated exposure.

2. **Numerical optimization (L-BFGS-B) of a richer objective.**
   When the objective is more than plain variance — e.g. it also penalizes cost, tracking error, tail risk, or volatility mismatch — there's no clean formula, so the code calls `scipy.optimize.minimize(method="L-BFGS-B")` with **box bounds** on each notional (`±max_notional`) and usually **several restart points** to avoid local minima. Examples:
   - `build_hedge_optimized`: minimize `Var(y_port+Xh)/Var(y_port) + cost_weight·Σ|h|·cost_bps`.
   - `build_hedge_cvar`: minimize the **tails** of the combined PnL (CVaR-style) instead of variance.
   - `build_hedge_robust` / `build_hedge_hybrid`: minimize a weighted blend of `−HER`, tracking error, cost, and (for robust) a volatility-matching term.

3. **Sparse / shrinkage / cost-aware variants of the regression.**
   - **Cost-aware LASSO / ElasticNet** (`build_hedge`, `build_hedge_costaware`): add an L1 penalty whose strength is proportional to each instrument's `COST_BPS`, so expensive instruments need a stronger signal to enter — yielding a **small, cheap basket**.
   - **Ledoit-Wolf shrinkage** (`build_hedge_shrinkage`, `build_hedge_regime_blend`): estimate a *shrunk* covariance matrix and solve the textbook min-variance hedge `h = −Σ_hh⁻¹ Σ_ph`; more stable when there are few observations.
   - **Analytic replication** (`build_hedge_replication`, `build_hedge_from_risk`): skip re-optimizing and instead convert the already-fitted **NAV or Risk weight matrix** straight into offsetting notionals.

All of them share the same plumbing (Section 9): build `y_port`, build `X = pnl_per_unit_notional`, solve for `h`, drop tiny positions, return a `Series` of notionals. They differ only in **what objective they optimize and how aggressively they regularize/limit cost.**

### C. How does the ensemble work, and why is it used?

**Why an ensemble at all?** There is no single hedge method that is best for every portfolio and every market regime:

- A pure min-variance hedge gives the highest in-sample HER but can use many instruments and rack up cost (hurting `HER − 2·cost`).
- A cost-aware sparse hedge trades a little HER for much lower cost.
- A shrinkage/regime hedge is more robust when the covariance is noisy or the recent regime differs from history.
- A tail-risk (CVaR) hedge protects against large adverse moves rather than average wobble.
- An analytic NAV/risk-replication hedge can be very clean when the factor models fit well.

Which of these wins depends on the *specific* `PORTFOLIO` passed in at runtime. Rather than guess, the code uses a **"generate-and-select" ensemble** (a bake-off): run *all* the strategies, **measure each one against the true objective**, and submit the best. This is an ensemble in the sense of *combining many models* — but the combination rule is **"argmax of the score,"** not averaging. It is essentially **model selection by cross-checking on the real metric.**

**The mechanism, concretely:**

1. **Generate** — `_gather_candidates(PORTFOLIO, nav_weights, EOD_PRICES)` runs ~18 distinct hedge builders (different methods × different universes × different tunings), each wrapped in its own `try/except` so a failure just drops that candidate. Each surviving result is stored as a `(name, notionals_series)` pair. The variety spans: optimized min-variance (loose/tight/liquid), robust, shrinkage (full/liquid), regime-blend (full/liquid), CVaR (full/liquid/tight), cost-aware, NAV-replication, risk-model hedge (full/liquid), and max-effort (full/liquid).

2. **Score** — every candidate is graded by `_hedge_score`, which **reconstructs the actual combined PnL** from the unit-PnL matrix, computes `HER = compute_her(y_port, hedge_pnl)` and `cost = compute_cost(...)`, and returns `HER − 2·cost`. Crucially, this uses the *same* evaluation primitives the grader uses, so the in-house score is a faithful proxy for the real score. Degenerate/empty baskets get `−1e18` so they can never win.

3. **Select** — sort candidates by score descending and take `scored[0][1]` — the notionals of the single best candidate. **No weighting or blending of baskets happens**; exactly one strategy's output is submitted.

```
cands  = _gather_candidates(PORTFOLIO, nav_weights, EOD_PRICES)   # ~18 baskets
scored = [(name, h, HER − 2·cost) for name, h in cands]          # grade each
scored.sort(by score, desc)
hedge_notionals = scored[0][1]                                    # keep the winner
```

So the "ensemble" is a **competition judged by the real objective**, which is why it reliably does at least as well as any single method it contains: the worst case is that the best single method wins, and often a method you wouldn't have hand-picked turns out best for that particular book.

### D. How is it all integrated (end-to-end wiring)?

The hedge stage is not isolated — it consumes the outputs of the NAV and Risk tasks and is wrapped in layered fallbacks:

```
nav_weights  (from fit_nav_improved / ridge)   ─┐
risk_weights (from fit_risk_improved / ridge)  ─┤
PORTFOLIO    (from stdin or default)           ─┤
EOD_PRICES                                     ─┘
        │
        ▼
_gather_candidates(PORTFOLIO, nav_weights, EOD_PRICES)
        │   • build_hedge_replication      USES nav_weights   ← NAV model feeds the hedge
        │   • build_hedge_from_risk        USES risk_weights  ← Risk model feeds the hedge
        │   • build_hedge_optimized / robust / shrinkage / regime / cvar / costaware / max_effort
        ▼
score each with _hedge_score  (HER − 2·cost, via compute_her + compute_cost)
        ▼
pick best  ──► hedge_notionals
        │
        ▼  (fallback chain if the bake-off returns nothing usable)
   build_hedge_hybrid ──► build_hedge_costaware ──► build_hedge(ridge)
        │   outer except ──► build_hedge(LIQUID, ridge) ──► empty Series
        ▼
hedging_basket = hedge_notionals.to_frame("Notional")   (index "Instrument")
```

Two integration points are worth calling out:

- **The NAV and Risk models are reused as hedge candidates.** `build_hedge_replication` turns the NAV weight matrix into offsetting notionals, and `build_hedge_from_risk` does the same with the risk weight matrix. So all three tasks are linked: better NAV/Risk fits give the bake-off stronger candidates to choose from.
- **Defense in depth.** If `_gather_candidates` yields nothing (e.g. scipy/sklearn missing, or all optimizers throw), the code falls back to `build_hedge_hybrid`, then `build_hedge_costaware`, then a plain ridge `build_hedge`, then a liquid-only ridge hedge, and finally an empty basket. A valid submission is *always* produced.

**One-sentence summary:** the hedge solves a **cost-penalized minimum-variance hedging problem**, attacks it with a dozen-plus regression/optimization methods, and uses a **select-the-best ensemble** — judged by the real `HER − 2·cost` metric — to submit whichever single basket actually performs best for the given portfolio.

---

## 9. Hedge Construction Functions (Task 3)

All hedge builders share a common skeleton:
1. Drop zero positions from `portfolio`.
2. Compute **unit PnL** via `pnl_per_unit_notional` — the per-unit-notional daily PnL of each instrument.
3. Build the **portfolio PnL series** `y_port = Σ portfolio_i · unitPnL_i`.
4. Solve for hedge notionals `h` that offset `y_port`.
5. **Drop tiny positions** (`drop_tol_mm`) and return the non-zero notionals as a `Series`.

The hedge identity throughout: combined PnL = `y_port + X·h`, and we want `Var(y_port + X·h)` minimized. Note the **sign convention** — most builders solve for the regression `h` and return `-h` (you short the replicating exposure to hedge it).

### `pnl_per_unit_notional(prices)` — lines 230–234
Per-instrument daily PnL for **one unit of notional**: ETFs scaled by `1/P_0`, par instruments by `1/100`. Returns a DataFrame `dates × instruments`. The shared feature matrix `X` for every hedge regression/optimization below.

### `build_hedge(portfolio, hedge_instruments, prices, method="lasso", ...)` — lines 237–270
**Method: cost-weighted LASSO (default) or ridge min-variance hedge.**
- Forms `y_port` and `X = unit_pnl[hedge_cols]`.
- **LASSO path:** sets per-feature penalty `cost_w = COST_BPS`. If `alpha` is None, auto-tunes it: standardize X, compute a relevance **score** `|Xnᵀy| / (T·cost_w)` per instrument, sort descending, and pick the **7th-best score (k=6)** as alpha — i.e. roughly target ~6 active instruments. Solve `lasso_cd`, return `h = −w`.
- **Ridge path:** closed-form min-variance `h = −(XᵀX + λI)⁻¹Xᵀy`.
The cost-weighted penalty means **expensive instruments need stronger signal** to enter the basket.

### `build_hedge_costaware(portfolio, hedge_instruments, prices, ...)` — lines 403–438
**Method: ElasticNetCV on cost-scaled features, with ridge fallback.**
Scales `X_scaled = X / cost_w` so the penalty effectively charges more for high-cost instruments, then fits `ElasticNetCV` over an L1-ratio grid `[0.1…0.9]` and alpha grid `logspace(−6,1,40)` with 5-fold CV. Un-scales the coefficients (`v / cost_w`) and returns `−w_orig`. Falls back to ridge min-variance if sklearn is unavailable or fails. This is the **workhorse seed** used by several other builders.

### `build_hedge_from_risk(portfolio, risk_w, prices, ...)` — lines 441–464
**Method: analytic hedge derived from the fitted risk weights.**
Rather than re-optimizing, it *translates* the risk-model betas directly into notionals: for each ETF position `n_etf` and each risk factor j with loading `w_ij`, add `−n_etf · w_ij / scale` to instrument j (where `scale = P_0/100` for par instruments, else 1, converting a return-beta into a notional). This produces a hedge that neutralizes exactly the **factor exposures** the risk model identified — a model-driven hedge rather than a data-driven optimization.

### `build_hedge_replication(portfolio, nav_w, prices, ...)` — lines 517–540
**Method: same analytic translation, but from the NAV weights.** Identical mechanics to `build_hedge_from_risk` but driven by `nav_w`. Since the NAV model *replicates* the ETF, the negative of that replication is a natural hedge. This is the `"rep"` candidate.

### `build_hedge_max_effort(portfolio, hedge_instruments, prices, max_notional=10000, ...)` — lines 467–514
**Method: numerical min-variance optimization (L-BFGS-B) with multiple restarts.**
- Computes an analytic min-variance start `h_minvar = −(XᵀX+λI)⁻¹Xᵀy` with a *tiny* λ (1e-9 → essentially OLS, maximal variance reduction).
- Objective = relative residual variance `Var(y_port + X·h)/Var(y_port)`.
- Runs `scipy.optimize.minimize` (L-BFGS-B, box bounds ±10000) from **four starting points** (the analytic solution, zeros, 1.3×, 0.7×) and keeps the best. Multi-start guards against local minima.
"Max effort" = chase the lowest possible variance with little regard for cost.

### `build_hedge_hybrid(portfolio, hedge_instruments, prices, her_w, te_w, cost_w_mult, ...)` — lines 543–582
**Method: cost-aware seed → constrained numerical optimization of a multi-term objective.**
1. **Seed** with `build_hedge_costaware` (so the optimizer only works over instruments that already matter). If the seed has ≤2 names, just return a costaware hedge.
2. Optimize over only the seed's instruments with objective
   `−her_w·HER + te_w·TrackingError + cost_w_mult·Cost`,
   i.e. **maximize HER, minimize tracking-error (std of combined PnL), minimize cost**, with tunable weights (2.2 / 9.5 / 0.009). Box bounds ±1200, L-BFGS-B.
This is a balanced objective that mirrors the real evaluation (HER vs cost) more directly than pure variance.

### `build_hedge_shrinkage(portfolio, hedge_instruments, prices, ...)` — lines 714–747
**Method: Ledoit-Wolf shrinkage covariance → analytic hedge.**
Stacks `[y | X]`, estimates a **shrunk covariance** with `LedoitWolf` (robust when T is small relative to the number of instruments), extracts `Σ_ph` (port-hedge covariances) and `Σ_hh` (hedge-hedge covariance), and solves the classic minimum-variance hedge `h = −Σ_hh⁻¹ Σ_ph`. Ridge fallback if sklearn is unavailable. Shrinkage stabilizes the covariance inverse, preventing the wild positions a raw sample covariance can produce.

### `build_hedge_regime_blend(portfolio, hedge_instruments, prices, short_window=60, blend_weight=0.6, ...)` — lines 750–784
**Method: blend of full-sample and recent-window shrinkage covariances.**
Computes Ledoit-Wolf covariance on the **full history** and on the **last 60 days**, then blends `cov = 0.6·recent + 0.4·full` before solving the min-variance hedge. This adapts the hedge toward the **current regime** while retaining long-run stability — a bias/variance compromise across time.

### `build_hedge_cvar(portfolio, hedge_instruments, prices, alpha=0.05, ...)` — lines 787–842
**Method: tail-risk (CVaR-style) optimization via L-BFGS-B.**
Instead of variance, the objective targets the **tails**: sort the combined PnL, take the worst `n_tail` and best `n_tail` observations, and minimize `−mean(left) + mean(right)` — i.e. shrink both downside and upside tails (symmetric tail compression). Starts from the analytic min-variance solution and optimizes from three restarts. Useful when the concern is large adverse moves, not average wobble.

### `build_hedge_robust(portfolio, hedge_instruments, prices, ...)` — lines 845–900
**Method: costaware seed → multi-term objective including a volatility-matching term.**
Like `hybrid`, but the objective adds a **`vol_mismatch`** penalty `|TE/std_port − 1|` (5.5 weight) on top of `−3.2·HER + 12·TE + 0.006·Cost`, and sizes the notional bound to the portfolio (`max(1200, 1.9·port_size)`). The vol-matching term discourages over-hedging (driving residual vol to zero in-sample, which often overfits) and aims for a residual whose volatility is sensible. Three restarts, L-BFGS-B.

### `build_hedge_optimized(portfolio, hedge_instruments, prices, max_notional=3000, cost_weight=1e-5, ...)` — lines 903–950
**Method: min-variance + linear cost penalty, single L-BFGS-B run.**
Objective = `Var(y_port+X·h)/Var(y_port) + cost_weight·Σ|h|·cost_w`. A clean, fast variance-plus-cost optimizer started from the analytic min-variance solution. This is the most-used candidate generator (`_gather_candidates` calls it three times with different `max_notional`/`drop_tol`/`cost_weight` settings).

---

## 10. The Hedge "Bake-Off": Candidate Generation, Scoring, Selection

### `_nav_relevant_instruments(portfolio, nav_w)` — lines 977–985
Returns the hedge instruments that actually carry a non-zero NAV weight for the held ETFs — a helper to restrict hedges to instruments the NAV model deems relevant (defined for completeness).

### `_hedge_score(notionals_series, portfolio, prices, cost_pen=2.0)` — lines 988–1005
**The objective function used to rank candidates.**
Reconstructs the portfolio PnL and hedge PnL from unit PnL, computes **HER** via `compute_her`, computes **transaction cost** via `compute_cost`, and returns `HER − cost_pen·cost`. Empty/degenerate baskets score `−1e18`. This directly encodes the trade-off "remove the most variance per dollar of cost."

### `_gather_candidates(portfolio, nav_w, prices)` — lines 1008–1142

> **Terminology used in the table below (important):**
> - **Builder** = the `build_hedge_*` *function* — i.e. the *method/algorithm* used to compute a hedge (min-variance, shrinkage, CVaR, replication, …). It answers *"how is the hedge computed."*
> - **Universe** = *which instruments the builder may use.* Two universes exist: **full** = `HEDGE_INSTRUMENTS` (all proxies: Proxy ETFs + futures + bonds + CDX), and **liquid** = `HEDGE_LIQUID` = `TSY_FUTURES + CDX` only (cheap, liquid macro instruments). It answers *"which menu of instruments can enter the basket."*
> - **Tuning** = the *parameter values* passed for that run — knobs like `max_notional` (per-position cap), `drop_tol_mm` (drop positions smaller than this), `cost_weight`/`alpha` (cost or tail aggressiveness). It answers *"how aggressive vs conservative the settings are."*
> - **Candidate** = one concrete *hedge basket* (a `Series` of notionals) = the **output of running one builder × one universe × one tuning**. Each gets a short name. This is why the *same builder* appears multiple times — e.g. `build_hedge_optimized` is run three ways (`opt_full`, `opt_tight`, `opt_liquid`), producing three different candidates.
>
> *Analogy:* builder = recipe, universe = allowed ingredients, tuning = oven temperature/time, candidate = the dish that comes out. The bake-off then tastes every dish (`_hedge_score`) and serves the best one.

**Runs the entire strategy zoo** and collects every non-empty result as a named candidate. It produces ~18 candidates by calling, each in its own try/except:

| Candidate | Builder | Universe / tuning |
|-----------|---------|-------------------|
| `rep` | `build_hedge_replication` | from NAV weights |
| `opt_full` | `build_hedge_optimized` | all instruments, loose, cheap-cost |
| `opt_tight` | `build_hedge_optimized` | all, tighter notional & higher cost |
| `opt_liquid` | `build_hedge_optimized` | liquid only (futures+CDX) |
| `robust` | `build_hedge_robust` | all |
| `shrink_full` / `shrink_liquid` | `build_hedge_shrinkage` | all / liquid |
| `regime_full` / `regime_liquid` | `build_hedge_regime_blend` | all / liquid |
| `cvar_full` / `cvar_liquid` / `cvar_tight` | `build_hedge_cvar` | varying α & universe |
| `costaware` | `build_hedge_costaware` | all |
| `risk_hedge` / `risk_hedge_liquid` | `build_hedge_from_risk` | from risk weights (all / liquid-filtered) |
| `max_effort` / `max_effort_liquid` | `build_hedge_max_effort` | all / liquid |

The point: different methods win for different portfolios and market regimes, so generate many and let the score decide.

### Final hedge selection (lines 1145–1168)
```
cands = _gather_candidates(...)
scored = [(name, h, _hedge_score(h, ...)) for name, h in cands]
scored.sort(by score desc)
hedge_notionals = best candidate
# fallbacks if empty:
#   build_hedge_hybrid → build_hedge_costaware → build_hedge(ridge)
# outer except: build_hedge(LIQUID, ridge) → empty Series
```
The winner is the candidate with the **highest `HER − 2·cost`**. Layered fallbacks guarantee *some* valid basket is always produced, even if sklearn/scipy are missing or every optimizer throws.

---

## 11. Output Assembly (lines 1180–1207)

- `nav_model = nav_weights`, `risk_model = risk_weights` (index named `"ETF"`).
- `hedging_basket` — the chosen notionals as a one-column `"Notional"` DataFrame, index named `"Instrument"`.
- `submission = {"nav": ..., "risk": ..., "hedge": ...}` with each DataFrame serialized via `.to_csv()`.
- `print(json.dumps(submission))` — the grader reads this single JSON line from stdout.

---

## 12. Summary — Methods → Problems Map

| Sub-problem | Core difficulty | Method chosen in the code | Why it fits |
|-------------|-----------------|---------------------------|-------------|
| NAV replication | Collinear proxies, non-stationarity, overfit risk | Ridge with hold-out alpha search + magnitude threshold + L1 normalization (`fit_nav_improved`); primary alt: correlation-screened, time-weighted ElasticNet ensemble | Regularization tames collinearity; validation picks the right shrinkage; normalization yields a clean replication portfolio that tracks NAV (low MAPE) |
| Risk decomposition | Must use only macro factors; separate rates vs credit | Two-stage orthogonal RidgeCV (rates → residual → credit), or single ridge w/ validation (`fit_risk_improved`) | Sequential regression gives clean factor attribution; ridge stabilizes betas |
| Hedging | Maximize HER while minimizing cost; robustness | A scored **ensemble** of ~18 hedge builders (min-var, cost-aware ElasticNet/LASSO, Ledoit-Wolf shrinkage, regime blend, CVaR tail, robust vol-matching, analytic risk/NAV replication), pick best `HER − 2·cost` | No single method dominates across portfolios/regimes; bake-off + the actual objective as the selector is the most direct route to a high score, with deep fallbacks for reliability |

**Cross-cutting techniques the code leans on repeatedly:**
- **Ridge regularization** as the stable default (closed form `(XᵀX+λI)⁻¹Xᵀy` with scale-invariant λ).
- **Cost-aware penalization** (penalty ∝ `COST_BPS`) so the basket prefers cheap futures/bonds.
- **Chronological train/validation splits** for honest hyperparameter selection (no look-ahead).
- **Ensembling / multi-start optimization** to avoid single-model and local-minimum fragility.
- **Defensive try/except + fallbacks** everywhere, so missing libraries or numerical failures degrade gracefully rather than crashing the submission.
