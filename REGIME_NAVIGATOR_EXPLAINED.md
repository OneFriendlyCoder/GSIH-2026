# Regime Navigator — Adaptive Portfolio Construction

A line-by-line, mathematics-heavy walk-through of `Regime_Navigator_Adaptive_Portfolio_Construction.py`, explaining what each block does, the math behind every formula, every quantitative-finance / ML technique used, and *why* each design choice was made.

---

## Table of Contents

1. [Problem in One Sentence](#1-problem-in-one-sentence)
2. [High-Level Architecture](#2-high-level-architecture)
3. [`__init__` — One-Time Setup](#3-__init__--one-time-setup)
4. [`allocate` Step 1 — Portfolio Value & Drawdown Tracking](#4-allocate-step-1--portfolio-value--drawdown-tracking)
5. [Step 2 — Live-Universe Mask](#5-step-2--live-universe-mask)
6. [Step 3 — Regime Detection](#6-step-3--regime-detection)
7. [Step 4 — Multi-Factor Alpha Model](#7-step-4--multi-factor-alpha-model)
8. [Step 4b — Alpha Smoothing, Sector Neutralization](#8-step-4b--alpha-smoothing-sector-neutralization)
9. [Step 5 — Top-40 Asset Selection](#9-step-5--top-40-asset-selection)
10. [Step 6 — Regime/Drawdown-Dependent Parameters](#10-step-6--regimedrawdown-dependent-parameters)
11. [Step 7 — Ledoit-Wolf Covariance Shrinkage](#11-step-7--ledoit-wolf-covariance-shrinkage)
12. [Step 8 — Asset-Specific Transaction-Cost Penalties](#12-step-8--asset-specific-transaction-cost-penalties)
13. [Step 9 — Analytical Mean-Variance QP via KKT + Cholesky](#13-step-9--analytical-mean-variance-qp-via-kkt--cholesky)
14. [Step 9b — Iterative Constraint Projection](#14-step-9b--iterative-constraint-projection)
15. [Step 10–11 — No-Trade Buffer & Net Exposure Normalization](#15-step-1011--no-trade-buffer--net-exposure-normalization)
16. [The Runner Harness](#16-the-runner-harness)
17. [Why This Architecture Works Across Regimes](#17-why-this-architecture-works-across-regimes)
18. [Where the Score Loss Probably Comes From](#18-where-the-score-loss-probably-comes-from)

---

## 1. Problem in One Sentence

At ~40 rebalance dates, output a `(100,)` weight vector that delivers strong **risk-adjusted** returns across 5 hidden market regimes (bull / bear / sideways / crisis / recovery), under hard constraints:

| # | Constraint | Limit |
|---|---|---|
| 1 | Max non-zero positions | ≤ 50 |
| 2 | Max single position | \|wᵢ\| ≤ 0.10 |
| 3 | Sector concentration | Σ\|wᵢ\| ≤ 0.30 per sector |
| 4 | Short limit | total short ≥ −0.30 |
| 5 | Gross exposure | Σ\|wᵢ\| ≤ 1.50 |
| 6 | Net exposure | 0.85 ≤ Σwᵢ ≤ 1.10 |
| 7 | Delisted | wᵢ = 0 if delisted |

The fundamental tension: any strategy tuned to one regime gets crushed in another. The whole code is built around **adapting** to detected regimes rather than committing to a single style.

---

## 2. High-Level Architecture

At every rebalance call, the pipeline is:

```
prices, fund, ind ──► (1)  update portfolio value & drawdown
                  ──► (2)  live-universe mask (drop delisted)
                  ──► (3)  regime detection (stress index → 4 regimes)
                  ──► (4)  cross-sectional alpha (5 factor blocks, regime-weighted)
                  ──► (5)  pick top-40 by alpha
                  ──► (6)  regime/drawdown-dependent risk & turnover params
                  ──► (7)  Ledoit-Wolf covariance on top-40
                  ──► (8)  per-asset transaction-cost vector
                  ──► (9)  closed-form QP via KKT + Cholesky
                  ──► (9b) iterative projection onto inequality constraints
                  ──► (10) no-trade buffer (respects 0.5% min-trade filter)
                  ──► (11) net-exposure rescale → return
```

There is **no persistent ML model** — only persistent **state**: `prev_weights`, `prev_alpha`, `portfolio_value`, `peak_value`, `ema_stress`. Everything else is recomputed each call from scratch.

---

## 3. `__init__` — One-Time Setup

```python
self.all_assets = sorted(prices['asset_id'].unique())
self.asset_to_idx = {a: i for i, a in enumerate(self.all_assets)}
```

A **fixed 100-element ordering** is established once. Every internal vector (alpha, weights, masks) is in this order; the runner remaps to the config's `asset_columns` order only at output time.

```python
latest_fund = fund_sorted.groupby('asset_id').last()
```

For each asset, pull the most recent fundamental snapshot from the **training slice only**, and cache:

- `self.sectors[asset]` — for sector-concentration constraint
- `self.market_caps[asset]` — LARGE / MID / SMALL
- `self.cap_impact[asset]` — 0 / 10 / 25 bps, mirroring the scorer's transaction-cost schedule

This caching is critical because sectors/cap class are looked up in the hot path on every rebalance.

**State variables** are zero-initialized:
- `prev_weights` — needed for the turnover-penalty term in the QP and the no-trade buffer
- `prev_alpha` — for exponential alpha smoothing
- `portfolio_value`, `peak_value` — drawdown tracker
- `ema_stress` — exponentially-smoothed regime indicator

---

## 4. `allocate` Step 1 — Portfolio Value & Drawdown Tracking

Method: `_update_portfolio_value` (lines 228–271).

For each day *t* between the previous rebalance and today, chain daily returns of the **previously-held** weights:

$$r_p(t) = \sum_{i=1}^{100} w_i^{\text{prev}} \cdot \left(\frac{P_i(t)}{P_i(t-1)} - 1\right)$$

$$V(t) = V(t-1)\cdot\bigl(1 + r_p(t)\bigr)$$

$$V_{\text{peak}} = \max\bigl(V_{\text{peak}},\, V(t)\bigr)$$

**Subtlety (lines 238–244):** the code scans the date series for **calendar gaps > 5 days**, which signal that a *new test window* has started. P/V is reset to start from after the gap rather than chaining across discontinuous price histories. The scorer feeds multiple windows through the same instance, so this isolation matters.

Later, **drawdown** (line 172) is used as a circuit-breaker signal:

$$\text{DD} = \frac{V_{\text{peak}} - V}{V_{\text{peak}}}$$

---

## 5. Step 2 — Live-Universe Mask

```python
latest_prices = prices_to_date[prices_to_date['date'] == current_date]
live_assets = set(latest_prices['asset_id'].values)
live_mask = np.array([a in live_assets for a in self.all_assets])
```

An asset is treated as **delisted** if it has no price row for `current_date`. The 100-bps penalty for non-zero weight on a delisted name is the harshest in the table, so dead assets get zeroed throughout the pipeline (alpha set to `−∞`, final weights forced to 0).

---

## 6. Step 3 — Regime Detection

This is the heart of the strategy. There's **no HMM, GMM, or k-means** — instead a hand-engineered **composite stress index** driven by domain knowledge of what historically marks regimes.

### 6.1 Per-indicator normalization (`_compute_stress`, lines 273–283)

Each macro signal is mapped to a [0, 1] "stress score" via a clipped linear ramp. The clip points encode an expert prior on "normal" vs "stressed" levels.

| Indicator | Formula | Reasoning |
|---|---|---|
| Implied vol (VIX-like) | `clip((IV − 12)/25, 0, 1)` | <12 = calm; >37 = crisis |
| HY credit spread | `clip((HY − 200)/600, 0, 1)` | <200 bps = benign; >800 = severe stress |
| Funding stress | `clip(FS, 0, 1)` | already normalized |
| Term spread | `clip(−TS/150, 0, 1)` | **inverted** — yield-curve *inversion* (TS<0) is bearish |
| 20-day realized vol | `clip((RV − 10)/30, 0, 1)` | <10 = quiet; >40 = panic |
| Sentiment | `clip((1 − sent)/2, 0, 1)` | sentiment ∈ [−1, 1]; lower = stress |

### 6.2 Weighted composite

$$\text{stress} = 0.25 \cdot \text{vol}_s + 0.25 \cdot \text{cred}_s + 0.15 \cdot \text{fund}_s + 0.15 \cdot \text{term}_s + 0.10 \cdot \text{rv}_s + 0.10 \cdot \text{sent}_s$$

Vol and credit get the largest weights because they're the most reliable leading indicators of crises (cf. VIX & HY spreads in 2008, March 2020). Term spread gets meaningful weight because yield-curve inversion has predicted nearly every U.S. recession.

### 6.3 Exponential smoothing (EMA)

```python
self.ema_stress = 0.7 * self.ema_stress + 0.3 * stress
```

This is a first-order IIR low-pass filter:

$$s_t = (1-\alpha)\, s_{t-1} + \alpha\, x_t,\quad \alpha = 0.3$$

Half-life:

$$\tau_{1/2} = \frac{\ln 2}{\ln(1/0.7)} \approx 1.94 \text{ rebalances} \approx 10 \text{ trading days}$$

The EMA prevents **whipsaw** — flipping between regimes on a single noisy IV print. It is the *only* mechanism that makes the regime label temporally persistent.

### 6.4 Threshold classification (`_get_regime`)

| stress range | label |
|---|---|
| > 0.65 | crisis |
| 0.38 – 0.65 | bear |
| 0.18 – 0.38 | normal |
| < 0.18 | bull |

Only **four** labels, not five. "Recovery" and "sideways" from the problem statement are handled implicitly: **recovery** = falling EMA stress in the bear/normal band (the EMA produces the smooth transition); **sideways** = normal. This is one of the design's weak points (see §18).

---

## 7. Step 4 — Multi-Factor Alpha Model

`_compute_alpha` (lines 322–428). This is a classical **multi-factor cross-sectional model** in the AQR / Barra / Fama-French tradition. Each asset gets a single composite alpha score that is a **regime-weighted sum of five factor blocks**.

### 7.1 Momentum factor

```python
mom20 = p_now / p_20 - 1
mom60 = p_now / p_60 - 1
```

Two horizons (1 month / 3 months) — medium-term momentum is the most-documented anomaly (Jegadeesh-Titman 1993).

**Risk adjustment** (line 353–354):

$$\text{mom}^{R}_i = \frac{\text{mom}_i}{\sigma_i}$$

where σᵢ is the **20-day annualized realized vol**:

$$\sigma_i = \text{std}\bigl(r_i\bigr)_{20d} \cdot \sqrt{252}$$

Dividing by vol turns raw return into a per-asset Sharpe-like stat — prevents high-vol names that merely *drifted up* from dominating the score.

### 7.2 Quality factor (line 408)

$$f_{\text{qual}} = 0.25\, z_{\text{ROE}} \;-\; 0.20\, z_{\text{D/E}} \;+\; 0.20\, z_{\text{FCF}} \;+\; 0.20\, z_{\text{earn}} \;+\; 0.15\, z_{\text{rev}}$$

Positive weights on profitability, free-cash-flow yield, earnings surprises, revenue growth; **negative** weight on leverage. This is Asness-style "Quality Minus Junk."

### 7.3 Value factor

$$f_{\text{val}} = -0.4\, z_{\text{P/E}} - 0.4\, z_{\text{P/B}} + 0.2\, z_{\text{div}}$$

Negative signs: a *low* P/E or P/B is "cheap" → desirable. Dividend yield is a positive tilt.

### 7.4 Low-vol factor

$$f_{\text{lv}} = -z_{\text{vol}}$$

The low-vol anomaly (Frazzini-Pedersen "Betting Against Beta") — low-vol stocks systematically outperform on a risk-adjusted basis.

### 7.5 Size factor

LARGE → 1, MID → 2, SMALL → 3, then z-scored. Higher z ⇒ smaller cap ⇒ captures the Fama-French SMB premium.

### 7.6 Robust z-scoring (`_robust_zscore`, lines 306–320)

Instead of $(x - \mu)/\sigma$, MAD-based z-scores:

$$\tilde{z}_i = \frac{x_i - \text{median}(x)}{1.4826 \cdot \text{MAD}(x)}, \qquad \text{MAD} = \text{median}\bigl(|x - \text{median}(x)|\bigr)$$

The factor `1.4826` makes MAD an unbiased estimator of σ when *x* is Gaussian. Then winsorize:

$$z \leftarrow \text{clip}(z, -3, 3)$$

**Why MAD instead of std?** MAD has a 50% breakdown point (vs std's 0%). A single outlier P/E ratio of 8000 from a near-zero-earnings company would otherwise destroy the entire value score. MAD shrugs it off.

### 7.7 Regime-dependent factor blending (lines 414–424)

| Regime | momentum | quality | value | low-vol | size |
|---|---|---|---|---|---|
| crisis | 0.05 | 0.40 | 0.15 | **0.35** | 0.05 |
| bear | 0.15 | 0.30 | 0.25 | 0.25 | 0.05 |
| normal | 0.40 | 0.15 | 0.20 | 0.10 | 0.15 |
| bull | **0.45** | 0.10 | 0.15 | 0.05 | 0.25 |

The intuition is empirically motivated:
- **Crisis / Bear:** load on Quality + Low-vol — defensive characteristics. Profitable, low-leverage, low-volatility names outperform in down markets.
- **Normal:** balanced — momentum dominant but everything contributes.
- **Bull:** Momentum + Size — risk-on regimes reward trend-following and small-cap exposure.

Final composite alpha:

$$\alpha_i = w_{\text{mom}}\, f_{\text{mom},i} + w_{\text{qual}}\, f_{\text{qual},i} + w_{\text{val}}\, f_{\text{val},i} + w_{\text{lv}}\, f_{\text{lv},i} + w_{\text{size}}\, f_{\text{size},i}$$

### 7.8 Look-ahead bias prevention (line 359)

```python
valid_fund = fundamentals[fundamentals['report_date'] <= current_dt]
```

Strict temporal filter on fundamentals. Combined with the 45-day reporting lag the problem statement embeds, this enforces causality — no peeking at unreleased earnings.

---

## 8. Step 4b — Alpha Smoothing, Sector Neutralization

### 8.1 Exponential alpha smoothing (line 139)

$$\alpha_t \leftarrow 0.65\, \alpha_t^{\text{raw}} + 0.35\, \alpha_{t-1}$$

Lowers turnover that would otherwise come from noisy z-scores oscillating. With 35% memory, the half-life is ≈ 1.6 rebalances ≈ 8 trading days.

### 8.2 Sector-neutral demeaning (lines 142–150)

For each sector S, subtract the sector-mean alpha (computed over **live** assets only):

$$\alpha_i \leftarrow \alpha_i - \overline{\alpha}_{\text{sector}(i)}$$

After this, every sector has mean-zero alpha within the live universe → no sector is structurally favored, the sector-concentration constraint binds less often, and the optimizer expresses bets *within* sectors rather than *across* them.

### 8.3 Dead-asset masking (lines 153, 426–427)

Set `alpha[~live_mask] = -∞`, so `argpartition` never picks them.

---

## 9. Step 5 — Top-40 Asset Selection

```python
n_select = min(40, n_live)
top_idx = np.argpartition(alpha, -n_select)[-n_select:]
top_idx = top_idx[np.argsort(alpha[top_idx])][::-1]
```

`argpartition` finds the top-40 in **O(n)** instead of O(n log n) — but more importantly, the rationale for 40 (not 50):

- The constraint cap is 50 non-zero positions, with a 25-bp penalty if exceeded.
- Leaving **headroom** means the constraint projection can drop a couple of positions due to sector/position-limit clipping without ever hitting the cap.
- 40 is small enough to keep position sizes meaningful (mean weight ≈ 2.5%) but large enough to diversify.

---

## 10. Step 6 — Regime / Drawdown-Dependent Parameters

```python
lam, gamma_base, target_net = self._get_regime_params(regime)
```

Three knobs returned by `_get_regime_params` (lines 295–304):

| Regime | λ (risk aversion) | γ (turnover penalty) | target net |
|---|---|---|---|
| crisis | 4.0 | 1.5 | 0.90 |
| bear | 2.5 | 1.0 | 0.95 |
| normal | 1.0 | 0.4 | 1.00 |
| bull | 0.6 | 0.3 | 1.05 |

Interpretation:
- **λ** scales the variance penalty in the QP — higher λ → more risk-averse → more diversification, less concentration on best-alpha names.
- **γ** scales the turnover penalty — higher γ → trade less when costs/uncertainty are high.
- **target_net** sets gross-market exposure — defensive in crisis, slight leverage in bull.

### 10.1 Drawdown circuit breaker (lines 172–178)

```python
if drawdown > 0.12:
    lam *= 2.0
    target_net = min(target_net, 0.90)
if drawdown > 0.20:
    lam *= 2.0
    target_net = 0.85
```

A **state-dependent overlay** on the regime parameters. Even if the macro regime says "normal," if the portfolio itself is bleeding, automatically:
1. Double-down on risk aversion → more diversified positions.
2. Cut net exposure → less market beta.

This directly targets the **Drawdown Control** and **Calmar Ratio** components of the scorecard.

---

## 11. Step 7 — Ledoit-Wolf Covariance Shrinkage

`_compute_covariance` (lines 430–471).

### 11.1 The problem with sample covariance

For *n* assets and *T* observations, sample covariance:

$$\hat{\Sigma}_{\text{sample}} = \frac{1}{T-1}\sum_{t=1}^{T}(r_t - \bar{r})(r_t - \bar{r})^\top$$

With 40 assets and only 60 days, T ≈ 1.5 n. The smallest eigenvalues are *severely* biased toward zero. A mean-variance optimizer then *loves* these directions — they look like "free risk-reduction" — and the resulting portfolio is dominated by estimation error. Michaud (1989) famously called naive MV "error maximization."

### 11.2 Ledoit-Wolf shrinkage

Take a convex combination of the noisy sample with a structured shrinkage target *F*:

$$\hat{\Sigma}_{\text{LW}} = \delta^\star F + (1 - \delta^\star)\, \hat{\Sigma}_{\text{sample}}$$

In `scikit-learn`'s implementation, $F = \mu I$ with $\mu = \text{tr}(\hat{\Sigma})/n$. The optimal shrinkage intensity $\delta^\star \in [0,1]$ minimizes the expected Frobenius error to the true Σ:

$$\delta^\star = \min\left(1,\; \max\left(0,\; \frac{\pi - \rho}{T \cdot \gamma^2}\right)\right)$$

where π is the asymptotic variance of the sample-cov entries, ρ is the asymptotic covariance with the target, and γ² is the squared bias of the target. The full derivation is in Ledoit-Wolf 2004; sklearn computes the closed form.

**Effect:** the smallest eigenvalues are pulled up, the largest pulled down → conditioning of Σ improves dramatically → the QP solution becomes stable.

### 11.3 Why a 60-day window?

```python
if len(rets) > 60:
    rets = rets.iloc[-60:]
```

A trade-off:
- **Too short** (<30 d): noisy, low DoF.
- **Too long** (>120 d): stale, misses recent regime shifts.

60 trading days ≈ 3 months — empirically a sweet spot for monthly-to-quarterly horizons.

### 11.4 Fallbacks (lines 446–460)

If LW fails or there are too few observations, fall back to `np.cov` with diagonal ridge `+1e-4·I`, or ultimately to `0.04·I` (4% annualized variance — a plausible prior).

---

## 12. Step 8 — Asset-Specific Transaction-Cost Penalties

```python
gamma_vec = np.array([
    gamma_base * (1.0 + self.cap_impact.get(self.all_assets[i], 0.0) / 20.0)
    for i in top_idx
])
```

The scorer charges transaction costs as:

$$c_i = 20\,\text{bps} + \text{cap\_impact}_i \cdot \text{turnover}_i$$

with cap impact = 0 / 10 / 25 bps for LARGE / MID / SMALL. The code mirrors this by scaling the per-asset γ:

$$\gamma_i = \gamma_{\text{base}} \cdot \Bigl(1 + \frac{\text{cap\_impact}_i}{20}\Bigr)$$

So a small-cap gets 2.25× the turnover penalty of a large-cap. The QP will therefore *trade large-caps in preference to small-caps*, all else equal — matching the actual cost structure.

---

## 13. Step 9 — Analytical Mean-Variance QP via KKT + Cholesky

`_optimize_weights` (lines 473–530). The most mathematically dense block. This is the core optimization.

### 13.1 The QP formulation

The unconstrained problem (with equality on net exposure):

$$\max_w \quad \alpha^\top w \;-\; \frac{\lambda}{2}\, w^\top \Sigma\, w \;-\; \frac{1}{2}\sum_i \gamma_i (w_i - w_i^{\text{prev}})^2$$

$$\text{s.t.} \quad \mathbf{1}^\top w = \text{target\_net}$$

The third term is the **quadratic transaction-cost penalty** (turnover penalized in L2 norm).

Expand the cost term:

$$\frac{1}{2}\sum_i \gamma_i (w_i - w_i^{\text{prev}})^2 = \frac{1}{2} w^\top \Gamma w - w^\top \Gamma w^{\text{prev}} + \text{const}$$

with $\Gamma = \text{diag}(\gamma_i)$.

The negative of the objective (so we minimize):

$$\min_w \quad \frac{1}{2} w^\top \underbrace{(\lambda \Sigma + \Gamma)}_{A} w \;-\; \underbrace{(\alpha + \Gamma w^{\text{prev}})}_{b}{}^\top w$$

$$\text{s.t.} \quad \mathbf{1}^\top w = \text{target\_net}$$

### 13.2 KKT conditions

Lagrangian:

$$\mathcal{L}(w, \mu) = \tfrac{1}{2} w^\top A w - b^\top w + \mu\,(\mathbf{1}^\top w - n_{\text{target}})$$

Stationarity:

$$\frac{\partial \mathcal{L}}{\partial w} = A w - b + \mu \mathbf{1} = 0 \;\Longrightarrow\; w = A^{-1}(b - \mu \mathbf{1})$$

Decompose:

$$w = \underbrace{A^{-1} b}_{w_0} - \mu \underbrace{A^{-1} \mathbf{1}}_{v}$$

Enforce $\mathbf{1}^\top w = n_{\text{target}}$:

$$\mathbf{1}^\top w_0 - \mu\, \mathbf{1}^\top v = n_{\text{target}}$$

$$\mu = \frac{\mathbf{1}^\top w_0 - n_{\text{target}}}{\mathbf{1}^\top v}$$

(The code uses `mu = (target_net - sum_w0) / sum_v` and `w = w0 + mu*v`, which is algebraically equivalent up to a sign convention.)

### 13.3 Cholesky factorization

$A = \lambda \Sigma + \Gamma + 10^{-5} I$ is symmetric positive-definite (LW Σ is PSD, Γ is positive-diagonal, plus tiny ridge for safety). Hence factor:

$$A = L L^\top$$

with *L* lower-triangular. To compute $w_0 = A^{-1} b$:

1. Forward-solve $L y_1 = b$ — O(n²)
2. Back-solve $L^\top w_0 = y_1$ — O(n²)

And the same with `1` in place of `b` gives *v*. Total cost: **O(n³)** for the factorization + 2× O(n²) for the solves, n = 40 → negligible (sub-millisecond).

**Why not just `np.linalg.solve(A, B)` once with both RHS stacked?** The Cholesky decomposition is reused — that's the *whole point* of the analytical formulation. The author wanted to avoid iterative numerical optimizers (cvxpy etc., which aren't even in the HackerRank image).

### 13.4 Falling back when A isn't PD (line 487)

```python
except np.linalg.LinAlgError:
    A += 1e-3 * np.eye(n)
    L = np.linalg.cholesky(A)
```

If Cholesky fails (numerical near-singularity), inflate the diagonal — a Tikhonov-style ridge — and retry.

---

## 14. Step 9b — Iterative Constraint Projection

The analytical KKT solution only enforces the **equality** `Σwᵢ = target_net`. Box bounds (0 ≤ wᵢ ≤ 0.10), sector caps, etc., are **inequality** constraints handled by **iterative proportional fitting** (a Dykstra-style projection):

```python
for _ in range(5):
    w = np.clip(w, 0.0, 0.10)         # box

    for sector, positions in sector_map.items():
        sec_sum = np.sum(w[positions])
        if sec_sum > 0.30:
            w[positions] *= 0.30 / sec_sum   # sector cap

    net = np.sum(w)
    if net > 1e-10 and abs(net - target_net) > 1e-6:
        w *= target_net / net                 # restore net

    w = np.clip(w, 0.0, 0.10)         # box again
```

Each iteration:
1. Clip to per-asset box [0, 0.10].
2. For any sector summing above 0.30, scale that sector down proportionally.
3. Rescale total to target net.
4. Re-clip (rescaling may have pushed names above 0.10).

This is a classical **alternating projection**. Each projection is a *contraction*, so the iterates converge — 5 iterations is empirically plenty.

**Observation:** the code forces `np.clip(w, 0.0, 0.10)` — i.e. **long-only**. The QP never produces shorts. The 30% short budget given by the problem is left unused. This is a deliberate simplification but possibly a source of un-captured alpha (see §18).

---

## 15. Step 10–11 — No-Trade Buffer & Net Exposure Normalization

### 15.1 No-trade buffer (`_apply_buffer`, lines 532–538)

```python
diff = np.abs(w - prev_w)
mask = diff < 0.005
w_new[mask] = prev_w[mask]
```

The backtester ignores trades smaller than 0.5%. Anything below that threshold becomes pure cost without execution → revert to previous weight, save the transaction fee.

This step directly targets the **Turnover Efficiency** scoring component.

### 15.2 Final net-exposure rescale (lines 212–219)

A safety net: even after the QP & projection, the buffer may push net out of [0.85, 1.10]. The final rescale guarantees compliance:

```python
if net < 0.85:  w *= 0.85 / net
elif net > 1.10: w *= 1.10 / net
```

Constraint #6 satisfied → no 25-bp penalty.

---

## 16. The Runner Harness

`main()` (lines 544–633) handles two input formats:

1. **Evaluation format**: `===CONFIG=== ... ===PRICES=== ... ===FUND=== ... ===IND===`. JSON config, CSV bodies.
2. **Local backtester format**: blank-line-separated sections.

Loops over `rebalance_dates`, calls `allocate`, and prints CSV with 8-decimal precision. If `asset_columns` is provided in config, weights are remapped to that exact order on output.

---

## 17. Why This Architecture Works Across Regimes

| Scoring component | How the code targets it |
|---|---|
| **Sharpe** | Risk-adjusted momentum + LW covariance + λ scaling |
| **Sortino** | Quality/low-vol tilt during stress reduces downside vol |
| **Calmar** | Drawdown circuit breaker scales exposure when V/V_peak < 0.88 |
| **Drawdown Control** | Same circuit breaker + defensive regime tilts |
| **Alpha Stability** | EMA on stress + EMA on alpha + regime-conditional factor weights |
| **Turnover Efficiency** | Quadratic γ-penalty in QP + 0.5% no-trade buffer + alpha smoothing |
| **Regime Adaptivity** | Whole design is regime-conditional |
| **Implementation Quality** | Long docstring; explicit use of fund/ind; explicit regime logic; no date literals |

The crucial **anti-overfitting** discipline:
- No date-keyed conditionals → passes anti-cheat scan.
- Regime detection is **macro-driven** (stress index), not price-driven on the test sample → generalizes.
- Robust z-scores (MAD) → resistant to outliers from new windows.
- LW shrinkage → resistant to estimation error in unseen covariance structure.

---

## 18. Where the Score Loss Probably Comes From

The recorded score is **44.15 / 100**. Plausible drag sources:

1. **Long-only by construction.** The optimizer clips at 0; the 30% short budget is wasted. In bear/crisis regimes, allowing modest shorts on the bottom-alpha names could materially help.
2. **Only 4 regimes; recovery is implicit.** A separate "recovery" tilt (post-crisis, falling stress) that re-engages momentum/size aggressively would likely improve Calmar/Sharpe.
3. **Fixed factor weights per regime.** Hand-tuned numbers (0.45 momentum in bull, etc.) may not transfer. A regime-conditional **ridge regression** of forward returns on factor exposures (fit on training slice only) would be more principled.
4. **Covariance horizon fixed at 60 days.** A short EMA covariance (e.g. RiskMetrics λ ≈ 0.94) would adapt faster in regime transitions.
5. **Top-K = 40 is hard-coded.** In a crisis with low signal dispersion, top-20 might give a better Sharpe via concentration; in bull markets, top-50 (at the cap) might harvest more breadth.
6. **No explicit risk-parity or hierarchical-risk-parity overlay.** Pure mean-variance with a single λ is known to underperform diversification-first methods on hidden test data because alpha forecasts have low signal-to-noise.
7. **Mean-reversion not used.** Only momentum is on the price side. A short-term reversal factor (5-day) would help in choppy/sideways regimes where momentum reverses.

The framework is sound; the parameter table and the long-only constraint are the most likely culprits behind the gap between 44 and a top score.

---

## Appendix A — Summary of ML / Quant Techniques in the File

| Technique | Where | Purpose |
|---|---|---|
| EMA (IIR low-pass) | `ema_stress`, `prev_alpha` blend | Reduce regime/alpha whipsaw |
| Composite stress index | `_compute_stress` | Detect macro regime |
| Threshold classification | `_get_regime` | Map stress → 4 labels |
| Robust z-score (MAD) | `_robust_zscore` | Outlier-resistant cross-sectional scoring |
| Winsorization | `np.clip(z, -3, 3)` | Cap tail influence |
| Sector neutralization | demean by sector | Remove industry tilt bias |
| Multi-factor model | `_compute_alpha` | Combine momentum/value/quality/lowvol/size |
| Regime-conditional weights | factor blending table | Adapt to detected regime |
| Ledoit-Wolf shrinkage | `_compute_covariance` | Stable cov for MV optimization |
| Mean-variance QP | `_optimize_weights` | Trade off return vs risk vs turnover |
| KKT analytical solution | μ = (n_target − Σw₀)/Σv | Closed-form equality constraint |
| Cholesky factorization | `np.linalg.cholesky(A)` | Fast PD linear solve |
| Tikhonov ridge | `A += 1e-3·I` fallback | Numerical stabilization |
| Iterative proportional fitting | 5-iter projection loop | Inequality-constraint satisfaction |
| Drawdown circuit breaker | DD-conditional λ/net scaling | Capital preservation |
| No-trade buffer | `_apply_buffer` | Match scorer's min-trade filter |

---

## Appendix B — Notation Cheat Sheet

| Symbol | Meaning |
|---|---|
| α ∈ ℝⁿ | Cross-sectional alpha vector |
| Σ ∈ ℝⁿˣⁿ | Asset return covariance (LW-shrunk) |
| Γ = diag(γᵢ) | Per-asset turnover penalty |
| λ | Risk-aversion (regime-dependent) |
| w, wᵖʳᵉᵛ | Current / previous weight vector |
| **1** | Vector of ones |
| n_target | Desired net exposure (regime-dependent) |
| s_t | EMA of stress index |
| DD | Drawdown = (V_peak − V) / V_peak |
| zᵢ | Robust z-score (MAD-based) |
| MAD | Median absolute deviation |
| 1.4826 | 1 / Φ⁻¹(3/4), MAD→σ consistency constant |

---
