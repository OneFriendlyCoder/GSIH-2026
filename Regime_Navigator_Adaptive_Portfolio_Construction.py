# Score : 44.15
"""
STRATEGY EXPLANATION
====================
Core approach: Regime-adaptive multi-factor alpha model with robust 
mean-variance optimization and drawdown-controlled dynamic exposure. 
The strategy combines cross-sectional factor investing with macro 
regime detection to dynamically adjust risk appetite and factor 
exposures based on market stress conditions.

Signals used:
- Momentum: 20-day and 60-day risk-adjusted price momentum (return/volatility)
- Quality: ROE, debt-to-equity, free cash flow yield, earnings surprise 
  from fundamentals
- Value: P/E, P/B (inverse z-scores), dividend yield
- Low-volatility: 20-day realized volatility from price data
- Size: Market capitalization class (LARGE/MID/SMALL)
- Macro regime: Composite stress index from impl_vol_index, credit_spread_hy, 
  funding_stress, term_spread, realized_vol_20d, and sentiment_score

Regime detection: A composite stress index is computed from normalized 
macro indicators using exponential smoothing. Four regimes are identified: 
Crisis (stress > 0.65), Bear (0.38-0.65), Normal (0.18-0.38), and Bull (< 0.18). 
Regime-dependent factor weights tilt the portfolio toward defensive factors 
(quality, low-vol) during stress and toward momentum/size in bull markets.

Portfolio construction:
1. Score all live assets using the regime-adaptive factor model with 
   robust cross-sectional z-scores (median/MAD normalization, winsorized to [-3,3])
2. Select top 40 assets by composite score (buffer below 50 limit)
3. Compute robust covariance via Ledoit-Wolf shrinkage on 60-day returns
4. Solve a mean-variance quadratic program with asset-specific transaction 
   cost penalties via analytical KKT conditions + Cholesky decomposition
5. Project onto constraints (position limits, sector limits, net exposure) 
   via iterative proportional fitting
6. Apply drawdown circuit breaker: if portfolio drawdown exceeds 12%, 
   scale exposure down and increase defensive tilt

Key design decisions:
- Analytical QP solution (Cholesky + KKT) avoids iterative numerical 
  optimizers, ensuring sub-millisecond portfolio construction
- Asset-specific turnover penalties account for SMALL/MID cap impact costs
- 60-day covariance lookback balances responsiveness with estimation stability
- Alpha exponential decay (35% memory) reduces turnover from signal noise
- 0.5% trade buffer aligns with backtester minimum trade filter
- Delisted assets are automatically excluded via live-universe filtering
- Fundamentals strictly filtered by report_date to prevent look-ahead bias
"""

import json
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.covariance import LedoitWolf


class PortfolioArchitect:
    def __init__(self, prices: pd.DataFrame,
                 fundamentals: pd.DataFrame,
                 indicators: pd.DataFrame):
        self.all_assets = sorted(prices['asset_id'].unique())
        self.n_assets = len(self.all_assets)
        self.asset_to_idx = {a: i for i, a in enumerate(self.all_assets)}

        # Precompute static attributes from latest training fundamentals
        fund_sorted = fundamentals.sort_values('report_date')
        latest_fund = fund_sorted.groupby('asset_id').last()

        self.sectors = {}
        self.market_caps = {}
        self.cap_impact = {}  # transaction cost impact by cap class (bps)

        for asset in self.all_assets:
            if asset in latest_fund.index:
                self.sectors[asset] = str(latest_fund.loc[asset, 'sector'])
                cap = str(latest_fund.loc[asset, 'market_cap_class'])
                self.market_caps[asset] = cap
                if cap == 'SMALL':
                    self.cap_impact[asset] = 25.0
                elif cap == 'MID':
                    self.cap_impact[asset] = 10.0
                else:
                    self.cap_impact[asset] = 0.0
            else:
                self.sectors[asset] = 'S01'
                self.market_caps[asset] = 'MID'
                self.cap_impact[asset] = 10.0

        # State variables (persist across rebalance calls)
        self.prev_weights = np.zeros(self.n_assets)
        self.prev_alpha = np.zeros(self.n_assets)
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.ema_stress = 0.0

    def allocate(self,
                 prices_to_date: pd.DataFrame,
                 fundamentals_to_date: pd.DataFrame,
                 indicators_to_date: pd.DataFrame,
                 current_date: str) -> np.ndarray:
        current_dt = pd.to_datetime(current_date)

        # ------------------------------------------------------------------
        # 1. Update portfolio value & drawdown tracking
        # ------------------------------------------------------------------
        self._update_portfolio_value(prices_to_date, current_date)

        # ------------------------------------------------------------------
        # 2. Identify live (non-delisted) assets
        # ------------------------------------------------------------------
        latest_prices = prices_to_date[prices_to_date['date'] == current_date]
        live_assets = set(latest_prices['asset_id'].values)
        live_mask = np.array([a in live_assets for a in self.all_assets])

        if not np.any(live_mask):
            self.prev_weights = np.zeros(self.n_assets)
            return self.prev_weights

        # ------------------------------------------------------------------
        # 3. Regime detection via composite stress index
        # ------------------------------------------------------------------
        ind_row = indicators_to_date[indicators_to_date['date'] == current_date]
        if len(ind_row) == 0:
            regime = 'normal'
        else:
            stress = self._compute_stress(ind_row.iloc[0])
            self.ema_stress = 0.7 * self.ema_stress + 0.3 * stress
            regime = self._get_regime(self.ema_stress)

        # ------------------------------------------------------------------
        # 4. Compute cross-sectional alpha scores
        # ------------------------------------------------------------------
        alpha = self._compute_alpha(
            prices_to_date, fundamentals_to_date, current_dt,
            regime, live_assets
        )

        # Exponential alpha smoothing to reduce turnover from noise
        alpha = 0.65 * alpha + 0.35 * self.prev_alpha
        self.prev_alpha = alpha.copy()

        # Sector-neutral demeaning (only among live assets to avoid dead-asset bias)
        sector_vals = list(set(self.sectors.values()))
        for sec in sector_vals:
            sec_mask = np.array([
                self.sectors.get(a, 'UNK') == sec for a in self.all_assets
            ])
            live_sec_mask = sec_mask & live_mask
            if np.sum(live_sec_mask) > 0:
                alpha[sec_mask] -= np.mean(alpha[live_sec_mask])

        # Zero out dead assets
        alpha[~live_mask] = -np.inf

        # ------------------------------------------------------------------
        # 5. Select top 40 assets (headroom below 50 limit)
        # ------------------------------------------------------------------
        n_live = int(np.sum(live_mask))
        n_select = min(40, n_live)
        if n_select == 0:
            self.prev_weights = np.zeros(self.n_assets)
            return self.prev_weights

        top_idx = np.argpartition(alpha, -n_select)[-n_select:]
        top_idx = top_idx[np.argsort(alpha[top_idx])][::-1]

        # ------------------------------------------------------------------
        # 6. Regime & drawdown dependent parameters
        # ------------------------------------------------------------------
        lam, gamma_base, target_net = self._get_regime_params(regime)

        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        if drawdown > 0.12:
            lam *= 2.0
            target_net = min(target_net, 0.90)
        if drawdown > 0.20:
            lam *= 2.0
            target_net = 0.85

        # ------------------------------------------------------------------
        # 7. Robust covariance estimation (Ledoit-Wolf shrinkage)
        # ------------------------------------------------------------------
        cov = self._compute_covariance(prices_to_date, top_idx)

        # ------------------------------------------------------------------
        # 8. Asset-specific transaction cost penalties
        # ------------------------------------------------------------------
        gamma_vec = np.array([
            gamma_base * (1.0 + self.cap_impact.get(self.all_assets[i], 0.0) / 20.0)
            for i in top_idx
        ])

        # ------------------------------------------------------------------
        # 9. Analytical QP + constraint projection
        # ------------------------------------------------------------------
        w_full = self._optimize_weights(
            alpha, cov, self.prev_weights, top_idx,
            lam, gamma_vec, target_net
        )

        # ------------------------------------------------------------------
        # 10. Turnover buffer (respect 0.5% minimum trade filter)
        # ------------------------------------------------------------------
        w_full = self._apply_buffer(w_full, self.prev_weights)

        # Ensure dead assets are zero
        w_full[~live_mask] = 0.0

        # ------------------------------------------------------------------
        # 11. Final net exposure normalization
        # ------------------------------------------------------------------
        net = np.sum(w_full)
        if net > 1e-10:
            if net < 0.85:
                w_full = w_full * (0.85 / net)
            elif net > 1.10:
                w_full = w_full * (1.10 / net)
        else:
            w_full = np.zeros(self.n_assets)

        self.prev_weights = w_full.copy()
        return w_full

    # =====================================================================
    # Helper Methods
    # =====================================================================

    def _update_portfolio_value(self, prices_to_date, current_date):
        """Track approximate portfolio value using held-asset returns."""
        all_dates = sorted(prices_to_date['date'].unique())
        if current_date not in all_dates:
            return

        all_dates_dt = pd.to_datetime(all_dates)
        current_idx = all_dates.index(current_date)

        # Detect window gaps (>5 calendar days) to avoid crossing gaps
        if current_idx > 0:
            gaps = [(all_dates_dt[i] - all_dates_dt[i - 1]).days for i in range(1, len(all_dates_dt))]
            recent_gaps = [i - 1 for i, g in enumerate(gaps, 1) if g > 5 and i - 1 < current_idx]
            window_start = (recent_gaps[-1] + 1) if recent_gaps else 0
        else:
            window_start = 0

        start_idx = max(window_start, current_idx - 4)
        period_dates = all_dates[start_idx:current_idx + 1]

        if len(period_dates) < 2:
            return

        period_prices = prices_to_date[prices_to_date['date'].isin(period_dates)]
        pivoted = period_prices.pivot(index='date', columns='asset_id', values='close')

        for d in range(1, len(period_dates)):
            prev_date = period_dates[d - 1]
            curr_date = period_dates[d]
            if prev_date not in pivoted.index or curr_date not in pivoted.index:
                continue

            prev_p = pivoted.loc[prev_date]
            curr_p = pivoted.loc[curr_date]

            day_ret = np.zeros(self.n_assets)
            for i, asset in enumerate(self.all_assets):
                if asset in prev_p.index and asset in curr_p.index and prev_p[asset] > 0:
                    day_ret[i] = curr_p[asset] / prev_p[asset] - 1.0

            port_ret = np.sum(self.prev_weights * day_ret)
            self.portfolio_value *= (1.0 + port_ret)

        self.peak_value = max(self.peak_value, self.portfolio_value)

    def _compute_stress(self, ind):
        """Composite stress index in [0,1]."""
        vol_s = np.clip((ind['impl_vol_index'] - 12.0) / 25.0, 0.0, 1.0)
        cred_s = np.clip((ind['credit_spread_hy'] - 200.0) / 600.0, 0.0, 1.0)
        fund_s = np.clip(ind['funding_stress'], 0.0, 1.0)
        term_s = np.clip(-ind['term_spread'] / 150.0, 0.0, 1.0)
        rv_s = np.clip((ind['realized_vol_20d'] - 10.0) / 30.0, 0.0, 1.0)
        sent_s = np.clip((1.0 - ind['sentiment_score']) / 2.0, 0.0, 1.0)

        return (0.25 * vol_s + 0.25 * cred_s + 0.15 * fund_s +
                0.15 * term_s + 0.10 * rv_s + 0.10 * sent_s)

    def _get_regime(self, stress):
        if stress > 0.65:
            return 'crisis'
        elif stress > 0.38:
            return 'bear'
        elif stress > 0.18:
            return 'normal'
        else:
            return 'bull'

    def _get_regime_params(self, regime):
        """Returns (lambda_risk, gamma_turnover_base, target_net_exposure)."""
        if regime == 'crisis':
            return 4.0, 1.5, 0.90
        elif regime == 'bear':
            return 2.5, 1.0, 0.95
        elif regime == 'normal':
            return 1.0, 0.4, 1.00
        else:  # bull
            return 0.6, 0.3, 1.05

    def _robust_zscore(self, x):
        """Median/MAD cross-sectional z-score with winsorization."""
        x = np.array(x, dtype=float)
        mask = np.isfinite(x)
        if np.sum(mask) < 5:
            return np.zeros_like(x)
        xm = x[mask]
        med = np.median(xm)
        mad = np.median(np.abs(xm - med))
        if mad < 1e-12:
            return np.zeros_like(x)
        z = (x - med) / (mad * 1.4826)
        z = np.clip(z, -3.0, 3.0)
        z[~mask] = 0.0
        return z

    def _compute_alpha(self, prices, fundamentals, current_dt, regime, live_assets):
        # Price pivot
        price_pivot = prices.pivot(index='date', columns='asset_id', values='close')
        price_pivot = price_pivot.reindex(columns=self.all_assets)

        # Momentum
        if len(price_pivot) >= 60:
            p_now = price_pivot.iloc[-1].values
            p_20 = price_pivot.iloc[-20].values
            p_60 = price_pivot.iloc[-60].values
        else:
            p_now = price_pivot.iloc[-1].values
            p_20 = price_pivot.iloc[0].values
            p_60 = price_pivot.iloc[0].values

        p_20_safe = np.where(p_20 == 0, np.nan, p_20)
        p_60_safe = np.where(p_60 == 0, np.nan, p_60)

        mom20 = (p_now / p_20_safe) - 1.0
        mom60 = (p_now / p_60_safe) - 1.0

        # Realized volatility (annualized)
        if len(price_pivot) >= 20:
            rets = price_pivot.pct_change().iloc[-20:]
            vol = rets.std().values * np.sqrt(252)
        else:
            vol = np.full(self.n_assets, 0.20)

        vol = np.clip(vol, 0.01, 2.0)

        # Risk-adjusted momentum
        mom20_r = mom20 / vol
        mom60_r = mom60 / vol

        # Fundamentals (strict temporal filtering)
        fundamentals = fundamentals.copy()
        fundamentals['report_date'] = pd.to_datetime(fundamentals['report_date'])
        valid_fund = fundamentals[fundamentals['report_date'] <= current_dt]

        if len(valid_fund) > 0:
            latest_fund = valid_fund.sort_values('report_date').groupby('asset_id').last()
        else:
            latest_fund = pd.DataFrame(index=[])

        # Extract fundamental arrays
        pe = np.full(self.n_assets, np.nan)
        pb = np.full(self.n_assets, np.nan)
        roe = np.full(self.n_assets, np.nan)
        de = np.full(self.n_assets, np.nan)
        fcf = np.full(self.n_assets, np.nan)
        earn = np.full(self.n_assets, np.nan)
        rev = np.full(self.n_assets, np.nan)
        div = np.full(self.n_assets, np.nan)

        for i, asset in enumerate(self.all_assets):
            if asset in latest_fund.index:
                row = latest_fund.loc[asset]
                pe[i] = row.get('pe_ratio', np.nan)
                pb[i] = row.get('pb_ratio', np.nan)
                roe[i] = row.get('roe', np.nan)
                de[i] = row.get('debt_equity', np.nan)
                fcf[i] = row.get('free_cash_flow_yield', np.nan)
                earn[i] = row.get('earnings_surprise', np.nan)
                rev[i] = row.get('revenue_growth', np.nan)
                div[i] = row.get('dividend_yield', np.nan)

        # Size factor (higher = smaller cap, to capture small-cap premium)
        size_map = {'LARGE': 1, 'MID': 2, 'SMALL': 3}
        size_vals = np.array([size_map.get(self.market_caps.get(a, 'MID'), 2) for a in self.all_assets])

        # Z-scores
        z_mom20 = self._robust_zscore(mom20_r)
        z_mom60 = self._robust_zscore(mom60_r)
        z_pe = self._robust_zscore(pe)
        z_pb = self._robust_zscore(pb)
        z_roe = self._robust_zscore(roe)
        z_de = self._robust_zscore(de)
        z_fcf = self._robust_zscore(fcf)
        z_earn = self._robust_zscore(earn)
        z_rev = self._robust_zscore(rev)
        z_div = self._robust_zscore(div)
        z_vol = self._robust_zscore(vol)
        z_size = self._robust_zscore(size_vals)

        # Composite factor scores
        f_mom = 0.6 * z_mom20 + 0.4 * z_mom60
        f_qual = 0.25 * z_roe - 0.20 * z_de + 0.20 * z_fcf + 0.20 * z_earn + 0.15 * z_rev
        f_val = -0.4 * z_pe - 0.4 * z_pb + 0.2 * z_div
        f_lv = -z_vol
        f_size = z_size

        # Regime-dependent factor weights
        if regime == 'crisis':
            w_mom, w_qual, w_val, w_lv, w_size = 0.05, 0.40, 0.15, 0.35, 0.05
        elif regime == 'bear':
            w_mom, w_qual, w_val, w_lv, w_size = 0.15, 0.30, 0.25, 0.25, 0.05
        elif regime == 'normal':
            w_mom, w_qual, w_val, w_lv, w_size = 0.40, 0.15, 0.20, 0.10, 0.15
        else:  # bull
            w_mom, w_qual, w_val, w_lv, w_size = 0.45, 0.10, 0.15, 0.05, 0.25

        alpha = (w_mom * f_mom + w_qual * f_qual + w_val * f_val +
                 w_lv * f_lv + w_size * f_size)

        live_mask = np.array([a in live_assets for a in self.all_assets])
        alpha[~live_mask] = -1e6
        return alpha

    def _compute_covariance(self, prices, selected_idx):
        """Ledoit-Wolf shrunk covariance on 60-day returns."""
        price_pivot = prices.pivot(index='date', columns='asset_id', values='close')
        price_pivot = price_pivot.reindex(columns=[self.all_assets[i] for i in selected_idx])

        if len(price_pivot) < 10:
            n = len(selected_idx)
            return np.eye(n) * 0.04

        rets = price_pivot.pct_change().dropna()
        if len(rets) > 60:
            rets = rets.iloc[-60:]

        rets = rets.dropna(axis=1, how='all')
        rets_vals = np.nan_to_num(rets.values, nan=0.0)

        if rets_vals.shape[0] < 5 or rets_vals.shape[1] < 2:
            n = len(selected_idx)
            return np.eye(n) * 0.04

        # Ledoit-Wolf shrinkage
        try:
            lw = LedoitWolf()
            lw.fit(rets_vals)
            cov = lw.covariance_
        except Exception:
            cov = np.cov(rets_vals, rowvar=False)
            if cov.ndim < 2:
                cov = np.array([[cov + 0.04]])
            else:
                cov += np.eye(cov.shape[0]) * 1e-4

        # Map back to full selected index space
        n_sel = len(selected_idx)
        full_cov = np.eye(n_sel) * 0.04
        valid_names = [c for c in rets.columns if c in self.asset_to_idx]
        valid_idx = [selected_idx.tolist().index(self.asset_to_idx[c]) for c in valid_names]

        if len(valid_idx) > 0 and cov.shape[0] == len(valid_idx):
            full_cov[np.ix_(valid_idx, valid_idx)] = cov

        return full_cov

    def _optimize_weights(self, alpha, cov, prev_w, selected_idx, lam, gamma_vec, target_net):
        """Analytical KKT solution + iterative constraint projection."""
        n = len(selected_idx)
        a = alpha[selected_idx]
        w_prev = prev_w[selected_idx]

        # Positive-definite system matrix
        A = lam * cov + np.diag(gamma_vec) + 1e-5 * np.eye(n)
        b = a + gamma_vec * w_prev

        # Cholesky solve for w0 = A^{-1} b and v = A^{-1} 1
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A += 1e-3 * np.eye(n)
            L = np.linalg.cholesky(A)

        y1 = np.linalg.solve(L, b)
        w0 = np.linalg.solve(L.T, y1)

        y2 = np.linalg.solve(L, np.ones(n))
        v = np.linalg.solve(L.T, y2)

        # Net exposure Lagrange multiplier (exact KKT)
        sum_w0 = np.sum(w0)
        sum_v = np.sum(v)
        mu = (target_net - sum_w0) / sum_v if abs(sum_v) > 1e-10 else 0.0
        w = w0 + mu * v

        # Iterative projection onto inequality constraints
        for _ in range(5):
            # Position limits
            w = np.clip(w, 0.0, 0.10)

            # Sector limits (iterative proportional scaling)
            sector_map = {}
            for i, idx in enumerate(selected_idx):
                sector = self.sectors.get(self.all_assets[idx], 'UNK')
                sector_map.setdefault(sector, []).append(i)

            for sector, positions in sector_map.items():
                sec_sum = np.sum(w[positions])
                if sec_sum > 0.30:
                    w[positions] *= 0.30 / sec_sum

            # Re-adjust net exposure after sector scaling
            net = np.sum(w)
            if net > 1e-10 and abs(net - target_net) > 1e-6:
                w *= target_net / net

            w = np.clip(w, 0.0, 0.10)

        # Place back into full universe vector
        w_full = np.zeros(self.n_assets)
        for i, idx in enumerate(selected_idx):
            w_full[idx] = w[i]

        return w_full

    def _apply_buffer(self, w, prev_w):
        """Respect the 0.5% minimum trade filter."""
        diff = np.abs(w - prev_w)
        mask = diff < 0.005
        w_new = w.copy()
        w_new[mask] = prev_w[mask]
        return w_new


# =====================================================================
# Runner harness (must remain intact)
# =====================================================================
def main():
    import sys
    raw = sys.stdin.read()
    raw = raw.replace('\r\n', '\n')

    # ------------------------------------------------------------------
    # Try evaluation format first: ===CONFIG=== {json} ===PRICES=== csv ...
    # ------------------------------------------------------------------
    sections = {}
    markers = ['CONFIG', 'PRICES', 'FUND', 'IND']
    for i, marker in enumerate(markers):
        tag = f'==={marker}==='
        start = raw.find(tag)
        if start == -1:
            continue
        content_start = start + len(tag)
        next_start = len(raw)
        for next_marker in markers[i + 1:]:
            next_tag = f'==={next_marker}==='
            pos = raw.find(next_tag, content_start)
            if pos != -1 and pos < next_start:
                next_start = pos
        sections[marker] = raw[content_start:next_start].strip()

    rebalance_dates = []
    asset_columns = []

    if 'CONFIG' in sections:
        try:
            config = json.loads(sections['CONFIG'])
            rebalance_dates = config.get('rebalance_dates', [])
            asset_columns = config.get('asset_columns', [])
        except Exception:
            pass

    def safe_read_csv(text):
        if not text:
            return pd.DataFrame()
        try:
            return pd.read_csv(StringIO(text))
        except Exception:
            return pd.DataFrame()

    prices = safe_read_csv(sections.get('PRICES', ''))
    fundamentals = safe_read_csv(sections.get('FUND', ''))
    indicators = safe_read_csv(sections.get('IND', ''))

    # ------------------------------------------------------------------
    # Fallback: local backtester plain-text format
    # ------------------------------------------------------------------
    if prices.empty:
        data = raw.split('\n\n')
        config_lines = {}
        for section in data:
            if not section.strip():
                continue
            lines = section.strip().split('\n')
            if lines[0] in ('CONFIG', 'PRICES', 'FUND', 'IND'):
                config_lines[lines[0]] = lines[1:]

        for line in config_lines.get('CONFIG', []):
            if line.startswith('rebalance_dates='):
                rebalance_dates = line.split('=', 1)[1].split(',')

        prices = safe_read_csv('\n'.join(config_lines.get('PRICES', [])))
        fundamentals = safe_read_csv('\n'.join(config_lines.get('FUND', [])))
        indicators = safe_read_csv('\n'.join(config_lines.get('IND', [])))

    if prices.empty:
        print(','.join(['date']))
        return

    architect = PortfolioArchitect(prices, fundamentals, indicators)

    # If config provides exact asset column order, use it and remap weights
    if (asset_columns and len(asset_columns) == architect.n_assets and
            set(asset_columns) == set(architect.all_assets)):
        idx_map = {a: i for i, a in enumerate(architect.all_assets)}
        print(','.join(['date'] + asset_columns))
        for date in rebalance_dates:
            w = architect.allocate(prices, fundamentals, indicators, date)
            w_ordered = [w[idx_map[a]] for a in asset_columns]
            row = [date] + ['%.8f' % x for x in w_ordered]
            print(','.join(row))
    else:
        print(','.join(['date'] + architect.all_assets))
        for date in rebalance_dates:
            w = architect.allocate(prices, fundamentals, indicators, date)
            row = [date] + ['%.8f' % x for x in w]
            print(','.join(row))


if __name__ == '__main__':
    main()