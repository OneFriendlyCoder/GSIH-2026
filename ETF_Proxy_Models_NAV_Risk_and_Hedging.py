import pandas as pd
import numpy as np
import json
import sys
import warnings
warnings.filterwarnings('ignore')

##### DO NOT MODIFY/DELETE THE BELOW CODE #############################################################################

#######################################################################################################################
# CONSTANTS STARTS
# BELOW ARE "NECESSARY" CONSTANTS
######################################################################################################################=

"""End-Of-Day (EOD ) prices of target ETFs and proxy instruments, you have to use this to build whatever model you wan to."""
EOD_PRICES     = pd.read_csv( "./eod_prices.csv", index_col = 0, parse_dates=True ).sort_index()

# Defn. of Instruments - use these while constructing your output dataframes
TARGET_ETFS  = [f"Target_ETF_{i}" for i in range(1, 6)]

# PROXY INSTRUMENTS AVAILABLE

# 1). ETFs
PROXY_ETFS   = [f"Proxy_ETF_{i}" for i in range(1, 11)]

# 2). TREASURY FUTURES
TSY_FUTURES  = ["TU", "FV", "TY", "US"]

# 3). TREASURY OTR ("one-the-run") BONDS
TSY_BONDS    = ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]

# 4). CREDIT DEFAULT SWAP INDICES
CDX          = ["CDX_IG_5Y", "CDX_IG_10Y", "CDX_HY_5Y", "CDX_HY_10Y"]

"""You can use all proxy instruments for 'NAV' but ONLY 'RISK_PROXY' instruments for building proxy model for 'Risk'."""

# AVAILABE PROXY INSTRUMENTS FOR NAV MODEL ( = ALL PROXY INSTRUMENTS )
ALL_PROXY    = PROXY_ETFS + TSY_FUTURES + TSY_BONDS + CDX

# AVAILABE PROXY INSTRUMENTS FOR RISK MODEL ( = ALL - PROXT ETFs INSTRUMENTS )
RISK_PROXY   = TSY_FUTURES + TSY_BONDS + CDX

# Par-quoted instruments (price per 100); ETFs are NAV per share
# EOD_PRICES has price per 100 par notional for PAR_QUOTED instruments, whereas, for ETFs (target & proxy), it's per share == NAV
PAR_QUOTED   = set(TSY_FUTURES + TSY_BONDS + CDX)

# Transaction cost for instruemnts, this will be applied on your hedging basket
# One-way transaction cost (bps on |market value|)
COST_BPS = {**{e: 2.0 for e in PROXY_ETFS},
            **{f: 0.5 for f in TSY_FUTURES},
            **{b: 0.5 for b in TSY_BONDS},
            **{c: 1.0 for c in CDX}}

#######################################################################################################################
# CONSTANTS END
#######################################################################################################################

##### DO NOT MODIFY/DELETE THE BELOW CODE #############################################################################

#######################################################################################################################
# HELPER FUNCTIONS

# BELOW FUNCTIONS ARE PROVIDED AS BOILER-PLATE CODE
# THESE FUNCTIONS ARE ALSO USED IN EVALUATION, RECOMMENDED TO USE THEM, BUT NOT MANDATORY
#######################################################################################################################

#######################################################################################################################
# HELPER FUNCTIONS START
#######################################################################################################################

def compute_returns(prices):
    """Compute simple daily returns from prices."""
    return prices.pct_change().dropna()

def predict_returns(weights, proxy_returns):
    """Predict target ETF returns from proxy returns and model weights."""
    return proxy_returns @ weights

def reconstruct_nav(weights, proxy_returns, base_nav):
    """Reconstruct NAV level from predicted returns chained off a base NAV."""
    pred_returns = predict_returns(weights, proxy_returns)
    return (1 + pred_returns).cumprod() * base_nav

def compute_mape(predicted_nav, actual_nav):
    """Compute mean absolute percentage error between predicted and actual NAV."""
    aligned = pd.concat([predicted_nav, actual_nav], axis=1, join="inner").dropna()
    aligned.columns = ["predicted", "actual"]
    return (abs(aligned["predicted"] - aligned["actual"]) / aligned["actual"]).mean() * 100
  
def compute_pnl_series(notionals: dict, prices: pd.DataFrame) -> pd.Series:
    """mm USD PnL_t = sum_i scaling_i * (P_i,t - P_i,t-1)."""
    if not notionals:
        return pd.Series(0.0, index=prices.index[1:])
    p0 = prices.iloc[0]
    scaled = pd.Series({c: n * (1.0 / p0[c] if c not in PAR_QUOTED else 1.0 / 100.0)
                        for c, n in notionals.items()})
    return prices[scaled.index].diff().iloc[1:].mul(scaled, axis=1).sum(axis=1)

def compute_cost(notionals: dict, prices: pd.DataFrame) -> float:
    """
    compute transaction cost of setting up the hedging basket
    One-way cost on |market value| at t=0:
      ETF : |MV| * bps / 1e4                       (Notional IS MV)
      Par : |N|/100 * P_0 * bps / 1e4              (par * dirty price / 100)
    """
    p0 = prices.iloc[0]
    return float(sum(
        abs(n) * COST_BPS[i] / 1e4 if i not in PAR_QUOTED
        else (abs(n) / 100.0) * p0[i] * COST_BPS[i] / 1e4
        for i, n in notionals.items()
    ))

def compute_her(pnl_port: pd.Series, pnl_hedge: pd.Series) -> float:
    """compute hedge effectiveness ratio.""" 
    var_p = pnl_port.var(ddof=0)
    return 0.0 if var_p <= 0 else float(
        1.0 - pnl_port.add(pnl_hedge, fill_value=0.0).var(ddof=0) / var_p)

#######################################################################################################################
# HELPER FUNCTIONS END
#######################################################################################################################


#######################################################################################################################
# YOUR CODE STARTS HERE
#######################################################################################################################

def _read_stdin_safely(timeout: float = 1.0) -> str:
    try:
        import select
        if sys.stdin is None or sys.stdin.isatty():
            return ""
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read()
    except Exception:
        pass
    return ""

TEST_INPUT = {}
_raw = _read_stdin_safely()
if _raw and _raw.strip():
    try:
        TEST_INPUT = json.loads(_raw)
    except Exception:
        TEST_INPUT = {}

_DEFAULT_PORTFOLIO = {
    "Target_ETF_1":  150.0,
    "Target_ETF_2": -120.0,
    "Target_ETF_3":  180.0,
    "Target_ETF_4": -100.0,
    "Target_ETF_5":  130.0,
}
if isinstance(TEST_INPUT.get("portfolio"), dict) and TEST_INPUT["portfolio"]:
    PORTFOLIO = {k: float(v) for k, v in TEST_INPUT["portfolio"].items()}
else:
    PORTFOLIO = dict(_DEFAULT_PORTFOLIO)


def fit_ridge_weights(target_returns, proxy_returns, alpha=1e-5):
    aligned = pd.concat([target_returns, proxy_returns], axis=1).dropna()
    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values
    n_feat = X.shape[1]
    XtX = X.T @ X
    lam = alpha * (np.trace(XtX) / n_feat if n_feat else 1.0)
    return np.linalg.solve(XtX + lam * np.eye(n_feat), X.T @ y)


def lasso_cd(X, y, alpha, weights=None, max_iter=1500, tol=1e-9):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if n == 0 or p == 0:
        return np.zeros(p)
    X_std = X.std(axis=0, ddof=0)
    X_std = np.where(X_std < 1e-12, 1.0, X_std)
    Xn = X / X_std
    if weights is None:
        weights = np.ones(p)
    weights = np.asarray(weights, dtype=float)

    w = np.zeros(p)
    col_sq = np.full(p, float(n))
    residual = y.copy()
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            old_w = w[j]
            if w[j] != 0.0:
                residual += Xn[:, j] * w[j]
            rho = float(Xn[:, j] @ residual)
            thresh = alpha * weights[j] * n
            if rho > thresh:
                w[j] = (rho - thresh) / col_sq[j]
            elif rho < -thresh:
                w[j] = (rho + thresh) / col_sq[j]
            else:
                w[j] = 0.0
            if w[j] != 0.0:
                residual -= Xn[:, j] * w[j]
            change = abs(w[j] - old_w)
            if change > max_change:
                max_change = change
        if max_change < tol * (np.max(np.abs(w)) + 1e-12):
            break
    return w / X_std


def fit_proxy_model(target_etfs, proxy_list, returns, method="ridge",
                    alpha=1e-5, index_name="ETF"):
    W = pd.DataFrame(0.0, index=list(target_etfs), columns=list(proxy_list))
    W.index.name = index_name
    available = [c for c in proxy_list if c in returns.columns]
    if not available:
        return W
    proxy_rets = returns[available]
    for etf in target_etfs:
        if etf not in returns.columns:
            continue
        try:
            w = fit_ridge_weights(returns[etf], proxy_rets, alpha)
            W.loc[etf, available] = w
        except Exception:
            pass
    return W


def pnl_per_unit_notional(prices):
    p0 = prices.iloc[0]
    scaling = pd.Series({c: (1.0 / 100.0 if c in PAR_QUOTED else 1.0 / p0[c])
                         for c in prices.columns})
    return prices.diff().iloc[1:].mul(scaling, axis=1)


def build_hedge(portfolio, hedge_instruments, prices,
                method="lasso", alpha=None, drop_tol_mm=1.0):
    portfolio = {k: float(v) for k, v in portfolio.items() if v}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols  = [c for c in hedge_instruments if c in unit_pnl.columns]
    if not target_cols or not hedge_cols:
        return pd.Series(dtype=float, name="Notional")

    y = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    X = unit_pnl[hedge_cols].values

    if method == "lasso":
        cost_w = np.array([COST_BPS.get(c, 1.0) for c in hedge_cols])
        if alpha is None:
            X_std = X.std(axis=0, ddof=0)
            X_std = np.where(X_std < 1e-12, 1.0, X_std)
            Xn = X / X_std
            score = np.abs(Xn.T @ y) / (X.shape[0] * cost_w)
            sorted_s = np.sort(score)[::-1]
            k = min(6, len(sorted_s) - 1)
            alpha = sorted_s[k] if k < len(sorted_s) else 0.0
        w = lasso_cd(X, y, alpha, weights=cost_w)
        h = -w
    else:
        XtX = X.T @ X
        lam = 1e-3 * np.trace(XtX) / XtX.shape[0]
        h = -np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X.T @ y)

    s = pd.Series(h, index=hedge_cols, name="Notional")
    s = s.where(s.abs() >= drop_tol_mm, 0.0)
    return s[s != 0.0]


RETURNS = compute_returns(EOD_PRICES)

try:
    from sklearn.linear_model import ElasticNetCV, RidgeCV, ElasticNet
    HAS_SK = True
except Exception:
    HAS_SK = False


def fit_nav_one_etf(etf, returns, proxy_list, corr_thresh=0.09,
                    low_var_ratio=0.35):
    full = pd.Series(0.0, index=proxy_list)
    if etf not in returns.columns:
        return full.values
    avail = [c for c in proxy_list if c in returns.columns]
    aligned = pd.concat([returns[etf], returns[avail]], axis=1).dropna()
    if len(aligned) < 25 or len(avail) == 0:
        return full.values
    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    y_var = float(y.var(ddof=0))
    proxy_avg_var = float(X.var(ddof=0).mean())
    if proxy_avg_var > 0 and y_var < low_var_ratio * proxy_avg_var:
        return full.values
    corrs = X.corrwith(y).abs().fillna(0.0)
    selected = corrs[corrs > corr_thresh].index.tolist()
    if len(selected) < 4:
        selected = corrs.nlargest(min(7, len(corrs))).index.tolist()
    Xsel = X[selected]
    n = len(y)
    sw = np.exp(np.linspace(-1.8, 0.0, n))
    sw = sw / sw.sum() * n
    coef = None
    if HAS_SK:
        coefs = []
        for l1 in (0.4, 0.65, 0.85):
            try:
                m = ElasticNetCV(
                    l1_ratio=l1,
                    alphas=np.logspace(-5.5, 0.5, 35),
                    cv=5, max_iter=15000, random_state=42,
                    n_jobs=1, fit_intercept=False,
                )
                m.fit(Xsel.values, y.values, sample_weight=sw)
                coefs.append(m.coef_)
            except Exception:
                continue
        if coefs:
            coef = np.median(np.stack(coefs, axis=0), axis=0)
    if coef is None or not np.any(np.abs(coef) > 1e-12):
        coef = fit_ridge_weights(y, Xsel, alpha=3e-5)
    coef = np.where(coef < -0.05 * np.abs(coef), -0.05 * np.abs(coef), coef)
    full.loc[selected] = coef
    return full.values


def fit_nav_model_all(target_etfs, proxy_list, returns):
    W = pd.DataFrame(0.0, index=list(target_etfs), columns=list(proxy_list))
    W.index.name = "ETF"
    for etf in target_etfs:
        try:
            W.loc[etf] = fit_nav_one_etf(etf, returns, proxy_list)
        except Exception:
            pass
    return W


def fit_risk_one_etf(etf, returns, rate_cols, credit_cols, all_cols):
    full = pd.Series(0.0, index=all_cols)
    if etf not in returns.columns:
        return full.values
    fut_cols = [c for c in TSY_FUTURES if c in returns.columns]
    cr_cols = [c for c in CDX if c in returns.columns]
    use_cols = fut_cols + cr_cols
    if not use_cols:
        return full.values
    aligned = pd.concat([returns[etf], returns[use_cols]], axis=1).dropna()
    if len(aligned) < 25:
        return full.values
    y = aligned.iloc[:, 0].values
    fut_coef = None
    if fut_cols:
        X_fut = aligned[fut_cols].values
        if HAS_SK:
            try:
                m = RidgeCV(alphas=np.logspace(-5, 2, 30), cv=6,
                            fit_intercept=False)
                m.fit(X_fut, y)
                fut_coef = m.coef_
            except Exception:
                fut_coef = None
        if fut_coef is None:
            fut_coef = fit_ridge_weights(pd.Series(y), pd.DataFrame(X_fut),
                                         alpha=8e-5)
        full.loc[fut_cols] = fut_coef
        y_resid = y - X_fut @ fut_coef
    else:
        y_resid = y
    if cr_cols:
        X_cr = aligned[cr_cols].values
        cr_coef = None
        if HAS_SK:
            try:
                m = RidgeCV(alphas=np.logspace(-5, 2, 30), cv=6,
                            fit_intercept=False)
                m.fit(X_cr, y_resid)
                cr_coef = m.coef_
            except Exception:
                cr_coef = None
        if cr_coef is None:
            cr_coef = fit_ridge_weights(pd.Series(y_resid),
                                        pd.DataFrame(X_cr), alpha=1e-4)
        full.loc[cr_cols] = cr_coef
    return full.values


def fit_risk_model_all(target_etfs, returns):
    W = pd.DataFrame(0.0, index=list(target_etfs), columns=list(RISK_PROXY))
    W.index.name = "ETF"
    rate_cols = TSY_FUTURES + TSY_BONDS
    credit_cols = CDX
    for etf in target_etfs:
        try:
            W.loc[etf] = fit_risk_one_etf(etf, returns, rate_cols,
                                          credit_cols, RISK_PROXY)
        except Exception:
            pass
    return W


def build_hedge_costaware(portfolio, hedge_instruments, prices,
                          drop_tol_mm=0.5):
    portfolio = {k: float(v) for k, v in portfolio.items() if v}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols = [c for c in hedge_instruments if c in unit_pnl.columns]
    if not target_cols or not hedge_cols:
        return pd.Series(dtype=float, name="Notional")
    y = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    X = unit_pnl[hedge_cols].values
    cost_w = np.array([COST_BPS.get(c, 1.0) for c in hedge_cols])
    h = None
    if HAS_SK and X.shape[0] >= 20:
        try:
            X_scaled = X / cost_w
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=np.logspace(-6, 1, 40),
                cv=5, max_iter=20000, random_state=0,
                n_jobs=1, fit_intercept=False,
            )
            model.fit(X_scaled, y)
            v = model.coef_
            w_orig = v / cost_w
            h = -w_orig
        except Exception:
            h = None
    if h is None:
        XtX = X.T @ X
        lam = 1e-3 * np.trace(XtX) / max(XtX.shape[0], 1)
        h = -np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X.T @ y)
    s = pd.Series(h, index=hedge_cols, name="Notional")
    s = s.where(s.abs() >= drop_tol_mm, 0.0)
    return s[s != 0.0]


def build_hedge_from_risk(portfolio, risk_w, prices, drop_tol_mm=0.1):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    p0 = prices.iloc[0]
    hedge = pd.Series(0.0, index=list(risk_w.columns))
    for etf, n_etf in portfolio.items():
        if etf not in risk_w.index:
            continue
        row = risk_w.loc[etf]
        for j in risk_w.columns:
            try:
                w_ij = float(row[j])
            except Exception:
                continue
            if w_ij == 0.0 or j not in p0.index:
                continue
            scale = float(p0[j]) / 100.0 if j in PAR_QUOTED else 1.0
            if scale == 0.0 or not np.isfinite(scale):
                continue
            hedge[j] = hedge[j] - n_etf * w_ij / scale
    hedge.name = "Notional"
    hedge = hedge.where(hedge.abs() >= drop_tol_mm, 0.0)
    return hedge[hedge != 0.0]


def build_hedge_max_effort(portfolio, hedge_instruments, prices,
                           max_notional=10000.0, drop_tol_mm=0.01):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols = [c for c in hedge_instruments if c in unit_pnl.columns]
    if not target_cols or not hedge_cols:
        return pd.Series(dtype=float, name="Notional")
    y_port = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    var_port = float(np.var(y_port, ddof=0))
    if var_port <= 0:
        return pd.Series(dtype=float, name="Notional")
    X = unit_pnl[hedge_cols].values
    n_feat = X.shape[1]
    XtX = X.T @ X
    lam = 1e-9 * np.trace(XtX) / max(n_feat, 1)
    try:
        h_minvar = -np.linalg.solve(XtX + lam * np.eye(n_feat), X.T @ y_port)
    except Exception:
        h_minvar = np.zeros(n_feat)

    def objective(h):
        comb = y_port + X @ h
        return float(np.var(comb, ddof=0)) / (var_port + 1e-12)

    h_best = h_minvar
    best_obj = objective(h_best)
    try:
        from scipy.optimize import minimize
        bounds = [(-max_notional, max_notional) for _ in hedge_cols]
        for x0 in (h_minvar, np.zeros(n_feat), h_minvar * 1.3, h_minvar * 0.7):
            try:
                res = minimize(objective, x0=x0, bounds=bounds,
                               method="L-BFGS-B",
                               options={"maxiter": 2500, "ftol": 1e-15,
                                        "gtol": 1e-12})
                if res.fun < best_obj:
                    best_obj = res.fun
                    h_best = res.x
            except Exception:
                continue
    except Exception:
        pass
    h_best = np.where(np.abs(h_best) >= drop_tol_mm, h_best, 0.0)
    s = pd.Series(h_best, index=hedge_cols, name="Notional")
    return s[s != 0.0]


def build_hedge_replication(portfolio, nav_w, prices, drop_tol_mm=0.5):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    p0 = prices.iloc[0]
    hedge = pd.Series(0.0, index=list(nav_w.columns))
    for etf, n_etf in portfolio.items():
        if etf not in nav_w.index:
            continue
        row = nav_w.loc[etf]
        for j in nav_w.columns:
            try:
                w_ij = float(row[j])
            except Exception:
                continue
            if w_ij == 0.0 or j not in p0.index:
                continue
            scale = float(p0[j]) / 100.0 if j in PAR_QUOTED else 1.0
            if scale == 0.0 or not np.isfinite(scale):
                continue
            hedge[j] = hedge[j] - n_etf * w_ij / scale
    hedge.name = "Notional"
    hedge = hedge.where(hedge.abs() >= drop_tol_mm, 0.0)
    return hedge[hedge != 0.0]


def build_hedge_hybrid(portfolio, hedge_instruments, prices,
                       her_w=2.2, te_w=9.5, cost_w_mult=0.009,
                       drop_tol_mm=0.55, max_notional=1200.0):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    seed = build_hedge_costaware(portfolio, hedge_instruments, prices,
                                 drop_tol_mm=0.0)
    if len(seed) <= 2:
        return build_hedge_costaware(portfolio, hedge_instruments, prices,
                                     drop_tol_mm=0.4)
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols = list(seed.index)
    y_port = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    X = unit_pnl[hedge_cols].values
    cost_w = np.array([COST_BPS.get(c, 1.0) for c in hedge_cols])
    var_port = float(np.var(y_port, ddof=0))
    if var_port <= 0:
        return seed.where(seed.abs() >= drop_tol_mm, 0.0).pipe(lambda s: s[s != 0.0])

    def objective(h):
        comb = y_port + X @ h
        her = 1.0 - float(np.var(comb, ddof=0)) / (var_port + 1e-12)
        te = float(np.std(comb, ddof=0))
        cost = float(np.sum(np.abs(h) * cost_w)) * 0.75
        return -her_w * her + te_w * te + cost_w_mult * cost

    try:
        from scipy.optimize import minimize
        bounds = [(-max_notional, max_notional) for _ in hedge_cols]
        res = minimize(objective, x0=seed.values, bounds=bounds,
                       method="L-BFGS-B",
                       options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-9})
        h_final = res.x
    except Exception:
        h_final = seed.values
    s = pd.Series(h_final, index=hedge_cols, name="Notional")
    s = s.where(s.abs() >= drop_tol_mm, 0.0)
    return s[s != 0.0]


def fit_nav_improved(target_etfs, proxy_list, returns,
                     alpha_range=None, mag_thresh=0.01):
    W = pd.DataFrame(0.0, index=list(target_etfs), columns=list(proxy_list))
    W.index.name = "ETF"
    if alpha_range is None:
        alpha_range = np.logspace(-6, 1, 50)
    available = [c for c in proxy_list if c in returns.columns]
    if not available:
        return W
    proxy_rets = returns[available]
    for etf in target_etfs:
        if etf not in returns.columns:
            continue
        try:
            aligned = pd.concat([returns[etf], proxy_rets], axis=1).dropna()
            if len(aligned) < 30:
                continue
            y = aligned.iloc[:, 0].values
            X = aligned.iloc[:, 1:].values
            n_train = int(0.8 * len(aligned))
            y_tr, y_val = y[:n_train], y[n_train:]
            X_tr, X_val = X[:n_train], X[n_train:]
            n_feat = X.shape[1]
            best_alpha = float(alpha_range[0])
            best_val_mse = np.inf
            XtX_tr = X_tr.T @ X_tr
            Xty_tr = X_tr.T @ y_tr
            I_p = np.eye(n_feat)
            for alpha in alpha_range:
                try:
                    w = np.linalg.solve(XtX_tr + alpha * I_p, Xty_tr)
                    y_pred = X_val @ w
                    val_mse = float(np.mean((y_val - y_pred) ** 2))
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_alpha = float(alpha)
                except Exception:
                    continue
            XtX = X.T @ X
            w = np.linalg.solve(XtX + best_alpha * I_p, X.T @ y)
            w = np.where(np.abs(w) > mag_thresh, w, 0.0)
            abs_sum = float(np.sum(np.abs(w)))
            if abs_sum > 0.1:
                w = w / abs_sum
            W.loc[etf, available] = w
        except Exception:
            pass
    return W


def fit_risk_improved(target_etfs, returns, low_var_ratio=0.20):
    W = pd.DataFrame(0.0, index=list(target_etfs), columns=list(RISK_PROXY))
    W.index.name = "ETF"
    use_cols = [c for c in (TSY_FUTURES + CDX) if c in returns.columns]
    if not use_cols:
        return W
    for etf in target_etfs:
        if etf not in returns.columns:
            continue
        try:
            aligned = pd.concat([returns[etf], returns[use_cols]], axis=1).dropna()
            if len(aligned) < 30:
                continue
            y = aligned.iloc[:, 0].values
            X = aligned.iloc[:, 1:].values
            y_var = float(np.var(y, ddof=0))
            X_avg_var = float(np.mean(np.var(X, axis=0, ddof=0)))
            if X_avg_var > 0 and y_var < low_var_ratio * X_avg_var:
                continue
            n_train = int(0.8 * len(aligned))
            y_tr, y_val = y[:n_train], y[n_train:]
            X_tr, X_val = X[:n_train], X[n_train:]
            n_feat = X.shape[1]
            best_alpha = 1e-7
            best_val_mse = np.inf
            XtX_tr = X_tr.T @ X_tr
            Xty_tr = X_tr.T @ y_tr
            I_p = np.eye(n_feat)
            for alpha in np.logspace(-8, -2, 40):
                try:
                    w = np.linalg.solve(XtX_tr + alpha * I_p, Xty_tr)
                    y_pred = X_val @ w
                    val_mse = float(np.mean((y_val - y_pred) ** 2))
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_alpha = float(alpha)
                except Exception:
                    continue
            XtX = X.T @ X
            w = np.linalg.solve(XtX + best_alpha * I_p, X.T @ y)
            W.loc[etf, use_cols] = w
        except Exception:
            pass
    return W


def fit_risk_model_simple(target_etfs, returns):
    W = pd.DataFrame(0.0, index=list(target_etfs), columns=list(RISK_PROXY))
    W.index.name = "ETF"
    use_cols = [c for c in RISK_PROXY if c in returns.columns]
    if not use_cols:
        return W
    for etf in target_etfs:
        if etf not in returns.columns:
            continue
        try:
            aligned = pd.concat([returns[etf], returns[use_cols]], axis=1).dropna()
            if len(aligned) < 25:
                continue
            y = aligned.iloc[:, 0].values
            X = aligned.iloc[:, 1:].values
            coef = None
            if HAS_SK:
                try:
                    from sklearn.linear_model import RidgeCV
                    m = RidgeCV(alphas=np.logspace(-6, 2, 50), cv=5,
                                fit_intercept=True)
                    m.fit(X, y)
                    coef = m.coef_
                except Exception:
                    coef = None
            if coef is None:
                coef = fit_ridge_weights(pd.Series(y), pd.DataFrame(X), alpha=1e-5)
            W.loc[etf, use_cols] = coef
        except Exception:
            pass
    return W


def build_hedge_shrinkage(portfolio, hedge_instruments, prices,
                          drop_tol_mm=0.1):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols = [c for c in hedge_instruments if c in unit_pnl.columns]
    if not target_cols or not hedge_cols:
        return pd.Series(dtype=float, name="Notional")
    y = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    X = unit_pnl[hedge_cols].values
    h = None
    try:
        from sklearn.covariance import LedoitWolf
        all_returns = np.column_stack([y.reshape(-1, 1), X])
        lw = LedoitWolf().fit(all_returns)
        cov = lw.covariance_
        sigma_ph = cov[0, 1:]
        sigma_hh = cov[1:, 1:]
        h = -np.linalg.solve(sigma_hh + 1e-8 * np.eye(len(hedge_cols)),
                             sigma_ph)
    except Exception:
        h = None
    if h is None:
        XtX = X.T @ X
        lam = 1e-7 * np.trace(XtX) / max(XtX.shape[0], 1)
        try:
            h = -np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X.T @ y)
        except Exception:
            h = np.zeros(len(hedge_cols))
    s = pd.Series(h, index=hedge_cols, name="Notional")
    s = s.where(s.abs() >= drop_tol_mm, 0.0)
    return s[s != 0.0]


def build_hedge_regime_blend(portfolio, hedge_instruments, prices,
                              short_window=60, blend_weight=0.6,
                              drop_tol_mm=0.1):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols = [c for c in hedge_instruments if c in unit_pnl.columns]
    if not target_cols or not hedge_cols:
        return pd.Series(dtype=float, name="Notional")
    y_s = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1)
    X_df = unit_pnl[hedge_cols]
    all_ret = pd.concat([y_s.rename("port"), X_df], axis=1).dropna()
    if len(all_ret) < 30:
        return pd.Series(dtype=float, name="Notional")
    n_hedge = len(hedge_cols)
    h = None
    try:
        from sklearn.covariance import LedoitWolf
        cov_full = LedoitWolf().fit(all_ret.values).covariance_
        sw = min(short_window, len(all_ret) - 1)
        recent = all_ret.iloc[-sw:]
        cov_rec = LedoitWolf().fit(recent.values).covariance_
        cov_blend = blend_weight * cov_rec + (1.0 - blend_weight) * cov_full
        sigma_ph = cov_blend[0, 1:]
        sigma_hh = cov_blend[1:, 1:]
        h = -np.linalg.solve(sigma_hh + 1e-8 * np.eye(n_hedge), sigma_ph)
    except Exception:
        h = None
    if h is None:
        return pd.Series(dtype=float, name="Notional")
    s = pd.Series(h, index=hedge_cols, name="Notional")
    s = s.where(s.abs() >= drop_tol_mm, 0.0)
    return s[s != 0.0]


def build_hedge_cvar(portfolio, hedge_instruments, prices,
                     alpha=0.05, drop_tol_mm=0.1, max_notional=5000.0):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols = [c for c in hedge_instruments if c in unit_pnl.columns]
    if not target_cols or not hedge_cols:
        return pd.Series(dtype=float, name="Notional")
    y = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    X = unit_pnl[hedge_cols].values
    T, p = X.shape
    if T < 20 or p == 0:
        return pd.Series(dtype=float, name="Notional")
    n_tail = max(2, int(alpha * T))

    def cvar_obj(h):
        net = y + X @ h
        sorted_net = np.sort(net)
        left = sorted_net[:n_tail]
        right = sorted_net[-n_tail:]
        return -float(np.mean(left)) + float(np.mean(right))

    def var_obj(h):
        net = y + X @ h
        return float(np.var(net, ddof=0))

    XtX = X.T @ X
    lam = 1e-7 * np.trace(XtX) / max(p, 1)
    try:
        h_init = -np.linalg.solve(XtX + lam * np.eye(p), X.T @ y)
    except Exception:
        h_init = np.zeros(p)

    h_best = h_init
    best_obj = cvar_obj(h_best)
    try:
        from scipy.optimize import minimize
        bounds = [(-max_notional, max_notional) for _ in hedge_cols]
        for x0 in (h_init, np.zeros(p), h_init * 0.7):
            try:
                res = minimize(cvar_obj, x0=x0, bounds=bounds,
                               method="L-BFGS-B",
                               options={"maxiter": 1500, "ftol": 1e-11})
                if res.fun < best_obj:
                    best_obj = res.fun
                    h_best = res.x
            except Exception:
                continue
    except Exception:
        pass

    h_best = np.where(np.abs(h_best) >= drop_tol_mm, h_best, 0.0)
    s = pd.Series(h_best, index=hedge_cols, name="Notional")
    return s[s != 0.0]


def build_hedge_robust(portfolio, hedge_instruments, prices, drop_tol_mm=0.1):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    if not target_cols:
        return pd.Series(dtype=float, name="Notional")
    y_port = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    var_port = float(np.var(y_port, ddof=0))
    std_port = float(np.std(y_port, ddof=0))
    if var_port <= 1e-12:
        return pd.Series(dtype=float, name="Notional")
    seed = build_hedge_costaware(portfolio, hedge_instruments, prices, 0.0)
    if seed.empty:
        return pd.Series(dtype=float, name="Notional")
    hedge_cols = list(seed.index)
    X = unit_pnl[hedge_cols].values
    cost_w = np.array([COST_BPS.get(c, 1.0) for c in hedge_cols])
    port_size = sum(abs(v) for v in portfolio.values())
    max_n = max(1200.0, port_size * 1.9)

    def objective(h):
        comb = y_port + X @ h
        var_combined = float(np.var(comb, ddof=0))
        her = 1.0 - var_combined / (var_port + 1e-12)
        te = float(np.std(comb, ddof=0))
        cost = float(np.sum(np.abs(h) * cost_w)) * 0.78
        vol_mismatch = abs(te / (std_port + 1e-8) - 1.0)
        return -3.2 * her + 12.0 * te + 0.006 * cost + 5.5 * vol_mismatch

    best_h = np.array(seed.values, dtype=float)
    best_obj = objective(best_h)
    try:
        from scipy.optimize import minimize
        bounds = [(-max_n, max_n) for _ in hedge_cols]
        starts = [
            seed.values.astype(float),
            seed.values.astype(float) * 0.6,
            np.zeros(len(hedge_cols)),
        ]
        for x0 in starts:
            try:
                res = minimize(objective, x0=x0, bounds=bounds,
                               method="L-BFGS-B",
                               options={"maxiter": 1000, "ftol": 1e-12})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_h = res.x
            except Exception:
                continue
    except Exception:
        pass
    s = pd.Series(best_h, index=hedge_cols, name="Notional")
    s = s.where(s.abs() >= drop_tol_mm, 0.0)
    return s[s != 0.0]


def build_hedge_optimized(portfolio, hedge_instruments, prices,
                          max_notional=3000.0, drop_tol_mm=0.1,
                          cost_weight=1e-5):
    portfolio = {k: float(v) for k, v in portfolio.items() if abs(v) > 1e-6}
    if not portfolio:
        return pd.Series(dtype=float, name="Notional")
    unit_pnl = pnl_per_unit_notional(prices)
    target_cols = [c for c in portfolio if c in unit_pnl.columns]
    hedge_cols = [c for c in hedge_instruments if c in unit_pnl.columns]
    if not target_cols or not hedge_cols:
        return pd.Series(dtype=float, name="Notional")
    y_port = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1).values
    var_port = float(np.var(y_port, ddof=0))
    if var_port <= 0:
        return pd.Series(dtype=float, name="Notional")
    X = unit_pnl[hedge_cols].values
    cost_w = np.array([COST_BPS.get(c, 1.0) for c in hedge_cols])
    n_feat = X.shape[1]

    XtX = X.T @ X
    lam = 1e-7 * np.trace(XtX) / max(n_feat, 1)
    try:
        h_minvar = -np.linalg.solve(XtX + lam * np.eye(n_feat), X.T @ y_port)
    except Exception:
        h_minvar = np.zeros(n_feat)

    def objective(h):
        comb = y_port + X @ h
        var_combined = float(np.var(comb, ddof=0))
        rel_var = var_combined / (var_port + 1e-12)
        tc = float(np.sum(np.abs(h) * cost_w))
        return rel_var + cost_weight * tc

    h_best = h_minvar
    try:
        from scipy.optimize import minimize
        bounds = [(-max_notional, max_notional) for _ in hedge_cols]
        res = minimize(objective, x0=h_minvar, bounds=bounds,
                       method="L-BFGS-B",
                       options={"maxiter": 1500, "ftol": 1e-14, "gtol": 1e-10})
        if res.fun < objective(h_best):
            h_best = res.x
    except Exception:
        pass

    h_best = np.where(np.abs(h_best) >= drop_tol_mm, h_best, 0.0)
    s = pd.Series(h_best, index=hedge_cols, name="Notional")
    return s[s != 0.0]


try:
    nav_weights = fit_nav_improved(TARGET_ETFS, ALL_PROXY, RETURNS)
    if (nav_weights != 0).sum().sum() < 5:
        nav_weights = fit_proxy_model(TARGET_ETFS, ALL_PROXY, RETURNS,
                                      method="ridge", alpha=1e-5)
except Exception:
    nav_weights = fit_proxy_model(TARGET_ETFS, ALL_PROXY, RETURNS,
                                  method="ridge", alpha=1e-5)
nav_weights.index.name = "ETF"

try:
    risk_weights = fit_risk_improved(TARGET_ETFS, RETURNS)
    if (risk_weights != 0).sum().sum() < 4:
        risk_weights = fit_proxy_model(TARGET_ETFS, RISK_PROXY, RETURNS,
                                       method="ridge", alpha=1e-5)
except Exception:
    risk_weights = fit_proxy_model(TARGET_ETFS, RISK_PROXY, RETURNS,
                                   method="ridge", alpha=1e-5)
risk_weights.index.name = "ETF"

HEDGE_INSTRUMENTS = PROXY_ETFS + TSY_FUTURES + TSY_BONDS + CDX
HEDGE_LIQUID = TSY_FUTURES + CDX


def _nav_relevant_instruments(portfolio, nav_w):
    used = set()
    for etf in portfolio:
        if etf in nav_w.index:
            row = nav_w.loc[etf]
            for j in nav_w.columns:
                if abs(float(row[j])) > 1e-10:
                    used.add(j)
    return [c for c in HEDGE_INSTRUMENTS if c in used]


def _hedge_score(notionals_series, portfolio, prices, cost_pen=2.0):
    if notionals_series is None or notionals_series.empty:
        return -1e18
    notionals_dict = {k: float(v) for k, v in notionals_series.items() if v != 0}
    if not notionals_dict:
        return -1e18
    try:
        unit_pnl = pnl_per_unit_notional(prices)
        target_cols = [c for c in portfolio if c in unit_pnl.columns]
        y_port = unit_pnl[target_cols].mul(pd.Series(portfolio), axis=1).sum(axis=1)
        h_cols = [c for c in notionals_dict if c in unit_pnl.columns]
        h_vec = pd.Series({c: notionals_dict[c] for c in h_cols})
        h_pnl = unit_pnl[h_cols].mul(h_vec, axis=1).sum(axis=1)
        her = compute_her(y_port, h_pnl)
        tc = compute_cost(notionals_dict, prices)
        return float(her) - cost_pen * float(tc)
    except Exception:
        return -1e18


def _gather_candidates(portfolio, nav_w, prices):
    candidates = []
    try:
        rep = build_hedge_replication(portfolio, nav_w, prices, drop_tol_mm=0.8)
        if not rep.empty:
            candidates.append(("rep", rep))
    except Exception:
        pass
    try:
        opt_full = build_hedge_optimized(portfolio, HEDGE_INSTRUMENTS, prices,
                                         max_notional=5000.0, drop_tol_mm=0.05,
                                         cost_weight=1e-6)
        if not opt_full.empty:
            candidates.append(("opt_full", opt_full))
    except Exception:
        pass
    try:
        opt_full_tight = build_hedge_optimized(portfolio, HEDGE_INSTRUMENTS, prices,
                                               max_notional=2500.0, drop_tol_mm=0.35,
                                               cost_weight=2e-5)
        if not opt_full_tight.empty:
            candidates.append(("opt_tight", opt_full_tight))
    except Exception:
        pass
    try:
        opt_liquid = build_hedge_optimized(portfolio, HEDGE_LIQUID, prices,
                                           max_notional=5000.0, drop_tol_mm=0.05,
                                           cost_weight=1e-6)
        if not opt_liquid.empty:
            candidates.append(("opt_liquid", opt_liquid))
    except Exception:
        pass
    try:
        robust = build_hedge_robust(portfolio, HEDGE_INSTRUMENTS, prices,
                                    drop_tol_mm=0.1)
        if not robust.empty:
            candidates.append(("robust", robust))
    except Exception:
        pass
    try:
        shrink_full = build_hedge_shrinkage(portfolio, HEDGE_INSTRUMENTS,
                                            prices, drop_tol_mm=0.1)
        if not shrink_full.empty:
            candidates.append(("shrink_full", shrink_full))
    except Exception:
        pass
    try:
        shrink_liquid = build_hedge_shrinkage(portfolio, HEDGE_LIQUID,
                                              prices, drop_tol_mm=0.05)
        if not shrink_liquid.empty:
            candidates.append(("shrink_liquid", shrink_liquid))
    except Exception:
        pass
    try:
        regime_full = build_hedge_regime_blend(portfolio, HEDGE_INSTRUMENTS,
                                               prices, short_window=60,
                                               blend_weight=0.6,
                                               drop_tol_mm=0.1)
        if not regime_full.empty:
            candidates.append(("regime_full", regime_full))
    except Exception:
        pass
    try:
        regime_liquid = build_hedge_regime_blend(portfolio, HEDGE_LIQUID,
                                                 prices, short_window=60,
                                                 blend_weight=0.6,
                                                 drop_tol_mm=0.05)
        if not regime_liquid.empty:
            candidates.append(("regime_liquid", regime_liquid))
    except Exception:
        pass
    try:
        cvar_full = build_hedge_cvar(portfolio, HEDGE_INSTRUMENTS, prices,
                                     alpha=0.05, drop_tol_mm=0.1,
                                     max_notional=5000.0)
        if not cvar_full.empty:
            candidates.append(("cvar_full", cvar_full))
    except Exception:
        pass
    try:
        cvar_liquid = build_hedge_cvar(portfolio, HEDGE_LIQUID, prices,
                                       alpha=0.05, drop_tol_mm=0.05,
                                       max_notional=5000.0)
        if not cvar_liquid.empty:
            candidates.append(("cvar_liquid", cvar_liquid))
    except Exception:
        pass
    try:
        cvar_tight = build_hedge_cvar(portfolio, HEDGE_INSTRUMENTS, prices,
                                      alpha=0.10, drop_tol_mm=0.1,
                                      max_notional=5000.0)
        if not cvar_tight.empty:
            candidates.append(("cvar_tight", cvar_tight))
    except Exception:
        pass
    try:
        ca = build_hedge_costaware(portfolio, HEDGE_INSTRUMENTS, prices,
                                   drop_tol_mm=0.4)
        if not ca.empty:
            candidates.append(("costaware", ca))
    except Exception:
        pass
    try:
        rh = build_hedge_from_risk(portfolio, risk_weights, prices,
                                   drop_tol_mm=0.1)
        if not rh.empty:
            candidates.append(("risk_hedge", rh))
    except Exception:
        pass
    try:
        rh_liquid = build_hedge_from_risk(portfolio, risk_weights, prices,
                                          drop_tol_mm=0.05)
        if not rh_liquid.empty:
            non_zero = rh_liquid[rh_liquid != 0]
            liquid_only = non_zero[[i for i in non_zero.index
                                    if i in HEDGE_LIQUID]]
            if not liquid_only.empty:
                candidates.append(("risk_hedge_liquid", liquid_only))
    except Exception:
        pass
    try:
        me = build_hedge_max_effort(portfolio, HEDGE_INSTRUMENTS, prices,
                                    max_notional=10000.0, drop_tol_mm=0.01)
        if not me.empty:
            candidates.append(("max_effort", me))
    except Exception:
        pass
    try:
        me_liquid = build_hedge_max_effort(portfolio, HEDGE_LIQUID, prices,
                                           max_notional=10000.0, drop_tol_mm=0.01)
        if not me_liquid.empty:
            candidates.append(("max_effort_liquid", me_liquid))
    except Exception:
        pass
    return candidates


try:
    cands = _gather_candidates(PORTFOLIO, nav_weights, EOD_PRICES)
    if cands:
        scored = [(name, h, _hedge_score(h, PORTFOLIO, EOD_PRICES, cost_pen=2.0))
                  for name, h in cands]
        scored.sort(key=lambda x: x[2], reverse=True)
        hedge_notionals = scored[0][1]
    else:
        hedge_notionals = pd.Series(dtype=float, name="Notional")
    if hedge_notionals is None or hedge_notionals.empty:
        hedge_notionals = build_hedge_hybrid(PORTFOLIO, HEDGE_INSTRUMENTS,
                                             EOD_PRICES)
    if hedge_notionals.empty:
        hedge_notionals = build_hedge_costaware(PORTFOLIO, HEDGE_INSTRUMENTS,
                                                EOD_PRICES, drop_tol_mm=0.4)
    if hedge_notionals.empty:
        hedge_notionals = build_hedge(PORTFOLIO, HEDGE_INSTRUMENTS, EOD_PRICES,
                                      method="ridge", drop_tol_mm=0.5)
except Exception:
    try:
        hedge_notionals = build_hedge(PORTFOLIO, HEDGE_LIQUID, EOD_PRICES,
                                      method="ridge", drop_tol_mm=0.5)
    except Exception:
        hedge_notionals = pd.Series(dtype=float, name="Notional")   

#######################################################################################################################
# YOUR CODE ENDS HERE
#######################################################################################################################

#######################################################################################################################
# SUBMISSION FORMAT - ASSIGN YOUR OUTPUT PANDAS DATAFRAMES BELOW
# YOU DO NOT NEED TO MODIFY THE DATAFRAMES FOR TASKS WHICH YOU HAVE NOT SOLVED, WE'LL DO PARTIAL GRADING 
# e.g., IF YOU'VE ONLY SOLVED THE 'NAV' TASK, JUST ASSIGN YOUR SOLUTION FOR IT TO 'nav_model' VARIABLE, AND LEAVE THE  # OTHER TWO UNTOUCHED
#######################################################################################################################

# TASK 1 - NAV ESTIMATION
nav_model = nav_weights
nav_model.index.name = "ETF"

# TASK 2 - RISK ESTIMATION
risk_model = risk_weights
risk_model.index.name = "ETF"

# TASK 3 - HEDGE CREATION
if isinstance(hedge_notionals, pd.Series):
    hedging_basket = hedge_notionals.to_frame(name="Notional")
else:
    hedging_basket = hedge_notionals.copy()
    if "Notional" not in hedging_basket.columns and hedging_basket.shape[1] >= 1:
        hedging_basket.columns = ["Notional"]
hedging_basket.index.name = "Instrument"

#######################################################################################################################
# SUBMISSION FORMAT - DO NOT MODIFY/DELETE THE BELOW CODE
#######################################################################################################################

submission = {
  "nav": nav_model.to_csv(),
  "risk": risk_model.to_csv(),
  "hedge": hedging_basket.to_csv() }

final_output = json.dumps( submission )
print( final_output )