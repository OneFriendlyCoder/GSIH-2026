import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import json
import math
import numpy as np
from scipy.optimize import least_squares, brentq

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

# =====================================================================
# Task 1 - yield curve
# =====================================================================

def build_curve(zero_curve):
    pairs = sorted([(float(z["maturity"]), float(z["rate"])) for z in zero_curve])
    mats = np.array([p[0] for p in pairs], dtype=float)
    rates = np.array([p[1] for p in pairs], dtype=float)
    dfs = np.exp(-rates * mats)
    fwds = np.zeros_like(mats)
    for i in range(len(mats) - 1):
        fwds[i] = -(math.log(dfs[i + 1]) - math.log(dfs[i])) / (mats[i + 1] - mats[i])
    fwds[-1] = fwds[-2]
    return mats, rates, dfs, fwds

def interp_rate(t, mats, rates):
    if t <= mats[0]: return float(rates[0])
    if t >= mats[-1]: return float(rates[-1])
    return float(np.interp(t, mats, rates))

def discount_factor(t, mats, rates):
    if t <= 1e-12: return 1.0
    return math.exp(-interp_rate(t, mats, rates) * t)

def fwd_instantaneous(t, mats, rates):
    if t <= mats[0]: return float(rates[0])
    if t >= mats[-1]: return float(rates[-1])
    idx = int(np.searchsorted(mats, t, side="right")) - 1
    idx = max(0, min(idx, len(mats) - 2))
    slope = (rates[idx + 1] - rates[idx]) / (mats[idx + 1] - mats[idx])
    r_at = rates[idx] + slope * (t - mats[idx])
    return r_at + t * slope

# =====================================================================
# Hull-White 1F primitives
# =====================================================================

def B_hw(a, tau):
    if tau <= 0.0: return 0.0
    if abs(a) < 1e-7: return tau * (1.0 - 0.5 * a * tau)
    return (1.0 - math.exp(-a * tau)) / a

def integrated_variance(a, sigmas, breakpoints, T):
    if T <= 0.0: return 0.0
    V = 0.0
    for k in range(len(sigmas)):
        t_lo = breakpoints[k]
        t_hi = breakpoints[k + 1]
        if t_lo >= T: break
        t_hi_eff = min(t_hi, T)
        if abs(a) < 1e-7:
            V += sigmas[k] ** 2 * (t_hi_eff - t_lo)
        else:
            V += (sigmas[k] ** 2 / (2.0 * a)) * (
                math.exp(-2.0 * a * (T - t_hi_eff)) - math.exp(-2.0 * a * (T - t_lo))
            )
    return V

def hw_A(t, T, a, sigmas, breakpoints, mats, rates):
    if t >= T: return 1.0
    P_t = discount_factor(t, mats, rates)
    P_T = discount_factor(T, mats, rates)
    f_t = fwd_instantaneous(t, mats, rates)
    B = B_hw(a, T - t)
    V_t = integrated_variance(a, sigmas, breakpoints, t)
    return (P_T / P_t) * math.exp(B * f_t - 0.5 * B * B * V_t)

# =====================================================================
# Task 2 - European swaption pricing (Jamshidian) + calibration
# =====================================================================

def jamshidian_payer_price(T0, payment_dates, taus, K, a, sigmas, breakpoints, mats, rates):
    n = len(payment_dates)
    if n == 0: return 0.0

    c = [K * tau for tau in taus]
    c[-1] += 1.0

    A_vals = [hw_A(T0, T, a, sigmas, breakpoints, mats, rates) for T in payment_dates]
    B_vals = [B_hw(a, T - T0) for T in payment_dates]

    def bond_minus_one(r):
        s = 0.0
        for i in range(n):
            s += c[i] * A_vals[i] * math.exp(-B_vals[i] * r)
        return s - 1.0

    try:
        f_lo = bond_minus_one(-10.0)
        f_hi = bond_minus_one(10.0)
        if f_lo * f_hi > 0:
            return 0.0
        r_star = brentq(bond_minus_one, -10.0, 10.0, xtol=1e-11, maxiter=100)
    except (ValueError, RuntimeError):
        return 0.0

    X_stars = [A_vals[i] * math.exp(-B_vals[i] * r_star) for i in range(n)]

    V_T0 = integrated_variance(a, sigmas, breakpoints, T0)
    V_T0_sqrt = math.sqrt(max(V_T0, 0.0))

    P_T0 = discount_factor(T0, mats, rates)
    P_Ts = [discount_factor(T, mats, rates) for T in payment_dates]

    price = 0.0
    for i in range(n):
        sigma_p = B_vals[i] * V_T0_sqrt
        X = X_stars[i]
        if sigma_p < 1e-14:
            ZBP_i = max(X * P_T0 - P_Ts[i], 0.0)
        else:
            h = (math.log(P_Ts[i] / (X * P_T0)) + 0.5 * sigma_p ** 2) / sigma_p
            ZBP_i = X * P_T0 * norm_cdf(-h + sigma_p) - P_Ts[i] * norm_cdf(-h)
        price += c[i] * ZBP_i

    return max(price, 0.0)

def prepare_swaption_specs(swaption_vols, mats, rates):
    specs = []
    for sv in swaption_vols:
        T0 = float(sv["expiry"])
        tenor = float(sv["tenor"])
        n_pay = int(round(tenor))
        payment_dates = [T0 + k for k in range(1, n_pay + 1)]
        taus = [1.0] * n_pay
        P_T0 = discount_factor(T0, mats, rates)
        P_Tn = discount_factor(T0 + tenor, mats, rates)
        annuity = sum(taus[k] * discount_factor(payment_dates[k], mats, rates) for k in range(n_pay))
        F = (P_T0 - P_Tn) / annuity
        sigma_N_mkt = float(sv["vol_bps"]) * 1e-4
        specs.append({
            "T0": T0, "payment_dates": payment_dates, "taus": taus,
            "K": F, "annuity": annuity, "sigma_N_mkt": sigma_N_mkt,
            "expiry": sv["expiry"], "tenor": sv["tenor"], "vol_bps": sv["vol_bps"],
        })
    return specs

def calibration_residuals(params, specs, breakpoints, mats, rates):
    a = float(params[0])
    sigmas = [float(s) for s in params[1:]]
    res = np.zeros(len(specs))
    for i, spec in enumerate(specs):
        try:
            price = jamshidian_payer_price(
                spec["T0"], spec["payment_dates"], spec["taus"], spec["K"],
                a, sigmas, breakpoints, mats, rates
            )
            sigma_N_model = price * math.sqrt(2.0 * math.pi / spec["T0"]) / spec["annuity"]
            
            # CRITICAL FIX: Scale to basis points so optimizer gradients don't vanish
            res[i] = (sigma_N_model - spec["sigma_N_mkt"]) * 10000.0
        except Exception:
            res[i] = 10000.0 # Huge penalty
    return res

def calibrate(specs, breakpoints, n_sigmas, mats, rates, x0=None):
    if x0 is None:
        # STRICTLY follow the prompt's suggested start to hit the intended global minimum
        x0 = np.array([0.05] + [0.008] * n_sigmas, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        
    lo = np.array([1e-6] + [1e-6] * n_sigmas, dtype=float)
    hi = np.array([1.0] + [0.5] * n_sigmas, dtype=float)
    x0 = np.minimum(np.maximum(x0, lo + 1e-9), hi - 1e-9)
    
    result = least_squares(
        calibration_residuals, x0,
        args=(specs, breakpoints, mats, rates),
        bounds=(lo, hi), method="trf",
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=500
    )
    a = float(result.x[0])
    sigmas = [float(s) for s in result.x[1:]]
    return a, sigmas, result

# =====================================================================
# Task 3 - Bermudan trinomial tree
# =====================================================================

def price_bermudan(spec, a, sigmas, breakpoints, mats, rates, steps_per_year=100, J_cap=100):
    strike = float(spec["strike"])
    notional = float(spec["notional"])
    swap_end = float(spec["swap_end"])
    exercise_dates = sorted([float(x) for x in spec["exercise_dates"]])

    dt = 1.0 / steps_per_year
    N_steps = int(round(swap_end * steps_per_year))

    exercise_steps = {int(round(ed * steps_per_year)) for ed in exercise_dates if 0 < int(round(ed * steps_per_year)) <= N_steps}

    bp_arr = np.array(breakpoints, dtype=float)
    sigma_step = np.zeros(N_steps)
    for i in range(N_steps):
        t_mid = (i + 0.5) * dt
        idx = int(np.searchsorted(bp_arr, t_mid, side="right")) - 1
        idx = max(0, min(idx, len(sigmas) - 1))
        sigma_step[i] = sigmas[idx]

    sigma_max = max(max(sigmas), 1e-8)
    dx = sigma_max * math.sqrt(3.0 * dt)
    j_max_natural = int(math.ceil(0.184 / (max(a, 1e-6) * dt)))
    j_max = max(20, min(j_max_natural, J_cap))

    J = 2 * j_max + 1
    j_arr = np.arange(-j_max, j_max + 1, dtype=float)

    k_arr = j_arr.astype(int).copy()
    k_arr[-1] = j_max - 1
    k_arr[0] = -j_max + 1

    a_dt = a * dt
    eta_arr = j_arr * (1.0 - a_dt) - k_arr
    eta_sq = eta_arr ** 2

    idx_kp1 = (k_arr + 1) + j_max
    idx_k = k_arr + j_max
    idx_km1 = (k_arr - 1) + j_max
    exp_jdx_dt = np.exp(-j_arr * dx * dt)

    Q = np.zeros(J)
    Q[j_max] = 1.0
    alpha = np.zeros(N_steps)

    for i in range(N_steps):
        sig_ratio_sq = (sigma_step[i] / sigma_max) ** 2
        P_next = discount_factor((i + 1) * dt, mats, rates)
        sum_Q_disc = max(float(np.dot(Q, exp_jdx_dt)), 1e-300)
        alpha[i] = -math.log(P_next / sum_Q_disc) / dt

        p_u = (sig_ratio_sq / 3.0 + eta_sq + eta_arr) * 0.5
        p_d = (sig_ratio_sq / 3.0 + eta_sq - eta_arr) * 0.5
        p_m = 1.0 - p_u - p_d

        Q_w = Q * (math.exp(-alpha[i] * dt) * exp_jdx_dt)
        Q_new = np.zeros(J)
        np.add.at(Q_new, idx_kp1, Q_w * p_u)
        np.add.at(Q_new, idx_k,   Q_w * p_m)
        np.add.at(Q_new, idx_km1, Q_w * p_d)
        Q = Q_new

    V = np.zeros(J)
    for i in range(N_steps - 1, -1, -1):
        sig_ratio_sq = (sigma_step[i] / sigma_max) ** 2
        p_u = (sig_ratio_sq / 3.0 + eta_sq + eta_arr) * 0.5
        p_d = (sig_ratio_sq / 3.0 + eta_sq - eta_arr) * 0.5
        p_m = 1.0 - p_u - p_d

        disc = math.exp(-alpha[i] * dt) * exp_jdx_dt
        cont = disc * (p_u * V[idx_kp1] + p_m * V[idx_k] + p_d * V[idx_km1])

        if i in exercise_steps:
            r_nodes = alpha[i] + j_arr * dx
            swap_pv = compute_swap_pv_vec(i * dt, swap_end, strike, r_nodes, a, sigmas, breakpoints, mats, rates)
            V = np.maximum(np.maximum(swap_pv, 0.0), cont)
        else:
            V = cont

    return notional * float(V[j_max])

def compute_swap_pv_vec(T_ex, swap_end, K, r_arr, a, sigmas, breakpoints, mats, rates):
    payment_dates = []
    t = swap_end
    while t > T_ex + 1e-9:
        payment_dates.append(t)
        t -= 1.0
    payment_dates.reverse()
    if not payment_dates: return np.zeros_like(r_arr)

    P_list = []
    for T_i in payment_dates:
        A_val = hw_A(T_ex, T_i, a, sigmas, breakpoints, mats, rates)
        B_val = B_hw(a, T_i - T_ex)
        P_list.append(A_val * np.exp(-B_val * r_arr))

    taus = [T_i - prev for T_i, prev in zip(payment_dates, [T_ex] + payment_dates[:-1])]
    annuity = sum(tau * P_i for tau, P_i in zip(taus, P_list))
    return (1.0 - P_list[-1]) - K * annuity

# =====================================================================
# Main
# =====================================================================

def main():
    data = json.loads(sys.stdin.read())
    zero_curve = data["zero_curve"]
    swaption_vols = data["swaption_vols"]
    spec = data["bermudan_spec"]

    exercise_dates = sorted([float(e) for e in spec["exercise_dates"]])
    swap_end = float(spec["swap_end"])
    breakpoints = sorted({0.0, *exercise_dates, swap_end})
    n_sigmas = len(breakpoints) - 1

    # Task 1
    mats, rates, dfs, fwds = build_curve(zero_curve)
    curve_out = [{"maturity": float(m), "discount_factor": round(float(dfs[i]), 4), "forward_rate": round(float(fwds[i]), 4)} for i, m in enumerate(mats)]

    # Task 2 - Base Calibration
    specs_base = prepare_swaption_specs(swaption_vols, mats, rates)
    
    # x0=None forces the strictly specified [0.05, 0.008...] guess
    a_cal, sig_cal, _ = calibrate(specs_base, breakpoints, n_sigmas, mats, rates, x0=None)
    x_base = np.array([a_cal] + sig_cal, dtype=float)

    # Task 3 - Pricing
    price = price_bermudan(spec, a_cal, sig_cal, breakpoints, mats, rates)

    # Task 4 - Vega
    vega_results = []
    for sv in swaption_vols:
        bumped_vols = [dict(sv2) for sv2 in swaption_vols]
        for bv in bumped_vols:
            if bv["expiry"] == sv["expiry"] and bv["tenor"] == sv["tenor"]:
                bv["vol_bps"] = float(bv["vol_bps"]) + 1.0
                
        specs_b = prepare_swaption_specs(bumped_vols, mats, rates)
        # Warm start using x_base so the optimizer doesn't jump valleys
        a_b, sig_b, _ = calibrate(specs_b, breakpoints, n_sigmas, mats, rates, x0=x_base)
        price_b = price_bermudan(spec, a_b, sig_b, breakpoints, mats, rates)
        vega_results.append({"expiry": sv["expiry"], "tenor": sv["tenor"], "vega_dollars_per_bp": round(price_b - price, 4)})

    # Task 4 - Delta
    BUMP = 1e-4
    delta_results = []
    for z in zero_curve:
        bumped_curve = [dict(z2) for z2 in zero_curve]
        for bz in bumped_curve:
            if bz["maturity"] == z["maturity"]:
                bz["rate"] = float(bz["rate"]) + BUMP
                
        mats_b, rates_b, _, _ = build_curve(bumped_curve)
        specs_d = prepare_swaption_specs(swaption_vols, mats_b, rates_b)
        
        # Warm start using x_base so the optimizer doesn't jump valleys
        a_d, sig_d, _ = calibrate(specs_d, breakpoints, n_sigmas, mats_b, rates_b, x0=x_base)
        price_d = price_bermudan(spec, a_d, sig_d, breakpoints, mats_b, rates_b)
        delta_results.append({"maturity": float(z["maturity"]), "delta_dollars_per_bp": round(price_d - price, 4)})

    calibration = {"model": "Hull-White", "parameters": {"mean_reversion": round(a_cal, 6)}}
    for i, s in enumerate(sig_cal):
        calibration["parameters"][f"sigma_{i + 1}"] = round(s, 6)

    output = {"curve": curve_out, "calibration": calibration, "price_dollars": round(price, 2), "vega": vega_results, "delta": delta_results}
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()