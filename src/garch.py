"""
GARCH(1,1) and TGARCH(1,1) with AR(1) mean equation and Student-t errors.
Estimated by maximum likelihood, hand-coded (no `arch` package).

Model, following Latief & Lefen (2018):

  Mean:      R_t = b0 + b1 * R_{t-1} + u_t
  GARCH:     s2_t = w + a * u_{t-1}^2 + b * s2_{t-1}
  TGARCH:    s2_t = w + a * u_{t-1}^2 + g * u_{t-1}^2 * d_{t-1} + b * s2_{t-1}
             where d_{t-1} = 1 if u_{t-1} < 0, else 0

  u_t | F_{t-1} ~ Student-t(nu), scaled to variance s2_t
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln


# ----------------------------------------------------------------------
# Variance recursions
# ----------------------------------------------------------------------

# Both recursions have the form  s2[t] = x[t] + beta * s2[t-1],  where x[t] collects the
# terms that depend only on lagged residuals. That is a first-order linear filter, so it
# can be evaluated by scipy.signal.lfilter instead of an interpreted Python loop.
#
# This is the single most important performance decision in the module. The likelihood is
# evaluated tens of thousands of times per fit, and the loop version accounted for ~92% of
# total runtime (1267 of 1375 microseconds per evaluation at T=4000). The filter form is
# bit-identical -- max|difference| = 0.0 for both models -- and ~38x faster.
#
# Putting the t=0 seed in x[0] (with zero filter state) reproduces s2[0] = var(u) exactly.

def _garch_variance(u, omega, alpha, beta):
    """GARCH(1,1) conditional variance path."""
    x = np.empty_like(u)
    x[0] = np.var(u)                       # initialise at sample variance
    x[1:] = omega + alpha * u[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], x)


def _tgarch_variance(u, omega, alpha, gamma, beta):
    """TGARCH(1,1) conditional variance path with leverage term."""
    d = (u < 0.0).astype(float)
    x = np.empty_like(u)
    x[0] = np.var(u)
    x[1:] = omega + alpha * u[:-1] ** 2 + gamma * u[:-1] ** 2 * d[:-1]
    return lfilter([1.0], [1.0, -beta], x)


# ----------------------------------------------------------------------
# Student-t log-likelihood
# ----------------------------------------------------------------------

def _student_t_loglik(u, s2, nu):
    """
    Log-likelihood for u_t ~ t(nu) standardised to have variance s2_t.
    The (nu-2) scaling makes the distribution unit-variance before scaling by s2.
    """
    if nu <= 2.0 or np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return -np.inf

    const = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
             - 0.5 * np.log(np.pi * (nu - 2)))

    ll = (const
          - 0.5 * np.log(s2)
          - (nu + 1) / 2 * np.log1p(u ** 2 / ((nu - 2) * s2)))

    total = np.sum(ll)
    return total if np.isfinite(total) else -np.inf


# ----------------------------------------------------------------------
# Parameterisation
# ----------------------------------------------------------------------
#
# The model constraints (omega > 0, alpha >= 0, beta >= 0, alpha + beta <= 1, nu > 2,
# |b1| < 1) used to be imposed by returning a flat 1e10 penalty whenever a trial
# parameter vector violated them. That turns the likelihood into a cliff, and a
# derivative-free simplex search does the worst possible thing at a cliff: it collapses
# onto the edge and stops. The symptom was omega pinned at the 1e-6 floor with
# alpha + beta = 1.000000 for CZK, DKK and RON -- the degenerate corner where a handful
# of conditional variances go to nearly zero and each near-zero sigma contributes a
# density spike that inflates the likelihood without describing the data.
#
# The fix is to make the constraints structural rather than punitive. Every parameter is
# now a smooth transform of an unconstrained (or box-bounded) coordinate, so the search
# CANNOT leave the admissible region and there is no cliff to fall off. The surface is
# differentiable everywhere, which also lets L-BFGS-B replace Nelder-Mead.
#
#   omega = exp(t_w),         t_w bounded below at log(OMEGA_MIN)
#   b1    = tanh(t_b)                                       -> |b1| < 1
#   p     = sigmoid(t_p)      total persistence             -> 0 < p < 1
#   w     = sigmoid(t_a)      alpha's share of p
#   nu    = NU_MIN + exp(t_n)                               -> nu > NU_MIN
#
#   GARCH:   alpha = p * w,              beta = p * (1 - w)
#   TGARCH:  g = tanh(t_g)
#            alpha = p * w * (1 - g),    gamma = 2 * p * w * g,   beta = p * (1 - w)
#            so alpha + 0.5 * gamma + beta = p exactly, alpha >= 0, and
#            alpha + gamma = p * w * (1 + g) >= 0, with the sign of gamma left free.
#
# The floor on omega is the substantive part of the fix, and it has to be RELATIVE to the
# series. An absolute floor of 1e-6 is no floor at all -- it admits the degenerate
# solutions. But an absolute floor of 1e-4 is far too aggressive at the other end: the
# krone's entire return variance is 0.000211 in percent-squared units, so 1e-4 would be
# 47% of it, and the "floor" would be dictating the fit rather than excluding an artifact.
# The six series differ in variance by three orders of magnitude (DKK 0.0002, HUF 0.304),
# so any single absolute number is wrong for most of them.
#
# Scaling by the sample variance fixes that. The degenerate optima found by the old
# simplex search sat at omega/var(r) between 1e-4 and 1e-5; a floor at 1e-3 of the sample
# variance excludes them for every series while remaining far below any economically
# meaningful variance intercept.
#
# Note that alpha + beta = 1 is NOT excluded. For a hard peg like the krone, permanent
# variance shocks are the honest description, and the boundary is approached to within
# 1e-5. What is excluded is omega collapsing to zero, which is a numerical artifact.

OMEGA_MIN_REL = 1e-3    # floor on omega, as a fraction of the sample variance
NU_MIN = 2.05
T_P_MAX = 12.0          # sigmoid(12) = 0.99999386: IGARCH in all but name


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))          # overflow-free logistic


def _logit(p, lo=1e-12):
    p = min(max(p, lo), 1.0 - lo)
    return np.log(p / (1.0 - p))


def _unpack(x, model):
    """Map unconstrained coordinates to natural parameters. Admissible by construction."""
    b0 = x[0]
    b1 = np.tanh(x[1])
    omega = np.exp(x[2])
    p = _sigmoid(x[3])
    w = _sigmoid(x[4])

    if model == "garch":
        alpha, gamma, beta = p * w, 0.0, p * (1.0 - w)
        nu = NU_MIN + np.exp(x[5])
    else:
        g = np.tanh(x[5])
        alpha = p * w * (1.0 - g)
        gamma = 2.0 * p * w * g
        beta = p * (1.0 - w)
        nu = NU_MIN + np.exp(x[6])

    return b0, b1, omega, alpha, gamma, beta, nu


def _pack(params, model, omega_floor):
    """Inverse of _unpack, for turning a natural-parameter start into a coordinate."""
    if model == "garch":
        b0, b1, omega, alpha, beta, nu = params
        gamma = 0.0
    else:
        b0, b1, omega, alpha, gamma, beta, nu = params

    p = alpha + 0.5 * gamma + beta
    w = (alpha + 0.5 * gamma) / p if p > 0 else 0.5

    x = [b0, np.arctanh(np.clip(b1, -0.99, 0.99)),
         np.log(max(omega, omega_floor * 1.5)),
         np.clip(_logit(p), -T_P_MAX, T_P_MAX), _logit(w)]

    if model == "garch":
        x.append(np.log(max(nu - NU_MIN, 1e-3)))
    else:
        denom = 2.0 * p * w
        g = gamma / denom if denom > 0 else 0.0
        x += [np.arctanh(np.clip(g, -0.99, 0.99)),
              np.log(max(nu - NU_MIN, 1e-3))]
    return np.asarray(x, dtype=float)


def _bounds(model, omega_floor):
    """Box bounds in coordinate space. These are the constraints; nothing else enforces
    them, and nothing else needs to."""
    b = [(None, None),                       # b0
         (-3.0, 3.0),                        # t_b1  -> |b1| < 0.995
         (np.log(omega_floor), np.log(1e4)),  # t_w   -> omega >= omega_floor
         (-T_P_MAX, T_P_MAX),                # t_p   -> persistence
         (-12.0, 12.0)]                      # t_a   -> alpha share
    if model == "tgarch":
        b.append((-3.0, 3.0))                # t_g   -> leverage split
    b.append((np.log(1e-3), np.log(200.0)))  # t_nu  -> 2.051 < nu < 202
    return b


def _neg_loglik_t(x, r, model):
    """Objective in coordinate space. Smooth, and finite everywhere by construction."""
    b0, b1, omega, alpha, gamma, beta, nu = _unpack(x, model)

    u = r[1:] - b0 - b1 * r[:-1]
    s2 = (_garch_variance(u, omega, alpha, beta) if model == "garch"
          else _tgarch_variance(u, omega, alpha, gamma, beta))

    ll = _student_t_loglik(u, s2, nu)
    # Only reachable through floating-point overflow in the recursion, not through a
    # constraint violation; a large finite value keeps the optimiser well-behaved.
    return 1e10 if not np.isfinite(ll) else -ll


# ----------------------------------------------------------------------
# Negative log-likelihood in natural parameters (kept for verification)
# ----------------------------------------------------------------------

def _neg_loglik(params, r, model):
    """Objective in natural parameters, with the old penalty barrier.

    No longer used for estimation -- it is retained so a fitted optimum can be
    re-evaluated in natural coordinates and checked against the transformed objective."""
    PENALTY = 1e10

    b0, b1 = params[0], params[1]

    if model == "garch":
        omega, alpha, beta, nu = params[2], params[3], params[4], params[5]
        gamma = 0.0
    else:
        omega, alpha, gamma, beta, nu = params[2], params[3], params[4], params[5], params[6]

    # --- parameter constraints -------------------------------------
    if omega <= 1e-6 or alpha < 0 or beta < 0 or nu <= 2.05:
        return PENALTY
    if abs(b1) >= 1:                       # stationary mean equation
        return PENALTY
    if model == "garch":
        persistence = alpha + beta
    else:
        persistence = alpha + 0.5 * gamma + beta   # E[d] = 0.5 for symmetric shocks
        if alpha + gamma < 0:
            return PENALTY
    if persistence > 1.0:                  # allow equality (IGARCH, e.g. hard pegs)
        return PENALTY

    # --- mean equation residuals -----------------------------------
    u = r[1:] - b0 - b1 * r[:-1]

    # --- variance path ---------------------------------------------
    if model == "garch":
        s2 = _garch_variance(u, omega, alpha, beta)
    else:
        s2 = _tgarch_variance(u, omega, alpha, gamma, beta)

    ll = _student_t_loglik(u, s2, nu)
    return PENALTY if not np.isfinite(ll) else -ll


# ----------------------------------------------------------------------
# Multi-start estimation
# ----------------------------------------------------------------------

def _start_grid(r, model):
    """
    Grid of starting values. Multi-start is necessary: the GARCH likelihood has a
    degenerate region as omega -> 0 where the likelihood is high but the fitted
    model is invalid. A single start can land there and report a spurious optimum.
    """
    var_r = np.var(r)
    b0_0, b1_0 = np.mean(r), 0.0

    starts = []
    for alpha in (0.03, 0.06, 0.10, 0.15, 0.20):
        for beta in (0.75, 0.82, 0.88, 0.94):
            if alpha + beta >= 0.999:
                continue
            omega = max(var_r * (1 - alpha - beta), 1e-4)
            for nu in (5.0, 8.0):
                if model == "garch":
                    starts.append([b0_0, b1_0, omega, alpha, beta, nu])
                else:
                    starts.append([b0_0, b1_0, omega, alpha * 0.7, alpha * 0.6, beta, nu])
    return starts


def fit_garch(returns, model="garch", scale=100.0, verbose=False):
    """
    Fit GARCH(1,1)-t or TGARCH(1,1)-t by maximum likelihood.

    Parameters
    ----------
    returns : array-like of daily log returns (in decimal form, e.g. 0.004)
    model   : "garch" or "tgarch"
    scale   : returns are multiplied by this before estimation for numerical
              stability, then results are scaled back. 100 puts returns in percent.

    Returns
    -------
    dict with parameter estimates, conditional volatility (original units),
    log-likelihood, AIC/BIC, and convergence information.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)] * scale

    names = (["b0", "b1", "omega", "alpha", "beta", "nu"] if model == "garch"
             else ["b0", "b1", "omega", "alpha", "gamma", "beta", "nu"])

    starts = _start_grid(r, model)
    omega_floor = OMEGA_MIN_REL * np.var(r)
    bnds = _bounds(model, omega_floor)
    best, n_valid, lls = None, 0, []

    for x0 in starts:
        try:
            res = minimize(_neg_loglik_t, _pack(x0, model, omega_floor), args=(r, model),
                           method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": 20000, "maxfun": 20000,
                                    "ftol": 1e-12, "gtol": 1e-8})
        except Exception:
            continue

        if not np.isfinite(res.fun) or res.fun >= 1e9:
            continue

        n_valid += 1
        lls.append(-res.fun)
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("No admissible optimum found across the start grid.")

    b0, b1, omega, alpha, gamma, beta, nu = _unpack(best.x, model)
    persistence = alpha + beta if model == "garch" else alpha + 0.5 * gamma + beta

    # Which box bounds are active at the optimum. omega sitting on its floor means the
    # likelihood still wants the degenerate corner and the floor is the only thing
    # holding it back -- that fit should not be reported without comment. Persistence on
    # its ceiling is the benign case: an IGARCH series pressed against alpha + beta = 1.
    at_bound = []
    if best.x[2] <= np.log(omega_floor) + 1e-6:
        at_bound.append("omega=floor")
    if best.x[3] >= T_P_MAX - 1e-6:
        at_bound.append("persistence=1")

    p = np.array([b0, b1, omega, alpha, beta, nu] if model == "garch"
                 else [b0, b1, omega, alpha, gamma, beta, nu])
    if model == "garch":
        gamma = None

    u = r[1:] - b0 - b1 * r[:-1]
    s2 = (_garch_variance(u, omega, alpha, beta) if model == "garch"
          else _tgarch_variance(u, omega, alpha, gamma, beta))

    ll = -best.fun
    k = len(p)
    n = len(u)

    out = {
        "model": model,
        "params": dict(zip(names, p)),
        "persistence": persistence,
        "loglik": ll,
        "aic": -2 * ll + 2 * k,
        "bic": -2 * ll + k * np.log(n),
        "nobs": n,
        # divide by `scale` to return volatility in the units of the input returns
        "cond_vol": np.sqrt(s2) / scale,
        "std_resid": u / np.sqrt(s2),
        "n_valid_starts": n_valid,
        "n_starts": len(starts),
        "converged": best.success,
        # How many starts reached a materially different optimum. This is the honest
        # multimodality measure: n_valid counts starts that finished, not starts that
        # agreed, so a run can report 26/26 while the answer is start-dependent.
        "n_distinct_optima": 1 + sum(1 for v in lls if ll - v > 0.1),
        "ll_spread": ll - min(lls) if lls else 0.0,
        "at_bound": at_bound,
        "omega_floor": omega_floor,
    }

    if verbose:
        flag = ("  [at bound: " + ", ".join(at_bound) + "]") if at_bound else ""
        print(f"{model.upper():7s} LL={ll:10.2f}  omega={omega:.5f}  "
              f"persistence={persistence:.6f}  nu={nu:.2f}  "
              f"starts {n_valid}/{out['n_starts']}  "
              f"distinct={out['n_distinct_optima']}{flag}")

    return out


# ----------------------------------------------------------------------
# Convenience wrapper: fit all currencies, return annual mean volatility
# ----------------------------------------------------------------------

def annual_volatility(returns_df, model="garch", verbose=True):
    """
    Fit `model` to each column of a daily returns DataFrame and return a
    long-format DataFrame of annual mean conditional volatility.

    returns_df : DataFrame, DatetimeIndex, one column per currency, log returns.
    """
    import pandas as pd

    fits, rows = {}, []

    for ccy in returns_df.columns:
        s = returns_df[ccy].dropna()
        fit = fit_garch(s.values, model=model, verbose=verbose)
        fits[ccy] = fit

        vol = pd.Series(fit["cond_vol"], index=s.index[1:])   # AR(1) loses one obs
        annual = vol.groupby(vol.index.year).mean()

        for year, v in annual.items():
            rows.append({"country": ccy, "year": int(year), "volatility": v})

    return pd.DataFrame(rows), fits


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Recovery check on simulated data with known parameters.
    rng = np.random.default_rng(0)
    T = 4000
    w_true, a_true, b_true, nu_true = 0.02, 0.08, 0.90, 6.0

    u = np.zeros(T)
    s2 = np.zeros(T)
    s2[0] = w_true / (1 - a_true - b_true)
    for t in range(1, T):
        s2[t] = w_true + a_true * u[t - 1] ** 2 + b_true * s2[t - 1]
        u[t] = np.sqrt(s2[t]) * rng.standard_t(nu_true) / np.sqrt(nu_true / (nu_true - 2))

    r = u / 100.0   # convert to decimal returns, as real data would be

    fit = fit_garch(r, model="garch", verbose=True)
    print("\ntrue :  omega=%.4f alpha=%.4f beta=%.4f nu=%.2f" %
          (w_true, a_true, b_true, nu_true))
    print("est  :  omega=%.4f alpha=%.4f beta=%.4f nu=%.2f" %
          (fit["params"]["omega"], fit["params"]["alpha"],
           fit["params"]["beta"], fit["params"]["nu"]))

    tfit = fit_garch(r, model="tgarch", verbose=True)
    print("tgarch gamma = %.4f (true 0, data is symmetric)" % tfit["params"]["gamma"])
