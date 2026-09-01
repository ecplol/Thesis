"""
Stage 2: panel regression of real GDP growth on GARCH exchange-rate volatility,
euro-area trade intensity, and their interaction.

Stage 1 (per-currency GARCH) is assumed done. This module takes its output --
a long DataFrame with columns [country, year, volatility] keyed by CURRENCY code --
and produces the estimated panel.

Specification
-------------
    growth_it = a_i + t_t + b1*vol_it + b2*trade_it + b3*(vol x trade)_it
                + controls_it + e_it

  growth      real GDP growth, Eurostat nama_10_gdp, unit CLV_PCH_PRE.
              NOT CP_MEUR: euro-denominated GDP moves mechanically with the
              exchange rate, so regressing it on exchange-rate volatility would
              put the regressor inside the dependent variable.
  vol         annual mean GARCH conditional volatility against EUR
  trade_int   (exports + imports with the 20 euro-area members) / GDP, in %
  controls    GFCF (% GDP), HICP inflation, government expenditure (% GDP)

Volatility and trade intensity are centered before the product is formed. The
interaction coefficient is unchanged by centering; what changes is the meaning
of b1 and b2, which become the marginal effects at the sample means rather than
at the (never-observed) point vol = trade = 0, and their collinearity with the
product term, which drops from severe to negligible.
"""

import glob
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# The volatility panel is keyed by currency, every Eurostat table by country.
# Merging without this map yields an empty frame rather than an error, which is
# why every merge below is followed by an assertion on the row count.
CCY_TO_GEO = {"CZK": "CZ", "DKK": "DK", "HUF": "HU",
              "PLN": "PL", "RON": "RO", "SEK": "SE"}

GEO = sorted(CCY_TO_GEO.values())
YEARS = range(2005, 2025)                       # 2024 inclusive
N_EXPECTED = len(GEO) * len(YEARS)              # 6 x 20 = 120

# Stage 1 returns volatility in decimal daily log-return units (~0.004). Left as
# is, a coefficient reads "percentage points of growth per unit of daily
# volatility" -- a one-unit move being roughly 250 times anything ever observed,
# which is why the raw estimates run into the hundreds. Multiplying by 100 puts
# the regressor in percent, so coefficients read per percentage point of daily
# volatility. This is a change of units only: t-statistics, p-values, R^2 and
# every marginal effect are identical.
VOL_SCALE = 100.0


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _read_eurostat(path, **filters):
    """Read a Eurostat CSV and filter on dimension columns, e.g. unit='CP_MEUR'."""
    df = pd.read_csv(path)
    for col, val in filters.items():
        if col not in df.columns:
            raise KeyError(f"{os.path.basename(path)}: no column '{col}' "
                           f"(has {sorted(c for c in df.columns if c.islower())})")
        df = df[df[col] == val]
    return df


def _find(pattern):
    """Locate a data file by glob; return None rather than raising, so a missing
    control degrades the specification instead of killing the run."""
    hits = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return hits[0] if hits else None


def load_gdp():
    """Real GDP growth (CLV_PCH_PRE) and nominal GDP level (CP_MEUR, the
    denominator for the ratio variables), both na_item = B1GQ."""
    path = _find("nama_10_gdp__custom_22321200_linear_2_0.csv")
    if path is None:
        raise FileNotFoundError("nama_10_gdp main file not found in data/")

    keep = ["geo", "TIME_PERIOD", "OBS_VALUE"]
    growth = (_read_eurostat(path, unit="CLV_PCH_PRE", na_item="B1GQ")[keep]
              .rename(columns={"geo": "country", "TIME_PERIOD": "year",
                               "OBS_VALUE": "growth"}))
    level = (_read_eurostat(path, unit="CP_MEUR", na_item="B1GQ")[keep]
             .rename(columns={"geo": "country", "TIME_PERIOD": "year",
                              "OBS_VALUE": "gdp_meur"}))
    return growth, level


def load_trade_intensity(gdp_level):
    """(exports + imports with the 20 euro-area members) / GDP, in %.

    Trade values are in euros, GDP in millions of euros, hence the 1e6."""
    path = _find("ds-045409*.csv")
    if path is None:
        raise FileNotFoundError("trade file ds-045409*.csv not found in data/")

    t = pd.read_csv(path)
    n_partners = t["partner"].nunique()
    if n_partners != 20:
        raise ValueError(f"expected the 20 EA members as partners, found {n_partners}")

    trade = (t.groupby(["reporter", "TIME_PERIOD"], as_index=False)["OBS_VALUE"].sum()
             .rename(columns={"reporter": "country", "TIME_PERIOD": "year",
                              "OBS_VALUE": "trade_eur"}))
    trade = trade.merge(gdp_level, on=["country", "year"], how="inner")
    trade["trade_int"] = trade["trade_eur"] / 1e6 / trade["gdp_meur"] * 100
    return trade[["country", "year", "trade_int"]]


def load_controls(gdp_level):
    """GFCF (% GDP), HICP inflation, government expenditure (% GDP).

    Returns a dict name -> DataFrame. A control whose file is absent is simply
    omitted; the caller reports which ones made it in."""
    out = {}

    # --- GFCF, % of GDP: P51G / B1GQ, both in current prices -----------
    # gfcf.csv covers 2005-2025 and supersedes the earlier 2016-only extract,
    # which had truncated the panel to 54 observations.
    path = _find("gfcf.csv") or _find("gross_fixed_capital_formation.csv")
    if path is not None:
        g = (_read_eurostat(path, unit="CP_MEUR", na_item="P51G")
             [["geo", "TIME_PERIOD", "OBS_VALUE"]]
             .rename(columns={"geo": "country", "TIME_PERIOD": "year",
                              "OBS_VALUE": "gfcf_meur"}))
        g = g.merge(gdp_level, on=["country", "year"], how="inner")
        g["gfcf_pct_gdp"] = g["gfcf_meur"] / g["gdp_meur"] * 100
        out["gfcf_pct_gdp"] = g[["country", "year", "gfcf_pct_gdp"]]

    # --- HICP, annual average rate of change ---------------------------
    path = _find("annual_average_rate_change.csv")
    if path is not None:
        h = (_read_eurostat(path, unit="RCH_A_AVG", coicop="CP00")
             [["geo", "TIME_PERIOD", "OBS_VALUE"]]
             .rename(columns={"geo": "country", "TIME_PERIOD": "year",
                              "OBS_VALUE": "hicp"}))
        out["hicp"] = h

    # --- general government expenditure, % of GDP ----------------------
    path = _find("general_government_exp.csv")
    if path is not None:
        e = (_read_eurostat(path, unit="PC_GDP", na_item="TE")
             [["geo", "TIME_PERIOD", "OBS_VALUE"]]
             .rename(columns={"geo": "country", "TIME_PERIOD": "year",
                              "OBS_VALUE": "gov_exp_pct_gdp"}))
        out["gov_exp_pct_gdp"] = e

    return out


# ----------------------------------------------------------------------
# Panel assembly
# ----------------------------------------------------------------------

def _check(df, stage, expect=N_EXPECTED, strict=True):
    """Assert the row count after a merge. A currency/country key mismatch
    produces zero rows silently, so this is the guard that catches it."""
    n = len(df)
    msg = f"{stage:34s} rows={n:4d} (expected {expect})"
    if n != expect:
        if n == 0:
            raise AssertionError(f"{msg}  <- EMPTY: key mismatch, "
                                 f"check the currency -> country map")
        if strict:
            raise AssertionError(f"{msg}  <- row count off; check keys/years")
        print(f"  ! {msg}")
    else:
        print(f"  . {msg}")
    return df


def build_panel(vol_df, verbose=True):
    """Assemble the estimation panel from the stage-1 volatility output.

    vol_df : long DataFrame with columns [country, year, volatility], where
             `country` holds CURRENCY codes (CZK, DKK, ...).
    """
    if verbose:
        print("=" * 72)
        print("PANEL ASSEMBLY")
        print("=" * 72)

    vol = vol_df.rename(columns={"volatility": "vol"})[["country", "year", "vol"]].copy()

    # currency -> country, before anything is merged
    unknown = set(vol["country"]) - set(CCY_TO_GEO)
    if unknown:
        raise KeyError(f"volatility panel has unmapped currency codes: {sorted(unknown)}")
    vol["country"] = vol["country"].map(CCY_TO_GEO)
    vol["year"] = vol["year"].astype(int)
    vol["vol"] = vol["vol"] * VOL_SCALE            # decimal -> percent, units only
    vol = vol[vol["year"].isin(YEARS)]
    _check(vol, "volatility (after ccy -> geo)")

    growth, level = load_gdp()
    for d in (growth, level):
        d["year"] = d["year"].astype(int)
    growth = growth[growth["country"].isin(GEO) & growth["year"].isin(YEARS)]
    level = level[level["country"].isin(GEO) & level["year"].isin(YEARS)]

    panel = vol.merge(growth, on=["country", "year"], how="inner")
    _check(panel, "+ real GDP growth")

    trade = load_trade_intensity(level)
    panel = panel.merge(trade, on=["country", "year"], how="inner")
    _check(panel, "+ trade intensity")

    for name, ctrl in load_controls(level).items():
        ctrl["year"] = ctrl["year"].astype(int)
        merged = panel.merge(ctrl, on=["country", "year"], how="left")
        _check(merged, f"+ {name}")
        cov = merged[name].notna().sum()
        if cov < N_EXPECTED:
            yrs = merged.loc[merged[name].notna(), "year"]
            print(f"    coverage {cov}/{N_EXPECTED} obs, years {yrs.min()}-{yrs.max()}")
        panel = merged

    # --- centered interaction ------------------------------------------
    panel["vol_c"] = panel["vol"] - panel["vol"].mean()
    panel["trade_c"] = panel["trade_int"] - panel["trade_int"].mean()
    panel["vol_x_trade"] = panel["vol_c"] * panel["trade_c"]

    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
    _check(panel, "final panel")
    return panel


# ----------------------------------------------------------------------
# Standard errors
# ----------------------------------------------------------------------

def driscoll_kraay_cov(y, X, time_idx, maxlags=2):
    """Full Driscoll-Kraay covariance matrix: Newey-West applied to moment conditions
    summed across countries within each year, so it is robust to heteroskedasticity,
    serial correlation, and -- the reason it is the headline here -- to cross-sectional
    dependence. Six small open economies facing the same euro-area shocks are anything
    but independent, which is exactly what cluster-by-country assumes.

    The whole matrix is returned, not just its diagonal, because a marginal effect from
    an interaction model needs the off-diagonal covariance term."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    e = y - X @ (XtX_inv @ X.T @ y)

    ht = pd.DataFrame(X * e[:, None]).groupby(np.asarray(time_idx)).sum().values
    S = ht.T @ ht
    for l in range(1, maxlags + 1):
        w = 1.0 - l / (maxlags + 1.0)                    # Bartlett kernel
        G = ht[l:].T @ ht[:-l]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv


def driscoll_kraay(y, X, time_idx, maxlags=2):
    """Driscoll-Kraay standard errors."""
    return np.sqrt(np.diag(driscoll_kraay_cov(y, X, time_idx, maxlags)))


def wald_f(params, cov, columns, terms, df_resid):
    """Wald F-test that every coefficient in `terms` is jointly zero, under an
    arbitrary covariance matrix (here Driscoll-Kraay).

    F = (Rb)' (R V R')^-1 (Rb) / q  ~  F(q, df_resid).

    This is the test that matters for an interaction model: trade_int and
    volatility x trade_int are two pieces of one hypothesis -- that trade
    exposure plays no role at all -- and the individual t-statistics can each be
    unremarkable while the pair is jointly informative, because the two
    coefficients are estimated together and their covariance is not zero."""
    from scipy import stats as st
    cols = list(columns)
    idx = [cols.index(t) for t in terms]
    q = len(idx)

    b = np.asarray(params, float)[idx]
    V = np.asarray(cov, float)[np.ix_(idx, idx)]

    stat = float(b @ np.linalg.solve(V, b) / q)
    return stat, q, int(df_resid), float(st.f.sf(stat, q, df_resid))


# Joint hypotheses worth reporting. Each is (label, terms).
JOINT_TESTS = [
    ("trade_c = vol_x_trade = 0          (no role for trade exposure)",
     ["trade_c", "vol_x_trade"]),
    ("vol_c   = vol_x_trade = 0          (no role for volatility)",
     ["vol_c", "vol_x_trade"]),
    ("vol_c = trade_c = vol_x_trade = 0  (none of the three)",
     ["vol_c", "trade_c", "vol_x_trade"]),
]


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------

CORE = ["vol_c", "trade_c", "vol_x_trade"]


def _design(panel, spec, regressors):
    """Build the design matrix. Fixed effects enter as dummies with one category
    dropped in each dimension (the constant carries the omitted cell)."""
    X = panel[regressors].copy()
    if spec in ("country", "twoway"):
        X = pd.concat([X, pd.get_dummies(panel["country"], prefix="c",
                                         drop_first=True, dtype=float)], axis=1)
    if spec == "twoway":
        X = pd.concat([X, pd.get_dummies(panel["year"], prefix="y",
                                         drop_first=True, dtype=float)], axis=1)
    return sm.add_constant(X)


def estimate(panel, controls=None, maxlags=2, verbose=True):
    """Pooled OLS, country FE, two-way FE. Driscoll-Kraay is the headline;
    OLS / HAC / cluster p-values are printed alongside so the reader can see how
    much of the inference is doing work.

    Returns (fits, est): fits maps spec -> (statsmodels result, DK se, DK p, DK cov),
    and est is the estimation sample the fits were computed on -- returned rather
    than recomputed so the diagnostics align residuals to the same rows."""
    if controls is None:
        controls = [c for c in ["gfcf_pct_gdp", "hicp", "gov_exp_pct_gdp"]
                    if c in panel.columns]
    regressors = CORE + list(controls)

    est = panel.dropna(subset=["growth"] + regressors).copy()
    if verbose and len(est) < len(panel):
        dropped = sorted(set(panel["year"]) - set(est["year"]))
        print(f"\nNOTE: {len(panel) - len(est)} obs dropped for missing controls "
              f"-> N = {len(est)}" + (f", years missing entirely: {dropped}" if dropped else ""))

    from scipy import stats as st
    fits = {}

    for spec, title in [("pooled", "POOLED OLS"),
                        ("country", "COUNTRY FIXED EFFECTS"),
                        ("twoway", "TWO-WAY FIXED EFFECTS (country + year)")]:
        X = _design(est, spec, regressors)
        y = est["growth"]

        ols = sm.OLS(y, X).fit()
        hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        clu = sm.OLS(y, X).fit(cov_type="cluster",
                               cov_kwds={"groups": est["country"].values})

        V_dk = driscoll_kraay_cov(y, X, est["year"].values, maxlags=maxlags)
        se_dk = np.sqrt(np.diag(V_dk))
        df_resid = int(ols.df_resid)
        p_dk = 2 * st.t.sf(np.abs(ols.params.values / se_dk), df_resid)
        fits[spec] = (ols, se_dk, p_dk, V_dk)

        if not verbose:
            continue

        print(f"\n{'=' * 88}\n{title}   N={int(ols.nobs)}  "
              f"R2={ols.rsquared:.4f}  within-df={df_resid}\n{'=' * 88}")
        print(f"{'':18s}{'coef':>10s}{'se(DK)':>10s}{'p(DK)':>9s}"
              f"{'p(OLS)':>9s}{'p(HAC)':>9s}{'p(clu)':>9s}")
        for v in regressors:
            i = list(X.columns).index(v)
            print(f"{v:18s}{ols.params[v]:10.4f}{se_dk[i]:10.4f}{p_dk[i]:9.4f}"
                  f"{ols.pvalues[v]:9.4f}{hac.pvalues[v]:9.4f}{clu.pvalues[v]:9.4f}")

        print(f"\n  Joint Wald tests (Driscoll-Kraay covariance)")
        print(f"  {'':64s}{'F':>8s}{'df':>10s}{'p':>9s}")
        for label, terms in JOINT_TESTS:
            if not set(terms) <= set(X.columns):
                continue
            F, q, dfr, p = wald_f(ols.params.values, V_dk, X.columns, terms, df_resid)
            print(f"  {label:64s}{F:8.3f}{f'({q},{dfr})':>10s}{p:9.4f}")

    if verbose:
        print("\nOnly 6 clusters: the cluster-robust p-values are biased toward zero"
              "\n(the asymptotics need many clusters) and are shown for comparison only.")
    return fits, est


def vif_table(panel, regressors, centered=True):
    """VIF for the regressors. Reported both ways because the centering decision
    is visible here and nowhere else: the raw product is highly collinear with its
    own components, the centered one is not."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cols = list(regressors)
    d = panel[cols].dropna()
    if not centered:
        d = d.copy()
        d["vol_x_trade"] = panel.loc[d.index, "vol"] * panel.loc[d.index, "trade_int"]
    X = sm.add_constant(d).values
    return pd.Series([variance_inflation_factor(X, i + 1) for i in range(len(cols))],
                     index=cols, name="VIF" + ("(centered)" if centered else "(raw)"))


# ----------------------------------------------------------------------

def between_within(est, regressors):
    """Between- vs within-country standard deviation for each regressor.

    Fixed effects absorb the between variation, so a regressor whose variation is
    mostly cross-sectional is identified off whatever little is left. A ratio far
    above 1 is a warning that the FE coefficient rests on a thin slice of the data,
    which no standard-error correction can repair."""
    rows = {}
    for v in regressors:
        g = est.groupby("country")[v]
        between = g.mean().std(ddof=1)
        within = (est[v] - g.transform("mean")).std(ddof=1)
        rows[v] = {"sd_between": between, "sd_within": within,
                   "ratio": between / within if within > 0 else np.inf}
    return pd.DataFrame(rows).T


def diagnostics(est, fit, regressors, label="two-way FE"):
    """Normality, heteroskedasticity, functional form, and serial correlation.

    fit : the statsmodels result to diagnose (the headline specification)."""
    from scipy import stats as st
    from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
    from statsmodels.stats.stattools import jarque_bera

    print(f"\n{'=' * 72}\nDIAGNOSTICS ({label}, N={int(fit.nobs)})\n{'=' * 72}")

    print("-- between/within sd, identification under FE " + "-" * 25)
    bw = between_within(est, regressors)
    print(bw.round(3).to_string())
    weak = bw.index[bw["ratio"] > 3].tolist()
    print(f"   ratio > 3 -> mostly cross-sectional: "
          f"{weak if weak else 'none'}")

    resid = np.asarray(fit.resid, float)

    jb, jb_p, skew, kurt = jarque_bera(resid)
    print(f"\n-- Jarque-Bera  stat={jb:.2f}  p={jb_p:.4f}  "
          f"skew={skew:+.3f}  excess kurt={kurt - 3:+.3f}")
    print("   Rejection means non-normal residuals. Harmless for consistency of the"
          "\n   slopes; it matters only for exact small-sample t and F inference,"
          "\n   which the robust standard errors already replace.")

    bp, bp_p, _, _ = het_breuschpagan(resid, fit.model.exog)
    print(f"\n-- Breusch-Pagan  stat={bp:.2f}  p={bp_p:.4f}")
    print("   Rejection means heteroskedasticity, already handled by DK/HAC/cluster.")

    for power in (2, 3):
        r = linear_reset(fit, power=power, use_f=True)
        print(f"\n-- Ramsey RESET (powers up to {power})  F={r.statistic:.3f}  "
              f"p={r.pvalue:.4f}")
    print("   Rejection means omitted non-linearity or a misspecified functional form."
          "\n   Non-rejection is the outcome you want here.")

    # Within-country AR(1). The lag MUST be taken inside each country: lagging the
    # stacked residual vector splices one country's 2024 onto the next country's
    # 2005 and manufactures a break at every panel boundary.
    d = est[["country", "year"]].copy()
    d["_res"] = resid
    d = d.sort_values(["country", "year"])
    d["_res_lag"] = d.groupby("country")["_res"].shift(1)
    d = d.dropna(subset=["_res_lag"])

    ar = sm.OLS(d["_res"], sm.add_constant(d["_res_lag"])).fit()
    rho, p_rho = ar.params["_res_lag"], ar.pvalues["_res_lag"]
    print(f"\n-- Within-country AR(1)  rho={rho:+.4f}  p={p_rho:.4f}  "
          f"n={int(ar.nobs)} (lagged within country, {len(est) - int(ar.nobs)} "
          f"first-year obs dropped)")
    print("   Significant rho means the DK lag truncation should be widened; "
          + ("no evidence of it here." if p_rho > 0.05 else "there IS evidence of it."))

    return {"jb_p": jb_p, "bp_p": bp_p, "ar1_rho": rho, "ar1_p": p_rho,
            "between_within": bw}


def marginal_effects(panel, controls=None, spec="twoway", maxlags=2,
                     at=(0.05, 0.25, 0.50, 0.75, 0.95), verbose=True):
    """Effect of volatility on growth at different levels of trade intensity.

    With an interaction, the coefficient on vol_c is not "the effect of volatility" --
    it is the effect at MEAN trade intensity only. The effect at trade intensity t is

        d(growth)/d(vol) = b1 + b3 * (t - mean(trade_int))

    and its variance needs the covariance term, not just the two variances:

        Var = V11 + c^2 * V33 + 2c * V13,     c = t - mean(trade_int)

    Reporting only b1 hides the fact that the effect changes sign across the sample."""
    from scipy import stats as st

    if controls is None:
        controls = [c for c in ["gfcf_pct_gdp", "hicp", "gov_exp_pct_gdp"]
                    if c in panel.columns]
    regressors = CORE + list(controls)
    est = panel.dropna(subset=["growth"] + regressors).copy()

    X = _design(est, spec, regressors)
    ols = sm.OLS(est["growth"], X).fit()
    V = driscoll_kraay_cov(est["growth"], X, est["year"].values, maxlags)
    cols = list(X.columns)
    i1, i3 = cols.index("vol_c"), cols.index("vol_x_trade")
    b1, b3 = ols.params["vol_c"], ols.params["vol_x_trade"]
    tbar = panel["trade_int"].mean()

    rows = []
    for q in at:
        t = panel["trade_int"].quantile(q)
        c = t - tbar
        me = b1 + b3 * c
        se = np.sqrt(V[i1, i1] + c ** 2 * V[i3, i3] + 2 * c * V[i1, i3])
        p = 2 * st.t.sf(abs(me / se), int(ols.df_resid))
        rows.append({"quantile": f"q{int(q * 100):02d}", "trade_int": t,
                     "dGrowth/dVol": me, "se": se, "p": p})
    out = pd.DataFrame(rows).set_index("quantile")

    if verbose:
        print(f"\n{'=' * 72}\nMARGINAL EFFECT OF VOLATILITY ({spec})\n{'=' * 72}")
        print(out.round(4).to_string())
        if b3 != 0:
            cross = tbar - b1 / b3
            print(f"\nEffect crosses zero at trade intensity {cross:.1f}% "
                  f"(sample range {panel['trade_int'].min():.1f}-"
                  f"{panel['trade_int'].max():.1f}%)")
        print("Country mean trade intensity:")
        print("  " + "  ".join(f"{k}={v:.1f}" for k, v in
                               panel.groupby("country")["trade_int"].mean()
                               .sort_values().items()))
    return out


def robustness(panel, controls=None, spec="twoway", maxlags=2, verbose=True):
    """Re-estimate the headline specification on subsamples.

    Centering is deliberately NOT recomputed per subsample: the interaction coefficient
    is invariant to centering, and holding the origin fixed keeps the main effects
    comparable across rows."""
    from scipy import stats as st

    if controls is None:
        controls = [c for c in ["gfcf_pct_gdp", "hicp", "gov_exp_pct_gdp"]
                    if c in panel.columns]
    regressors = CORE + list(controls)

    CRISIS = [2008, 2009, 2020]
    cuts = [("full sample", panel.index == panel.index),
            ("excl. DKK (pegged)", panel["country"] != "DK"),
            (f"excl. crisis {CRISIS}", ~panel["year"].isin(CRISIS)),
            ("2010-2024 only", panel["year"] >= 2010),
            ("excl. COVID 2020-21", ~panel["year"].isin([2020, 2021]))]
    cuts += [(f"drop {c}", panel["country"] != c) for c in sorted(panel["country"].unique())]

    rows = []
    for lab, mask in cuts:
        est = panel[np.asarray(mask)].dropna(subset=["growth"] + regressors)
        if est["country"].nunique() < 2 or len(est) < len(regressors) + 10:
            continue
        X = _design(est, spec, regressors)
        ols = sm.OLS(est["growth"], X).fit()
        V = driscoll_kraay_cov(est["growth"], X, est["year"].values, maxlags)
        se = np.sqrt(np.diag(V))
        cols = list(X.columns)
        rec = {"N": int(ols.nobs)}
        for v in ["vol_c", "trade_c", "vol_x_trade"]:
            i = cols.index(v)
            rec[v] = ols.params[v]
            rec[v + "_p"] = 2 * st.t.sf(abs(ols.params[v] / se[i]), int(ols.df_resid))
        # Joint test on the trade pair, cut by cut: a single coefficient can lose
        # significance in a subsample simply because the SEs widen, while the pair
        # still rejects. Only if this column dies too has the finding actually gone.
        rec["F_trade"], _, _, rec["F_trade_p"] = wald_f(
            ols.params.values, V, cols, ["trade_c", "vol_x_trade"], int(ols.df_resid))
        rows.append(pd.Series(rec, name=lab))

    out = pd.DataFrame(rows)
    if verbose:
        print(f"\n{'=' * 88}\nROBUSTNESS ({spec}, Driscoll-Kraay p-values)\n{'=' * 88}")
        print(out.round(4).to_string())
        b = out["vol_x_trade"]
        print(f"\nInteraction across {len(out)} cuts: range [{b.min():+.4f}, {b.max():+.4f}], "
              f"sign flips = {int((np.sign(b) != np.sign(b.iloc[0])).sum())}, "
              f"significant at 5% in {int((out['vol_x_trade_p'] < 0.05).sum())}/{len(out)}")
        print(f"Joint F on (trade_c, vol_x_trade): rejects at 5% in "
              f"{int((out['F_trade_p'] < 0.05).sum())}/{len(out)} cuts")
    return out


def run(vol_df, controls=None, headline="twoway"):
    """Full stage 2: assemble, report collinearity, estimate, diagnose."""
    panel = build_panel(vol_df)

    ctrl = controls if controls is not None else [
        c for c in ["gfcf_pct_gdp", "hicp", "gov_exp_pct_gdp"] if c in panel.columns]
    regressors = CORE + list(ctrl)

    print("\n" + "=" * 72 + "\nCOLLINEARITY\n" + "=" * 72)
    print(pd.concat([vif_table(panel, regressors, centered=True),
                     vif_table(panel, regressors, centered=False)], axis=1)
          .round(2).to_string())
    print("Flag: VIF > 10.")

    fits, est = estimate(panel, controls=ctrl)
    diagnostics(est, fits[headline][0], regressors,
                label={"pooled": "pooled OLS", "country": "country FE",
                       "twoway": "two-way FE"}[headline])
    return panel, fits
