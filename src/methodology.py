#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.optimize import minimize
import seaborn as sns


# In[79]:


files ={
    "CZK": "/Users/eddiechiu/Documents/GitHub/Thesis/data/Czech_koruna.csv",
    "DKK": "/Users/eddiechiu/Documents/GitHub/Thesis/data/Danish_krone.csv",
    "HUF": "/Users/eddiechiu/Documents/GitHub/Thesis/data/Hungarian_forint.csv",
    "PLN": "/Users/eddiechiu/Documents/GitHub/Thesis/data/Polish_zloty.csv",
    "RON": "/Users/eddiechiu/Documents/GitHub/Thesis/data/Romanian_leu.csv",
    "SEK": "/Users/eddiechiu/Documents/GitHub/Thesis/data/Swedish_krona.csv",
    "USD_EUR": "/Users/eddiechiu/Documents/GitHub/Thesis/data/USD_EUR.csv",
}


# In[80]:


def load_data(path, currency):
    df = pd.read_csv(path)
    val_col = [c for c in df.columns if c not in ("DATE", "TIME PERIOD")][0]
    out = df[["DATE", val_col]].copy()
    out.columns = ["date", currency]
    out["date"] = pd.to_datetime(out["date"])
    out[currency] = pd.to_numeric(out[currency], errors="coerce")
    return out.set_index("date").sort_index()

rates = pd.concat([load_data(p, c) for c, p in files.items()], axis=1)


# In[81]:


rates = rates.loc["2000-01-01":"2024-12-31"] 
rates = rates.dropna()


# In[82]:


print(rates.head())
print(rates.shape)


# In[83]:


returns = np.log(rates / rates.shift(1)).dropna()


# In[84]:


model = arch_model(returns["PLN"] * 100, vol="Garch", p=1, q=1, dist="t")
result = model.fit(disp="off")

print(result.summary())


# In[85]:


results = {}

for currency in returns.columns:
    model = arch_model(returns[currency] * 100, vol="Garch", p=1, q=1, dist="t", rescale=True)
    results[currency] = model.fit(disp="off")
    print(f"\n=== {currency} ===")
    print(results[currency].summary())


# In[86]:


summary_table = pd.DataFrame({
    currency: {
        "alpha": res.params["alpha[1]"],
        "beta": res.params["beta[1]"],
        "alpha+beta": res.params["alpha[1]"] + res.params["beta[1]"],
        "nu": res.params["nu"],
        "converged": res.convergence_flag == 0,
    }
    for currency, res in results.items()
}).T

print(summary_table)


# In[87]:


returns["RON"].plot(figsize=(10,3), title="RON log returns")


# In[88]:


#new period 
rates_new = rates.loc["2005-01-01":"2025-12-31"] 
rates_new = rates_new.dropna()

returns_new = np.log(rates_new / rates_new.shift(1)).dropna()


# In[89]:


results_new = {}

for currency in returns_new.columns:
    model = arch_model(returns_new[currency] * 100, vol="Garch", p=1, q=1, dist="t", rescale=True,)
    results_new[currency] = model.fit(disp="off", options={"maxiter": 30000}) 

    print(f"\n=== {currency} ===")
    print(results_new[currency].summary())
    print(results_new[currency].cov_type)


# In[90]:


summary_table = pd.DataFrame({
    currency: {
        "alpha": res.params["alpha[1]"],
        "beta": res.params["beta[1]"],
        "alpha+beta": res.params["alpha[1]"] + res.params["beta[1]"],
        "nu": res.params["nu"],
        "converged": res.convergence_flag == 0,
    }
    for currency, res in results_new.items()
}).T

print(summary_table)


# In[91]:


returns_new["PLN"].plot(figsize=(10,3), title="PLN log returns")



# In[92]:


#maxiter adjustment dont work
#fix the CZK convergence issue by adjusting the optimization method and maxiter parameter


# In[93]:


#standardized residuals
std_resid = {}

for currency, result in results_new.items():
    std_resid[currency] = result.std_resid

std_resid_df = pd.DataFrame(std_resid)
std_resid_df = std_resid_df.dropna()

print(std_resid_df.shape)
print(std_resid_df.head())
print(std_resid_df.isna().sum())


# In[94]:


print(std_resid_df.index.min(), "→", std_resid_df.index.max())
print("PLN original result count:", len(results["PLN"].std_resid))


# In[95]:


# epsilon_t = standardized residuals from each currency's GARCH fit
epsilon = std_resid_df.values   # T x n matrix, rows = time, columns = currencies
T, n = epsilon.shape

# S = unconditional correlation matrix of epsilon_t (paper's S / Q_bar)
S = np.corrcoef(epsilon.T)

def dcc_loglik(params, epsilon, S):
    alpha, beta = params
    if alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    T, n = epsilon.shape
    Q_t = S.copy()
    loglik = 0

    for t in range(1, T):
        eps_prev = epsilon[t-1].reshape(-1, 1)

        # Eq. 22: Q_t = (1-alpha-beta)S + alpha * eps_{t-1} eps_{t-1}' + beta * Q_{t-1}
        Q_t = (1 - alpha - beta) * S + alpha * (eps_prev @ eps_prev.T) + beta * Q_t

        # Eq. 23-24: R_t = (I ⊙ Q_t^{1/2})^{-1} Q_t (I ⊙ Q_t^{1/2})^{-1}
        q_diag_sqrt = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(q_diag_sqrt, q_diag_sqrt)

        eps_t = epsilon[t]
        try:
            R_t_inv = np.linalg.inv(R_t)
            sign, logdet_R_t = np.linalg.slogdet(R_t)
            if sign <= 0:
                return 1e10
            # QML log-likelihood contribution (Eq. 25, Gaussian form)
            loglik += -0.5 * (logdet_R_t + eps_t @ R_t_inv @ eps_t)
        except np.linalg.LinAlgError:
            return 1e10

    return -loglik  # scipy minimizes, so return negative log-likelihood

# Estimate alpha, beta via QML
result = minimize(
    dcc_loglik, x0=[0.05, 0.90],
    args=(epsilon, S), method="Nelder-Mead"
)
alpha_hat, beta_hat = result.x
print("Converged:", result.success)
print("alpha:", alpha_hat, "beta:", beta_hat, "alpha+beta:", alpha_hat + beta_hat)


# In[96]:


#MLE reestimation of DCC parameters a,b
def compute_R_series(epsilon, S, alpha, beta):
    T, n = epsilon.shape
    Q_t = S.copy()
    R_series = np.zeros((T, n, n))

    for t in range(T):
        q_diag_sqrt = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(q_diag_sqrt, q_diag_sqrt)
        R_series[t] = R_t

        if t < T - 1:
            eps_t = epsilon[t].reshape(-1, 1)
            Q_t = (1 - alpha - beta) * S + alpha * (eps_t @ eps_t.T) + beta * Q_t

    return R_series

R_series = compute_R_series(epsilon, S, alpha_hat, beta_hat)
print(R_series.shape)


# In[97]:


currencies = std_resid_df.columns.tolist()
n = len(currencies)

pairwise_corr = {}
for i in range(n):
    for j in range(i+1, n):
        pair_name = f"{currencies[i]}_{currencies[j]}"
        pairwise_corr[pair_name] = R_series[:, i, j]

corr_df = pd.DataFrame(pairwise_corr, index=std_resid_df.index)
print(corr_df.head())
print(corr_df.describe())


# In[98]:


print(R_series.shape)
print(corr_df.describe())


# In[99]:


mean_corr = {}
for i in range(len(currencies)):
    for j in range(len(currencies)):
        if i == j:
            mean_corr[(currencies[i], currencies[j])] = 1.0
        else:
            key = f"{currencies[min(i,j)]}_{currencies[max(i,j)]}"
            if key in corr_df.columns:
                mean_corr[(currencies[i], currencies[j])] = corr_df[key].mean()


corr_matrix = pd.DataFrame(index=currencies, columns=currencies, dtype=float)
for (a, b), v in mean_corr.items():
    corr_matrix.loc[a, b] = v

plt.figure(figsize=(7,6))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("DCC Correlation: Non-Euro Currencies (2005-2025)")
plt.show()


# In[100]:


pairwise_series = {}

for i in range(len(currencies)):
    for j in range(i+1, len(currencies)):  # unique pairs only
        key = f"{currencies[i]},{currencies[j]}"
        pairwise_series[key] = R_series[:, i, j]

pairwise_df = pd.DataFrame(pairwise_series, index=std_resid_df.index)
print(pairwise_df.shape)  # (T, 15)

n_pairs = len(pairwise_df.columns)

fig, axes = plt.subplots(
    nrows=n_pairs, ncols=1,
    figsize=(14, 3 * n_pairs),   # scales height with number of pairs
    sharex=True
)

for ax, col in zip(axes, pairwise_df.columns):
    ax.plot(pairwise_df.index, pairwise_df[col], linewidth=0.8, color="steelblue")
    ax.set_title(col, loc="left", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    #ax.set_ylim(-1, 1)

axes[-1].set_xlabel("Date")
fig.suptitle("Dynamic Conditional Correlation Over Time — Each Currency Pair", fontsize=14, y=1.0)
plt.tight_layout()
plt.show()


# In[101]:


pairwise_annual = pairwise_df.resample("Y").mean()
print(pairwise_annual.shape)   # should be roughly (20, 15) — 20 years, 15 pairs
print(pairwise_annual.head())

rho_annual = pd.DataFrame(index=pairwise_annual.index)

for ccy in currencies:
    others = [c for c in currencies if c != ccy]
    pair_cols = []
    for o in others:
        key1 = f"{ccy},{o}"
        key2 = f"{o},{ccy}"
        if key1 in pairwise_annual.columns:
            pair_cols.append(key1)
        elif key2 in pairwise_annual.columns:
            pair_cols.append(key2)
    rho_annual[ccy] = pairwise_annual[pair_cols].mean(axis=1)

print(rho_annual)


# In[102]:


rho_long = rho_annual.reset_index().melt(id_vars="date", var_name="currency", value_name="rho")
rho_long["year"] = rho_long["date"].dt.year
rho_long = rho_long.drop(columns="date")

print(rho_long.head())
print(rho_long.shape)  # roughly 20 years x 6 currencies = ~120 row


# In[103]:


#https://ec.europa.eu/eurostat/databrowser/view/nama_10_gdp__custom_22265205/default/table
df = pd.read_csv("/Users/eddiechiu/Documents/GitHub/Thesis/data/nama_10_gdp__custom_22265205_linear_2_0.csv")
print(df.head())
print(df.columns.tolist())


# In[104]:


gdp_raw = pd.read_csv("/Users/eddiechiu/Documents/GitHub/Thesis/data/nama_10_gdp__custom_22265205_linear_2_0.csv")

gdp = gdp_raw[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
gdp.columns = ["country", "year", "gdp_meur"]

# filter to your six countries and your window
your_countries = ["PL", "CZ", "HU", "RO", "SE", "DK"]
gdp = gdp[gdp["country"].isin(your_countries)]
gdp = gdp[(gdp["year"] >= 2005) & (gdp["year"] <= 2025)]

# pivot so each country is a column, year is the index — easy to inspect
gdp_wide = gdp.pivot(index="year", columns="country", values="gdp_meur")
print(gdp_wide)


# In[105]:


#https://ec.europa.eu/eurostat/databrowser/view/ds-045409__custom_22289115/default/table
trade_raw = pd.read_csv("/Users/eddiechiu/Documents/GitHub/Thesis/data/ds-045409__custom_22289115_linear_2_0 (1).csv")

print("Shape:", trade_raw.shape)
print("Columns:", trade_raw.columns.tolist())
print(trade_raw.head())

print("exports shape:", exports.shape)
print("imports shape:", imports.shape)
print(trade_final.head(20))
print(trade_final.shape)
print(trade_final["country"].unique())
print(trade_raw[["flow", "FLOW"]].head(10))
print(trade_raw["flow"].dtype, trade_raw["FLOW"].dtype)
print(trade_raw["flow"].unique())
print(trade_raw["FLOW"].unique())


# In[106]:


# explicitly select uppercase FLOW (the text labels), not lowercase flow (numeric codes)
trade = trade_raw[["reporter", "partner", "TIME_PERIOD", "FLOW", "OBS_VALUE"]].copy()
trade.columns = ["country", "partner", "year", "flow_label", "value"]

print(trade["flow_label"].unique())   # should show ['IMPORT' 'EXPORT']

# filter to your window
trade = trade[(trade["year"] >= 2005) & (trade["year"] <= 2025)]

# split
exports = trade[trade["flow_label"] == "EXPORT"]
imports = trade[trade["flow_label"] == "IMPORT"]

print("exports shape:", exports.shape)
print("imports shape:", imports.shape)

# sum across all 20 Eurozone partners, per reporting country per year
exports_summed = exports.groupby(["country", "year"])["value"].sum().reset_index()
exports_summed.columns = ["country", "year", "exports_to_EA"]

imports_summed = imports.groupby(["country", "year"])["value"].sum().reset_index()
imports_summed.columns = ["country", "year", "imports_from_EA"]

trade_final = exports_summed.merge(imports_summed, on=["country", "year"])
trade_final["trade_volume_EA"] = trade_final["exports_to_EA"] + trade_final["imports_from_EA"]
trade_final["trade_balance_EA"] = trade_final["exports_to_EA"] - trade_final["imports_from_EA"]

trade_final["exports_to_EA_meur"] = trade_final["exports_to_EA"] / 1_000_000
trade_final["imports_from_EA_meur"] = trade_final["imports_from_EA"] / 1_000_000
trade_final["trade_volume_EA_meur"] = trade_final["trade_volume_EA"] / 1_000_000

print(trade_final.head(20))
print(trade_final.shape)
print(trade_final["country"].unique())


# In[107]:


czk_trade = trade_final[trade_final["country"] == "CZ"]
czk_gdp = gdp_wide["CZ"].reset_index()
czk_gdp.columns = ["year", "gdp_meur"]

check = czk_trade.merge(czk_gdp, on="year")
check["trade_pct_gdp"] = check["trade_volume_EA_meur"] / check["gdp_meur"] * 100
print(check[["year", "trade_volume_EA_meur", "gdp_meur", "trade_pct_gdp"]])

gdp_long = gdp_wide.reset_index().melt(id_vars="year", var_name="country", value_name="gdp_meur")

trade_gdp_check = trade_final.merge(gdp_long, on=["country", "year"])
trade_gdp_check["trade_pct_gdp"] = trade_gdp_check["trade_volume_EA_meur"] / trade_gdp_check["gdp_meur"] * 100

print(trade_gdp_check.groupby("country")["trade_pct_gdp"].describe())


# In[ ]:
