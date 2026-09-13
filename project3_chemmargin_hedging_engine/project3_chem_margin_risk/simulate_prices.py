"""
simulate_prices.py
-------------------
Simulates 3 years of daily prices for a naphtha-fed ethylene cracker:

    naphtha_price   ($/ton)  -- feedstock
    ethylene_price  ($/ton)  -- product

Both are modeled as correlated mean-reverting (Ornstein-Uhlenbeck) processes,
which is the standard way commodity prices are modeled in energy/petrochemical
trading desks (prices fluctuate but are pulled back toward a long-run
equilibrium set by global supply/demand and substitute fuels).

Calibration (long-run mean, volatility, mean-reversion speed) is set to
realistic historical ranges for naphtha and ethylene:
    - naphtha:  ~$650/ton, ~25% annualized volatility
    - ethylene: ~$1050/ton, ~20% annualized volatility
    - correlation ~0.75 (feedstock and product track global oil/energy cycles
      together, but not perfectly -- that residual "basis" is exactly the
      margin risk this project manages)

From these two price series we compute the plant's crack margin per ton of
ethylene produced:

    margin_per_ton = ethylene_price - (naphtha_price / yield) - processing_cost

where yield is an *effective* net feed conversion factor (accounting for
co-product credits from propylene/butadiene/aromatics sales) rather than
raw ethylene stoichiometric yield -- see NOTE in constants below.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)

N_DAYS = 1260  # ~5 trading years (more history for robust train/test split)
DT = 1 / 252

# Naphtha (feedstock)
NAPHTHA_MEAN = 650.0
NAPHTHA_VOL = 0.25
NAPHTHA_KAPPA = 3.0  # mean reversion speed

# Ethylene (product)
ETHYLENE_MEAN = 1050.0
ETHYLENE_VOL = 0.20
ETHYLENE_KAPPA = 2.5

CORRELATION = 0.75
# NOTE: a real steam cracker's stoichiometric ethylene yield from naphtha is
# only ~30%, but the plant also sells co-products (propylene, butadiene,
# aromatics, fuel gas), which effectively credit back much of the naphtha
# cost. YIELD here is therefore an *effective* net feed conversion (yield
# net of co-product credits), calibrated so the resulting margin matches
# real-world naphtha cracker margins, which are typically thin ($20-150/ton)
# -- this is explicitly a simplification and is noted as such in the README.
YIELD = 0.75
PROCESSING_COST = 150.0  # $/ton ethylene (utilities, labor, maintenance, capex charge)


def simulate_ou_pair(n_days, mean1, vol1, kappa1, mean2, vol2, kappa2, corr):
    prices1 = np.zeros(n_days)
    prices2 = np.zeros(n_days)
    prices1[0] = mean1
    prices2[0] = mean2

    cov = np.array([[1, corr], [corr, 1]])
    L = np.linalg.cholesky(cov)

    for t in range(1, n_days):
        z = RNG.normal(0, 1, 2)
        z = L @ z  # correlate the two shocks
        dW1, dW2 = z[0] * np.sqrt(DT), z[1] * np.sqrt(DT)

        prices1[t] = prices1[t-1] + kappa1 * (mean1 - prices1[t-1]) * DT + vol1 * prices1[t-1] * dW1
        prices2[t] = prices2[t-1] + kappa2 * (mean2 - prices2[t-1]) * DT + vol2 * prices2[t-1] * dW2

    return prices1, prices2


def main():
    naphtha, ethylene = simulate_ou_pair(
        N_DAYS, NAPHTHA_MEAN, NAPHTHA_VOL, NAPHTHA_KAPPA,
        ETHYLENE_MEAN, ETHYLENE_VOL, ETHYLENE_KAPPA, CORRELATION
    )

    dates = pd.bdate_range("2023-01-02", periods=N_DAYS)
    df = pd.DataFrame({
        "date": dates,
        "naphtha_price": naphtha,
        "ethylene_price": ethylene,
    })
    df["margin_per_ton"] = df["ethylene_price"] - (df["naphtha_price"] / YIELD) - PROCESSING_COST

    df.to_csv("data/price_and_margin_history.csv", index=False)
    print(f"Generated {len(df)} trading days of price/margin history.")
    print(df.describe())
    print(f"\nRealized correlation (naphtha vs ethylene): {df['naphtha_price'].corr(df['ethylene_price']):.2f}")


if __name__ == "__main__":
    main()
