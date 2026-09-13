"""
forecast_and_hedge.py
----------------------
Three linked analyses on top of the simulated price/margin history:

1. FORECASTING
   Train a gradient-boosted regressor on lagged price/margin features to
   predict next-day margin -- gives the plant a short-term view of where
   margin is heading.

2. RISK QUANTIFICATION (Monte Carlo VaR / CVaR)
   Simulate 5,000 possible 30-day-ahead margin paths (bootstrapped from
   historical daily margin changes) for a plant producing a fixed monthly
   volume, and compute:
     - 95% Value-at-Risk (VaR): the loss not expected to be exceeded 95% of the time
     - 95% Conditional VaR (CVaR): the average loss in the worst 5% of cases
   This is the exact risk metric used by commodity trading desks and bank
   risk teams.

3. HEDGE RATIO OPTIMIZATION
   Model hedging a fraction h of monthly production via futures (locking in
   today's margin for that fraction). Hedging reduces risk but typically
   costs a small basis/transaction cost. Sweep h from 0% to 100% to build a
   risk-return frontier (expected margin vs. 95% VaR), analogous to a
   Markowitz efficient frontier, and select the hedge ratio that minimizes
   VaR subject to a minimum acceptable expected margin.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

MONTHLY_VOLUME_TONS = 10_000     # ethylene tons/month this plant produces
HORIZON_DAYS = 21                # ~1 trading month ahead
N_MC_PATHS = 5_000
HEDGE_TRANSACTION_COST_PER_TON = 3.0  # $/ton cost of putting on a futures hedge
# Minimum acceptable expected monthly margin the plant's finance team requires
# -- set relative to the fully-unhedged expected margin at run time (see main()).


def build_features(df):
    df = df.copy()
    for lag in [1, 5, 10, 20]:
        df[f"margin_lag{lag}"] = df["margin_per_ton"].shift(lag)
    df["margin_roll_mean10"] = df["margin_per_ton"].rolling(10).mean()
    df["margin_roll_std10"] = df["margin_per_ton"].rolling(10).std()
    df["naphtha_lag1"] = df["naphtha_price"].shift(1)
    df["ethylene_lag1"] = df["ethylene_price"].shift(1)
    # Domain-informed features: these capture "how stressed is the plant
    # right now" far better than raw lagged prices, since risk of a future
    # loss-making stretch depends on how deep into negative territory the
    # margin already is, and how persistent that's been.
    df["margin_roll_mean50"] = df["margin_per_ton"].rolling(50).mean()
    df["dist_below_longrun"] = df["margin_roll_mean50"] - df["margin_per_ton"]
    df["pct_negative_last20"] = (df["margin_per_ton"] < 0).rolling(20).mean()
    df["min_margin_last10"] = df["margin_per_ton"].rolling(10).min()
    df["target_next_margin"] = df["margin_per_ton"].shift(-1)
    # Early-warning label: does margin dip below $0/ton (a loss-making day)
    # at any point in the NEXT 10 trading days? This is the practically
    # useful question for a risk desk ("do we need to act now?"), and is
    # more learnable than exact next-day noise (see note below).
    fwd_min = df["margin_per_ton"].shift(-1).rolling(10, min_periods=10).min().shift(-9)
    df["risk_event_next10d"] = (fwd_min < 0).astype(int)
    return df.dropna()


def train_forecaster(df):
    feat_cols = [c for c in df.columns if c not in
                 ["date", "naphtha_price", "ethylene_price", "margin_per_ton",
                  "target_next_margin", "risk_event_next10d"]]
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    # (a) Point regression on next-day margin -- reported honestly, including
    # the naive persistence baseline. For a near-mean-reverting spread with
    # high day-to-day noise, beating a persistence baseline on raw next-day
    # value is genuinely hard (this mirrors real commodity price behavior,
    # which is close to a martingale at the daily level) -- worth stating
    # plainly rather than dressing up, and it's exactly why (b) below reframes
    # the ML problem into one that's both more learnable and more useful.
    reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=0)
    reg.fit(train[feat_cols], train["target_next_margin"])
    pred = reg.predict(test[feat_cols])
    mae = mean_absolute_error(test["target_next_margin"], pred)
    r2 = r2_score(test["target_next_margin"], pred)
    naive_mae = mean_absolute_error(test["target_next_margin"], test["margin_per_ton"])

    print("=== (a) Next-Day Margin Point Forecast (regression) ===")
    print(f"Model MAE: ${mae:.2f}/ton   R2: {r2:.3f}")
    print(f"Naive (persistence) baseline MAE: ${naive_mae:.2f}/ton")
    print("Note: daily margin changes are close to unpredictable noise (as expected")
    print("for a near-mean-reverting spread), so this is reported for transparency --")
    print("see the early-warning classifier below for the practically useful model.")

    # (b) Early-warning classifier: will there be a loss-making day (margin < 0)
    # within the next 10 trading days?
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=0)
    clf.fit(train[feat_cols], train["risk_event_next10d"])
    clf_pred = clf.predict(test[feat_cols])
    clf_proba = clf.predict_proba(test[feat_cols])[:, 1]
    f1 = f1_score(test["risk_event_next10d"], clf_pred)
    auc = roc_auc_score(test["risk_event_next10d"], clf_proba)

    print("\n=== (b) 10-Day Margin Risk Early-Warning Classifier ===")
    print(f"F1: {f1:.3f}   ROC-AUC: {auc:.3f}")
    print("(Predicts: will the plant see at least one loss-making day in the next 10")
    print("trading days? -- the actionable signal for triggering a hedge decision.)")

    return reg, clf, feat_cols, test, pred, (mae, r2, naive_mae, f1, auc)


def fit_ar1_meanreversion(margin_series):
    """Fit margin_t - margin_{t-1} = kappa*(mu - margin_{t-1}) + noise via OLS.
    This recovers the mean-reversion speed (kappa), long-run mean (mu), and
    residual noise std directly from the data -- needed so the Monte Carlo
    simulation below correctly reflects that a margin currently below its
    long-run average is expected to drift back up (and vice versa). A naive
    bootstrap of historical day-to-day changes would lose this state-
    dependent drift entirely, since raw changes average out to ~zero over a
    stationary series -- worth flagging explicitly as a common simulation
    pitfall."""
    y = margin_series.diff().dropna().values
    x = margin_series.shift(1).dropna().values[:len(y)]
    # y = kappa*mu - kappa*x + noise  ->  linear regression of y on x
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    kappa = -slope
    mu = intercept / kappa if kappa != 0 else margin_series.mean()
    resid = y - (slope * x + intercept)
    sigma = resid.std()
    return kappa, mu, sigma


def monte_carlo_var(margin_series, current_margin, horizon_days, n_paths, volume_tons):
    """Simulate horizon-day-ahead total monthly margin outcomes using a
    mean-reverting (AR(1)/Ornstein-Uhlenbeck) model fit to the historical
    margin series, starting from today's actual margin level."""
    kappa, mu, sigma = fit_ar1_meanreversion(margin_series)
    rng = np.random.default_rng(3)

    paths = np.full(n_paths, current_margin)
    for _ in range(horizon_days):
        shocks = rng.normal(0, sigma, n_paths)
        paths = paths + kappa * (mu - paths) + shocks

    total_dollar_margin = paths * volume_tons
    return total_dollar_margin, (kappa, mu, sigma)


def compute_var_cvar(pnl_distribution, confidence=0.95):
    expected = pnl_distribution.mean()
    var_threshold = np.percentile(pnl_distribution, (1 - confidence) * 100)
    var = expected - var_threshold
    cvar_tail = pnl_distribution[pnl_distribution <= var_threshold]
    cvar = expected - cvar_tail.mean()
    return expected, var, cvar, var_threshold


def hedge_frontier(unhedged_pnl, current_margin, volume_tons):
    """Sweep hedge ratio h in [0,1]. Hedging fraction h locks in today's
    margin for that fraction of volume (zero variance on that portion) at a
    transaction cost, leaving (1-h) exposed to the full unhedged
    distribution."""
    hedge_ratios = np.linspace(0, 1, 21)
    rows = []
    locked_dollar_margin = current_margin * volume_tons

    for h in hedge_ratios:
        hedged_portion = h * locked_dollar_margin - h * volume_tons * HEDGE_TRANSACTION_COST_PER_TON
        unhedged_portion = (1 - h) * unhedged_pnl  # array over MC paths
        total = hedged_portion + unhedged_portion
        expected, var, cvar, _ = compute_var_cvar(total)
        rows.append({"hedge_ratio": h, "expected_margin": expected, "VaR_95": var, "CVaR_95": cvar})

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv("data/price_and_margin_history.csv", parse_dates=["date"])
    feat_df = build_features(df)

    reg, clf, feat_cols, test, pred, fmetrics = train_forecaster(feat_df)

    # --- Risk quantification ---
    current_margin = df["margin_per_ton"].iloc[-1]

    unhedged_pnl, (kappa, mu, sigma) = monte_carlo_var(
        df["margin_per_ton"], current_margin, HORIZON_DAYS, N_MC_PATHS, MONTHLY_VOLUME_TONS
    )
    exp0, var0, cvar0, _ = compute_var_cvar(unhedged_pnl)

    print(f"\nFitted mean-reversion model: kappa={kappa:.3f}, long-run mean=${mu:.2f}/ton, daily sigma=${sigma:.2f}/ton")
    print(f"Current margin: ${current_margin:.2f}/ton (long-run mean: ${mu:.2f}/ton)")
    print("\n=== 21-Trading-Day-Ahead Monthly Margin Risk (Unhedged) ===")
    print(f"Expected monthly margin: ${exp0:,.0f}")
    print(f"95% VaR: ${var0:,.0f}  (potential shortfall not exceeded 95% of the time)")
    print(f"95% CVaR: ${cvar0:,.0f}  (average shortfall in the worst 5% of scenarios)")

    # --- Hedge optimization: present the full risk-return frontier ---
    # Rather than picking one "optimal" hedge ratio, sweep h from 0-100% and
    # let the finance team choose a point based on their risk tolerance --
    # exactly how a Markowitz efficient frontier is used in practice.
    frontier = hedge_frontier(unhedged_pnl, current_margin, MONTHLY_VOLUME_TONS)

    # Two illustrative risk-tolerance policies a finance team might set:
    conservative = frontier.iloc[(frontier["VaR_95"] - 0.25 * var0).abs().idxmin()]
    moderate = frontier.iloc[(frontier["VaR_95"] - 0.60 * var0).abs().idxmin()]

    print("\n=== Hedge Ratio Efficient Frontier ===")
    if current_margin > mu:
        print(f"Current margin (${current_margin:.2f}/ton) is ABOVE its long-run mean")
        print(f"(${mu:.2f}/ton), so the model expects some mean-reversion DOWN over the")
        print("next month -- here hedging both locks in today's elevated margin AND")
        print("reduces risk, so more hedging dominates on both axes. (The interesting")
        print("trade-off case is the reverse: margin below the long-run mean, where")
        print("hedging trades away expected upside for lower risk -- see README.)")
    else:
        print(f"Current margin (${current_margin:.2f}/ton) is BELOW its long-run mean")
        print(f"(${mu:.2f}/ton), so the model expects some mean-reversion UP over the")
        print("next month -- fully hedging today would lock in the depressed margin and")
        print("forgo that expected recovery. The frontier below quantifies that trade-off:")
    print(f"\n  0% hedge (status quo):    Expected ${exp0:,.0f}, 95% VaR ${var0:,.0f}")
    print(f"  {moderate['hedge_ratio']*100:.0f}% hedge (moderate risk cut): Expected ${moderate['expected_margin']:,.0f}, 95% VaR ${moderate['VaR_95']:,.0f}")
    print(f"  {conservative['hedge_ratio']*100:.0f}% hedge (conservative):    Expected ${conservative['expected_margin']:,.0f}, 95% VaR ${conservative['VaR_95']:,.0f}")
    print(f"  100% hedge (fully locked): Expected ${frontier.iloc[-1]['expected_margin']:,.0f}, 95% VaR ${frontier.iloc[-1]['VaR_95']:,.0f}")
    best = moderate  # used for the annotated plot marker

    frontier.to_csv("data/hedge_frontier.csv", index=False)

    with open("results_metrics.txt", "w") as f:
        f.write("(a) Next-Day Margin Point Forecast (regression, reported for transparency)\n")
        f.write(f"  Model MAE: ${fmetrics[0]:.2f}/ton   R2: {fmetrics[1]:.3f}\n")
        f.write(f"  Naive persistence baseline MAE: ${fmetrics[2]:.2f}/ton\n")
        f.write("  (Daily margin changes are close to unpredictable noise, as expected for\n")
        f.write("   a near-mean-reverting spread -- this motivates (b) below.)\n\n")
        f.write("(b) 10-Day Margin Risk Early-Warning Classifier\n")
        f.write(f"  F1: {fmetrics[3]:.3f}   ROC-AUC: {fmetrics[4]:.3f}\n\n")
        f.write(f"Fitted mean-reversion model: kappa={kappa:.3f}, long-run mean=${mu:.2f}/ton, daily sigma=${sigma:.2f}/ton\n\n")
        f.write("21-Day Monthly Margin Risk (Unhedged)\n")
        f.write(f"  Expected: ${exp0:,.0f}   95% VaR: ${var0:,.0f}   95% CVaR: ${cvar0:,.0f}\n\n")
        f.write("Hedge Ratio Efficient Frontier (illustrative policy points)\n")
        f.write(f"  0% hedge:   Expected ${exp0:,.0f}, 95% VaR ${var0:,.0f}\n")
        f.write(f"  {moderate['hedge_ratio']*100:.0f}% hedge:  Expected ${moderate['expected_margin']:,.0f}, 95% VaR ${moderate['VaR_95']:,.0f}\n")
        f.write(f"  {conservative['hedge_ratio']*100:.0f}% hedge:  Expected ${conservative['expected_margin']:,.0f}, 95% VaR ${conservative['VaR_95']:,.0f}\n")
        f.write(f"  100% hedge: Expected ${frontier.iloc[-1]['expected_margin']:,.0f}, 95% VaR ${frontier.iloc[-1]['VaR_95']:,.0f}\n")

    # --- Plot 1: price + margin history ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df["date"], df["naphtha_price"], label="Naphtha ($/ton)", color="tab:brown")
    axes[0].plot(df["date"], df["ethylene_price"], label="Ethylene ($/ton)", color="tab:blue")
    axes[0].set_ylabel("$/ton")
    axes[0].legend()
    axes[0].set_title("Simulated Feedstock & Product Prices")

    axes[1].plot(df["date"], df["margin_per_ton"], color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Margin ($/ton)")
    axes[1].set_title("Cracker Margin Over Time")
    plt.tight_layout()
    plt.savefig("plots/price_and_margin_history.png", dpi=150)
    plt.close()

    # --- Plot 2: early-warning classifier probability vs actual margin ---
    clf_proba_full = clf.predict_proba(test[feat_cols])[:, 1]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(test["date"].values, test["margin_per_ton"].values, color="tab:green", label="Actual margin ($/ton)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Margin ($/ton)", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")

    ax2 = ax1.twinx()
    ax2.plot(test["date"].values, clf_proba_full, color="tab:red", alpha=0.7,
             label="Predicted risk probability")
    ax2.set_ylabel("P(loss-making day within 10d)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1)

    plt.title(f"Early-Warning Classifier: Risk Probability vs Actual Margin (AUC={fmetrics[4]:.2f})")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    plt.tight_layout()
    plt.savefig("plots/forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Monte Carlo distribution with VaR/CVaR ---
    plt.figure(figsize=(8, 5))
    plt.hist(unhedged_pnl, bins=60, alpha=0.7, color="tab:blue")
    plt.axvline(exp0, color="black", linestyle="-", label=f"Expected: ${exp0:,.0f}")
    plt.axvline(exp0 - var0, color="orange", linestyle="--", label=f"95% VaR: ${var0:,.0f}")
    plt.axvline(exp0 - cvar0, color="red", linestyle="--", label=f"95% CVaR: ${cvar0:,.0f}")
    plt.xlabel("Monthly margin ($)")
    plt.ylabel("Frequency (of 5,000 simulated paths)")
    plt.title("Monte Carlo Simulated 1-Month-Ahead Margin (Unhedged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/montecarlo_var.png", dpi=150)
    plt.close()

    # --- Plot 4: hedge frontier ---
    plt.figure(figsize=(8, 5.5))
    plt.plot(frontier["VaR_95"], frontier["expected_margin"], marker="o", markersize=3)
    plt.scatter([best["VaR_95"]], [best["expected_margin"]], color="red", s=80, zorder=5,
                label=f"Chosen hedge: {best['hedge_ratio']*100:.0f}%")
    for _, row in frontier.iloc[::4].iterrows():
        plt.annotate(f"{row['hedge_ratio']*100:.0f}%", (row["VaR_95"], row["expected_margin"]),
                     fontsize=8, alpha=0.7)
    plt.xlabel("95% VaR ($) -- lower is less risky")
    plt.ylabel("Expected monthly margin ($)")
    plt.title("Hedge Ratio Efficient Frontier")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/hedge_frontier.png", dpi=150)
    plt.close()

    print("\nSaved plots to plots/, metrics to results_metrics.txt")


if __name__ == "__main__":
    main()
