# ChemMargin — Margin Risk & Hedging Optimization Engine for a Petrochemical Plant

**A tool that combines process economics, market risk modeling, and machine
learning to answer a question every commodity-exposed manufacturer actually
faces: how much should we hedge, and when should we worry?**

## The problem
A naphtha-fed ethylene cracker buys a volatile feedstock (naphtha) and sells
a volatile product (ethylene). Its profit is the spread between the two —
the "crack margin" — which can swing from healthy to loss-making within
weeks as global oil and petrochemical markets move. This is a real,
multi-billion-dollar risk that treasury and trading desks at companies like
Dow, BASF, SABIC, and Reliance manage daily, and it's a natural fit for
someone who understands both the chemical process economics *and* financial
risk modeling — which is exactly why this project exists.

## What it does
1. **Simulates realistic market dynamics** — feedstock and product prices as
   correlated mean-reverting (Ornstein-Uhlenbeck) stochastic processes,
   calibrated to real-world commodity volatility (~20-25% annualized) and
   correlation (~0.7-0.8), the standard approach used in energy/commodity
   trading desks.
2. **Forecasts near-term risk with ML** — rather than chasing an unpredictable
   exact next-day price (which is genuinely close to noise for a
   mean-reverting spread, and the project reports that honestly with a naive
   baseline comparison), it trains a classifier to answer the practically
   useful question: *"will this plant see a loss-making day in the next 10
   trading days?"* — achieving ROC-AUC 0.80 using domain-informed features
   (deviation from long-run margin, recent loss frequency, rolling
   volatility).
3. **Quantifies risk with Monte Carlo VaR/CVaR** — simulates 5,000 possible
   one-month-ahead outcomes using a properly calibrated mean-reverting model
   (fit via AR(1) regression on the historical margin series — not a naive
   i.i.d. bootstrap, which would silently discard the mean-reversion signal
   and is a common mistake in this kind of project) to compute 95%
   Value-at-Risk and Conditional VaR in dollar terms.
4. **Optimizes the hedge ratio** — builds a full risk-return efficient
   frontier (à la Markowitz, applied to a physical plant instead of a
   portfolio) showing how much expected margin a finance team gives up for
   each unit of risk reduction, so hedging decisions are quantified, not
   guessed.

## Why this project is worth highlighting
Most "predictive maintenance" or "sales forecasting" resume projects use an
off-the-shelf dataset and a standard model. This one:
- Required understanding **both** the underlying chemical process economics
  (yield, co-product credits, processing cost) **and** financial risk theory
  (VaR/CVaR, mean-reversion calibration, efficient frontiers) — genuinely
  rare combination.
- Catches and fixes a real modeling pitfall mid-build (a naive historical
  bootstrap for Monte Carlo silently loses mean-reversion signal) — a strong
  interview talking point that shows real understanding, not just calling
  `.fit()`.
- Produces an actual business recommendation (a hedge ratio with a
  transparent expected-margin-vs-risk trade-off) rather than just a metric.

## Results
| Component | Metric |
|---|---|
| Next-day margin regression (reported for transparency) | MAE $16.9/ton vs. naive baseline $8.2/ton — see note in code on why this is expected |
| 10-day loss-event early-warning classifier | ROC-AUC 0.80, F1 0.54 |
| Unhedged 1-month 95% VaR | ~$630K on a 10,000 ton/month plant |
| Hedge frontier | Quantifies exact $ trade-off between hedge ratio, expected margin, and VaR |

See `plots/` for the full set of visuals, including the early-warning
classifier's risk-probability signal spiking correctly ahead of an actual
loss-making stretch in the simulated data.

## How to run
```bash
pip install -r requirements.txt
python simulate_prices.py        # generates data/price_and_margin_history.csv
python forecast_and_hedge.py     # forecasting, VaR, hedge frontier, all plots
```

## Project structure
```
├── simulate_prices.py       # correlated mean-reverting price simulation
├── forecast_and_hedge.py    # ML forecasting + Monte Carlo VaR + hedge optimization
├── data/                    # generated price/margin history
├── models/                   # (extend: save trained models here)
├── plots/                    # all evaluation & result plots
├── results_metrics.txt      # final metrics
└── requirements.txt
```

## Honest limitations (good to mention proactively in interviews)
- Prices are simulated with a calibrated stochastic model, not pulled from a
  live market feed — the pipeline is built so real historical naphtha/ethylene
  or crude/product price data (e.g., from EIA, ICIS, or a Bloomberg terminal)
  drops in with no structural changes.
- The hedge model assumes a simple linear futures hedge; a real desk would
  also consider options-based hedges (collars, caps) for asymmetric payoffs.
- Yield/co-product economics are simplified to a single effective conversion
  factor rather than a full multi-product mass balance.

## Extensions
- Swap in real market data (EIA/ICIS/exchange APIs)
- Add options-based hedging strategies (collars) to the frontier
- Extend to a multi-plant, multi-commodity portfolio (this is where it
  starts looking like a genuine trading-desk risk system)
