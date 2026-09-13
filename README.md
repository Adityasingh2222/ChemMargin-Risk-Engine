# ChemMargin: Petrochemical Cracker Margin & Hedging Model

I built this project to analyze and manage financial risk for an ethylene steam cracker plant producing 10,000 tonnes per month using naphtha feedstock.

In chemical manufacturing, raw material costs fluctuate constantly[cite: 2, 3]. When naphtha prices rise or ethylene prices drop, a plant can easily slide into negative operating margins[cite: 3]. I developed this pipeline to connect plant operating economics with data science and quantitative financial risk to address three key operational questions[cite: 2, 3]:
1. When is the plant at risk of running at a cash loss[cite: 2, 3]?
2. What is the worst-case financial downside over the next operating month[cite: 2, 3]?
3. How much production should be locked into futures contracts to protect profits[cite: 3]?

---

## What I Built

* **Stochastic Price & Margin Simulation:** Built a correlated time-series generator using an Ornstein–Uhlenbeck mean-reverting process to model how naphtha and ethylene prices move and revert toward long-run equilibrium over time[cite: 2, 3].
* **Plant Economics Model:** Modeled operational cash margins per tonne based on feedstock consumption stoichiometry, utility fuel usage, and conversion costs[cite: 2, 3].
* **Early Warning Classifier:** Trained a supervised machine learning model using scikit-learn to identify negative-margin regimes 10 days before they occur, reaching an ROC-AUC of 0.80[cite: 1, 3].
* **Monte Carlo Risk Engine:** Executed a 5,000-path simulation to evaluate distribution-level downside risk, calculating an unhedged 21-day 95% Value-at-Risk (VaR) of $632,092[cite: 3].
* **Hedge Ratio Optimizer:** Mapped the trade-off curve across futures hedge ratios from 0% (fully exposed to spot fluctuations) to 100% (locked margin baseline)[cite: 3].
* **Interactive Web Console:** Created a browser dashboard (`index.html`) featuring dynamic SVG graphs and an interactive slider to demonstrate real-time risk mitigation[cite: 3].

---

## Key Results

| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Long-Run Mean Spread** | $25.88 / tonne[cite: 3] | Equilibrium cash margin over production costs[cite: 2, 3] |
| **Current Baseline** | $118.08 / tonne[cite: 3] | Starting operating margin in the simulated period[cite: 3] |
| **95% VaR (Unhedged)** | $632,092[cite: 3] | Maximum expected loss at 95% confidence over 21 days[cite: 2, 3] |
| **Early Warning ROC-AUC** | 0.80[cite: 3] | Model accuracy predicting cash loss events 10 days ahead[cite: 2, 3] |
| **Fully Hedged Risk (100%)** | $0 Market VaR[cite: 3] | Complete elimination of monthly spot variance[cite: 3] |

---

## Project Structure

* `index.html` — Interactive browser console with an interactive hedge slider and dynamic SVG charts[cite: 3].
* `simulate_prices.py` — Generates correlated stochastic price paths for naphtha, ethylene, and net margins[cite: 1, 2].
* `forecast_and_hedge.py` — Fits time-series dynamics, trains the ML classifier, executes Monte Carlo simulations, and maps the hedge frontier[cite: 1, 3].
* `requirements.txt` — Python package dependencies (pandas, numpy, scikit-learn, matplotlib).
* `data/` — CSV files containing simulated margin histories and computed frontier values.
* `plots/` — Output charts showing margins, forecast comparisons, and the hedge frontier[cite: 1].

---

## How to Run It

### 1. View the Dashboard
Open `index.html` directly in any web browser[cite: 3]. It runs natively with vanilla JavaScript and CSS without requiring a local server[cite: 3].

### 2. Run the Code Locally
```bash
pip install -r requirements.txt
python simulate_prices.py
python forecast_and_hedge.py
