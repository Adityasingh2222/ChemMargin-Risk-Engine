# ChemMargin: Petrochemical Cracker Margin & Hedging Model

I built this project to analyze and manage financial risk for an ethylene steam cracker plant producing 10,000 tonnes per month using naphtha feedstock. 

In chemical manufacturing, raw material costs fluctuate constantly. If naphtha prices jump or ethylene prices drop, a plant can easily slide into negative operating margins[cite: 3]. I wanted to combine plant economics with data science and risk analysis to solve three practical questions:
1. When is the plant at risk of running at a loss[cite: 2, 3]?
2. What is the worst-case financial downside over the next operating month[cite: 2, 3]?
3. How much production should be locked into futures contracts to protect profits[cite: 3]?

---

## What I Built

* **Simulated Price and Spread Data:** Built a correlated time-series generator using a mean-reverting process (Ornstein–Uhlenbeck) to model how naphtha and ethylene prices move and revert over time[cite: 2, 3].
* **Plant Economics Model:** Modeled operational cash margins per tonne based on feedstock consumption, utility fuel usage, and plant operating costs[cite: 2, 3].
* **Early Warning Classifier:** Trained a machine learning model using scikit-learn to spot negative-margin periods 10 days before they happen (achieved an ROC-AUC of 0.80)[cite: 1, 3].
* **Monte Carlo Risk Simulation:** Ran 5,000 simulated price paths to calculate the 21-day 95% Value-at-Risk (VaR), showing an unhedged plant exposure of around $632,000[cite: 3].
* **Hedge Ratio Calculator:** Calculated the risk vs. expected profit trade-off across futures hedge ratios from 0% (fully exposed to spot prices) to 100% (fully locked)[cite: 3].
* **Interactive Web Console:** Created a clean HTML/JS dashboard (`index.html`) where anyone can use a slider to see expected profit and risk change in real time[cite: 3].

---

## Project Structure

* `index.html` — Interactive browser dashboard with real-time hedge slider and charts[cite: 3].
* `simulate_prices.py` — Python script that creates the synthetic commodity prices and margins[cite: 1, 2].
* `forecast_and_hedge.py` — Script that trains the ML classifier, runs the Monte Carlo simulation, and computes hedge values[cite: 1, 3].
* `requirements.txt` — Required Python packages (pandas, numpy, scikit-learn, matplotlib).
* `data/` — CSV files storing generated price history and hedge results.
* `plots/` — Output charts showing margins, forecast comparisons, and the hedge frontier[cite: 1].

---

## Key Results

* **Mean Margin:** Long-run average margin settled at $25.88 per tonne[cite: 3].
* **Downside Risk:** Unhedged 95% Value-at-Risk was calculated at $632,092 for a 21-day horizon[cite: 3].
* **Classification Performance:** The early-warning classifier achieved an 0.80 ROC-AUC score on test data[cite: 3].
* **Hedging Impact:** Locking in 100% of production via futures completely removes monthly margin uncertainty, while partial hedging balances upside against downside protection[cite: 3].

---

## How to Run It

### 1. View the Dashboard
Just open `index.html` in any web browser[cite: 3]. It runs completely in the browser with no extra setup or server needed[cite: 3].

### 2. Run the Code Locally
If you want to run or modify the Python scripts[cite: 1]:

```bash
# Install packages
pip install -r requirements.txt

# Step 1: Generate price history
python simulate_prices.py

# Step 2: Run ML training, Monte Carlo risk, and hedging optimization
python forecast_and_hedge.py
