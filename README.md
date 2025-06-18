# Monte Carlo Option Pricing

A Python project for pricing European and Asian options using Monte Carlo simulation. This project demonstrates the use of probabilistic techniques to estimate option prices and compares results with analytical solutions where available.

## 📌 Project Overview

This project implements a Monte Carlo engine for simulating stock price paths based on Geometric Brownian Motion (GBM), and uses these paths to price European and Asian options. It also explores variance reduction techniques to improve simulation efficiency.

## 🔢 Features

- Pricing of:
  - European Call and Put options
  - Asian Call and Put options (arithmetic average)
- Stock price simulation using Geometric Brownian Motion
- Risk-neutral valuation using discounting
- Variance reduction techniques:
  - Antithetic Variates
  - Control Variates (optional)
- Convergence diagnostics and performance evaluation
- Comparison with Black-Scholes closed-form solution (for European options)
- Full LaTeX documentation of the project

## 🧠 Mathematical Model

Simulated stock paths follow:

S_{t+\Delta t} = S_t \cdot \exp\left[\left(r - \frac{1}{2} \sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z\right]


Where:
- \( S_t \) = asset price at time \( t \)
- \( r \) = risk-free rate
- \( \sigma \) = volatility
- \( Z \sim \mathcal{N}(0,1) \)

### European Option Payoff:
\[
\max(S_T - K, 0) \quad \text{or} \quad \max(K - S_T, 0)
\]

### Asian Option Payoff (arithmetic average):
\[
\max\left(\frac{1}{n} \sum_{i=1}^n S_{t_i} - K, 0\right)
\]

## 🗂️ Project Structure

monte-carlo-option-pricing/
│
├── data/ # Store any test data or results
├── plots/ # Output plots and diagnostics
├── src/
│ ├── option.py # Option classes (European, Asian)
│ ├── monte_carlo.py # Core pricing engine
│ ├── utils.py # Helper functions, RNG, etc.
│ ├── plotting.py # Convergence plots, histograms
│ └── main.py # Main script to run simulations
│
├── results/ # Final prices, errors, etc.
├── report/
│ └── report.tex # LaTeX documentation
│
├── README.md
└── requirements.txt


## 🧪 Example Usage

```bash
python main.py --option european --type call --S0 100 --K 105 --T 1 --r 0.05 --sigma 0.2 --paths 100000
