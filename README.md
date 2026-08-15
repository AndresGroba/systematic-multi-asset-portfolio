# Systematic Multi-Asset Portfolio Strategy

> End-to-end quantitative asset allocation framework combining market signals, Machine Learning, Black-Litterman and dynamic portfolio optimisation.

**MSc Quantitative Finance — AFI Global Education | Group Project**

---

## Overview

This project develops a systematic portfolio management strategy over a diversified universe of **42 ETFs plus a money-market asset**, covering multiple regions, sectors and asset classes.

The strategy follows a predominantly **contrarian investment philosophy**, seeking attractive opportunities during market dislocations while dynamically adapting portfolio allocation to the prevailing market regime.

The complete investment pipeline combines traditional quantitative finance with Machine Learning:

**Market Data → Regime Detection → Composite Signals → Machine Learning → Black-Litterman → Merton Allocation → Davis-Norman No-Trade Bands → Portfolio**

The framework includes historical backtesting, transaction costs, walk-forward validation, robustness analysis and explicit controls against look-ahead bias.

---

## Strategy Architecture

### 1. Market Regime Detection

The framework classifies market conditions into three regimes:

- **Normal**
- **Caution**
- **Crisis**

The detected regime dynamically modifies the intensity of the strategy's views and portfolio rebalancing behaviour.

### 2. Composite Quantitative Signal

Each ETF receives an attractiveness score based on several market indicators:

- Momentum
- Short-term reversal
- Long-term trend
- Drawdown
- Volatility

The signal combines medium-term trend information with a contrarian component designed to identify potential recovery opportunities following significant market declines.

### 3. Machine Learning

Machine Learning is incorporated as a **signal refinement layer**.

**XGBoost** estimates the probability that each ETF will generate a positive return over the following 21 trading days using momentum, volatility, trend and drawdown features.

**K-Means clustering** groups ETFs according to their historical behaviour, helping reduce redundant exposures and improve effective diversification.

### 4. Black-Litterman

The refined quantitative signals are incorporated as investor views within the **Black-Litterman model**, combining them with equilibrium expected returns and the covariance structure of the investment universe.

### 5. Portfolio Allocation

Portfolio weights are determined using a **Merton-inspired optimal allocation framework**.

To limit unnecessary turnover and explicitly account for transaction costs, **Davis-Norman-inspired no-trade bands** determine when portfolio positions should actually be rebalanced.

---

## Backtesting & Validation

The strategy was evaluated using an end-to-end historical backtesting framework including:

- Transaction costs
- Dynamic risk-free rates
- EUR currency conversion
- Walk-forward validation
- Out-of-sample analysis
- Parameter sensitivity tests
- Random portfolio comparisons
- Bootstrap analysis of Sharpe ratios
- Explicit look-ahead bias controls

All portfolio decisions use only information available at each historical decision date.

---

## Historical Performance

### Full Backtest — 2013–2026

| Metric | Strategy | S&P 500 | MSCI World |
|---|---:|---:|---:|
| Total Return | 421.96% | 510.90% | 343.65% |
| CAGR | 13.80% | 15.21% | 12.36% |
| Volatility | 17.56% | 18.39% | 18.26% |
| Sharpe Ratio | 0.78 | 0.81 | 0.68 |
| Max Drawdown | -34.06% | -33.09% | -33.33% |

Over the complete backtest period, the strategy outperformed the **MSCI World** in both CAGR and Sharpe ratio, while remaining below the S&P 500.

### 2020–2026 Subperiod

| Metric | Strategy | S&P 500 | MSCI World |
|---|---:|---:|---:|
| Total Return | **164.36%** | 138.16% | 112.70% |
| CAGR | **16.52%** | 14.62% | 12.60% |
| Volatility | 20.29% | 21.31% | 20.66% |
| Sharpe Ratio | **0.77** | 0.66 | 0.60 |
| Max Drawdown | -33.50% | -33.09% | -33.33% |

During the more volatile 2020–2026 period, the strategy outperformed both benchmarks in total return, CAGR and Sharpe ratio.

> **Note:** These results should not be interpreted as evidence of statistically significant alpha. Bootstrap confidence intervals for the Sharpe ratios overlap with those of the benchmarks. The project therefore focuses on robustness and portfolio construction rather than claiming persistent market outperformance.

---

## Robustness Analysis

Several tests were implemented to evaluate the stability of the strategy and reduce the risk of overfitting:

- Walk-forward validation
- Parameter sensitivity analysis
- Randomised parameter tests
- Random portfolio comparison
- Machine Learning contribution analysis
- Bootstrap confidence intervals
- Explicit no-look-ahead verification

These tests provide additional evidence on the robustness of the overall investment architecture beyond the headline backtest results.

---

## Interactive Dashboard

The project also includes a **Streamlit dashboard** for exploring portfolio performance, benchmark comparisons, drawdowns, portfolio composition and robustness studies.

The dashboard can be run locally using:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

---

## Repository Structure

```text
.
├── dashboard/          # Interactive Streamlit dashboard
├── outputs/            # Backtest and robustness results
├── scripts/            # Strategy execution and validation scripts
├── src/                # Core quantitative framework
│   ├── backtest/
│   ├── data/
│   ├── metrics/
│   ├── models/
│   ├── portfolio/
│   ├── visualization/
│   └── walkforward/
├── requirements.txt
└── README.md
```

---

## Technologies & Methods

**Programming**

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `yfinance` · `Streamlit` · `Plotly`

**Quantitative Finance**

`Black-Litterman` · `Merton Allocation` · `Davis-Norman` · `Portfolio Optimisation` · `Backtesting`

**Machine Learning**

`XGBoost` · `K-Means` · `Walk-Forward Validation`

---

## Academic Context

This project was developed collaboratively as part of the **MSc in Quantitative Finance at AFI Global Education**.

### Quantara Capital

- Enrique Barrajón
- **Andrés Groba**
- Pablo López
- David Pérez
- Jaime Rubio

The project was developed collaboratively, with the team contributing across strategy design, quantitative modelling, implementation, analysis and presentation.

---

## Disclaimer

This repository contains an academic quantitative finance project developed for educational and research purposes.

Historical and simulated performance does not guarantee future results and should not be interpreted as investment advice.
