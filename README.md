# Honest Backtester

A daily backtester for single-instrument equity strategies, built so that the most common sources of false backtest results are structurally impossible rather than left to the user's discipline.

**Scope: one instrument, daily closing prices, positions of long / short / flat.** It does not handle multi-asset portfolios, intraday data, options, futures roll, leverage, or order types. Running a strategy outside that scope through this engine will produce a wrong answer silently, so don't.

## What it enforces

| | Typical backtester | This one |
|---|---|---|
| Lookahead | user's responsibility | impossible — the strategy is handed only data up to the current bar |
| Position lag | optional | always one bar |
| Transaction costs | default off | default on, 5 bps |
| Out-of-sample split | manual | built into the object |
| Statistical significance | not reported | t-statistic and verdict on every result |

## Why the t-statistic

A Sharpe ratio means nothing without the length of the period it was measured over. A Sharpe of 1.0 over six months is noise; over ten years it is a result. The engine reports

```
t = Sharpe × √(years)
```

and flags anything below 1.96 as **not distinguishable from zero**. Most retail backtest reports would fail this test and never mention it.

## Usage

A strategy is any function that receives the data available so far and returns a position: `1` long, `-1` short, `0` flat.

```python
prices = yf.download(["SPY"], start="2019-01-01", end="2026-01-01")["Close"].dropna()
bt = Backtest(prices, cost_bps=5)

def momentum(data):
    if len(data) < 50:
        return 0
    return 1 if data.iloc[-50:].mean() > 0 else -1

bt.report(momentum, "50-day momentum")
bt.compare({"momentum": momentum, "buy and hold": lambda d: 1})
```

The function cannot see future data because it is never given any — `data` is truncated at the current bar by the engine, not by convention.

## Worked example

SPY, 2024–2025, 5 bps costs:

| Strategy | Return | Volatility | Sharpe | t-stat | Significant |
|---|---|---|---|---|---|
| 50-day momentum | 11.01% | 16.00% | 0.44 | 0.62 | No |
| Buy and hold | 21.39% | 16.37% | 1.06 | 1.50 | No |

Two things this surfaces that a conventional report would not. The momentum rule returned roughly half of buy-and-hold at the same volatility and drawdown — the trading added nothing. And buy-and-hold's Sharpe of 1.06 looks strong but carries a t-statistic of 1.50: over two years, even that result is not statistically distinguishable from zero.

Forcing every strategy to be shown beside the benchmark is deliberate. A strategy reported alone always looks better than it is.

## Tests

`test_engine()` verifies the engine against cases with known answers:

- flat prices produce exactly zero return
- rising prices profit a long position
- long and short returns are exact mirrors
- higher costs strictly reduce returns
- the first bar is always flat, confirming the one-bar lag

The lag test is the important one: a lookahead bug makes every strategy appear profitable, and nothing else in the output would reveal it.

An earlier version of this engine returned near-zero results for every strategy because a single-column DataFrame was being multiplied against a Series, producing misaligned indices. It printed a confident, formatted, entirely wrong report. These tests exist because of that.

## Limitations

- Single instrument only.
- Daily closes; no intraday, no slippage model, no partial fills.
- Costs are a flat per-trade rate; no spread model or market impact.
- Positions are discrete (−1, 0, 1); no position sizing.
- No borrow costs or short-availability constraints.

## Running it

Open `honest_backtester.ipynb` in Google Colab and run the cells in order. Requires `numpy`, `pandas`, `yfinance`.
