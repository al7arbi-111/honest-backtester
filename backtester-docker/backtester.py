import numpy as np
import pandas as pd
import yfinance as yf


class Backtest:
    """Daily backtester for single-instrument strategies.

    Lookahead is structurally impossible: the strategy function is only ever
    handed data up to the current bar. Positions lag by one bar. Costs are on
    by default.
    """

    def __init__(self, prices, cost_bps=5, train_end="2023-12-31"):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.cost = cost_bps / 10000
        self.train_end = train_end

    def train(self):
        return self.returns.loc[:self.train_end]

    def test(self):
        return self.returns.loc[self.train_end:].iloc[1:]

    def run(self, strategy, data=None):
        data = self.test() if data is None else data
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            data = data.iloc[:, 0]
        raw = pd.Series(
            [float(strategy(data.iloc[:i])) for i in range(1, len(data) + 1)],
            index=data.index, dtype=float)
        pos = raw.shift(1).fillna(0)
        gross = pos * data
        fees = pos.diff().abs().fillna(0) * self.cost
        return gross - fees

    def stats(self, net, rf=0.04):
        n = len(net)
        ann_r = net.mean() * 252
        ann_v = net.std() * 252 ** 0.5
        sharpe = (ann_r - rf) / ann_v if ann_v > 0 else np.nan
        eq = (1 + net).cumprod()
        active = net[net != 0]
        t_stat = sharpe * np.sqrt(n / 252) if ann_v > 0 else np.nan
        return {
            "days": n,
            "annual_return": ann_r,
            "volatility": ann_v,
            "sharpe": sharpe,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
            "win_rate": (active > 0).mean() if len(active) else np.nan,
            "t_stat": t_stat,
            "significant": abs(t_stat) > 1.96 if not np.isnan(t_stat) else False,
        }

    def report(self, strategy, name="strategy"):
        s = self.stats(self.run(strategy))
        print(f"=== {name} ===")
        print(f"  period        {s['days']} days")
        print(f"  annual ret    {s['annual_return']*100:.1f}%")
        print(f"  volatility    {s['volatility']*100:.1f}%")
        print(f"  sharpe        {s['sharpe']:.2f}")
        print(f"  max drawdown  {s['max_drawdown']*100:.1f}%")
        print(f"  win rate      {s['win_rate']*100:.1f}%")
        print(f"  t-statistic   {s['t_stat']:.2f}")
        verdict = "significant" if s["significant"] else "NOT distinguishable from zero"
        print(f"  verdict       {verdict}")
        print()

    def compare(self, strategies):
        rows = []
        for name, fn in strategies.items():
            s = self.stats(self.run(fn))
            rows.append({
                "strategy": name,
                "return": s["annual_return"] * 100,
                "vol": s["volatility"] * 100,
                "sharpe": s["sharpe"],
                "t_stat": s["t_stat"],
                "significant": s["significant"],
            })
        return pd.DataFrame(rows).set_index("strategy").round(2)


def test_engine():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    flat = pd.Series(100.0, index=idx)
    up = pd.Series(np.arange(100, 200.0), index=idx)

    b = Backtest(pd.DataFrame({"X": flat}), cost_bps=0)
    assert abs(b.run(lambda d: 1).sum()) < 1e-9, "flat prices must give zero return"

    b = Backtest(pd.DataFrame({"X": up}), cost_bps=0, train_end="2024-02-01")
    long_r = b.run(lambda d: 1).sum()
    short_r = b.run(lambda d: -1).sum()
    assert long_r > 0, "rising prices must profit a long"
    assert abs(long_r + short_r) < 1e-9, "long and short must be exact mirrors"

    b_free = Backtest(pd.DataFrame({"X": up}), cost_bps=0, train_end="2024-02-01")
    b_paid = Backtest(pd.DataFrame({"X": up}), cost_bps=10, train_end="2024-02-01")
    flip = lambda d: 1 if len(d) % 2 else -1
    assert b_paid.run(flip).sum() < b_free.run(flip).sum(), "costs must reduce returns"

    assert b.run(lambda d: 1).iloc[0] == 0, "first bar must be flat"

    print("all tests passed\n")


def momentum(data):
    if len(data) < 50:
        return 0
    return 1 if data.iloc[-50:].mean() > 0 else -1


def buy_and_hold(data):
    return 1


if __name__ == "__main__":
    test_engine()

    prices = yf.download(["SPY"], start="2019-01-01", end="2026-01-01",
                         progress=False)["Close"].dropna()
    bt = Backtest(prices, cost_bps=5)

    bt.report(momentum, "50-day momentum")
    bt.report(buy_and_hold, "buy and hold")

    print(bt.compare({"momentum": momentum, "buy and hold": buy_and_hold}))