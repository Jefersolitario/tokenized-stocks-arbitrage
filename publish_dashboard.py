"""Publishes the Sunday-evening fade for the portfolio dashboard, in its results contract (version 1).

Root: dashboard/ . One strategy, `sunday_fade_1k`, with the role "monitor" (shown for comparison, never added to the
portfolio figures). One row per weekend, dated by the UTC day on which the trades are closed.

  backtest rows   the study's two-leg 1% Sunday-evening cell on its 29 weekends (results/posthoc_by_weekend.csv):
                  trade prints and assumed costs, in-sample.
  paper rows      weekends worked out by paper_eval.py at recorded book prices (results/paper/<date>_trades.csv,
                  primary cell, official reference).

PnL is in USD at 1,000 USD for the stock leg of every trade. The two modes are never joined into one curve.
Published rows never change: a run that would alter one stops with an error.

Usage:  python publish_dashboard.py        (from the repository root, after paper_eval.py)
"""
import csv
import datetime as dt
import json
import pathlib
import subprocess

import polars as pl

import common

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE / "dashboard"
SID = "sunday_fade_1k"
TICKET = 1_000.0
MANIFEST = {
    "id": SID,
    "name": "Tokenized stocks: Sunday-evening fade",
    "market": "Hyperliquid stock contracts (xyz)",
    "currency": "USD",
    "days_per_year": 52,
    "description": (
        "Every Sunday at 19:00 New York: for each tokenized-stock contract whose move since Friday's close differs by 1% or more from "
        "beta times the Nasdaq-100 contract's move, trade 1,000 USD against the difference (sell the stock contract if it is above, buy "
        "if below) with the opposite position in the index contract, and close both at 22:00. One row per weekend, dated by the UTC day "
        "of the close. Backtest rows are the study's 29 weekends at trade prints with assumed costs (in-sample). Paper rows use recorded "
        "order books, the venue's fees and funding. Research only: no money is traded."),
    "ticket_usd": TICKET,
    "role": "monitor",
    "source_repo": "tokenized-stocks-arbitrage",
}


def backtest_rows():
    off = common.load_official()
    sessions = off.filter(pl.col("ticker") == "QQQ").sort("date")["date"].to_list()
    w = pl.read_csv(HERE / "results" / "posthoc_by_weekend.csv").filter(
        (pl.col("venue") == "hyperliquid") & (pl.col("entry") == "Rm") & (pl.col("signal") == "neutral") & (pl.col("exit") == "Rp") & (pl.col("X") == 0.01))
    rows = []
    for r in w.sort("weekend_d0").iter_rows(named=True):
        d0 = dt.date.fromisoformat(r["weekend_d0"])
        reopen = min(d for d in sessions if d > d0)          # the trades are closed in the first hours (UTC) of the reopening day
        rows.append({"date": reopen, "mode": "backtest", "pnl": r["sum_net_bps"] * TICKET / 1e4, "capital": 0.0, "n_trades": r["trades"]})
    return rows


def paper_rows():
    rows = []
    for p in sorted((HERE / "results" / "paper").glob("*_trades.csv")):
        sunday = dt.date.fromisoformat(p.name[:10])
        net = [float(r["net_bps"]) for r in csv.DictReader(open(p)) if r["cell"] == "primary" and r["reference"] == "official"]
        rows.append({"date": sunday + dt.timedelta(days=1), "mode": "paper", "pnl": sum(net) * TICKET / 1e4, "capital": 0.0, "n_trades": len(net)})
    return rows


def main():
    schema = {"date": pl.Date, "mode": pl.String, "pnl": pl.Float64, "capital": pl.Float64, "n_trades": pl.Int32}
    new = pl.DataFrame(backtest_rows() + paper_rows(), schema=schema).sort("mode", "date")
    out = ROOT / SID
    out.mkdir(parents=True, exist_ok=True)
    daily = out / "daily.parquet"
    if daily.exists():                                        # published history is append-only
        old = pl.read_parquet(daily)
        both = old.join(new, on=["date", "mode"], how="left", suffix="_new")
        gone = both.filter(pl.col("pnl_new").is_null())
        moved = both.filter((pl.col("pnl") - pl.col("pnl_new")).abs() > 1e-6)
        if gone.height or moved.height:
            raise SystemExit(f"refusing to change published rows: {gone.height} would disappear, {moved.height} would change\n{moved}")
    new.write_parquet(daily)
    back, paper = new.filter(pl.col("mode") == "backtest"), new.filter(pl.col("mode") == "paper")
    version = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    manifest = dict(MANIFEST, backtest_start=str(back["date"].min()), paper_start=str(paper["date"].min()) if paper.height else None, code_version=version)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    index = {"contract_version": 1, "source": "tokenized-stocks-arbitrage", "strategies": [SID],
             "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "data_through": str(new["date"].max())}
    (ROOT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"{SID}: {back.height} backtest weekends, total {back['pnl'].sum():.0f} USD; {paper.height} paper weekend(s), total {paper['pnl'].sum():.0f} USD; "
          f"data through {index['data_through']}")


if __name__ == "__main__":
    main()
