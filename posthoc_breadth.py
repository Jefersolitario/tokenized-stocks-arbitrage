"""POST-HOC description, written after the results were seen: how broad is the Sunday-evening FADE across contracts?

    python posthoc_breadth.py        (after analyze.py)

Hyperliquid weekends, entry 19:00 and exit 22:00 New York time, FADE, both forms, thresholds 1%, 2%, 3%, all weekends.
Writes results/posthoc_by_contract.csv (one row per form, threshold and contract) and prints a digest.
"""
import numpy as np
import polars as pl

import analyze as an
from common import RES

BPS = 1e4
# Funds and leveraged products among the contracts, judged from the tickers (approximate: a few recent tickers could not be classified and count as single stocks).
ETF_LIKE = {"SMH", "SOXL", "TLT", "XBI", "MAGS", "KORU", "EWT", "EWY", "EWJ", "EWZ", "KSTR", "URNM", "XLE", "MUU", "NCLD", "SNXX", "LYTE", "DRAM"}


def main():
    an.COSTS = an.Costs()
    k = an.prep(pl.read_parquet(RES / "panel_hl.parquet"), "hyperliquid").filter(pl.col("kind") == "weekend")
    rows = []
    for signal in ("neutral", "raw"):
        c = an.candidates(k, "hyperliquid", "Rm", "Rp", signal)
        for X in an.X_GRID:
            kk = an.cell(c, X, "fade")
            d = pl.DataFrame(dict(contract=c["coin"][kk["take"]], net=kk["net"] * BPS, beta=c["beta"][kk["take"]], core=c["core"][kk["take"]], rth_only=c["rth_only"][kk["take"]],
                                  ntl=c["ntl"][kk["take"]], cl=c["cl"][kk["take"]]))
            g = d.group_by("contract").agg(trades=pl.len(), weekends=pl.col("cl").n_unique(), mean_net_bps=pl.col("net").mean(), sum_net_bps=pl.col("net").sum(),
                                           hit=(pl.col("net") > 0).mean(), hedge_ratio_mean=pl.col("beta").mean(), entry_hour_usd_median=pl.col("ntl").median(),
                                           core=pl.col("core").first(), regular_hours_only=pl.col("rth_only").first()).sort("contract")
            g = g.with_columns(signal=pl.lit(signal), X=pl.lit(X), etf_like=pl.col("contract").str.replace("xyz:", "").is_in(sorted(ETF_LIKE)))
            rows.append(g)
            top = g.sort("trades", descending=True).head(10)
            tot = float(g["sum_net_bps"].sum())
            etf = d.filter(pl.col("contract").str.replace("xyz:", "").is_in(sorted(ETF_LIKE)))["net"]
            stk = d.filter(~pl.col("contract").str.replace("xyz:", "").is_in(sorted(ETF_LIKE)))["net"]
            print(f"{signal} X={X}: {d.height} trades in {g.height} contracts; contracts with a positive mean: {int((g['mean_net_bps'] > 0).sum())}; "
                  f"ten most traded contracts: {top['trades'].sum() / d.height:.0%} of trades, {top['sum_net_bps'].sum() / tot:.0%} of the summed PnL; "
                  f"largest single contract: {g['sum_net_bps'].max() / tot:.0%} of the summed PnL; "
                  f"funds and leveraged products: {len(etf)} trades, mean {etf.mean() if len(etf) else float('nan'):.1f} bps; single stocks: {len(stk)} trades, mean {stk.mean() if len(stk) else float('nan'):.1f} bps")
            if signal == "neutral":
                b = d["beta"].to_numpy(); n = d["net"].to_numpy()
                for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 3.0001)):
                    m = (b >= lo) & (b < hi) if hi <= 3 else (b >= 3)
                    if m.any():
                        print(f"    hedge ratio in [{lo}, {min(hi, 3)}{']' if hi > 3 else ')'}: {int(m.sum())} trades, mean net {n[m].mean():.1f} bps")
    pl.concat(rows).select("signal", "X", "contract", "trades", "weekends", "mean_net_bps", "sum_net_bps", "hit", "hedge_ratio_mean", "entry_hour_usd_median", "core",
                           "regular_hours_only", "etf_like").write_csv(RES / "posthoc_by_contract.csv")


if __name__ == "__main__":
    main()
