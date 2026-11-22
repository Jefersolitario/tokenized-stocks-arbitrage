"""POST-HOC execution check, written after the results were seen.

    python posthoc_next_trade.py        (after analyze.py)

The pre-registered rule enters at the venue price AT the signal time, i.e. the last trade before it (close of the
hourly candle ending at the signal time). A bot cannot trade at a price that has already printed. Here the signal is
unchanged, but the entry is the FIRST trade after the signal time (open of the hourly candle that starts at the signal
time), for the stock contract and for the index contract. Two exits: the last trade before the exit time (as in the
pre-registered rule) and the first trade after it. Hyperliquid weekends, Sunday-evening window (19:00 -> 22:00 New York).

Writes results/posthoc_next_trade.csv.
"""
import numpy as np
import polars as pl

import analyze as an
import posthoc as ph
from common import DATA, RES

BPS = 1e4


def main():
    an.COSTS = an.Costs()
    c = pl.col
    k = an.prep(pl.read_parquet(RES / "panel_hl.parquet"), "hyperliquid").filter(c("kind") == "weekend")
    hv = an.halves(k)
    c1 = pl.read_parquet(DATA / "hl_candles_1h.parquet").filter(c("n") > 0)
    nxt = c1.select("coin", "t", o=c("o").cast(pl.Float64), ntl=c("v") * c("c").cast(pl.Float64), n=c("n").cast(pl.Int64))
    idx = nxt.filter(c("coin") == "xyz:XYZ100").select("t", io=c("o"))
    for tcol in ("Rm", "Rp"):
        k = k.join(nxt.select("coin", c("t").alias(tcol), c("o").alias(f"N_{tcol}"), c("ntl").alias(f"N_{tcol}_ntl"), c("n").alias(f"N_{tcol}_n")), on=["coin", tcol], how="left")
        k = k.join(idx.select(c("t").alias(tcol), c("io").alias(f"NI_{tcol}")), on=tcol, how="left")
    # pseudo clocks: RmN / RpN = first trade after Rm / Rp.  Funding payments are those of the original window.
    k = k.with_columns(rs_RmN_Rp=c("P_Rp") / c("N_Rm") - 1, ri_RmN_Rp=c("I_Rp") / c("NI_Rm") - 1, fs_RmN_Rp=c("fs_Rm_Rp"), fi_RmN_Rp=c("fi_Rm_Rp"),
                       rs_RmN_RpN=c("N_Rp") / c("N_Rm") - 1, ri_RmN_RpN=c("NI_Rp") / c("NI_Rm") - 1, fs_RmN_RpN=c("fs_Rm_Rp"), fi_RmN_RpN=c("fi_Rm_Rp"),
                       P_RmN_ntl=c("N_Rm_ntl"), P_RmN_n=c("N_Rm_n"),
                       slip_stock_bps=(c("N_Rm") / c("P_Rm") - 1) * BPS)
    rows = []
    for signal in ("neutral", "raw"):
        base = an.candidates(k, "hyperliquid", "Rm", "Rp", signal)
        n1 = an.candidates(k, "hyperliquid", "RmN", "Rp", signal, sig_time="Rm")
        n2 = an.candidates(k, "hyperliquid", "RmN", "RpN", signal, sig_time="Rm")
        variants = [("pre-registered: entry at the last trade before 19:00, exit at the last trade before 22:00", base, None, "primary"),
                    ("entry at the first trade after 19:00, exit at the last trade before 22:00", n1, None, "primary"),
                    ("entry at the first trade after 19:00, exit at the first trade after 22:00", n2, None, "primary"),
                    ("first trade after 19:00 -> first trade after 22:00, at least 60 trades in the hour after entry", n2, lambda q: q["ntrades"] >= 60, "primary"),
                    ("first trade after 19:00 -> first trade after 22:00, at least 100,000 USD traded in the hour after entry", n2, lambda q: q["ntl"] >= 1e5, "primary"),
                    ("first trade after 19:00 -> first trade after 22:00, costs: half-spread 10 bps on every leg", n2, None, "hs10"),
                    ("first trade after 19:00 -> first trade after 22:00, costs: fee 4.5 bps and half-spread 10 bps", n2, None, "fee4.5_hs10")]
        for X in an.X_GRID:
            for name, q, f, case in variants:
                for half in ("sel", "eval", "all"):
                    m = np.isin(q["cl"], hv[half])
                    if f is not None:
                        m &= f(q)
                    s, kk = ph.stat(q, X, m, hv[half], case)
                    rows.append(dict(venue="hyperliquid", signal=signal, X=X, side="fade", variant=name, half=half, **s))
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_csv(RES / "posthoc_next_trade.csv")
    # The give-back slope with the entry price measured independently of the signal price: the regressor uses the last
    # trade before 19:00, the outcome starts at the first trade after 19:00, so a stale or bouncing last trade cannot
    # create the negative slope by itself.
    k2 = k.with_columns(y_next=c("P_Rp") / c("N_Rm") - 1, y_next2=c("N_Rp") / c("N_Rm") - 1)
    sl = []
    for lab, y in (("last trade before 19:00 -> last trade before 22:00 (pre-registered)", "venue_reopen"),
                   ("first trade after 19:00 -> last trade before 22:00", "y_next"), ("first trade after 19:00 -> first trade after 22:00", "y_next2")):
        for fe in (False, True):
            r = an.slope_row(k2, lab, "v_Rm", y, fe=fe, placebo=False)
            sl.append(dict(outcome=lab, fixed_effects=fe, n=r.get("n"), weekends=r.get("clusters"), slope=r.get("slope"), lo=r.get("lo"), hi=r.get("hi"), r2=r.get("r2")))
    sl = pl.DataFrame(sl)
    sl.write_csv(RES / "posthoc_next_trade_slopes.csv")
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=200, fmt_str_lengths=90, float_precision=3):
        print(sl)
    # how far the first trade after 19:00 is from the last trade before it, in the direction of the fade (bps; positive = the fade already lost that much)
    b = an.candidates(k.filter(c("slip_stock_bps").is_finite()), "hyperliquid", "Rm", "Rp", "neutral")
    kk = an.cell(b, 0.01, "fade")
    sl = k.filter(c("slip_stock_bps").is_finite()).filter(pl.all_horizontal([c(x).is_finite() for x in ("e_Rm", "rs_Rm_Rp", "ri_Rm_Rp", "beta")]))["slip_stock_bps"].to_numpy()[kk["take"]]
    adverse = -kk["d"] * sl
    print(f"two-leg fade 1%: first trade after 19:00 vs last trade before it, stock leg, in the direction against the fade: "
          f"mean {adverse.mean():.1f} bps, median {np.median(adverse):.1f}, p10 {np.percentile(adverse, 10):.1f}, p90 {np.percentile(adverse, 90):.1f} (n {len(adverse)})")
    with pl.Config(tbl_rows=140, tbl_cols=30, tbl_width_chars=260, fmt_str_lengths=110, float_precision=1):
        print(out.select("signal", "X", "variant", "half", "trades", "weekends_with_trades", "gross_bps", "cost_bps", "net_bps", "net_lo", "net_hi", "p_net_le_0", "wk_t", "entry_hour_usd_median"))


if __name__ == "__main__":
    main()
