"""POST-HOC checks, written after the pre-registered results were seen. They add no new hypothesis: they attack the
cells that looked profitable (fading the weekend move before external pricing resumes) with execution-realism filters.

    python posthoc.py        (after analyze.py)

Writes results/posthoc_execution.csv, posthoc_by_weekend.csv, posthoc_path.csv, posthoc_presample.csv.
Variants of each FADE cell: entry delayed one hour (signal unchanged), entry-hour volume of at least 100,000 / 10,000
USD, at least 100 trades in the entry hour, core subset, contracts with / without overnight external pricing, the
short and the long side separately, harsher cost cases, and the worst and best result after dropping any one weekend.
"""
import numpy as np
import polars as pl

import analyze as an
import stats as st
from common import RES

BPS = 1e4
DELAY = {"T0": "T0p1", "Rm": "Rmp1", "T1": "T1p1"}


def stat(c, X, mask, cls, case="primary", side="fade"):
    k = an.cell(c, X, side, mask, case)
    s = st.cell_stats(k["net"] * BPS, c["cl"][k["take"]], cls)
    ntl = c["ntl"][k["take"]]
    fin = np.isfinite(ntl)
    return dict(trades=s["trades"], weekends_with_trades=s["clusters_with_trades"], weekends=s["clusters"],
                gross_bps=float(np.mean(k["gross"]) * BPS) if len(k["gross"]) else np.nan, funding_bps=float(np.mean(k["fund"]) * BPS) if len(k["fund"]) else np.nan,
                cost_bps=float(np.mean(k["cost"]) * BPS) if len(k["cost"]) else np.nan, net_bps=s["mean"], net_lo=s["lo"], net_hi=s["hi"], p_net_le_0=s["p_le_0"],
                wk_net_bps=s["wk_mean"], wk_t=s["wk_t"], wk_p_le_0=s["wk_p_le_0"], hit=s["hit"],
                entry_hour_usd_median=float(np.median(ntl[fin])) if fin.any() else np.nan, share_entry_hour_ge_100k=float(np.mean(ntl[fin] >= 1e5)) if fin.any() else np.nan), k


def presample():
    """The same reopening window on the 2h candle grid, for the weekends BEFORE the 1h sample (Nov 2025 - Apr 2026):
    entry 00:00 UTC on the reopening day (19:00 New York in winter time, 20:00 in summer time), exit 04:00 UTC.
    These weekends were not used to find the effect. Funding is assumed at the main sample's median hourly rate."""
    c = pl.col
    med = float(pl.read_parquet(an.DATA / "hl_funding.parquet")["rate"].median())
    p = pl.read_parquet(RES / "panel_hl2h.parquet").filter(c("kind") == "weekend")
    p = p.with_columns(gap=c("open1") / c("close0") - 1, cl=c("d0").cast(pl.Int32), g_Z0=c("P_Z0") / c("close0") - 1, m_Z0=c("I_Z0") / c("ndx_close0") - 1,
                       rs=c("P_Z4") / c("P_Z0") - 1, ri=c("I_Z4") / c("I_Z0") - 1, winter=(c("Z0") - c("Rm")).abs() < 1000)
    p = p.with_columns(e_Z0=c("g_Z0") - c("beta") * c("m_Z0"))
    p = p.filter((c("g_Z0").abs() <= 0.30) & ~c("split_near") & c("rs").is_finite())
    rows = []
    samples = [("before the 1h sample (not used to find the effect)", ~c("in_1h_sample")), ("before the 1h sample, winter time (entry 19:00 New York)", ~c("in_1h_sample") & c("winter")),
               ("before the 1h sample, summer time (entry 20:00 New York)", ~c("in_1h_sample") & ~c("winter")), ("inside the 1h sample (cross-check of the 2h definition)", c("in_1h_sample"))]
    for lab, f in samples:
        d = p.filter(f)
        for fe in (False, True):
            r = an.slope_row(d, lab, "g_Z0", "rs", fe=fe, placebo=False)
            rows.append(dict(sample=lab, what="slope of the venue's move 00:00-04:00 UTC on its spread at 00:00 UTC" + (", weekend fixed effects" if fe else ""), signal="", X=None,
                             trades=r.get("n"), weekends_with_trades=r.get("clusters"), weekends=r.get("clusters"), value=r.get("slope"), lo=r.get("lo"), hi=r.get("hi")))
        for signal, scol in (("neutral", "e_Z0"), ("raw", "g_Z0")):
            need = [scol, "rs"] + (["ri", "beta"] if signal == "neutral" else [])
            q = d.filter(pl.all_horizontal([c(k).is_finite() for k in need]))
            if q.height == 0:
                continue
            sg, rs, cl = an.arr(q, scol, "rs", "cl"); coins = q["coin"].to_list(); n = q.height
            cost = 2 * an.COSTS.per_side(coins, "primary", "hyperliquid") / BPS
            R, fund_long = rs.copy(), np.full(n, -med * 4)
            if signal == "neutral":
                ri, beta = an.arr(q, "ri", "beta")
                R = rs - beta * ri; fund_long = -med * 4 + beta * med * 4
                cost = cost + 2 * beta * an.COSTS.per_side(["xyz:XYZ100"] * n, "primary", "hyperliquid") / BPS
            allc = np.unique(d["cl"].to_numpy())
            for X in an.X_GRID:
                take = np.abs(sg) > X
                dd = -np.sign(sg[take]); gross = dd * R[take]; net = gross + dd * fund_long[take] - cost[take]
                sn = st.cell_stats(net * BPS, cl[take], allc)
                rows.append(dict(sample=lab, what="FADE, entry 00:00 UTC, exit 04:00 UTC: mean net PnL per trade (bps)", signal=signal, X=X, trades=sn["trades"],
                                 weekends_with_trades=sn["clusters_with_trades"], weekends=sn["clusters"], value=sn["mean"], lo=sn["lo"], hi=sn["hi"], p_le_0=sn["p_le_0"],
                                 gross_bps=float(np.mean(gross) * BPS) if take.any() else np.nan, cost_bps=float(np.mean(cost[take]) * BPS) if take.any() else np.nan,
                                 wk_t=sn["wk_t"], wk_p_le_0=sn["wk_p_le_0"], hit=sn["hit"]))
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_csv(RES / "posthoc_presample.csv")
    with pl.Config(tbl_rows=120, tbl_cols=30, tbl_width_chars=250, fmt_str_lengths=90, float_precision=2):
        print(out)


def main():
    an.COSTS = an.Costs()
    panels = {"hyperliquid": an.prep(pl.read_parquet(RES / "panel_hl.parquet"), "hyperliquid"), "binance": an.prep(pl.read_parquet(RES / "panel_bn.parquet"), "binance")}
    configs = {"hyperliquid": [("Rm", "neutral", "Rp"), ("Rm", "raw", "Rp"), ("T0", "neutral", "Oplus"), ("T0", "raw", "Oplus"), ("T0", "neutral", "C1")],
               "binance": [("Rm", "neutral", "Rp"), ("Rm", "raw", "Rp"), ("T0", "neutral", "Oplus"), ("T0", "raw", "Oplus")]}
    rows, byw = [], []
    for venue, df in panels.items():
        k = df.filter(pl.col("kind") == "weekend")
        hv = an.halves(k)
        for entry, signal, exit_ in configs[venue]:
            base = an.candidates(k, venue, entry, exit_, signal)
            delayed = an.candidates(k, venue, DELAY[entry], exit_, signal, sig_time=entry)
            variants = [("base", base, None, "primary"),
                        ("entry delayed one hour (signal unchanged)", delayed, None, "primary"),
                        ("entry-hour volume >= 100,000 USD", base, lambda c: c["ntl"] >= 1e5, "primary"),
                        ("entry-hour volume >= 10,000 USD", base, lambda c: c["ntl"] >= 1e4, "primary"),
                        ("at least 100 trades in the entry hour", base, lambda c: c["ntrades"] >= 100, "primary"),
                        ("delayed one hour and entry-hour volume >= 100,000 USD", delayed, lambda c: c["ntl"] >= 1e5, "primary"),
                        ("core subset", base, lambda c: c["core"], "primary"),
                        ("short side only (venue above the signal's reference)", base, lambda c: c["s"] > 0, "primary"),
                        ("long side only (venue below the signal's reference)", base, lambda c: c["s"] < 0, "primary"),
                        ("costs: 10,000 USD order book cost (reference line)", base, None, "ref_10k"),
                        ("costs: half-spread 10 bps on every leg", base, None, "hs10"),
                        ("costs: fee 4.5 bps and half-spread 10 bps", base, None, "fee4.5_hs10"),
                        ("costs: maker fee, no spread", base, None, "maker")]
            if venue == "hyperliquid":
                variants += [("contracts with overnight external pricing (24/5)", base, lambda c: ~c["rth_only"], "primary"),
                             ("contracts priced externally in regular hours only (no Sunday reopening)", base, lambda c: c["rth_only"], "primary")]
            for X in an.X_GRID:
                for name, c, f, case in variants:
                    for half in ("sel", "eval", "all"):
                        m = np.isin(c["cl"], hv[half])
                        if f is not None:
                            m &= f(c)
                        s, _ = stat(c, X, m, hv[half], case)
                        rows.append(dict(venue=venue, entry=entry, signal=signal, exit=exit_, X=X, side="fade", variant=name, half=half, **s))
                # leave one weekend out, whole sample
                m = np.isin(base["cl"], hv["all"]); kk = an.cell(base, X, "fade", m)
                cl = base["cl"][kk["take"]]; net = kk["net"] * BPS
                loo = [net[cl != w].mean() for w in np.unique(cl)] if len(net) else []
                if loo:
                    rows.append(dict(venue=venue, entry=entry, signal=signal, exit=exit_, X=X, side="fade", variant="drop any one weekend: worst / best mean net", half="all",
                                     trades=len(net), weekends_with_trades=len(loo), weekends=len(hv["all"]), net_bps=float(net.mean()), net_lo=float(min(loo)), net_hi=float(max(loo))))
                # per weekend
                d0 = base["d0"][kk["take"]]
                for w in np.unique(cl):
                    j = cl == w
                    byw.append(dict(venue=venue, entry=entry, signal=signal, exit=exit_, X=X, weekend_d0=str(d0[j][0]), half="sel" if w in hv["sel"] else "eval", trades=int(j.sum()),
                                    mean_gross_bps=float(kk["gross"][j].mean() * BPS), mean_net_bps=float(net[j].mean()), sum_net_bps=float(net[j].sum()),
                                    share_short=float((kk["d"][j] < 0).mean())))
    pl.DataFrame(rows, infer_schema_length=None).write_csv(RES / "posthoc_execution.csv")
    pl.DataFrame(byw).write_csv(RES / "posthoc_by_weekend.csv")

    # where along the weekend the venue's move is given back (description): later move on the cumulative move so far
    path = []
    c = pl.col
    for venue, df in panels.items():
        k = df.filter(c("kind") == "weekend").with_columns(
            seg_T0_Rm=c("P_Rm") / c("P_T0") - 1, seg_Rm_Rmp1=c("P_Rmp1") / c("P_Rm") - 1, seg_Rmp1_Rp=c("P_Rp") / c("P_Rmp1") - 1, seg_Rp_T1=c("P_T1") / c("P_Rp") - 1,
            seg_T1_Oplus=c("P_Oplus") / c("P_T1") - 1, v_Rmp1=c("P_Rmp1") / c("P_F") - 1, bm_Rm=c("beta") * c("m_Rm"))
        segs = [("Sun 20:00 UTC -> Sun 19:00 ET (index futures reopen at 18:00 ET)", "v_T0", "seg_T0_Rm"),
                ("Sun 19:00 ET -> 20:00 ET (last hour of internal pricing)", "v_Rm", "seg_Rm_Rmp1"),
                ("Sun 20:00 ET -> 22:00 ET (external pricing resumes at 20:05 ET)", "v_Rmp1", "seg_Rmp1_Rp"),
                ("Sun 22:00 ET -> Mon 08:00 UTC", "v_Rp", "seg_Rp_T1"), ("Mon 08:00 UTC -> end of the cash open hour", "v_T1", "venue_post")]
        for lab, x, y in segs:
            for sub, f in (("all", pl.lit(True)), ("core", c("core"))) + ((("24/5 external pricing", ~c("rth_only")), ("regular-hours external pricing only", c("rth_only"))) if venue == "hyperliquid" else ()):
                r = an.slope_row(k.filter(f), f"{venue}: {lab} [{sub}]", x, y, placebo=False)
                path.append(dict(venue=venue, segment=lab, subset=sub, n=r.get("n"), weekends=r.get("clusters"), slope=r.get("slope"), lo=r.get("lo"), hi=r.get("hi"), r2=r.get("r2")))
                r = an.slope_row(k.filter(f), f"{venue}: {lab} [{sub}] fe", x, y, fe=True, placebo=False)
                path.append(dict(venue=venue, segment=lab + " - weekend fixed effects", subset=sub, n=r.get("n"), weekends=r.get("clusters"), slope=r.get("slope"), lo=r.get("lo"), hi=r.get("hi"), r2=r.get("r2")))
        # do stock contracts lag the index contract at R-?  slope of the stock's spread on beta x index move (1 = no lag)
        r = an.slope_row(k, f"{venue}: g(R-) on beta x m(R-)", "bm_Rm", "g_Rm", placebo=False)
        path.append(dict(venue=venue, segment="spread g(R-) on beta x index-contract move m(R-) [1 = stocks have followed the index]", subset="all", n=r.get("n"), weekends=r.get("clusters"),
                         slope=r.get("slope"), lo=r.get("lo"), hi=r.get("hi"), r2=r.get("r2")))
        k2 = k.with_columns(bm_T1=c("beta") * c("m_T1"))
        r = an.slope_row(k2, f"{venue}: g(T1) on beta x m(T1)", "bm_T1", "g_T1", placebo=False)
        path.append(dict(venue=venue, segment="spread g(T1) on beta x index-contract move m(T1) [same, after external pricing resumed]", subset="all", n=r.get("n"), weekends=r.get("clusters"),
                         slope=r.get("slope"), lo=r.get("lo"), hi=r.get("hi"), r2=r.get("r2")))
    pl.DataFrame(path, infer_schema_length=None).write_csv(RES / "posthoc_path.csv")
    presample()
    with pl.Config(tbl_rows=120, tbl_cols=30, tbl_width_chars=250, fmt_str_lengths=80, float_precision=2):
        e = pl.read_csv(RES / "posthoc_execution.csv", infer_schema_length=None)
        print(e.filter((pl.col("venue") == "hyperliquid") & (pl.col("entry") == "Rm") & (pl.col("signal") == "neutral") & (pl.col("X") == 0.01)).select(
            "variant", "half", "trades", "weekends_with_trades", "gross_bps", "cost_bps", "net_bps", "net_lo", "net_hi", "p_net_le_0", "wk_t", "wk_p_le_0", "entry_hour_usd_median"))
        print(pl.read_csv(RES / "posthoc_path.csv", infer_schema_length=None))


if __name__ == "__main__":
    main()
