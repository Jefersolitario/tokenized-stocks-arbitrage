"""Prints every key number of REPORT.md from the result files.   python report_numbers.py [section]

Sections: key (the numbers quoted in the write-up), sample, h1, slopes, basis, rule, placebo, costs, variants, extended, premium, audit, posthoc, late, weekend, all (default).
Nothing is computed here except formatting: the numbers come from results/*.csv written by analyze.py, audit.py and
spreads.py.
"""
import json
import sys

import polars as pl

from common import RES

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(110); pl.Config.set_float_precision(3)
pl.Config.set_tbl_hide_dataframe_shape(True); pl.Config.set_tbl_hide_column_data_types(True)


def rd(name):
    return pl.read_csv(RES / name, infer_schema_length=None)


def h1():
    print("\n== H1: slope of the official Monday gap on the venue's weekend move, weekend-level bootstrap ==")
    print(rd("h1.csv"))


def slopes():
    s = rd("test1_slopes.csv")
    cols = ["label", "n", "clusters", "slope", "lo", "hi", "r2", "intercept_bps", "rmse_y_minus_x_bps", "rmse_y_bps", "medabs_y_minus_x_bps", "medabs_y_bps",
            "share_x_closer", "sign_agree_y_ge_25bps", "placebo_p2_5", "placebo_p97_5", "placebo_share_ge_real", "placebo_lag1"]
    print("\n== Test 1 / 4: all slopes (y = official gap unless the label says the venue's later move) ==")
    print(s.select([c for c in cols if c in s.columns]))
    j = json.loads((RES / "summary.json").read_text())
    for k, v in j.items():
        if k.startswith("per_contract"):
            print(k, v)


def basis():
    print("\n== Test 2: leftover basis (bps) ==")
    print(rd("test2_basis.csv"))
    r = rd("test2_regressions.csv")
    print(r.select("label", "n", "clusters", "slope", "lo", "hi", "r2", "intercept_bps"))


def rule():
    print("\n== Test 3 / 4: selection on the earlier half, one evaluation on the later half (bps per trade, net of costs) ==")
    s = rd("test3_selected.csv")
    print(s.select("family", "entry", "signal", "exit", "X", "side", "fallback", "sel_trades", "sel_net_bps", "sel_lo", "sel_hi", "eval_trades", "eval_clusters_with_trades", "eval_clusters",
                   "eval_gross_bps", "eval_funding_bps", "eval_cost_bps", "eval_net_bps", "eval_lo", "eval_hi", "eval_p_net_le_0", "eval_p_bh", "eval_p_bh10", "eval_hit",
                   "eval_wk_net_bps", "eval_wk_lo", "eval_wk_hi", "eval_wk_t", "eval_wk_p_le_0", "eval_entry_hour_usd_median", "eval_entry_hour_usd_p10", "eval_entry_hour_usd_min",
                   "eval_share_entry_hour_ge_100k", "eval_share_entry_hour_ge_10k"))
    g = rd("test3_grid.csv")
    print("\n-- full grid, FADE side (FOLLOW gross = -FADE gross; FOLLOW net = -FADE gross - costs + funding of the opposite sign) --")
    print(g.filter((pl.col("side") == "fade")).select("family", "entry", "signal", "exit", "X", "half", "trades", "clusters_with_trades", "clusters", "gross_bps", "gross_lo", "gross_hi",
                                                      "t_cluster_gross", "funding_bps", "cost_bps", "fee_bps", "net_bps", "net_lo", "net_hi", "p_net_le_0", "wk_net_bps", "wk_t", "wk_p_le_0", "hit",
                                                      "entry_hour_usd_median", "entry_hour_usd_p10", "entry_hour_usd_min", "share_entry_hour_ge_100k", "share_entry_hour_ge_10k", "funding_missing"))
    print("\n-- full grid, FOLLOW side, net --")
    print(g.filter((pl.col("side") == "follow")).select("family", "entry", "signal", "exit", "X", "half", "trades", "gross_bps", "net_bps", "net_lo", "net_hi", "p_net_le_0"))


def placebo():
    print("\n== Placebo: signals reassigned to other closures (2,000 permutations), cluster t of the gross FADE PnL ==")
    p = rd("test3_placebo.csv")
    print(p.select("family", "scope", "entry", "signal", "exit", "X", "t_fade_real", "placebo_t_p2_5", "placebo_t_p97_5", "placebo_p_fade", "placebo_p_follow", "t_fade_lag1_placebo", "is_best_of_grid"))
    print(p.group_by("family", "scope").agg(pl.col("best_abs_t_real").first(), pl.col("placebo_best_abs_t_p95").first(), pl.col("placebo_p_best_of_grid").first()).sort("family", "scope"))


def costs():
    print("\n== Cost sensitivity of the selected cell, later half (bps per trade) ==")
    print(rd("test3_costs.csv"))
    print("\n== Fees and measured half-spreads ==")
    sp = rd("spreads.csv")
    print(sp.group_by("venue").agg(contracts=pl.len(), thin_book_1k=pl.col("thin_book").sum(), thin_book_10k=pl.col("thin_book_10k").sum(), quoted_half_median=pl.col("quoted_half_bps").median(),
                                   cost1k_median=pl.col("cost_1000_bps").median(), cost1k_p90=pl.col("cost_1000_bps").quantile(0.9), cost1k_max=pl.col("cost_1000_bps").max(),
                                   cost10k_median=pl.col("cost_10000_bps").median(), cost10k_p90=pl.col("cost_10000_bps").quantile(0.9), cost50k_median=pl.col("cost_50000_bps").median(),
                                   half_spread_used_median=pl.col("half_spread_bps").median(), half_spread_10k_median=pl.col("half_spread_10k_bps").median(),
                                   taker_median=pl.col("taker_bps").median(), first=pl.col("first_snapshot_utc").min(), last=pl.col("last_snapshot_utc").max()))
    core = ["xyz:XYZ100", "xyz:SP500"] + [f"xyz:{t}" for t in ("NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "MU", "SMH")]
    print(sp.filter(pl.col("contract").is_in(core) | pl.col("contract").is_in(["NVDABUSDT", "TSLABUSDT", "SPYBUSDT", "QQQBUSDT", "MUBUSDT"])).select(
        "venue", "contract", "snapshots", "quoted_half_bps", "cost_1000_bps", "cost_10000_bps", "cost_50000_bps", "bid_depth_usd", "ask_depth_usd", "half_spread_bps", "half_spread_10k_bps", "taker_bps", "maker_bps"))
    print(rd("hl_fees.csv").group_by("growth_mode", "delisted").agg(n=pl.len(), taker_bps=pl.col("taker_bps").first(), maker_bps=pl.col("maker_bps").first()).sort("growth_mode", "delisted"))


def variants():
    print("\n== Variants of the primary configuration (selected threshold and side) ==")
    print(rd("test3_variants.csv"))


def extended():
    print("\n== Extended sample (2h candles), weekends before the 1h sample: the rule ==")
    if (RES / "test3_extended.csv").exists():
        print(rd("test3_extended.csv"))


def premium():
    print("\n== Hyperliquid hourly funding premium (perp against the venue's reference price) by hour type ==")
    print(rd("premium_by_regime.csv"))


def audit():
    print("\n== Clock audit: median |venue / official - 1| in bps at offsets around the session times ==")
    print(rd("audit_clock.csv"))
    c = rd("audit_coverage.csv")
    print(c.group_by("venue").agg(contracts=pl.len(), first=pl.col("first").min(), last=pl.col("last").max(), with_missing_hours=(pl.col("missing_hours") > 0).sum(),
                                  missing_hours_total=pl.col("missing_hours").sum(), zero_trade_candles=pl.col("zero_trade").sum(), median_hourly_usd=pl.col("median_hourly_usd").median()))
    lv = rd("audit_levels.csv")
    print(lv.filter(((pl.col("ratio_median") - 1).abs() > 0.02) | (pl.col("dropped_level") > 0) | (pl.col("dropped_split") > 0)))


def posthoc():
    print("\n== POST-HOC execution checks of the FADE cells before external pricing resumes (bps per trade, net) ==")
    e = rd("posthoc_execution.csv")
    cols = ["venue", "entry", "signal", "exit", "X", "variant", "half", "trades", "weekends_with_trades", "weekends", "gross_bps", "funding_bps", "cost_bps", "fee_bps", "net_bps", "net_lo", "net_hi",
            "p_net_le_0", "wk_net_bps", "wk_t", "wk_p_le_0", "hit", "entry_hour_usd_median", "share_entry_hour_ge_100k", "hedge_ratio_median", "hedge_ratio_p10", "hedge_ratio_p90"]
    print(e.filter(pl.col("entry") == "Rm").select(cols))
    print(e.filter(pl.col("entry") == "T0").select(cols))
    print("\n== POST-HOC: where along the weekend the venue's move is given back (slope of the segment's move on the move so far) ==")
    print(rd("posthoc_path.csv"))
    print("\n== POST-HOC: the reopening test on the weekends before the 1h sample (2h candles) ==")
    print(rd("posthoc_presample.csv"))
    print("\n== POST-HOC: every weekend clock, form and threshold in one best-of-grid placebo (cluster t of the gross FADE PnL) ==")
    a = rd("posthoc_placebo_all_cells.csv")
    print(a.group_by("family", "scope").agg(cells=pl.col("cells_in_grid").first(), best_abs_t_real=pl.col("best_abs_t_real").first(), placebo_best_p95=pl.col("placebo_best_abs_t_p95").first(),
                                            placebo_best_p99=pl.col("placebo_best_abs_t_p99").first(), p_best_of_grid=pl.col("placebo_p_best_of_grid").first(),
                                            shifts=pl.col("shifts").first(), shift_best_max=pl.col("shift_best_abs_t_max").first()).sort("family", "scope"))
    print(a.select("family", "scope", "entry", "signal", "exit", "X", "t_fade_real", "placebo_p_fade", "placebo_p_vs_best_of_grid", "shift_p_vs_best_of_grid", "t_fade_lag1_placebo", "is_best_of_grid"))
    b = rd("posthoc_by_weekend.csv")
    print("\n== POST-HOC: per-weekend mean net PnL per trade of the FADE cells ==")
    print(b.group_by("venue", "entry", "signal", "exit", "X").agg(weekends_with_trades=pl.len(), weekends_positive=(pl.col("mean_net_bps") > 0).sum(), median_weekend_bps=pl.col("mean_net_bps").median(),
                                                                 worst_weekend_bps=pl.col("mean_net_bps").min(), best_weekend_bps=pl.col("mean_net_bps").max(), trades=pl.col("trades").sum()).sort("venue", "entry", "signal", "exit", "X"))
    print(b.filter((pl.col("venue") == "hyperliquid") & (pl.col("entry") == "Rm") & (pl.col("signal") == "neutral") & (pl.col("X") == 0.01)))
    late()


def late():
    """The two checks added last (both POST-HOC): entry at the first trade after the signal time, and the fee-setting history."""
    if (RES / "posthoc_next_trade.csv").exists():
        print("\n== POST-HOC: entry at the first trade AFTER the signal time (Hyperliquid, Sunday evening; bps per trade, net) ==")
        n = rd("posthoc_next_trade.csv")
        print(n.select("signal", "X", "variant", "half", "trades", "weekends_with_trades", "weekends", "gross_bps", "cost_bps", "net_bps", "net_lo", "net_hi", "p_net_le_0", "wk_t", "wk_p_le_0",
                       "entry_hour_usd_median"))
        print(rd("posthoc_next_trade_slopes.csv"))
    if (RES / "posthoc_fee_history.csv").exists():
        print("\n== POST-HOC: fee setting at the time of each trade (metadata field lastFeeScaleChangeTime) ==")
        c = rd("posthoc_fee_history_contracts.csv")
        print(f"contracts {c.height}; last change inside the 1h sample: {int(c['change_inside_1h_sample'].sum())}; fewest days from a contract's last change to its first sampled weekend: "
              f"{c['days_from_change_to_first_sampled_weekend'].min()}; changes more than 2 days after the first 2h candle: "
              f"{c.filter(pl.col('days_from_first_candle_to_change') > 2).select('contract', 'first_2h_candle', 'last_fee_scale_change', 'taker_bps_now').rows()}")
        h = rd("posthoc_fee_history.csv")
        print(f"rule cells checked: {h.height}; trades entered before their contract's last fee-scale change, summed over all cells: {int(h['trades_before_last_change'].sum())}")


def weekend():
    print("\n== This weekend (20-23 Nov 2026), description only ==")
    print(rd("this_weekend.csv"))
    if (RES / "test1_capture_ratio.csv").exists():
        print("\n== Share of the venue's move at T1 that the cash open confirmed (gap / v), description only ==")
        print(rd("test1_capture_ratio.csv"))
    if (RES / "coverage_check.csv").exists():
        print("\n== Coverage of the nominal 95% cluster-bootstrap interval on synthetic data ==")
        print(rd("coverage_check.csv"))


def key():
    """The numbers quoted in the write-up, one line each, with the file they come from."""
    f = lambda x, d=1: "n/a" if x is None else f"{x:.{d}f}"
    S, G, SEL = rd("test1_slopes.csv"), rd("test3_grid.csv"), rd("test3_selected.csv")
    E, PA, PRE, BW = rd("posthoc_execution.csv"), rd("posthoc_placebo_all_cells.csv"), rd("posthoc_presample.csv"), rd("posthoc_by_weekend.csv")
    B, PATH, SP = rd("test2_basis.csv"), rd("posthoc_path.csv"), rd("spreads.csv")

    def slope(label, tag):
        r = S.filter(pl.col("label") == label)
        if r.height == 0 or r["slope"][0] is None:
            print(f"{tag}: (not available) {label}"); return
        r = r.row(0, named=True)
        print(f"{tag}: slope {f(r['slope'], 3)} [{f(r['lo'], 3)}, {f(r['hi'], 3)}], R2 {f(r['r2'], 3)}, n {r['n']}, closures {r['clusters']}, "
              f"rmse(y-x) {f(r['rmse_y_minus_x_bps'])} vs rmse(y) {f(r['rmse_y_bps'])} bps, x closer {f(r['share_x_closer'], 3)}, sign agree {f(r['sign_agree_y_ge_25bps'], 3)}, "
              f"placebo 97.5% {f(r.get('placebo_p97_5'), 3)}, lag-1 placebo {f(r.get('placebo_lag1'), 3)}   <- test1_slopes.csv | {label}")

    def cellrow(family, entry, signal, exit_, X, side, half, tag):
        r = G.filter((pl.col("family") == family) & (pl.col("entry") == entry) & (pl.col("signal") == signal) & (pl.col("exit") == exit_) & (pl.col("X") == X) & (pl.col("side") == side) & (pl.col("half") == half))
        if r.height == 0:
            print(f"{tag}: (not available)"); return
        r = r.row(0, named=True)
        print(f"{tag}: {family} {entry}->{exit_} {signal} X={X} {side} [{half}]: trades {r['trades']} in {r['clusters_with_trades']} of {r['clusters']} closures; gross {f(r['gross_bps'])}, funding {f(r['funding_bps'], 2)}, "
              f"cost {f(r['cost_bps'])} (fees {f(r['fee_bps'])}), net {f(r['net_bps'])} [{f(r['net_lo'])}, {f(r['net_hi'])}] bps, p(net<=0) {f(r['p_net_le_0'], 3)}; per-closure t {f(r['wk_t'], 2)} (p {f(r['wk_p_le_0'], 3)}); hit {f(r['hit'], 2)}; "
              f"entry-hour USD median {f(r['entry_hour_usd_median'], 0)}, p10 {f(r['entry_hour_usd_p10'], 0)}, min {f(r['entry_hour_usd_min'], 0)}, share>=100k {f(r['share_entry_hour_ge_100k'], 2)}, share>=10k {f(r['share_entry_hour_ge_10k'], 2)}; "
              f"funding missing {r['funding_missing']}   <- test3_grid.csv")

    print("\n== KEY NUMBERS ==")
    for venue in ("hyperliquid", "binance"):
        slope(f"{venue} weekend PRIMARY: gap on v(T1), all contracts", f"H1 primary [{venue}]")
        slope(f"{venue} weekend: gap on v(T1), fixed effects (stock-specific part)", f"H1 fixed effects [{venue}]")
        slope(f"{venue} weekend: gap on v(T1), core subset", f"H1 core [{venue}]")
        slope(f"{venue} weekend: gap on v(T1), earlier half", f"H1 earlier half [{venue}]")
        slope(f"{venue} weekend: gap on v(T1), later half", f"H1 later half [{venue}]")
        slope(f"{venue} weekend: gap on v(T0)", f"T0 [{venue}]")
        slope(f"{venue} weekend: gap on v(T0), fixed effects", f"T0 fixed effects [{venue}]")
        slope(f"{venue} weekend: gap on v(R-)", f"R- [{venue}]")
        slope(f"{venue} weekend: gap on v(R+)", f"R+ [{venue}]")
        slope(f"{venue} weekend: venue's later move P(O+)/P(T1)-1 on v(T1)", f"later move after T1 [{venue}]")
        slope(f"{venue} weekend: venue's move T0 to T1, P(T1)/P(T0)-1, on v(T0)", f"move T0->T1 [{venue}]")
        slope(f"{venue} weekend: venue's move across the Sunday reopening P(R+)/P(R-)-1 on v(R-)", f"reopening [{venue}]")
        slope(f"{venue} weekend: venue's move across the Sunday reopening, fixed effects", f"reopening fixed effects [{venue}]")
        slope(f"{venue} weekend: venue's move across the Sunday reopening, core subset", f"reopening core [{venue}]")
        slope(f"{venue} night PRIMARY: gap on v(T1), all contracts", f"nights [{venue}]")
        slope(f"{venue} night: venue's later move P(O+)/P(T1)-1 on v(T1)", f"nights later move [{venue}]")
    for lab in ("xyz:XYZ100 weekend: ETF gap on v(T1), 1h sample", "xyz:XYZ100 weekend: ETF gap on v(T0), 1h sample", "xyz:SP500 weekend: ETF gap on v(T1), 1h sample", "xyz:SP500 weekend: ETF gap on v(T0), 1h sample",
                "xyz:XYZ100 weekend: ETF gap on venue/index close - 1 at T1, 2h candles, all since listing", "xyz:XYZ100 weekend: ETF gap on venue/index close - 1 at T0, 2h candles, all since listing",
                "extended 2h sample, weekends before the 1h sample: gap on g(T1)", "extended 2h sample, weekends before the 1h sample: gap on g(T0)",
                "extended 2h sample, weekends before the 1h sample: gap on g(T1), weekend fixed effects", "extended 2h sample, all weekends since listing: gap on g(T1)"):
        slope(lab, "index / extended")
    print()
    print(SEL.select("family", "entry", "signal", "exit", "X", "side", "sel_trades", "sel_net_bps", "sel_lo", "sel_hi", "eval_trades", "eval_clusters_with_trades", "eval_clusters", "eval_gross_bps", "eval_funding_bps",
                     "eval_cost_bps", "eval_net_bps", "eval_lo", "eval_hi", "eval_p_net_le_0", "eval_p_bh", "eval_p_bh10", "eval_wk_t", "eval_wk_p_le_0"))
    for X in (0.01, 0.02, 0.03):
        for half in ("sel", "eval", "all"):
            cellrow("hl_weekend", "T1", "neutral", "Oplus", X, "fade", half, "Monday clock, two-leg FADE")
    for X in (0.01, 0.02, 0.03):
        for half in ("sel", "eval", "all"):
            cellrow("hl_weekend", "T1", "raw", "Oplus", X, "fade", half, "Monday clock, single-leg FADE")
    for signal in ("neutral", "raw"):
        for X in (0.01, 0.02, 0.03):
            for half in ("sel", "eval", "all"):
                cellrow("hl_weekend_reopen", "Rm", signal, "Rp", X, "fade", half, "Sunday evening FADE")
    for signal in ("neutral", "raw"):
        for X in (0.01, 0.02, 0.03):
            for half in ("sel", "eval", "all"):
                cellrow("hl_weekend", "T0", signal, "Oplus", X, "fade", half, "Sunday 20:00 UTC FADE to the open hour")
    for signal in ("neutral", "raw"):
        for X in (0.01, 0.03):
            for half in ("sel", "eval", "all"):
                cellrow("bn_weekend_reopen", "Rm", signal, "Rp", X, "fade", half, "Binance Sunday evening FADE")
    print()
    print(PA.group_by("family", "scope").agg(cells=pl.col("cells_in_grid").first(), best_abs_t_real=pl.col("best_abs_t_real").first(), placebo_best_p95=pl.col("placebo_best_abs_t_p95").first(),
                                             placebo_best_p99=pl.col("placebo_best_abs_t_p99").first(), p_best_of_grid=pl.col("placebo_p_best_of_grid").first(), shifts=pl.col("shifts").first(),
                                             shift_best_max=pl.col("shift_best_abs_t_max").first()).sort("family", "scope"))
    print(PA.filter(pl.col("entry").is_in(["Rm", "T1", "T0", "Rm+1h (signal at Rm)"]) & pl.col("exit").is_in(["Rp", "Oplus"]) & pl.col("signal").is_in(["neutral", "raw"])).select(
        "family", "scope", "entry", "signal", "exit", "X", "t_fade_real", "placebo_p_fade", "placebo_p_vs_best_of_grid", "shift_p_vs_best_of_grid", "t_fade_lag1_placebo").sort("family", "scope", "entry", "signal", "exit", "X"))
    print(E.filter((pl.col("entry") == "Rm") & (pl.col("signal") == "neutral") & pl.col("X").is_in([0.01, 0.03])).select(
        "venue", "X", "variant", "half", "trades", "weekends_with_trades", "gross_bps", "funding_bps", "cost_bps", "fee_bps", "net_bps", "net_lo", "net_hi", "p_net_le_0", "wk_t", "wk_p_le_0",
        "entry_hour_usd_median", "share_entry_hour_ge_100k", "hedge_ratio_median", "hedge_ratio_p10", "hedge_ratio_p90"))
    print(PRE)
    print(BW.group_by("venue", "entry", "signal", "exit", "X").agg(weekends_with_trades=pl.len(), weekends_positive=(pl.col("mean_net_bps") > 0).sum(), median_weekend_bps=pl.col("mean_net_bps").median(),
                                                                  worst_weekend_bps=pl.col("mean_net_bps").min(), best_weekend_bps=pl.col("mean_net_bps").max(), trades=pl.col("trades").sum()).sort("venue", "entry", "signal", "exit", "X"))
    print(B.filter(pl.col("basis").is_in(["B_pre30", "B_open30", "B_close1", "B_close0", "B_close1_usd", "B_openhour"])))
    print(PATH.filter(pl.col("subset").is_in(["all", "earlier half", "later half"])))
    core = ["xyz:XYZ100", "xyz:SP500"] + [f"xyz:{t}" for t in ("NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "MU", "SMH")]
    print(SP.filter(pl.col("contract").is_in(core)).select("contract", "snapshots", "quoted_half_bps", "cost_1000_bps", "cost_10000_bps", "cost_50000_bps", "bid_depth_usd", "ask_depth_usd", "half_spread_bps", "half_spread_10k_bps", "taker_bps", "maker_bps"))
    print(SP.group_by("venue").agg(contracts=pl.len(), cost1k_median=pl.col("cost_1000_bps").median(), cost1k_p75=pl.col("cost_1000_bps").quantile(0.75), cost1k_p90=pl.col("cost_1000_bps").quantile(0.9),
                                   cost10k_median=pl.col("cost_10000_bps").median(), cost10k_p90=pl.col("cost_10000_bps").quantile(0.9), thin_10k=pl.col("thin_book_10k").sum(),
                                   first=pl.col("first_snapshot_utc").min(), last=pl.col("last_snapshot_utc").max(), snapshots=pl.col("snapshots").max()))
    if (RES / "posthoc_next_trade.csv").exists():
        n = rd("posthoc_next_trade.csv")
        print(n.filter((pl.col("signal") == "neutral") & pl.col("X").is_in([0.01, 0.03])).select("X", "variant", "half", "trades", "weekends_with_trades", "gross_bps", "cost_bps", "net_bps", "net_lo", "net_hi",
                                                                                                 "wk_t", "entry_hour_usd_median"))
        print(rd("posthoc_next_trade_slopes.csv"))
    if (RES / "posthoc_fee_history.csv").exists():
        c, h = rd("posthoc_fee_history_contracts.csv"), rd("posthoc_fee_history.csv")
        print(f"fee history: contracts {c.height}; last fee-scale change inside the 1h sample {int(c['change_inside_1h_sample'].sum())}; fewest days from last change to first sampled weekend "
              f"{c['days_from_change_to_first_sampled_weekend'].min()}; trades entered before their contract's last change (all {h.height} cells) {int(h['trades_before_last_change'].sum())}"
              f"   <- posthoc_fee_history*.csv")


def sample():
    j = json.loads((RES / "summary.json").read_text())
    print("\n== How many variants were run (rows per result file) ==")
    for name, what in (("test1_slopes.csv", "slopes (each row one regression)"), ("test1_per_contract.csv", "per-contract slopes"), ("test2_basis.csv", "basis summaries"),
                       ("test2_regressions.csv", "basis regressions"), ("test3_grid.csv", "rule cells x sample halves (pre-registered grid)"), ("test3_selected.csv", "configurations selected and evaluated"),
                       ("test3_costs.csv", "cost cases of the selected cells"), ("test3_variants.csv", "variants of the primary configuration x halves"), ("test3_extended.csv", "extended-sample rule cells"),
                       ("test3_placebo.csv", "placebo rows, pre-registered families"), ("posthoc_execution.csv", "POST-HOC execution variants x halves"), ("posthoc_path.csv", "POST-HOC path slopes"),
                       ("posthoc_presample.csv", "POST-HOC pre-sample replication rows"), ("posthoc_placebo_all_cells.csv", "POST-HOC best-of-all-clocks placebo rows"),
                       ("posthoc_next_trade.csv", "POST-HOC next-trade entry variants x halves"), ("posthoc_next_trade_slopes.csv", "POST-HOC next-trade slopes"),
                       ("posthoc_fee_history.csv", "POST-HOC fee-history cells x halves")):
        if (RES / name).exists():
            d = rd(name)
            extra = ""
            if name == "test3_grid.csv":
                extra = f" = {d.select('family', 'entry', 'signal', 'exit', 'X', 'side', 'long_only').unique().height} distinct cells in {d['family'].n_unique()} families"
            if name == "posthoc_placebo_all_cells.csv":
                extra = " = " + ", ".join(f"{r[0]}: {r[1]} cells" for r in d.group_by("family").agg(pl.col("cells_in_grid").first()).rows())
            print(f"  {name}: {d.height} rows ({what}){extra}")
    print("\n== Samples ==")
    for k, v in j.items():
        if k.endswith("_clusters") or k in ("seed", "bootstrap_resamples", "permutations", "median_hourly_funding_rate", "primary_rule"):
            print(k, v)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    funcs = dict(sample=sample, h1=h1, slopes=slopes, basis=basis, rule=rule, placebo=placebo, costs=costs, variants=variants, extended=extended, premium=premium, audit=audit,
                 posthoc=posthoc, weekend=weekend, key=key, late=late)
    for name, f in funcs.items():
        if what in ("all", name):
            f()
