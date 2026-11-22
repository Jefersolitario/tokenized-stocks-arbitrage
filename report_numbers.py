"""Prints every key number of REPORT.md from the result files.   python report_numbers.py [section]

Sections: h1, slopes, basis, rule, placebo, costs, variants, extended, premium, audit, sample, all (default).
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
                   "eval_gross_bps", "eval_funding_bps", "eval_cost_bps", "eval_net_bps", "eval_lo", "eval_hi", "eval_p_net_le_0", "eval_p_bh", "eval_hit", "eval_entry_hour_usd_median", "eval_entry_hour_usd_p10"))
    g = rd("test3_grid.csv")
    print("\n-- full grid, FADE side (FOLLOW gross = -FADE gross; FOLLOW net = -FADE gross - costs + funding of the opposite sign) --")
    print(g.filter((pl.col("side") == "fade")).select("family", "entry", "signal", "exit", "X", "half", "trades", "clusters_with_trades", "clusters", "gross_bps", "gross_lo", "gross_hi",
                                                      "t_cluster_gross", "funding_bps", "cost_bps", "net_bps", "net_lo", "net_hi", "p_net_le_0", "hit", "entry_hour_usd_median", "funding_missing"))
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
    print(sp.group_by("venue").agg(contracts=pl.len(), thin_book=pl.col("thin_book").sum(), quoted_half_median=pl.col("quoted_half_bps").median(), cost1k_median=pl.col("cost_1000_bps").median(),
                                   cost10k_median=pl.col("cost_10000_bps").median(), cost10k_p90=pl.col("cost_10000_bps").quantile(0.9), cost50k_median=pl.col("cost_50000_bps").median(),
                                   half_spread_used_median=pl.col("half_spread_bps").median(), taker_median=pl.col("taker_bps").median(), first=pl.col("first_snapshot_utc").min(), last=pl.col("last_snapshot_utc").max()))
    core = ["xyz:XYZ100", "xyz:SP500"] + [f"xyz:{t}" for t in ("NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "MU", "SMH")]
    print(sp.filter(pl.col("contract").is_in(core) | pl.col("contract").is_in(["NVDABUSDT", "TSLABUSDT", "SPYBUSDT", "QQQBUSDT", "MUBUSDT"])).select(
        "venue", "contract", "snapshots", "quoted_half_bps", "cost_1000_bps", "cost_10000_bps", "cost_50000_bps", "bid_depth_usd", "ask_depth_usd", "half_spread_bps", "taker_bps", "maker_bps"))
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


def sample():
    j = json.loads((RES / "summary.json").read_text())
    print("\n== Samples ==")
    for k, v in j.items():
        if k.endswith("_clusters") or k in ("seed", "bootstrap_resamples", "permutations", "median_hourly_funding_rate", "primary_rule"):
            print(k, v)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    funcs = dict(sample=sample, h1=h1, slopes=slopes, basis=basis, rule=rule, placebo=placebo, costs=costs, variants=variants, extended=extended, premium=premium, audit=audit)
    for name, f in funcs.items():
        if what in ("all", name):
            f()
