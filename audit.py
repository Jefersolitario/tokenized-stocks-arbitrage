"""Data-quality and clock audit (no hypothesis outcome is computed here).

    python audit.py

1. Clock: the venue price should be closest to the official close at the cash close time and to the official open at
   the cash open time. If the session clock (time zone, daylight saving) or the meaning of Yahoo's open/close were
   wrong, the minimum would sit at another offset.
2. Instrument and split check: venue price / official close at the close time, per contract.
3. Candle coverage: first and last candle, missing hours, zero-trade candles per contract.
Writes results/audit_*.csv.
"""
import polars as pl

from common import DATA, H, INDEX, RES, close_ms, load_official, open_ms, price_table


def clock_audit(pt, c30, off, key_map, label):
    days = off.select("ticker", "date", "open", "close").join(key_map, on="ticker", how="inner")
    days = days.with_columns(C=pl.col("date").map_elements(close_ms, return_dtype=pl.Int64), O=pl.col("date").map_elements(open_ms, return_dtype=pl.Int64))
    out = []
    for k in range(-3, 4):
        j = days.with_columns(t_end=pl.col("C") + k * H).join(pt.select("coin", "t_end", "px"), on=["coin", "t_end"], how="inner")
        d = (j["px"] / j["close"] - 1).abs() * 1e4
        out.append(dict(venue=label, at="close", offset_hours=float(k), n=j.height, median_abs_bps=d.median(), mean_abs_bps=d.mean(), p90_abs_bps=d.quantile(0.9)))
    for k in range(-2, 4):
        j = days.with_columns(t=pl.col("O") + k * (H // 2)).join(c30.select("coin", "t", pl.col("o").cast(pl.Float64).alias("px")), on=["coin", "t"], how="inner")
        d = (j["px"] / j["open"] - 1).abs() * 1e4
        out.append(dict(venue=label, at="open (30m candle open)", offset_hours=k / 2, n=j.height, median_abs_bps=d.median(), mean_abs_bps=d.mean(), p90_abs_bps=d.quantile(0.9)))
    return out


def main():
    RES.mkdir(exist_ok=True)
    off = load_official()
    c1 = pl.read_parquet(DATA / "hl_candles_1h.parquet")
    c30 = pl.read_parquet(DATA / "hl_candles_30m.parquet").filter(pl.col("n") > 0)
    k1 = pl.read_parquet(DATA / "bn_klines_1h.parquet")
    k30 = pl.read_parquet(DATA / "bn_klines_30m.parquet").filter(pl.col("n") > 0).rename({"symbol": "coin"})
    coins = [c for c in c1["coin"].unique().to_list() if c not in INDEX]
    hl_map = pl.DataFrame({"coin": coins, "ticker": [c.split(":")[1] for c in coins]})
    syms = [s for s in k1["symbol"].unique().to_list() if s != "USDCUSDT"]
    bn_map = pl.DataFrame({"coin": syms, "ticker": [{"BRKB": "BRK-B"}.get(s[:-5], s[:-5]) for s in syms]})
    rows = clock_audit(price_table(c1, "coin", H), c30, off, hl_map, "hyperliquid") + clock_audit(price_table(k1, "symbol", H, ntl="qv"), k30, off, bn_map, "binance")
    clock = pl.DataFrame(rows)
    clock.write_csv(RES / "audit_clock.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(clock)

    # coverage per contract
    cov = []
    for label, df, key in (("hyperliquid 1h", c1, "coin"), ("binance 1h", k1, "symbol")):
        g = df.sort(key, "t").group_by(key, maintain_order=True).agg(
            first=pl.from_epoch(pl.col("t").min(), time_unit="ms"), last=pl.from_epoch(pl.col("t").max(), time_unit="ms"), candles=pl.len(),
            hours_spanned=(pl.col("t").max() - pl.col("t").min()) // H + 1, zero_trade=(pl.col("n") == 0).sum(),
            median_hourly_usd=(pl.col("v") * pl.col("c").cast(pl.Float64)).median() if key == "coin" else pl.col("qv").median())
        cov.append(g.rename({key: "contract"}).with_columns(venue=pl.lit(label), missing_hours=pl.col("hours_spanned") - pl.col("candles")))
    cov = pl.concat(cov)
    cov.write_csv(RES / "audit_coverage.csv")
    print(cov.group_by("venue").agg(contracts=pl.len(), first=pl.col("first").min(), last=pl.col("last").max(), with_missing_hours=(pl.col("missing_hours") > 0).sum(),
                                    zero_trade_candles=pl.col("zero_trade").sum(), median_hourly_usd=pl.col("median_hourly_usd").median()))

    # instrument / split check per contract, from the built panels
    lv = []
    for name in ("panel_hl", "panel_bn"):
        p = pl.read_parquet(RES / f"{name}.parquet").filter(pl.col("P_F").is_not_null() & pl.col("close0").is_not_null())
        r = p.with_columns(ratio=pl.col("P_F") / pl.col("close0"))
        lv.append(r.group_by("coin").agg(n=pl.len(), ratio_median=pl.col("ratio").median(), ratio_min=pl.col("ratio").min(), ratio_max=pl.col("ratio").max(),
                                         dropped_level=(~pl.col("level_ok")).sum(), dropped_split=pl.col("split_near").sum()).with_columns(panel=pl.lit(name)))
    lv = pl.concat(lv).sort("panel", "coin")
    lv.write_csv(RES / "audit_levels.csv")
    bad = lv.filter(((pl.col("ratio_median") - 1).abs() > 0.02) | (pl.col("dropped_level") > 0))
    with pl.Config(tbl_rows=60, tbl_width_chars=160):
        print("contracts whose venue/official ratio at the close is off (median off by >2% or any observation off by >30%):")
        print(bad)


if __name__ == "__main__":
    main()
