"""Builds the observation panels: one row per contract and closure (pair of consecutive sessions d0 -> d1).

    python build_panel.py

Reads data/*.parquet (venue candles and funding, downloaded by the workflow) and official/daily.csv.
Writes results/panel_hl.parquet (Hyperliquid, 1h candles), panel_hl2h.parquet (2h candles, extended history),
panel_idx.parquet (index contracts), panel_bn.parquet (Binance stock tokens).
Every venue price is the close of the candle ending exactly at the stated time, with at least one trade.
"""
import numpy as np
import polars as pl

from common import DATA, H, INDEX, RES, Funding, add_px, load_official, load_splits, price_table, rolling_beta, session_pairs

ENTRIES, EXITS = ["T0", "T1", "T0p1", "T1p1"], ["Oplus", "C1"]
WINDOWS = [(e, x) for e in ENTRIES for x in EXITS] + [("Rm", "Rp"), ("Rmp1", "Rp")]       # (entry, exit) pairs that need a funding sum


def with_official(obs, off, spl):
    o = off.select("ticker", "date", "open", "close", "dividend")
    obs = obs.join(o.select("ticker", pl.col("date").alias("d0"), pl.col("close").alias("close0")), on=["ticker", "d0"], how="left")
    obs = obs.join(o.select("ticker", pl.col("date").alias("d1"), pl.col("open").alias("open1"), pl.col("close").alias("close1"),
                            pl.col("dividend").alias("div1")), on=["ticker", "d1"], how="left")
    # within 7 calendar days of a split of that ticker
    near = (obs.join(spl.select("ticker", pl.col("date").alias("sdate")), on="ticker", how="inner")
            .filter((pl.col("d1") >= pl.col("sdate") - pl.duration(days=7)) & (pl.col("d0") <= pl.col("sdate") + pl.duration(days=7)))
            .select("ticker", "d0").unique().with_columns(split_near=pl.lit(True)))
    return obs.join(near, on=["ticker", "d0"], how="left").with_columns(pl.col("split_near").fill_null(False))


def add_funding(obs, fund, coin_of, prefix):
    """Adds <prefix>_<entry>_<exit> = sum of hourly funding rates in (entry, exit] and <prefix>n_... = records found."""
    coins = obs[coin_of].to_list() if isinstance(coin_of, str) and coin_of in obs.columns else [coin_of] * obs.height
    arr = {c: obs[c].to_numpy().astype("float64") for c in {k for w in WINDOWS for k in w} if c in obs.columns}
    coins = np.array(coins)
    new = {}
    for e, x in WINDOWS:
        s = np.full(obs.height, np.nan); n = np.full(obs.height, np.nan)
        for coin in np.unique(coins):
            m = coins == coin
            s[m], n[m] = fund.window(coin, arr[e][m], arr[x][m])
        new[f"{prefix}_{e}_{x}"] = s; new[f"{prefix}n_{e}_{x}"] = n
    return obs.with_columns(**{k: pl.Series(v) for k, v in new.items()})


def main():
    RES.mkdir(exist_ok=True)
    off, spl = load_official(), load_splits()
    pairs = session_pairs(off)
    betas = rolling_beta(off)
    c1 = pl.read_parquet(DATA / "hl_candles_1h.parquet")
    c30 = pl.read_parquet(DATA / "hl_candles_30m.parquet").filter(pl.col("n") > 0)
    c2 = pl.read_parquet(DATA / "hl_candles_2h.parquet")
    fund = Funding(pl.read_parquet(DATA / "hl_funding.parquet"))
    pt, pt2 = price_table(c1, "coin", H), price_table(c2, "coin", 2 * H)
    t_first = c1["t"].min()
    pairs1 = pairs.filter(pl.col("F") >= t_first + H)                      # closures the 1h candles can cover
    times = ["F", "T0", "T1", "T0p1", "T1p1", "Oplus", "C1", "Rm", "Rmp1", "Rp"]

    def index_legs(obs, table, cols):
        for pref, coin in (("I", "xyz:XYZ100"), ("J", "xyz:SP500")):
            sub = table.filter(pl.col("coin") == coin)
            for tcol in cols:
                obs = add_px(obs, sub, tcol, f"{pref}_{tcol}", by_coin=False)
        return obs

    # ---------------- Hyperliquid, 1h candles: stocks and ETFs
    coins = sorted(c for c in c1["coin"].unique().to_list() if c not in INDEX)
    obs = pairs1.join(pl.DataFrame({"coin": coins}), how="cross").with_columns(ticker=pl.col("coin").str.replace("xyz:", "", literal=True))
    obs = with_official(obs, off, spl)
    for tcol in times:
        obs = add_px(obs, pt, tcol, f"P_{tcol}", extra=tcol in ("T0", "T1", "T0p1", "T1p1", "F", "Rm", "Rmp1"))
    obs = obs.join(c30.select("coin", pl.col("t").alias("O"), pl.col("o").cast(pl.Float64).alias("P_open30")), on=["coin", "O"], how="left")
    obs = obs.join(c30.select("coin", (pl.col("t") + H // 2).alias("O"), pl.col("c").cast(pl.Float64).alias("P_preopen30")), on=["coin", "O"], how="left")
    obs = index_legs(obs, pt, times)
    obs = obs.join(betas, on=["ticker", "d0"], how="left")
    obs = add_funding(obs, fund, "coin", "fs"); obs = add_funding(obs, fund, "xyz:XYZ100", "fi"); obs = add_funding(obs, fund, "xyz:SP500", "fj")
    obs = obs.with_columns(level_ok=((pl.col("P_F") / pl.col("close0") - 1).abs() <= 0.30))
    obs.sort("d0", "coin").write_parquet(RES / "panel_hl.parquet", compression="zstd")
    print("panel_hl", obs.shape, "closures", obs.select(pl.col("d0").n_unique()).item(), "coins", len(coins))

    # ---------------- Hyperliquid, 2h candles: extended history (times on even UTC hours only)
    coins2 = sorted(c for c in c2["coin"].unique().to_list() if c not in INDEX)
    pairs2 = pairs.filter(pl.col("F") >= c2["t"].min() + 2 * H)
    o2 = pairs2.join(pl.DataFrame({"coin": coins2}), how="cross").with_columns(ticker=pl.col("coin").str.replace("xyz:", "", literal=True))
    o2 = with_official(o2, off, spl)
    t2 = ["F", "T0", "T1", "X16", "C1", "Z0", "Z4"]
    for tcol in t2:
        o2 = add_px(o2, pt2, tcol, f"P_{tcol}", extra=tcol in ("T0", "T1", "Z0"))
    o2 = index_legs(o2, pt2, t2)
    ndx = off.filter(pl.col("ticker") == "^NDX").select(pl.col("date").alias("d0"), pl.col("close").alias("ndx_close0"))
    o2 = o2.join(ndx, on="d0", how="left").join(betas, on=["ticker", "d0"], how="left")
    o2 = o2.with_columns(in_1h_sample=pl.col("F") >= t_first + H)
    o2.sort("d0", "coin").write_parquet(RES / "panel_hl2h.parquet", compression="zstd")
    print("panel_hl2h", o2.shape, "closures", o2.select(pl.col("d0").n_unique()).item(), "coins", len(coins2))

    # ---------------- index contracts against SPY / QQQ (returns) and the index closes (levels)
    rows = []
    for coin, (etf, idx) in INDEX.items():
        x = pairs.with_columns(coin=pl.lit(coin), ticker=pl.lit(etf))
        x = with_official(x, off, spl)
        lv = off.filter(pl.col("ticker") == idx)
        x = x.join(lv.select(pl.col("date").alias("d0"), pl.col("close").alias("idx_close0")), on="d0", how="left")
        x = x.join(lv.select(pl.col("date").alias("d1"), pl.col("close").alias("idx_close1"), pl.col("open").alias("idx_open1")), on="d1", how="left")
        for tcol in times:
            x = add_px(x, pt, tcol, f"P_{tcol}", extra=tcol in ("T0", "T1"))
        for tcol in t2:
            x = add_px(x, pt2, tcol, f"Q_{tcol}")
        x = x.join(c30.select("coin", pl.col("t").alias("O"), pl.col("o").cast(pl.Float64).alias("P_open30")), on=["coin", "O"], how="left")
        x = add_funding(x, fund, "coin", "fs")
        rows.append(x)
    idxp = pl.concat(rows).with_columns(in_1h_sample=pl.col("F") >= t_first + H)
    idxp.sort("d0", "coin").write_parquet(RES / "panel_idx.parquet", compression="zstd")
    print("panel_idx", idxp.shape)

    # ---------------- Binance stock tokens (USDT quoted), 1h klines; hedge leg = Hyperliquid xyz:XYZ100
    k1 = pl.read_parquet(DATA / "bn_klines_1h.parquet")
    k30 = pl.read_parquet(DATA / "bn_klines_30m.parquet").filter(pl.col("n") > 0)
    ptb = price_table(k1, "symbol", H, ntl="qv")
    usdc = ptb.filter(pl.col("coin") == "USDCUSDT")
    syms = sorted(s for s in k1["symbol"].unique().to_list() if s != "USDCUSDT")
    special = {"BRKB": "BRK-B"}
    tick = {s: special.get(s[:-5], s[:-5]) for s in syms}                 # NVDABUSDT -> NVDA
    pairsb = pairs.filter(pl.col("F") >= k1.filter(pl.col("symbol") != "USDCUSDT")["t"].min() + H)
    ob = pairsb.join(pl.DataFrame({"coin": syms, "ticker": [tick[s] for s in syms]}), how="cross")
    ob = with_official(ob, off, spl)
    for tcol in times:
        ob = add_px(ob, ptb, tcol, f"P_{tcol}", extra=tcol in ("T0", "T1", "T0p1", "T1p1", "F", "Rm", "Rmp1"))
        ob = add_px(ob, usdc, tcol, f"U_{tcol}", by_coin=False)
    ob = ob.join(k30.select(pl.col("symbol").alias("coin"), pl.col("t").alias("O"), pl.col("o").alias("P_open30")), on=["coin", "O"], how="left")
    ob = ob.join(k30.select(pl.col("symbol").alias("coin"), (pl.col("t") + H // 2).alias("O"), pl.col("c").alias("P_preopen30")), on=["coin", "O"], how="left")
    ob = index_legs(ob, pt, times)
    ob = ob.join(betas, on=["ticker", "d0"], how="left")
    ob = add_funding(ob, fund, "xyz:XYZ100", "fi")
    ob = ob.with_columns(level_ok=((pl.col("P_F") / pl.col("close0") - 1).abs() <= 0.30))
    ob.sort("d0", "coin").write_parquet(RES / "panel_bn.parquet", compression="zstd")
    print("panel_bn", ob.shape, "closures", ob.select(pl.col("d0").n_unique()).item(), "tokens", len(syms))


if __name__ == "__main__":
    main()
