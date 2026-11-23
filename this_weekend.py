"""Description only, no test: where the venues stand this weekend (20-23 Nov 2026) against Friday's official close.

    python this_weekend.py      -> results/this_weekend.csv

Monday's open is not known when this is run, so nothing here says how the weekend ended.
"""
import datetime as dt

import polars as pl

from common import CORE, DATA, H, RES, close_ms, load_official, ms, price_table


def main():
    off = load_official()
    d0 = dt.date(2026, 11, 20)
    F = close_ms(d0)
    marks = {"at Friday's cash close time": F, "Fri 20:00 New York (internal pricing starts)": ms(d0, 20, 0, __import__("common").NY),
             "Sat 10:00 UTC": ms(dt.date(2026, 11, 21), 10), "Sun 00:00 UTC": ms(dt.date(2026, 11, 22), 0), "Sun 10:00 UTC": ms(dt.date(2026, 11, 22), 10)}
    pt = price_table(pl.read_parquet(DATA / "hl_candles_1h.parquet"), "coin", H)
    ptb = price_table(pl.read_parquet(DATA / "bn_klines_1h.parquet"), "symbol", H, ntl="qv")
    rows = []
    for t in CORE + ["AMD", "PLTR", "QQQ", "SPY"]:
        r = off.filter((pl.col("ticker") == t) & (pl.col("date") == d0))
        if r.height == 0:
            continue
        close = r["close"][0]; day = close / r["open"][0] - 1
        for venue, table, coin in (("hyperliquid", pt, f"xyz:{t}"), ("binance", ptb, f"{t}BUSDT")):
            sub = table.filter(pl.col("coin") == coin)
            if sub.height == 0:
                continue
            row = dict(ticker=t, venue=venue, friday_close=close, friday_open_to_close_pct=day * 100)
            for lab, tm in marks.items():
                x = sub.filter(pl.col("t_end") == tm)
                row[f"spread_pct {lab}"] = (x["px"][0] / close - 1) * 100 if x.height else None
            last = sub.sort("t_end").tail(1)
            row["latest_hour_end_utc"] = dt.datetime.fromtimestamp(last["t_end"][0] / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
            row["spread_pct latest"] = (last["px"][0] / close - 1) * 100
            rows.append(row)
    out = pl.DataFrame(rows)
    RES.mkdir(exist_ok=True); out.write_csv(RES / "this_weekend.csv")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250, float_precision=2):
        print(out)


if __name__ == "__main__":
    main()
