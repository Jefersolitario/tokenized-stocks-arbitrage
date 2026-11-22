"""Shared definitions for the weekend-basis study: session clock, loaders, price lookups."""
import datetime as dt, gzip, json, os, pathlib
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
# WB_DATA / WB_RES redirect the venue data and the outputs (used only by the synthetic known-answer test)
DATA = pathlib.Path(os.environ.get("WB_DATA", HERE / "data"))
OFF = HERE / "official"
RES = pathlib.Path(os.environ.get("WB_RES", HERE / "results"))
NY, UTC = ZoneInfo("America/New_York"), dt.timezone.utc
H = 3_600_000
SEED = 20261122
EARLY = {dt.date(2025, 7, 3), dt.date(2025, 11, 28), dt.date(2025, 12, 24), dt.date(2026, 11, 27), dt.date(2026, 12, 24)}
CORE = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "MU", "SMH"]
INDEX = {"xyz:XYZ100": ("QQQ", "^NDX"), "xyz:SP500": ("SPY", "^GSPC")}
# xyz contracts the venue documentation lists with external prices in regular hours only (every night is an internal session)
RTH_ONLY = ["DKNG", "BX", "ZM", "EBAY", "BB", "BOT", "SHAZ", "NCLD", "ICE"]


def ms(d, hh, mm=0, tz=UTC):
    return int(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz).timestamp() * 1000)


def open_ms(d):
    return ms(d, 9, 30, NY)


def close_ms(d):
    return ms(d, 13 if d in EARLY else 16, 0, NY)


def load_official():
    return pl.read_csv(OFF / "daily.csv.gz", schema_overrides={"dividend": pl.Float64, "split": pl.Float64}).with_columns(pl.col("date").str.to_date())


def load_splits():
    return pl.read_csv(OFF / "splits.csv").with_columns(pl.col("date").str.to_date())


def session_pairs(off):
    """Consecutive sessions (d0, d1) from SPY's calendar with every clock time the tests use, in epoch ms."""
    sess = off.filter(pl.col("ticker") == "SPY").sort("date")["date"].to_list()
    rows = []
    for d0, d1 in zip(sess[:-1], sess[1:]):
        between = [d0 + dt.timedelta(days=k) for k in range(1, (d1 - d0).days)]
        wk = any(x.weekday() == 5 for x in between) and any(x.weekday() == 6 for x in between)
        kind = "weekend" if wk else ("night" if (d1 - d0).days == 1 else "other")
        prev = d1 - dt.timedelta(days=1)
        o, t1, t0 = open_ms(d1), ms(d1, 8), ms(prev, 20)
        rows.append(dict(d0=d0, d1=d1, kind=kind, gap_days=(d1 - d0).days, F=close_ms(d0), O=o, Oplus=o + H // 2, C1=close_ms(d1),
                         T1=t1, T1p1=t1 + H, T0=t0, T0p1=t0 + H, X16=ms(d1, 16),
                         Rm=ms(prev, 19, 0, NY), Rp=ms(prev, 22, 0, NY)))      # one hour before / two hours after external pricing resumes
    return pl.DataFrame(rows)


def price_table(candles, key, step, ntl=None):
    """Venue price at time t = close of the candle ending exactly at t, with at least one trade."""
    n = (pl.col("v") * pl.col("c").cast(pl.Float64)) if ntl is None else pl.col(ntl)
    return candles.filter(pl.col("n") > 0).select(pl.col(key).alias("coin"), (pl.col("t") + step).alias("t_end"),
                                                  pl.col("c").cast(pl.Float64).alias("px"), n.alias("ntl"), pl.col("n").cast(pl.Int64))


def add_px(obs, pt, time_col, name, by_coin=True, extra=False):
    cols = [pl.col("t_end").alias(time_col), pl.col("px").alias(name)]
    if extra:
        cols += [pl.col("ntl").alias(name + "_ntl"), pl.col("n").alias(name + "_n")]
    if by_coin:
        return obs.join(pt.select("coin", *cols), on=["coin", time_col], how="left")
    return obs.join(pt.select(*cols), on=time_col, how="left")


class Funding:
    """Sum of hourly funding rates over funding times in (a, b], per coin, with a completeness count."""

    def __init__(self, f):
        self.by = {}
        f = f.with_columns(th=(pl.col("t") // H) * H).sort("coin", "th")
        for (coin,), g in f.group_by("coin", maintain_order=True):
            th = g["th"].to_numpy(); r = g["rate"].to_numpy()
            self.by[coin] = (th, np.concatenate([[0.0], np.cumsum(r)]))

    def window(self, coin, a, b):
        """a, b: int arrays (ms). Returns (sum of rates, number of funding records) for (a, b]; NaN where a or b is missing."""
        a = np.asarray(a, dtype="float64"); b = np.asarray(b, dtype="float64")
        out = np.full(a.shape, np.nan); cnt = np.full(a.shape, np.nan)
        if coin not in self.by:
            return out, cnt
        th, cum = self.by[coin]
        ok = ~(np.isnan(a) | np.isnan(b))
        ia = np.searchsorted(th, a[ok], side="right"); ib = np.searchsorted(th, b[ok], side="right")
        out[ok] = cum[ib] - cum[ia]; cnt[ok] = ib - ia
        return out, cnt


def rolling_beta(off, bench="QQQ", window=60, min_obs=30):
    """beta of each ticker's close-to-close return on the benchmark's, over the `window` sessions ending at each date."""
    wide = off.select("ticker", "date", "close_file").pivot(on="ticker", index="date", values="close_file").sort("date")
    dates = wide["date"].to_list()
    b = wide[bench].to_numpy().astype(float); rb = np.full(len(b), np.nan); rb[1:] = b[1:] / b[:-1] - 1
    rows = []
    for t in wide.columns:
        if t == "date":
            continue
        p = wide[t].to_numpy().astype(float); r = np.full(len(p), np.nan); r[1:] = p[1:] / p[:-1] - 1
        for i in range(len(p)):
            lo = max(0, i - window + 1)
            x, y = rb[lo:i + 1], r[lo:i + 1]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= min_obs and np.var(x[ok]) > 0:
                beta = float(np.clip(np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok], ddof=1), 0.0, 3.0)); n = int(ok.sum())
            else:
                beta, n = 1.0, int(ok.sum())
            rows.append((t, dates[i], beta, n))
    return pl.DataFrame(rows, schema=["ticker", "d0", "beta", "beta_n"], orient="row")


def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
