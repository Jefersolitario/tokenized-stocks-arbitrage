"""Known-answer test of the whole pipeline on synthetic venue data.   python synth_test.py

Builds fake venue candles from the real official daily bars with a planted behaviour, runs build_panel.py, spreads.py
and analyze.py on them (in a scratch directory), and checks the outputs against values recomputed here directly from
the official bars with separate code.

Planted behaviour, every contract and closure: after the cash close the venue price moves to
close0 * (1 + gap / 0.8) and stays there (so the venue overshoots the true gap by 25%), then prints the official open
at the open, stays flat for 30 minutes and moves in a straight line to the official close. On weekends the venue
first overshoots further, to close0 * (1 + gap / 0.64), and comes back to gap / 0.8 when external pricing resumes
(20:00 New York time on the day before the reopening session).
Expected: Test 1 slope 0.8 exactly with R2 = 1 at T1 (give-back 20%) and 0.64 at T0 and R-, venue-only reversal slopes
close to -0.2 (T1 to the open hour, and across the reopening), and FADE rules whose gross PnL equals the recomputed
give-back; funding and costs equal the planted constants.
"""
import datetime as dt
import gzip
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
SCR = pathlib.Path(os.environ.get("WB_SCRATCH", HERE.parent.parent.parent / "scratch" / "synth"))
B_TRUE, B_PRE, R0, H = 0.8, 0.64, 6.25e-6, 3_600_000
TICKERS = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "MU", "SMH", "AMD", "PLTR", "COIN", "CVX", "LLY"]


def build():
    sys.path.insert(0, str(HERE))
    from common import NY, close_ms, load_official, ms, open_ms, session_pairs
    off = load_official()
    data = SCR / "data"; data.mkdir(parents=True, exist_ok=True)
    t0, t1 = ms(dt.date(2026, 4, 28), 4), ms(dt.date(2026, 11, 22), 11)
    t0_2h = ms(dt.date(2025, 11, 12), 20)
    rows1, rows30, rows2, fund = [], [], [], []
    legs = [(f"xyz:{t}", t, 1.0) for t in TICKERS] + [("xyz:XYZ100", "QQQ", None), ("xyz:SP500", "SPY", None)]
    lvl = {"xyz:XYZ100": "^NDX", "xyz:SP500": "^GSPC"}
    for coin, tick, _ in legs:
        o = off.filter(pl.col("ticker") == tick).sort("date")
        dates, op, cl = o["date"].to_list(), o["open"].to_numpy(), o["close"].to_numpy()
        if coin in lvl:                                   # index contracts trade at the index level, with the ETF's returns
            lv = off.filter(pl.col("ticker") == lvl[coin]).sort("date")["close"].to_numpy()
            scale = lv / cl; op, cl = op * scale, cl * scale
        om = np.array([open_ms(d) for d in dates]); cm = np.array([close_ms(d) for d in dates])
        at_open = {int(t): float(p) for t, p in zip(om, op)}
        # weekend closures: time external pricing resumes (20:00 New York on the day before the next session); nights: none
        reopen = [ms(dates[i + 1] - dt.timedelta(days=1), 20, 0, NY) if (dates[i + 1] - dates[i]).days >= 3 else None for i in range(len(dates) - 1)]

        def price(t):
            i = np.searchsorted(om, t, side="right") - 1          # last session whose open is <= t
            if i < 0:
                return np.nan
            if t <= cm[i]:                                        # inside session i
                if t <= om[i] + H // 2:
                    return op[i]
                w = (t - om[i] - H // 2) / (cm[i] - om[i] - H // 2)
                return op[i] + w * (cl[i] - op[i])
            if i + 1 >= len(om):                                  # after the last session: flat
                return cl[i]
            gap = op[i + 1] / cl[i] - 1
            w = min(1.0, (t - cm[i]) / (0.6 * (om[i + 1] - cm[i])))
            b = B_PRE if reopen[i] is not None and t < reopen[i] else B_TRUE
            return cl[i] * (1 + w * gap / b)
        for step, rows, a in ((H, rows1, t0), (H // 2, rows30, ms(dt.date(2026, 8, 10), 0)), (2 * H, rows2, t0_2h)):
            for t in range(a, t1 + 1, step):
                po, pc = price(t), price(t + step)
                if step == H // 2:
                    hod = (t % 86_400_000) / H
                    if not (12 <= hod < 16 or 19 <= hod < 22):
                        continue
                    po = at_open.get(t, po)                       # first trade at the open is the official open
                if np.isnan(po) or np.isnan(pc):
                    continue
                rows.append((coin, t, po, max(po, pc), min(po, pc), pc, 200_000.0 / pc, 10))
        for t in range(t0 - 200 * H, t1 + 1, H):
            fund.append((coin, t + 37, R0, 0.0))
    f32 = {k: pl.Float32 for k in ("o", "h", "l", "c")}
    sch = ["coin", "t", "o", "h", "l", "c", "v", "n"]
    pl.DataFrame(rows1, schema=sch, orient="row").cast(f32).cast({"n": pl.Int32}).write_parquet(data / "hl_candles_1h.parquet")
    for name, rows in (("30m", rows30), ("2h", rows2)):
        pl.DataFrame(rows, schema=sch, orient="row").cast(f32).cast({"n": pl.Int32}).drop("h", "l").write_parquet(data / f"hl_candles_{name}.parquet")
    pl.DataFrame(fund, schema=["coin", "t", "rate", "premium"], orient="row").write_parquet(data / "hl_funding.parquet")
    # Binance: the same paths under token names, quoted in a "USDT" worth exactly one dollar
    b1 = pl.DataFrame(rows1, schema=sch, orient="row").filter(~pl.col("coin").is_in(list(lvl)))
    tok = lambda df: df.with_columns(symbol=pl.col("coin").str.replace("xyz:", "", literal=True) + "BUSDT", qv=pl.col("v") * pl.col("c")).select("symbol", "t", "o", "h", "l", "c", "v", "qv", pl.col("n").cast(pl.Int32))
    usdc = pl.DataFrame({"symbol": "USDCUSDT", "t": list(range(t0, t1 + 1, H)), "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0, "qv": 1.0, "n": 5}).cast({"n": pl.Int32})
    pl.concat([tok(b1), usdc]).write_parquet(data / "bn_klines_1h.parquet")
    tok(pl.DataFrame(rows30, schema=sch, orient="row").filter(~pl.col("coin").is_in(list(lvl)))).write_parquet(data / "bn_klines_30m.parquet")
    # books: 50,000 USD a level, 1 bp apart, starting 0.5 bp from the mid  -> cost of a 10,000 USD order = 0.5 bp -> floor 1 bp
    book = lambda px: {"bids": [[f"{px * (1 - (0.5 + k) / 1e4):.6f}", f"{50_000 / px:.6f}", 1] for k in range(20)], "asks": [[f"{px * (1 + (0.5 + k) / 1e4):.6f}", f"{50_000 / px:.6f}", 1] for k in range(20)]}
    coins = [c for c, _, _ in legs]
    snap = [{"taken_utc_ms_start": t1, "taken_utc_ms_end": t1, "tag": "synthetic", "books": {c: dict(time=t1, **book(100.0)) for c in coins}}]
    for name, obj in (("hl_books.json.gz", snap), ("bn_books.json.gz", {"taken_utc_ms_start": t1, "books": {f"{t}BUSDT": {k: [x[:2] for x in v] for k, v in book(100.0).items()} for t in TICKERS}})):
        with gzip.open(data / name, "wt") as fh:
            json.dump(obj, fh)
    (data / "hl_meta_xyz.json").write_text(json.dumps({"meta": {"universe": [{"name": c, "deployerFeeScale": "1.0", "growthMode": "enabled", "maxLeverage": 10} for c in coins]}}))
    return off, session_pairs(off)


def run(script):
    env = dict(os.environ, WB_DATA=str(SCR / "data"), WB_RES=str(SCR / "results"))
    r = subprocess.run([sys.executable, str(HERE / script)], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-6000:]); raise SystemExit(f"{script} failed on the synthetic data")


def check(off, pairs):
    from common import rolling_beta
    res = SCR / "results"
    S = pl.read_csv(res / "test1_slopes.csv", infer_schema_length=None)
    row = lambda label: S.filter(pl.col("label") == label).row(0, named=True)
    for venue in ("hyperliquid", "binance"):
        for kind in ("weekend", "night"):
            r = row(f"{venue} {kind} PRIMARY: gap on v(T1), all contracts")
            assert abs(r["slope"] - B_TRUE) < 2e-3 and r["r2"] > 0.999 and r["lo"] <= r["slope"] <= r["hi"], (venue, kind, r)
            r2 = row(f"{venue} {kind}: venue's later move P(O+)/P(T1)-1 on v(T1)")
            assert -0.26 < r2["slope"] < -0.14, (venue, kind, r2)
        for lab, exp in (("gap on v(T0)", B_PRE), ("gap on v(R-)", B_PRE), ("gap on v(R+)", B_TRUE)):
            r = row(f"{venue} weekend: {lab}")
            assert abs(r["slope"] - exp) < 2e-3, (lab, r)
        r = row(f"{venue} weekend: venue's move across the Sunday reopening P(R+)/P(R-)-1 on v(R-)")
        assert -0.26 < r["slope"] < -0.14, r
    # independent recomputation of the raw FADE rule (entry T1, exit O+), Hyperliquid weekends, straight from the official bars
    o = off.select("ticker", "date", "open", "close")
    first_F = int(pl.read_parquet(SCR / "data" / "hl_candles_1h.parquet")["t"].min()) + H
    wk = pairs.filter((pl.col("kind") == "weekend") & (pl.col("F") >= first_F) & (pl.col("d1") <= dt.date(2026, 11, 20)))
    cls = sorted(wk["d0"].to_list()); half = int(np.ceil(len(cls) / 2))
    G = pl.read_csv(res / "test3_grid.csv", infer_schema_length=None)
    betas = rolling_beta(off)
    ndx = {r["date"]: r["close"] for r in o.filter(pl.col("ticker") == "^NDX").iter_rows(named=True)}
    # the synthetic xyz:XYZ100 is QQQ rescaled each day to the index level, so its gap is recomputed the same way
    q = {r["date"]: (r["open"] * ndx[r["date"]] / r["close"], ndx[r["date"]]) for r in o.filter(pl.col("ticker") == "QQQ").iter_rows(named=True)}
    for signal in ("raw", "neutral"):
        for X in (0.01, 0.02, 0.03):
            for half_name, sub in (("sel", cls[:half]), ("eval", cls[half:])):
                gross, n = [], 0
                for t in TICKERS:
                    px = {r["date"]: (r["open"], r["close"]) for r in o.filter(pl.col("ticker") == t).iter_rows(named=True)}
                    bt = {r["d0"]: r["beta"] for r in betas.filter(pl.col("ticker") == t).iter_rows(named=True)}
                    for r in wk.filter(pl.col("d0").is_in(sub)).iter_rows(named=True):
                        gap = px[r["d1"]][0] / px[r["d0"]][1] - 1
                        g = gap / B_TRUE; ret = (1 + gap) / (1 + g) - 1
                        if signal == "neutral":
                            gq = q[r["d1"]][0] / q[r["d0"]][1] - 1
                            s = g - bt[r["d0"]] * gq / B_TRUE; ret = ret - bt[r["d0"]] * ((1 + gq) / (1 + gq / B_TRUE) - 1)
                        else:
                            s = g
                        if abs(s) > X:
                            gross.append(-np.sign(s) * ret); n += 1
                got = G.filter((pl.col("family") == "hl_weekend") & (pl.col("entry") == "T1") & (pl.col("signal") == signal) & (pl.col("exit") == "Oplus") & (pl.col("X") == X) & (pl.col("side") == "fade") & (pl.col("half") == half_name)).row(0, named=True)
                assert got["trades"] == n, (signal, X, half_name, got["trades"], n)
                if n:
                    exp = float(np.mean(gross) * 1e4)
                    assert abs(got["gross_bps"] - exp) < 0.05, (signal, X, half_name, got["gross_bps"], exp)   # venue prices are stored as float32
                    # funding: FADE direction d pays d * hours * R0 on the stock leg (and receives beta * that on the hedge); costs: 2 * (0.9 + 1.0) bps per unit
                    assert abs(got["cost_bps"] - (3.8 if signal == "raw" else got["cost_bps"])) < 1e-6 and got["cost_bps"] >= 3.8 - 1e-9, got
                    assert abs(got["net_bps"] - (got["gross_bps"] + got["funding_bps"] - got["cost_bps"])) < 1e-6, got
                    assert abs(got["funding_bps"]) <= 7 * R0 * 1e4 * 4 + 1e-9, got
                    assert got["funding_missing"] == 0 and abs(got["entry_hour_usd_median"] - 200_000) < 1.0, got
    # the reopening family, raw signal: enter at R- on g(R-) = gap / 0.64, exit at R+ where the venue is at gap / 0.8
    for X in (0.01, 0.02, 0.03):
        for half_name, sub in (("sel", cls[:half]), ("eval", cls[half:])):
            gross = []
            for t in TICKERS:
                px = {r["date"]: (r["open"], r["close"]) for r in o.filter(pl.col("ticker") == t).iter_rows(named=True)}
                for r in wk.filter(pl.col("d0").is_in(sub)).iter_rows(named=True):
                    gap = px[r["d1"]][0] / px[r["d0"]][1] - 1
                    sgl = gap / B_PRE
                    if abs(sgl) > X:
                        gross.append(-np.sign(sgl) * ((1 + gap / B_TRUE) / (1 + gap / B_PRE) - 1))
            got = G.filter((pl.col("family") == "hl_weekend_reopen") & (pl.col("signal") == "raw") & (pl.col("X") == X) & (pl.col("side") == "fade") & (pl.col("half") == half_name)).row(0, named=True)
            assert got["trades"] == len(gross), ("reopen", X, half_name, got["trades"], len(gross))
            if gross:
                assert abs(got["gross_bps"] - float(np.mean(gross) * 1e4)) < 0.05 and abs(got["cost_bps"] - 3.8) < 1e-6, ("reopen", X, half_name, got)
                assert abs(abs(got["funding_bps"]) - 0) <= 3 * R0 * 1e4 + 1e-9, got
    pr = pl.read_parquet(res / "panel_hl.parquet").filter(pl.col("kind") == "weekend")
    assert np.allclose(pr["fsn_Rm_Rp"].to_numpy(), 3.0) and np.allclose(pr["fs_Rm_Rp"].to_numpy(), 3 * R0), "reopening funding window"
    # funding windows: (T1, O+] holds exactly the hours between 08:00 UTC and the end of the open hour
    p = pl.read_parquet(res / "panel_hl.parquet").filter(pl.col("kind") == "weekend")
    exp_n = np.round(((p["Oplus"] - p["T1"]) / H).to_numpy()); got_n = p["fsn_T1_Oplus"].to_numpy()
    assert np.allclose(exp_n, got_n) and np.allclose(p["fs_T1_Oplus"].to_numpy(), exp_n * R0), "funding window"
    assert set(np.unique(exp_n)) <= {6.0, 7.0}, np.unique(exp_n)
    # leftover basis: the planted venue prints the official open at the open and the official close at the close
    Bz = pl.read_csv(res / "test2_basis.csv")
    for col in ("B_open30", "B_openhour", "B_close1", "B_close0"):
        b = Bz.filter((pl.col("venue") == "hyperliquid") & (pl.col("basis") == col) & (pl.col("subset") == "all"))
        assert b.height == 2 and (b["p99_abs_bps"] < 0.02).all(), (col, b)
    h1 = pl.read_csv(res / "h1.csv")
    r = h1.filter((pl.col("venue") == "hyperliquid") & pl.col("test").str.starts_with("PRIMARY")).row(0, named=True)
    assert r["h1_verdict"].startswith("SUPPORTED") and abs(r["give_back_share"] - 0.2) < 2e-3, r
    print("synthetic known-answer test passed: slope", round(r["slope"], 4), "give-back", round(r["give_back_share"], 4), "weekends", r["weekends"])


if __name__ == "__main__":
    off, pairs = build()
    for script in ("build_panel.py", "spreads.py", "analyze.py"):
        run(script)
    check(off, pairs)
