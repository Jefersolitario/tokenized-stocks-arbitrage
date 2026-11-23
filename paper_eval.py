"""Works out the paper trades of PAPER_RULE.md for one recorded Sunday evening. Reads local files only.

Usage:  python paper_eval.py 2026-11-22          (the New York date of the Sunday; run from this folder)

Inputs: quotes/<date>/ (record_quotes.py), official/daily.csv.gz (Friday's official closes and the hedge ratios, as in
the study), results/hl_fees.csv (the venue's taker fee per contract). Writes results/paper/<date>_trades.csv and
<date>_summary.json, and prints the summary.

Cells, as fixed in PAPER_RULE.md: the primary (signal 19:00, entry 19:00, exit 22:00 New York, 1,000 USD stock leg) and
three that decide nothing (signal at 18:00; primary without entries costing over 20 bps; 10,000 USD). The rule skips a
contract whose Friday official close is missing; a second set of rows, marked reference = "venue", shows those
contracts with the venue's own price at Friday's cash close as a stand-in, and is not part of the rule.
"""
import csv
import datetime as dt
import gzip
import json
import pathlib
import statistics as st
import sys
from zoneinfo import ZoneInfo

import polars as pl

import common

HERE = pathlib.Path(__file__).resolve().parent
NY, UTC = ZoneInfo("America/New_York"), dt.timezone.utc
INDEX = "xyz:XYZ100"
THRESHOLD, MID_MAX_AGE, BOOK_MAX_LATE = 0.01, 30_000, 60_000


def ny_ms(day, hh, mm=0):
    return int(dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=NY).timestamp() * 1000)


def levels(text):
    return [(float(a), float(b)) for a, b in (x.split(":") for x in text.split("|") if x)]


def walk_usd(side, usd):
    """Average price of a market order of `usd` notional against the recorded levels; None if they are too thin."""
    left, qty = float(usd), 0.0
    for px, sz in side:
        take = min(left, px * sz)
        qty += take / px
        left -= take
        if left <= 1e-9:
            return usd / qty
    return None


def walk_qty(side, qty):
    left, paid = float(qty), 0.0
    for px, sz in side:
        take = min(left, sz)
        paid += take * px
        left -= take
        if left <= 1e-12:
            return paid / qty
    return None


class Recording:
    def __init__(self, folder):
        self.books, self.ctx, mids = {}, {}, []
        for p in sorted(folder.glob("books_*.csv.gz")):
            for r in csv.DictReader(gzip.open(p, "rt")):
                self.books.setdefault(r["coin"], []).append(r)
        for v in self.books.values():
            v.sort(key=lambda r: int(r["recv_ms"]))
        for p in sorted(folder.glob("mids_*.csv.gz")):
            rd = csv.reader(gzip.open(p, "rt"))
            col = {c: i for i, c in enumerate(next(rd))}        # each file carries its own list of contracts
            mids += [(int(row[0]), col, row) for row in rd]
        mids.sort(key=lambda x: x[0])
        self.mid_t, self.mid_rows = [x[0] for x in mids], mids
        for p in sorted(folder.glob("ctx_*.csv.gz")):
            for r in csv.DictReader(gzip.open(p, "rt")):
                self.ctx.setdefault(r["coin"], []).append((int(r["recv_ms"]), r["funding"]))
        self.hour = {}
        for p in sorted(folder.glob("candles_1h_*.csv.gz")):
            for r in csv.DictReader(gzip.open(p, "rt")):
                self.hour[(r["coin"], int(r["t"]))] = float(r["c"])

    def mid_at(self, coin, t):
        """Last 5-second mid at or before t, no older than 30 s."""
        lo, hi = 0, len(self.mid_t)
        while lo < hi:
            m = (lo + hi) // 2
            if self.mid_t[m] <= t:
                lo = m + 1
            else:
                hi = m
        if lo == 0 or t - self.mid_t[lo - 1] > MID_MAX_AGE:
            return None
        _, col, row = self.mid_rows[lo - 1]
        v = row[col[coin]] if coin in col else ""
        return float(v) if v else None

    def books_from(self, coin, t):
        return (r for r in self.books.get(coin, []) if int(r["recv_ms"]) >= t)

    def funding_before(self, coin, t):
        last = None
        for ms, f in self.ctx.get(coin, []):
            if ms >= t:
                break
            last = f
        return float(last) if last not in (None, "") else None


def fill(rec, coin, t, buy, usd=None, qty=None, wait=BOOK_MAX_LATE):
    """Market order at the first snapshot at or after t (within `wait` ms; None = any later one). Returns a dict or None."""
    for r in rec.books_from(coin, t):
        if wait is not None and int(r["recv_ms"]) - t > wait:
            return None
        bids, asks = levels(r["bids"]), levels(r["asks"])
        if not bids or not asks:
            continue
        px = walk_usd(asks if buy else bids, usd) if usd is not None else walk_qty(asks if buy else bids, qty)
        if px is None:
            if wait is None:
                continue
            return None
        mid = (bids[0][0] + asks[0][0]) / 2.0
        return {"px": px, "mid": mid, "cost_bps": (px / mid - 1.0) * 1e4 * (1 if buy else -1), "late_ms": int(r["recv_ms"]) - t}
    return None


def trade(rec, coin, d, beta, size, fees, t_in, t_out):
    """d = +1 buys the stock contract and sells the index contract; d = -1 the opposite. Result in bps of the stock leg."""
    s_in = fill(rec, coin, t_in, buy=d > 0, usd=size)
    i_in = fill(rec, INDEX, t_in, buy=d < 0, usd=beta * size) if beta > 0 else None
    if s_in is None or (beta > 0 and i_in is None):
        return None
    qs = size / s_in["px"]
    s_out = fill(rec, coin, t_out, buy=d < 0, qty=qs, wait=None)
    if s_out is None:
        return None
    pnl = d * qs * (s_out["px"] - s_in["px"])
    mid = d * (s_out["mid"] / s_in["mid"] - 1.0) * size
    fee = fees.get(coin, 0.9) / 1e4 * (size + qs * s_out["px"])
    idx_cost = 0.0
    if beta > 0:
        qi = beta * size / i_in["px"]
        i_out = fill(rec, INDEX, t_out, buy=d > 0, qty=qi, wait=None)
        if i_out is None:
            return None
        pnl += -d * qi * (i_out["px"] - i_in["px"])
        mid += -d * beta * (i_out["mid"] / i_in["mid"] - 1.0) * size
        fee += fees.get(INDEX, 0.9) / 1e4 * (beta * size + qi * i_out["px"])
        idx_cost = (i_in["cost_bps"] + i_out["cost_bps"]) * beta
    fund, hours, h = 0.0, 0, (t_in // 3_600_000 + 1) * 3_600_000
    while h <= t_out:                                   # hourly payments while the position is open
        fs, fi = rec.funding_before(coin, h), rec.funding_before(INDEX, h)
        fund += -d * (fs or 0.0) * size + d * (fi or 0.0) * beta * size
        hours += 1
        h += 3_600_000
    k = 1e4 / size
    return {"gross_mid_bps": mid * k, "exec_bps": pnl * k, "spread_bps": (mid - pnl) * k, "fee_bps": fee * k, "funding_bps": fund * k,
            "net_bps": (pnl - fee + fund) * k, "stock_in_cost_bps": s_in["cost_bps"], "stock_out_cost_bps": s_out["cost_bps"],
            "index_cost_bps": idx_cost, "funding_payments": hours, "entry_late_s": s_in["late_ms"] / 1000, "exit_late_s": s_out["late_ms"] / 1000}


def summarise(rows):
    v = [r["net_bps"] for r in rows]
    if not v:
        return {"trades": 0}
    mean = lambda key: round(st.mean(r[key] for r in rows), 1)  # noqa: E731
    fills = [r["stock_in_cost_bps"] for r in rows] + [r["stock_out_cost_bps"] for r in rows]
    return {"trades": len(v), "net_mean_bps": round(st.mean(v), 1), "net_median_bps": round(st.median(v), 1),
            "share_positive": round(sum(x > 0 for x in v) / len(v), 2), "worst_bps": round(min(v), 1), "best_bps": round(max(v), 1),
            "gross_mid_bps": mean("gross_mid_bps"), "spread_bps": mean("spread_bps"), "fee_bps": mean("fee_bps"), "funding_bps": mean("funding_bps"),
            "stock_fill_cost_mean_bps": round(st.mean(fills), 1), "stock_fill_cost_median_bps": round(st.median(fills), 1),
            "long_stock_trades": sum(r["d"] > 0 for r in rows), "mean_beta": round(st.mean(r["beta"] for r in rows), 2),
            "usd_at_this_size": round(sum(v) * rows[0]["size"] / 1e4, 0)}


def main(day_text):
    sunday = dt.date.fromisoformat(day_text)
    rec = Recording(HERE / "quotes" / day_text)
    off = common.load_official()
    friday = off.filter((pl.col("ticker") == "QQQ") & (pl.col("date") < sunday))["date"].max()
    assert (sunday - friday).days <= 4, f"official prices end on {friday}: refresh official/daily.csv.gz first"
    close = dict(off.filter(pl.col("date") == friday).select("ticker", "close").iter_rows())
    beta = {r[0]: r[2] for r in common.rolling_beta(off).filter(pl.col("d0") == friday).iter_rows()}
    fees = {r["contract"]: float(r["taker_bps"]) for r in csv.DictReader(open(HERE / "results" / "hl_fees.csv"))}
    t18, t19, t22 = ny_ms(sunday, 18), ny_ms(sunday, 19), ny_ms(sunday, 22)
    fri_close_ms = common.close_ms(friday)
    idx_ref = rec.hour[(INDEX, fri_close_ms - 3_600_000)]
    cells = {"primary": (t19, 1_000), "signal_18": (t18, 1_000), "usd_10000": (t19, 10_000)}
    out, skipped = [], {"no_friday_close": [], "no_mid_at_signal": [], "no_book": []}
    for coin in sorted(c for c in rec.books if c not in (INDEX, "xyz:SP500")):
        tick = coin.split(":")[1]
        refs = [("official", close[tick])] if tick in close else []
        if not refs:
            skipped["no_friday_close"].append(tick)
            if (coin, fri_close_ms - 3_600_000) in rec.hour:
                refs = [("venue", rec.hour[(coin, fri_close_ms - 3_600_000)])]
        for ref_name, ref_px in refs:
            b = beta.get(tick, 1.0)
            for cell, (t_sig, size) in cells.items():
                ms, mi = rec.mid_at(coin, t_sig), rec.mid_at(INDEX, t_sig)
                if ms is None or mi is None:
                    skipped["no_mid_at_signal"].append(f"{tick}/{cell}")
                    continue
                signal = ms / ref_px - 1.0 - b * (mi / idx_ref - 1.0)
                if abs(signal) < THRESHOLD:
                    continue
                d = -1 if signal > 0 else 1
                res = trade(rec, coin, d, b, size, fees, t19, t22)
                if res is None:
                    skipped["no_book"].append(f"{tick}/{cell}")
                    continue
                out.append(dict(cell=cell, reference=ref_name, coin=coin, d=d, beta=round(b, 3), size=size, signal_pct=round(signal * 100, 3),
                                spread_pct=round((ms / ref_px - 1) * 100, 3), index_move_pct=round((mi / idx_ref - 1) * 100, 3),
                                **{k: round(v, 2) for k, v in res.items()}))
    summary = {"sunday": day_text, "friday": str(friday), "index_reference": idx_ref, "index_move_19_pct": round((rec.mid_at(INDEX, t19) / idx_ref - 1) * 100, 3),
               "contracts_recorded": len(rec.books) - 2, "skipped": skipped, "cells": {}}
    for ref in ("official", "venue"):
        rows = [r for r in out if r["reference"] == ref]
        prim = [r for r in rows if r["cell"] == "primary"]
        summary["cells"][ref] = {"primary": summarise(prim), "signal_18": summarise([r for r in rows if r["cell"] == "signal_18"]),
                                 "entry_cost_20_or_less": summarise([r for r in prim if r["stock_in_cost_bps"] <= 20.0]),
                                 "usd_10000": summarise([r for r in rows if r["cell"] == "usd_10000"])}
    res_dir = HERE / "results" / "paper"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / f"{day_text}_trades.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else ["cell"])
        w.writeheader()
        w.writerows(out)
    (res_dir / f"{day_text}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=1))
    for ref, cs in summary["cells"].items():
        for name, s in cs.items():
            print(f"{ref:8s} {name:22s}", s)
    return out, summary


if __name__ == "__main__":
    main(sys.argv[1])
