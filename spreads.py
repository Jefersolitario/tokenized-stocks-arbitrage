"""Half-spreads and depth from the order-book snapshots taken by the workflow, and the fee per contract.

    python spreads.py

Writes results/spreads.csv: per contract, the quoted half-spread and the cost against the mid of a market order of
1,000 / 10,000 / 50,000 USD (average of buy and sell), median over the snapshots; visible depth; the half-spread the
tests use (1,000 USD order, floor 1 bp) with the 10,000 USD figure as a reference; and the taker and
maker fee implied by the venue metadata and the venue's documented fee formula (Hyperliquid HIP-3: tier-0 base
4.5 / 1.5 bps x 2 for deployer fee scale 1.0, x 0.1 when growth mode is enabled; Binance spot: 10 bps assumed, not verified).
"""
import datetime as dt
import json

import numpy as np
import polars as pl

from common import DATA, RES, read_json_gz

SIZES = (1_000, 10_000, 50_000)


def walk(levels, usd):
    """Average fill price of a market order of `usd` notional walking `levels` [(px, sz)]; None if the book is too thin."""
    need, cost, qty = usd, 0.0, 0.0
    for px, sz in levels:
        take = min(sz, need / px)
        cost += take * px; qty += take; need -= take * px
        if need <= 1e-9:
            return cost / qty
    return None


def book_stats(bids, asks):
    bids = [(float(b[0]), float(b[1])) for b in bids]; asks = [(float(a[0]), float(a[1])) for a in asks]
    if not bids or not asks:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    out = {"mid": mid, "quoted_half_bps": (asks[0][0] - bids[0][0]) / mid / 2 * 1e4,
           "bid_depth_usd": sum(p * s for p, s in bids), "ask_depth_usd": sum(p * s for p, s in asks),
           "far_bid_bps": (mid - bids[-1][0]) / mid * 1e4, "far_ask_bps": (asks[-1][0] - mid) / mid * 1e4}
    for usd in SIZES:
        b, s = walk(asks, usd), walk(bids, usd)
        out[f"cost_{usd}_bps"] = np.nan if b is None or s is None else ((b / mid - 1) + (1 - s / mid)) / 2 * 1e4
    return out


def main():
    RES.mkdir(exist_ok=True)
    rows = []
    snaps = read_json_gz(DATA / "hl_books.json.gz")
    f2 = DATA / "hl_books_after_funding.json.gz"
    if f2.exists():
        snaps += read_json_gz(f2)
    for snap in snaps:
        when = dt.datetime.fromtimestamp(snap["taken_utc_ms_start"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        for coin, bk in snap["books"].items():
            st = book_stats(bk["bids"], bk["asks"])
            if st:
                rows.append(dict(venue="hyperliquid", contract=coin, snapshot_utc=when, **st))
    bsnap = read_json_gz(DATA / "bn_books.json.gz")
    when = dt.datetime.fromtimestamp(bsnap["taken_utc_ms_start"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    for sym, bk in bsnap["books"].items():
        st = book_stats(bk["bids"], bk["asks"])
        if st:
            rows.append(dict(venue="binance", contract=sym, snapshot_utc=when, **st))
    raw = pl.DataFrame(rows)
    raw.write_csv(RES / "spreads_snapshots.csv")
    agg = raw.group_by("venue", "contract").agg(snapshots=pl.len(), first_snapshot_utc=pl.col("snapshot_utc").min(), last_snapshot_utc=pl.col("snapshot_utc").max(),
                                                **{c: pl.col(c).median() for c in raw.columns if c.endswith("_bps") or c.endswith("_usd")})
    meta = json.loads((DATA / "hl_meta_xyz.json").read_text())["meta"]["universe"]
    def scale(u):
        # Hyperliquid fee documentation: scaleIfHip3 = deployerFeeScale < 1 ? deployerFeeScale + 1 : deployerFeeScale * 2; growthModeScale = growthMode ? 0.1 : 1
        d = float(u.get("deployerFeeScale", "1.0"))
        return (d + 1 if d < 1 else d * 2) * (0.1 if u.get("growthMode") == "enabled" else 1.0)
    # tier-0 base perp rates: taker 0.045% = 4.5 bps, maker 0.015% = 1.5 bps
    fee = {u["name"]: (4.5 * scale(u), 1.5 * scale(u), u.get("growthMode") == "enabled", bool(u.get("isDelisted")), u.get("maxLeverage")) for u in meta}
    agg = agg.with_columns(
        taker_bps=pl.when(pl.col("venue") == "binance").then(10.0).otherwise(pl.col("contract").map_elements(lambda c: fee.get(c, (9.0,))[0], return_dtype=pl.Float64)),
        maker_bps=pl.when(pl.col("venue") == "binance").then(10.0).otherwise(pl.col("contract").map_elements(lambda c: fee.get(c, (9.0, 3.0))[1], return_dtype=pl.Float64)))
    # pre-registered half-spread (addendum 3): cost of a 1,000 USD market order against the mid, floor 1 bp; 5 bps if no usable snapshot.
    # The same for a 10,000 USD order is kept as a reference line.
    hs = lambda col: pl.when(pl.col(col).is_finite()).then(pl.max_horizontal(pl.col(col), pl.lit(1.0))).otherwise(5.0)
    agg = agg.with_columns(half_spread_bps=hs("cost_1000_bps"), half_spread_10k_bps=hs("cost_10000_bps"),
                           thin_book=~pl.col("cost_1000_bps").is_finite(), thin_book_10k=~pl.col("cost_10000_bps").is_finite()).sort("venue", "contract")
    agg.write_csv(RES / "spreads.csv")
    feetab = pl.DataFrame([dict(contract=k, taker_bps=v[0], maker_bps=v[1], growth_mode=v[2], delisted=v[3], max_leverage=v[4]) for k, v in fee.items()])
    feetab.write_csv(RES / "hl_fees.csv")
    with pl.Config(tbl_rows=30, tbl_width_chars=200, tbl_cols=20):
        print(agg.group_by("venue").agg(contracts=pl.len(), thin_1k=pl.col("thin_book").sum(), thin_10k=pl.col("thin_book_10k").sum(), quoted_half_median=pl.col("quoted_half_bps").median(),
                                        cost1k_median=pl.col("cost_1000_bps").median(), cost1k_p90=pl.col("cost_1000_bps").quantile(0.9),
                                        cost10k_median=pl.col("cost_10000_bps").median(), cost10k_p90=pl.col("cost_10000_bps").quantile(0.9),
                                        half_spread_used_median=pl.col("half_spread_bps").median(), taker_median=pl.col("taker_bps").median()))
        print(feetab.group_by("growth_mode", "delisted").agg(n=pl.len(), taker=pl.col("taker_bps").first()))


if __name__ == "__main__":
    main()
