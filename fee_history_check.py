"""POST-HOC check of the fee assumption, written after the results were seen.

    python fee_history_check.py        (after analyze.py)

The fee per fill used in the study comes from the contract metadata as downloaded on 22 Nov 2026 (growth mode on or
off, deployer fee scale). The metadata also carries `lastFeeScaleChangeTime` per contract. The download holds no
history of the setting, so for a trade entered BEFORE that time the fee actually charged is not known.
This script counts those trades and re-prices them under a conservative assumption: before its last change a contract
was charged the full builder-market fee (9 bps per taker fill at the base tier) instead of the growth-mode fee.
The index leg (xyz:XYZ100) is treated the same way.

Writes results/posthoc_fee_history_contracts.csv and results/posthoc_fee_history.csv.
"""
import json
from datetime import datetime, timezone

import numpy as np
import polars as pl

import analyze as an
import stats as st
from common import DATA, RES

BPS = 1e4
FULL_FEE = 9.0          # bps per taker fill of a builder market outside growth mode (base tier, deployer fee scale 1.0)


def change_dates():
    """contract -> date of the last fee-scale change (None when the field is missing or the epoch)."""
    u = json.loads((DATA / "hl_meta_xyz.json").read_text())["meta"]["universe"]
    out = {}
    for x in u:
        s = x.get("lastFeeScaleChangeTime") or ""
        d = None
        if s and not s.startswith("1970"):
            d = datetime.fromisoformat(s[:19]).replace(tzinfo=timezone.utc).date()
        out[x["name"]] = d
    return out


def main():
    an.COSTS = an.Costs()
    chg = change_dates()
    taker = an.COSTS.taker
    p = an.prep(pl.read_parquet(RES / "panel_hl.parquet"), "hyperliquid")
    k = p.filter(pl.col("kind") == "weekend")
    hv = an.halves(k)
    sample_start = k["d0"].min()

    # per contract: listing proxy (first 2h candle), last change, whether the change falls inside the 1h sample
    first = pl.read_parquet(DATA / "hl_candles_2h.parquet").group_by("coin").agg(first_ms=pl.col("t").min())
    first = {r["coin"]: datetime.fromtimestamp(r["first_ms"] / 1000, tz=timezone.utc).date() for r in first.iter_rows(named=True)}
    first_wk = {r["coin"]: r["d0"] for r in k.group_by("coin").agg(pl.col("d0").min()).iter_rows(named=True)}   # first sampled weekend (its Friday)
    rows = []
    for coin in sorted(set(k["coin"].to_list()) | {"xyz:XYZ100", "xyz:SP500"}):
        d, f, w = chg.get(coin), first.get(coin), first_wk.get(coin, sample_start)
        rows.append(dict(contract=coin, taker_bps_now=taker.get(coin), first_2h_candle=str(f) if f else "", last_fee_scale_change=str(d) if d else "",
                         days_from_first_candle_to_change=(d - f).days if d and f else None,
                         change_inside_1h_sample=bool(d and d > sample_start),
                         first_sampled_weekend=str(w), days_from_change_to_first_sampled_weekend=(w - d).days if d else None))
    con = pl.DataFrame(rows)
    con.write_csv(RES / "posthoc_fee_history_contracts.csv")
    n_in = int(con["change_inside_1h_sample"].sum())
    late = con.filter(pl.col("days_from_first_candle_to_change") > 2)
    print(f"contracts: {con.height}; last fee-scale change inside the 1h sample (after {sample_start}): {n_in}; "
          f"change more than 2 days after the first 2h candle: {late.height}; within 2 days of it or before it: "
          f"{con.filter(pl.col('days_from_first_candle_to_change') <= 2).height}; no change time: {con.filter(pl.col('last_fee_scale_change') == '').height}")
    print(f"fewest days from a contract's last fee-scale change to its first sampled weekend: {con['days_from_change_to_first_sampled_weekend'].min()} "
          f"(positive = every sampled weekend of every contract comes after its last change)")
    print("changes more than 2 days after the first candle:", late.select("contract", "first_2h_candle", "last_fee_scale_change", "taker_bps_now").rows())

    idx_chg = chg.get("xyz:XYZ100")
    out = []
    for entry, signal, exit_ in [("Rm", "neutral", "Rp"), ("Rm", "raw", "Rp"), ("T1", "neutral", "Oplus"), ("T1", "raw", "Oplus")]:
        c = an.candidates(k, "hyperliquid", entry, exit_, signal)
        d0 = c["d0"].astype("datetime64[D]")
        cd = np.array([np.datetime64(chg[x]) if chg.get(x) else np.datetime64("1970-01-01") for x in c["coin"]], dtype="datetime64[D]")
        unknown = cd > d0                                  # the setting changed after the Friday of this weekend
        tk = np.array([taker.get(x, FULL_FEE) for x in c["coin"]], float)
        extra = np.where(unknown, 2 * np.maximum(FULL_FEE - tk, 0.0), 0.0) / BPS
        if signal == "neutral" and idx_chg is not None:    # index leg, same treatment
            iu = np.datetime64(idx_chg) > d0
            extra = extra + np.where(iu, 2 * c["beta"] * max(FULL_FEE - taker.get("xyz:XYZ100", FULL_FEE), 0.0), 0.0) / BPS
        for X in an.X_GRID:
            for side in ("fade", "follow"):
                for half in ("sel", "eval", "all"):
                    m = np.isin(c["cl"], hv[half])
                    kk = an.cell(c, X, side, m)
                    take = kk["take"]
                    base = st.cell_stats(kk["net"] * BPS, c["cl"][take], hv[half])
                    cons = st.cell_stats((kk["net"] - extra[take]) * BPS, c["cl"][take], hv[half])
                    known = st.cell_stats(kk["net"][~unknown[take]] * BPS, c["cl"][take][~unknown[take]], hv[half])
                    out.append(dict(entry=entry, signal=signal, exit=exit_, X=X, side=side, half=half, trades=base["trades"],
                                    trades_before_last_change=int(unknown[take].sum()), share_before_last_change=float(unknown[take].mean()) if take.any() else np.nan,
                                    net_bps=base["mean"], net_lo=base["lo"], net_hi=base["hi"],
                                    net_bps_full_fee_before_change=cons["mean"], cons_lo=cons["lo"], cons_hi=cons["hi"], cons_p_le_0=cons["p_le_0"], cons_wk_t=cons["wk_t"],
                                    trades_setting_known=known["trades"], weekends_setting_known=known["clusters_with_trades"],
                                    net_bps_setting_known=known["mean"], known_lo=known["lo"], known_hi=known["hi"], known_wk_t=known["wk_t"]))
    res = pl.DataFrame(out, infer_schema_length=None)
    res.write_csv(RES / "posthoc_fee_history.csv")
    with pl.Config(tbl_rows=80, tbl_cols=30, tbl_width_chars=250, float_precision=2):
        print(res.filter(pl.col("side") == "fade").select("entry", "signal", "X", "half", "trades", "trades_before_last_change", "net_bps", "net_lo", "net_hi",
                                                          "net_bps_full_fee_before_change", "cons_lo", "cons_hi", "cons_wk_t",
                                                          "trades_setting_known", "weekends_setting_known", "net_bps_setting_known", "known_lo", "known_hi", "known_wk_t"))


if __name__ == "__main__":
    main()
