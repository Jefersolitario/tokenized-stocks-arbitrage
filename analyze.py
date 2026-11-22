"""All pre-registered tests of the weekend-basis study.

    python analyze.py

Reads results/panel_*.parquet (build_panel.py), results/spreads.csv and results/hl_fees.csv (spreads.py).
Writes results/test1_slopes.csv, test1_per_contract.csv, test2_basis.csv, test2_regressions.csv, test3_grid.csv,
test3_selected.csv, test3_placebo.csv, test3_costs.csv, premium_by_regime.csv, h1.csv and summary.json.
One cluster = one weekend (or one night). Seeds are fixed in common.SEED.
"""
import datetime as dt
import json

import numpy as np
import polars as pl

import stats as st
from common import CORE, DATA, H, NY, RES, RTH_ONLY, SEED, load_official

X_GRID = (0.01, 0.02, 0.03)
ENTRIES, EXITS = ["T0", "T1", "T0p1", "T1p1"], ["Oplus", "C1"]
WINDOWS = [(e, x) for e in ENTRIES for x in EXITS] + [("Rm", "Rp"), ("Rmp1", "Rp")]
GRID = lambda entries: [(e, sg, x) for e in entries for sg in ("neutral", "raw") for x in EXITS]
REOPEN = [("Rm", "neutral", "Rp"), ("Rm", "raw", "Rp")]
BPS = 1e4


# ------------------------------------------------------------------ preparation
def prep(p, venue):
    c = pl.col
    p = p.with_columns(gap=c("open1") / c("close0") - 1, day1=c("close1") / c("open1") - 1, cl=c("d0").cast(pl.Int32),
                       core=c("ticker").is_in(CORE), rth_only=c("ticker").is_in(RTH_ONLY))
    for T in ("T0", "T1", "Rm", "Rp"):
        p = p.with_columns(**{f"v_{T}": c(f"P_{T}") / c("P_F") - 1, f"g_{T}": c(f"P_{T}") / c("close0") - 1,
                              f"m_{T}": c(f"I_{T}") / c("I_F") - 1, f"mS_{T}": c(f"J_{T}") / c("J_F") - 1})
        p = p.with_columns(**{f"e_{T}": c(f"g_{T}") - c("beta") * c(f"m_{T}"), f"eS_{T}": c(f"g_{T}") - c("beta") * c(f"mS_{T}")})
    new = {}
    for e, x in WINDOWS:
        new[f"rs_{e}_{x}"] = c(f"P_{x}") / c(f"P_{e}") - 1
        new[f"ri_{e}_{x}"] = c(f"I_{x}") / c(f"I_{e}") - 1
        new[f"rj_{e}_{x}"] = c(f"J_{x}") / c(f"J_{e}") - 1
    p = p.with_columns(**new)
    adj = (lambda t: c(f"U_{t}")) if venue == "binance" else (lambda t: pl.lit(1.0))
    p = p.with_columns(B_open30=c("P_open30") / c("open1") - 1, B_pre30=c("P_preopen30") / c("open1") - 1, B_openhour=c("P_Oplus") / c("open1") - 1,
                       B_close1=c("P_C1") / c("close1") - 1, B_close0=c("P_F") / c("close0") - 1,
                       B_close1_usd=c("P_C1") / adj("C1") / c("close1") - 1, B_close0_usd=c("P_F") / adj("F") / c("close0") - 1)
    p = p.with_columns(venue_post=c("P_Oplus") / c("P_T1") - 1, venue_post_T0=c("P_Oplus") / c("P_T0") - 1, venue_reopen=c("P_Rp") / c("P_Rm") - 1,
                       venue_T0_T1=c("P_T1") / c("P_T0") - 1)
    return p.filter(c("level_ok").fill_null(False) & ~c("split_near") & c("close0").is_not_null() & c("open1").is_not_null())


def arr(df, *cols):
    return [df[x].to_numpy().astype(float) if df[x].dtype != pl.String else df[x].to_numpy() for x in cols]


# ------------------------------------------------------------------ test 1
def slope_row(df, label, xcol, ycol="gap", fe=False, placebo=True, **tags):
    d = df.filter(pl.col(xcol).is_finite() & pl.col(ycol).is_finite())
    out = dict(label=label, x=xcol, y=ycol, fixed_effects=fe, **tags)
    if d.height < 5:
        return dict(out, n=d.height)
    x, y, cl = arr(d, xcol, ycol, "cl"); unit = d["coin"].to_numpy()
    r = st.ols_cluster_boot(x, y, cl, fe=fe)
    xx, yy = (st.demean_within(x, cl), st.demean_within(y, cl)) if fe else (x, y)
    big = np.abs(yy) >= 0.0025
    out.update(r)
    out.update(intercept_bps=r["intercept"] * BPS, give_back=1 - r["slope"], give_back_lo=1 - r["hi"], give_back_hi=1 - r["lo"],
               rmse_y_minus_x_bps=float(np.sqrt(np.mean((yy - xx) ** 2)) * BPS), rmse_y_bps=float(np.sqrt(np.mean(yy ** 2)) * BPS),
               medabs_y_minus_x_bps=float(np.median(np.abs(yy - xx)) * BPS), medabs_y_bps=float(np.median(np.abs(yy)) * BPS),
               share_x_closer=float(np.mean(np.abs(yy - xx) < np.abs(yy))), sign_agree_y_ge_25bps=float(np.mean(np.sign(xx[big]) == np.sign(yy[big]))) if big.any() else np.nan,
               sd_x_bps=float(np.std(xx) * BPS), sd_y_bps=float(np.std(yy) * BPS))
    if placebo:
        pb = st.slope_placebo(cl, unit, xx, yy)
        out.update(placebo_mean=pb["perm_mean"], placebo_p2_5=pb["perm_p2_5"], placebo_p97_5=pb["perm_p97_5"], placebo_share_ge_real=pb["share_perm_ge_real"], placebo_lag1=pb["lag1"])
    return out


def per_contract(df, xcol, min_obs=10):
    rows = []
    for (coin,), g in df.filter(pl.col(xcol).is_finite() & pl.col("gap").is_finite()).group_by("coin"):
        if g.height >= min_obs:
            a, b, r2 = st.ols(g[xcol].to_numpy(), g["gap"].to_numpy())
            rows.append(dict(coin=coin, n=g.height, slope=b, r2=r2))
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"coin": pl.String, "n": pl.Int64, "slope": pl.Float64, "r2": pl.Float64})


# ------------------------------------------------------------------ test 3
class Costs:
    def __init__(self):
        sp = pl.read_csv(RES / "spreads.csv"); fees = pl.read_csv(RES / "hl_fees.csv")
        self.hs = {r["contract"]: r["half_spread_bps"] for r in sp.iter_rows(named=True)}            # 1,000 USD order (primary)
        self.hs10k = {r["contract"]: r["half_spread_10k_bps"] for r in sp.iter_rows(named=True)}     # 10,000 USD order (reference line)
        self.taker = {r["contract"]: r["taker_bps"] for r in fees.iter_rows(named=True)}
        self.maker = {r["contract"]: r["maker_bps"] for r in fees.iter_rows(named=True)}

    def per_side(self, coins, case, venue):
        """bps per side for each coin under a cost case (fee + half-spread; the maker case pays the maker fee and no spread)."""
        n = len(coins)
        hs = np.array([self.hs.get(k, 5.0) for k in coins], float); hs10k = np.array([self.hs10k.get(k, 5.0) for k in coins], float)
        if venue == "binance":                       # spot: 10 bps taker and maker assumed (not verified for these pairs)
            tk = np.full(n, 10.0); mk = np.full(n, 10.0); f45 = tk; f9 = tk
        else:
            tk = np.array([self.taker.get(k, 9.0) for k in coins], float); mk = np.array([self.maker.get(k, 3.0) for k in coins], float)
            f45 = np.full(n, 4.5); f9 = np.full(n, 9.0)
        return {"primary": tk + hs, "fee_only": tk + 0.0, "ref_10k": tk + hs10k, "fee4.5": f45 + hs, "fee9": f9 + hs, "hs2": tk + 2.0, "hs10": tk + 10.0, "maker": mk + 0.0,
                "fee4.5_hs10": f45 + 10.0}[case]


COSTS = None
CASES = ["primary", "fee_only", "ref_10k", "fee4.5", "fee9", "hs2", "hs10", "maker", "fee4.5_hs10"]   # fee_only: taker fees with no spread, to split the cost


def candidates(df, venue, entry, exit_, signal, hedge="I", sig_time=None):
    """Every observation with a valid signal and valid entry/exit prices, with the outcome per unit of long exposure."""
    sig_time = sig_time or entry
    neutral = signal == "neutral"
    scol = (f"e_{sig_time}" if hedge == "I" else f"eS_{sig_time}") if neutral else f"g_{sig_time}"
    hcoin, hp, hf = ("xyz:XYZ100", "ri", "fi") if hedge == "I" else ("xyz:SP500", "rj", "fj")
    need = [scol, f"rs_{entry}_{exit_}"] + ([f"{hp}_{entry}_{exit_}", "beta"] if neutral else [])
    d = df.filter(pl.all_horizontal([pl.col(k).is_finite() for k in need]))
    s, rs = arr(d, scol, f"rs_{entry}_{exit_}")
    coins = d["coin"].to_list(); n = d.height
    fs = d[f"fs_{entry}_{exit_}"].to_numpy().astype(float) if f"fs_{entry}_{exit_}" in d.columns else np.zeros(n)
    fund_missing = ~np.isfinite(fs); fs = np.where(fund_missing, 0.0, fs)
    R, fund_long = rs.copy(), -fs
    cost = {k: 2 * COSTS.per_side(coins, k, venue) / BPS for k in CASES}
    beta = np.zeros(n)                                    # hedge ratio per trade (0 = no hedge leg)
    if neutral:
        rh, beta = arr(d, f"{hp}_{entry}_{exit_}", "beta")
        fh = d[f"{hf}_{entry}_{exit_}"].to_numpy().astype(float); fh = np.where(np.isfinite(fh), fh, 0.0)
        R = rs - beta * rh; fund_long = -fs + beta * fh
        for k in CASES:
            cost[k] = cost[k] + 2 * beta * COSTS.per_side([hcoin] * n, k, "hyperliquid") / BPS
    ntl = d[f"P_{entry}_ntl"].to_numpy().astype(float) if f"P_{entry}_ntl" in d.columns else np.full(n, np.nan)
    ntr = d[f"P_{entry}_n"].to_numpy().astype(float) if f"P_{entry}_n" in d.columns else np.full(n, np.nan)
    return dict(s=s, R=R, fund_long=fund_long, cost=cost, cl=d["cl"].to_numpy(), coin=np.array(coins), ntl=ntl, ntrades=ntr, fund_missing=fund_missing, beta=beta,
                d0=d["d0"].to_numpy(), rth_only=d["rth_only"].to_numpy(), core=d["core"].to_numpy())


def cell(c, X, side, mask=None, case="primary", clusters=None):
    take = np.abs(c["s"]) > X
    if mask is not None:
        take &= mask
    d = -np.sign(c["s"][take]) if side == "fade" else np.sign(c["s"][take])
    gross = d * c["R"][take]; fund = d * c["fund_long"][take]; cost = c["cost"][case][take]
    return dict(take=take, d=d, gross=gross, fund=fund, cost=cost, net=gross + fund - cost)


def grid_rows(c, clusters_by_half, tags, long_only=False):
    rows = []
    for X in X_GRID:
        for side in ("fade", "follow"):
            for half, cls in clusters_by_half.items():
                m = np.isin(c["cl"], cls)
                if long_only:
                    m &= (-np.sign(c["s"]) > 0) if side == "fade" else (np.sign(c["s"]) > 0)
                k = cell(c, X, side, m)
                s_net = st.cell_stats(k["net"] * BPS, c["cl"][k["take"]], cls)
                s_gross = st.cell_stats(k["gross"] * BPS, c["cl"][k["take"]], cls, B=2000)
                ntl = c["ntl"][k["take"]]
                rows.append(dict(**tags, X=X, side=side, half=half, long_only=long_only, trades=s_net["trades"], clusters_with_trades=s_net["clusters_with_trades"],
                                 clusters=s_net["clusters"], gross_bps=s_gross["mean"], gross_lo=s_gross["lo"], gross_hi=s_gross["hi"], t_cluster_gross=s_gross["t_cluster"],
                                 funding_bps=float(np.mean(k["fund"]) * BPS) if len(k["fund"]) else np.nan, cost_bps=float(np.mean(k["cost"]) * BPS) if len(k["cost"]) else np.nan,
                                 fee_bps=float(np.mean(c["cost"]["fee_only"][k["take"]]) * BPS) if len(k["cost"]) else np.nan,
                                 net_bps=s_net["mean"], net_lo=s_net["lo"], net_hi=s_net["hi"], p_net_le_0=s_net["p_le_0"], median_net_bps=s_net["median"], hit=s_net["hit"],
                                 wk_net_bps=s_net["wk_mean"], wk_lo=s_net["wk_lo"], wk_hi=s_net["wk_hi"], wk_t=s_net["wk_t"], wk_p_le_0=s_net["wk_p_le_0"],
                                 entry_hour_usd_median=float(np.nanmedian(ntl)) if len(ntl) and np.isfinite(ntl).any() else np.nan,
                                 entry_hour_usd_p10=float(np.nanpercentile(ntl, 10)) if len(ntl) and np.isfinite(ntl).any() else np.nan,
                                 entry_hour_usd_min=float(np.nanmin(ntl)) if len(ntl) and np.isfinite(ntl).any() else np.nan,
                                 share_entry_hour_ge_100k=float(np.mean(ntl >= 1e5)) if len(ntl) else np.nan, share_entry_hour_ge_10k=float(np.mean(ntl >= 1e4)) if len(ntl) else np.nan,
                                 funding_missing=int(c["fund_missing"][k["take"]].sum())))
    return rows


def halves(df):
    cls = np.sort(df["cl"].unique().to_numpy())
    k = int(np.ceil(len(cls) / 2))
    return {"sel": cls[:k], "eval": cls[k:], "all": cls}


def run_family(df, venue, kind, configs, family, long_only=False):
    """Full grid, selection on the earlier half, single evaluation on the later half, cost sensitivities."""
    hv = halves(df)
    grid, selected, sens, cands = [], [], [], {}
    for entry, signal, exit_ in configs:
        tags = dict(family=family, venue=venue, kind=kind, entry=entry, signal=signal, exit=exit_)
        c = candidates(df, venue, entry, exit_, signal); cands[(entry, signal, exit_)] = c
        rows = grid_rows(c, hv, tags, long_only=long_only); grid += rows
        sel = [r for r in rows if r["half"] == "sel" and r["trades"] >= 8]
        best = max(sel, key=lambda r: r["net_bps"]) if sel else None
        X, side = (best["X"], best["side"]) if best else (0.01, "fade")
        pick = lambda half, X=X, side=side: next(r for r in rows if r["half"] == half and r["X"] == X and r["side"] == side)
        ev, se = pick("eval"), pick("sel")
        selected.append(dict(**tags, long_only=long_only, X=X, side=side, fallback=best is None,
                             sel_trades=se["trades"], sel_net_bps=se["net_bps"], sel_lo=se["net_lo"], sel_hi=se["net_hi"],
                             eval_trades=ev["trades"], eval_clusters_with_trades=ev["clusters_with_trades"], eval_clusters=ev["clusters"],
                             eval_gross_bps=ev["gross_bps"], eval_funding_bps=ev["funding_bps"], eval_cost_bps=ev["cost_bps"],
                             eval_net_bps=ev["net_bps"], eval_lo=ev["net_lo"], eval_hi=ev["net_hi"], eval_p_net_le_0=ev["p_net_le_0"], eval_hit=ev["hit"],
                             eval_wk_net_bps=ev["wk_net_bps"], eval_wk_lo=ev["wk_lo"], eval_wk_hi=ev["wk_hi"], eval_wk_t=ev["wk_t"], eval_wk_p_le_0=ev["wk_p_le_0"],
                             eval_entry_hour_usd_median=ev["entry_hour_usd_median"], eval_entry_hour_usd_p10=ev["entry_hour_usd_p10"], eval_entry_hour_usd_min=ev["entry_hour_usd_min"],
                             eval_share_entry_hour_ge_100k=ev["share_entry_hour_ge_100k"], eval_share_entry_hour_ge_10k=ev["share_entry_hour_ge_10k"]))
        m = np.isin(c["cl"], hv["eval"])
        if long_only:
            m &= (-np.sign(c["s"]) > 0) if side == "fade" else (np.sign(c["s"]) > 0)
        for case in CASES:
            k = cell(c, X, side, m, case)
            s_ = st.cell_stats(k["net"] * BPS, c["cl"][k["take"]], hv["eval"], B=2000)
            sens.append(dict(**tags, long_only=long_only, X=X, side=side, case=case, trades=s_["trades"], cost_bps=float(np.mean(k["cost"]) * BPS) if len(k["cost"]) else np.nan,
                             net_bps=s_["mean"], net_lo=s_["lo"], net_hi=s_["hi"]))
    sel_df = pl.DataFrame(selected)
    sel_df = sel_df.with_columns(eval_p_bh=pl.Series(st.bh(sel_df["eval_p_net_le_0"].to_numpy())))
    return pl.DataFrame(grid), sel_df, pl.DataFrame(sens), cands, hv


def placebo_family(cands, hv, family, n_perm=st.N_PERM):
    """Signals reassigned to other closures (same permutation for every cell). Statistic: cluster t of the gross FADE PnL.
    Returns per-cell rows and the best-of-grid comparison, on the whole sample and on the later half."""
    out = []
    for scope in ("all", "eval"):
        cls = hv[scope]
        mats = {}
        for key, c in cands.items():
            m = np.isin(c["cl"], cls)
            coins = np.unique(c["coin"][m])
            if len(coins) == 0:
                continue
            ic = np.searchsorted(cls, c["cl"][m]); iu = np.searchsorted(coins, c["coin"][m])
            S = np.full((len(cls), len(coins)), np.nan); R = S.copy()
            S[ic, iu] = c["s"][m]; R[ic, iu] = c["R"][m]
            mats[key] = (S, R)

        def tstats(perm=None):
            res = {}
            for key, (S, R) in mats.items():
                S2 = S if perm is None else S[perm]
                for X in X_GRID:
                    with np.errstate(invalid="ignore"):
                        P = np.where((np.abs(S2) > X) & np.isfinite(R), -np.sign(S2) * R, np.nan)
                    res[key + (X,)] = st.cluster_t(P)
            return res
        real = tstats()
        rng = np.random.default_rng(SEED)
        perms = [tstats(rng.permutation(len(cls))) for _ in range(n_perm)]
        keys = list(real)
        M = np.array([[p[k] for k in keys] for p in perms])
        maxabs = np.nanmax(np.abs(M), axis=1)
        rv = np.array([real[k] for k in keys])
        best = int(np.nanargmax(np.abs(rv))) if np.isfinite(rv).any() else 0
        # shifted weekends: every circular shift of the signal rows by 1 .. N-1 closures, same statistic
        shifts = [tstats(np.roll(np.arange(len(cls)), s)) for s in range(1, len(cls))]
        Ms = np.array([[p[k] for k in keys] for p in shifts]) if shifts else np.zeros((0, len(keys)))
        shift_max = np.nanmax(np.abs(Ms), axis=1) if len(Ms) else np.array([])
        for j, k in enumerate(keys):
            out.append(dict(family=family, scope=scope, entry=k[0], signal=k[1], exit=k[2], X=k[3], t_fade_real=rv[j], cells_in_grid=len(keys),
                            placebo_p_vs_best_of_grid=float(np.mean(maxabs >= abs(rv[j]))) if np.isfinite(rv[j]) else np.nan,
                            placebo_best_abs_t_p99=float(np.nanpercentile(maxabs, 99)),
                            shifts=int(len(shift_max)), shift_best_abs_t_max=float(shift_max.max()) if len(shift_max) else np.nan,
                            shift_best_abs_t_p95=float(np.percentile(shift_max, 95)) if len(shift_max) else np.nan,
                            shift_p_vs_best_of_grid=float((1 + np.sum(shift_max >= abs(rv[j]))) / (1 + len(shift_max))) if np.isfinite(rv[j]) and len(shift_max) else np.nan,
                            placebo_p_fade=float(np.nanmean(M[:, j] >= rv[j])) if np.isfinite(rv[j]) else np.nan,
                            placebo_p_follow=float(np.nanmean(M[:, j] <= rv[j])) if np.isfinite(rv[j]) else np.nan,
                            placebo_t_p2_5=float(np.nanpercentile(M[:, j], 2.5)) if np.isfinite(M[:, j]).any() else np.nan,
                            placebo_t_p97_5=float(np.nanpercentile(M[:, j], 97.5)) if np.isfinite(M[:, j]).any() else np.nan,
                            is_best_of_grid=j == best, best_abs_t_real=float(np.nanmax(np.abs(rv))), placebo_best_abs_t_p95=float(np.nanpercentile(maxabs, 95)),
                            placebo_p_best_of_grid=float(np.mean(maxabs >= np.nanmax(np.abs(rv))))))
        # lagged placebo: previous closure's signal
        lag = {}
        for key, (S, R) in mats.items():
            for X in X_GRID:
                with np.errstate(invalid="ignore"):
                    P = np.where((np.abs(S[:-1]) > X) & np.isfinite(R[1:]), -np.sign(S[:-1]) * R[1:], np.nan)
                lag[key + (X,)] = st.cluster_t(P)
        for row in out:
            if row["scope"] == scope and row["family"] == family:
                row["t_fade_lag1_placebo"] = lag.get((row["entry"], row["signal"], row["exit"], row["X"]), np.nan)
    return pl.DataFrame(out)


# ------------------------------------------------------------------ test 2
def basis_rows(df, venue):
    rows = []
    cols = ["B_pre30", "B_open30", "B_openhour", "B_close1", "B_close0"] + (["B_close1_usd", "B_close0_usd"] if venue == "binance" else [])
    for kind, lab in (("weekend", "reopening day after a weekend"), ("night", "other days")):
        for sub, f in (("all", pl.lit(True)), ("core", pl.col("core"))):
            d0 = df.filter((pl.col("kind") == kind) & f)
            for col in cols:
                d = d0.filter(pl.col(col).is_finite())
                if d.height == 0:
                    continue
                v = d[col].to_numpy() * BPS; cl = d["cl"].to_numpy()
                med, lo, hi = st.median_abs_boot(v, cl)
                rows.append(dict(venue=venue, day=lab, subset=sub, basis=col, n=d.height, clusters=int(len(np.unique(cl))), mean_bps=float(v.mean()), median_bps=float(np.median(v)),
                                 median_abs_bps=med, median_abs_lo=lo, median_abs_hi=hi, p90_abs_bps=float(np.percentile(np.abs(v), 90)), p99_abs_bps=float(np.percentile(np.abs(v), 99))))
    return rows


def basis_regressions(df, venue):
    rows = []
    d = df.with_columns(dB=pl.col("B_close1") - pl.col("B_open30"))
    for kind in ("weekend", "night"):
        k = d.filter(pl.col("kind") == kind)
        rows.append(slope_row(k, f"{venue} {kind}: (B_close - B_open) on B_open [-1 = opening basis gone by the close]", "B_open30", "dB", placebo=False))
        rows.append(slope_row(k, f"{venue} {kind}: B_open on spread g(T1) [0 = no basis left by a large spread]", "g_T1", "B_open30", placebo=False))
        rows.append(slope_row(k, f"{venue} {kind}: B_pre-open (last trade before the open) on g(T1)", "g_T1", "B_pre30", placebo=False))
        rows.append(slope_row(k, f"{venue} {kind}: B_close(d1) on B_close(d0) [persistence of the standing premium]", "B_close0", "B_close1", placebo=False))
    return rows


def premium_by_regime(off):
    f = pl.read_parquet(DATA / "hl_funding.parquet")
    sess = set(off.filter(pl.col("ticker") == "SPY")["date"].to_list())
    th = np.sort(f["t"].unique().to_numpy() // H * H)
    th = np.unique(th)
    reg = {}
    for t in th:
        a = dt.datetime.fromtimestamp((t - H) / 1000, NY); b = dt.datetime.fromtimestamp(t / 1000, NY)
        am, bm = a.hour * 60 + a.minute, (b.hour * 60 + b.minute) or 1440
        wd, is_sess = a.weekday(), a.date() in sess
        if (wd == 4 and am >= 20 * 60) or wd == 5 or (wd == 6 and bm <= 20 * 60 and b.date() == a.date()):
            r = "weekend, internal pricing (Fri 20:00 - Sun 20:00 ET)"
        elif is_sess and am >= 10 * 60 and bm <= 16 * 60 and b.date() == a.date():
            r = "cash session, full hours (10:00-16:00 ET)"
        elif is_sess and am < 9 * 60 + 30 < bm:
            r = "hour containing the cash open"
        elif is_sess and am >= 4 * 60 and bm <= 9 * 60 + 30:
            r = "pre-market (04:00-09:00 ET)"
        elif is_sess and am >= 16 * 60 and bm <= 20 * 60 and b.date() == a.date():
            r = "after-hours (16:00-20:00 ET)"
        elif (am >= 20 * 60 and wd in (6, 0, 1, 2, 3)) or (bm <= 4 * 60 and wd in (0, 1, 2, 3, 4)):
            r = "overnight (20:00-04:00 ET, Sun-Thu nights)"
        else:
            r = "other (holidays, session edges)"
        reg[int(t)] = r
    f = f.with_columns(th=(pl.col("t") // H) * H).with_columns(regime=pl.col("th").replace_strict(reg, return_dtype=pl.String),
                                                               ticker=pl.col("coin").str.replace("xyz:", "", literal=True))
    f = f.with_columns(group=pl.when(pl.col("ticker").is_in(["XYZ100", "SP500"])).then(pl.lit("index contracts")).when(pl.col("ticker").is_in(CORE)).then(pl.lit("core stocks")).otherwise(pl.lit("other stocks")))
    g = f.group_by("group", "regime").agg(hours=pl.len(), contracts=pl.col("coin").n_unique(), premium_mean_bps=pl.col("premium").mean() * BPS,
                                          premium_median_abs_bps=pl.col("premium").abs().median() * BPS, premium_p90_abs_bps=pl.col("premium").abs().quantile(0.9) * BPS,
                                          funding_mean_bps_per_hour=pl.col("rate").mean() * BPS, funding_p90_abs_bps_per_hour=pl.col("rate").abs().quantile(0.9) * BPS)
    return g.sort("group", "regime"), float(f["rate"].median())


# ------------------------------------------------------------------ extended sample, index contracts
def extended(p2, sel_primary, med_rate):
    c = pl.col
    p = p2.with_columns(gap=c("open1") / c("close0") - 1, cl=c("d0").cast(pl.Int32), g_T1=c("P_T1") / c("close0") - 1, g_T0=c("P_T0") / c("close0") - 1,
                        m_T1=c("I_T1") / c("ndx_close0") - 1, rs=c("P_X16") / c("P_T1") - 1, ri=c("I_X16") / c("I_T1") - 1)
    p = p.with_columns(e_T1=c("g_T1") - c("beta") * c("m_T1"), level_ok=(c("g_T1").abs() <= 0.30))
    p = p.filter((c("kind") == "weekend") & c("level_ok").fill_null(False) & ~c("split_near") & c("gap").is_finite())
    rows = []
    for lab, f in (("extended 2h sample, weekends before the 1h sample", ~c("in_1h_sample")), ("extended 2h sample, weekends inside the 1h sample (cross-check)", c("in_1h_sample")),
                   ("extended 2h sample, all weekends since listing", pl.lit(True))):
        d = p.filter(f)
        rows.append(slope_row(d, f"{lab}: gap on g(T1)", "g_T1", sample="hl_2h"))
        rows.append(slope_row(d, f"{lab}: gap on g(T0)", "g_T0", sample="hl_2h"))
        rows.append(slope_row(d, f"{lab}: gap on g(T1), weekend fixed effects", "g_T1", fe=True, sample="hl_2h"))
    # the rule, primary configuration's selected threshold and side, on the weekends before the 1h sample
    d = p.filter(~c("in_1h_sample") & c("e_T1").is_finite() & c("rs").is_finite() & c("ri").is_finite() & c("beta").is_finite())
    trades = []
    if d.height:
        s, rs, ri, beta, cl = arr(d, "e_T1", "rs", "ri", "beta", "cl"); coins = d["coin"].to_list()
        hours = ((d["X16"] - d["T1"]) / H).to_numpy().astype(float)
        cost = 2 * COSTS.per_side(coins, "primary", "hyperliquid") / BPS + 2 * beta * COSTS.per_side(["xyz:XYZ100"] * d.height, "primary", "hyperliquid") / BPS
        for X in X_GRID:
            for side in ("fade", "follow"):
                take = np.abs(s) > X
                dd = -np.sign(s[take]) if side == "fade" else np.sign(s[take])
                gross = dd * (rs[take] - beta[take] * ri[take]); fund = dd * (-med_rate * hours[take] + beta[take] * med_rate * hours[take])
                net = gross + fund - cost[take]
                sn = st.cell_stats(net * BPS, cl[take], np.unique(cl)); sg = st.cell_stats(gross * BPS, cl[take], np.unique(cl), B=2000)
                trades.append(dict(sample="hl_2h before the 1h sample", entry="T1", signal="neutral", exit="16:00 UTC", X=X, side=side,
                                   is_selected_cell=bool(sel_primary and X == sel_primary[0] and side == sel_primary[1]), trades=sn["trades"],
                                   clusters_with_trades=sn["clusters_with_trades"], clusters=sn["clusters"], gross_bps=sg["mean"], gross_lo=sg["lo"], gross_hi=sg["hi"],
                                   t_cluster_gross=sg["t_cluster"], cost_bps=float(np.mean(cost[take]) * BPS) if take.any() else np.nan, net_bps=sn["mean"], net_lo=sn["lo"], net_hi=sn["hi"], p_net_le_0=sn["p_le_0"]))
    return rows, trades, p


def index_rows(idx):
    c = pl.col
    p = idx.with_columns(gap=c("open1") / c("close0") - 1, cl=c("d0").cast(pl.Int32), v_T1=c("P_T1") / c("P_F") - 1, v_T0=c("P_T0") / c("P_F") - 1,
                         q_T1=c("Q_T1") / c("idx_close0") - 1, q_T0=c("Q_T0") / c("idx_close0") - 1, post=c("P_Oplus") / c("P_T1") - 1,
                         B_close1=c("P_C1") / c("idx_close1") - 1, B_close0=c("P_F") / c("idx_close0") - 1)
    rows = []
    for coin in ("xyz:XYZ100", "xyz:SP500"):
        d = p.filter(c("coin") == coin)
        for kind in ("weekend", "night"):
            k = d.filter(c("kind") == kind)
            rows.append(slope_row(k, f"{coin} {kind}: ETF gap on v(T1), 1h sample", "v_T1", placebo=False, sample="index"))
            if kind == "weekend":
                rows.append(slope_row(k, f"{coin} {kind}: ETF gap on v(T0), 1h sample", "v_T0", placebo=False, sample="index"))
                rows.append(slope_row(k.filter(~c("in_1h_sample")), f"{coin} {kind}: ETF gap on venue/index close - 1 at T1, 2h candles, before the 1h sample", "q_T1", placebo=False, sample="index"))
                rows.append(slope_row(k, f"{coin} {kind}: ETF gap on venue/index close - 1 at T1, 2h candles, all since listing", "q_T1", placebo=False, sample="index"))
                rows.append(slope_row(k, f"{coin} {kind}: ETF gap on venue/index close - 1 at T0, 2h candles, all since listing", "q_T0", placebo=False, sample="index"))
            rows.append(slope_row(k, f"{coin} {kind}: venue's later move P(O+)/P(T1)-1 on v(T1)", "v_T1", "post", placebo=False, sample="index"))
    return rows, p


# ------------------------------------------------------------------ main
def main():
    global COSTS
    COSTS = Costs()
    off = load_official()
    hl = prep(pl.read_parquet(RES / "panel_hl.parquet"), "hyperliquid")
    bn = prep(pl.read_parquet(RES / "panel_bn.parquet"), "binance")
    summary = {"seed": SEED, "bootstrap_resamples": st.B_BOOT, "permutations": st.N_PERM}
    slopes, percon, capture = [], [], []

    # ---- Test 1 (and 4): slopes
    for venue, df in (("hyperliquid", hl), ("binance", bn)):
        for kind in ("weekend", "night"):
            k = df.filter(pl.col("kind") == kind)
            tag = dict(sample=f"{venue}_1h", venue=venue, kind=kind)
            slopes.append(slope_row(k, f"{venue} {kind} PRIMARY: gap on v(T1), all contracts", "v_T1", **tag))
            slopes.append(slope_row(k, f"{venue} {kind}: gap on g(T1)", "g_T1", **tag))
            slopes.append(slope_row(k, f"{venue} {kind}: gap on v(T1), fixed effects (stock-specific part)", "v_T1", fe=True, **tag))
            slopes.append(slope_row(k.filter(pl.col("core")), f"{venue} {kind}: gap on v(T1), core subset", "v_T1", **tag))
            slopes.append(slope_row(k, f"{venue} {kind}: venue's later move P(O+)/P(T1)-1 on v(T1)", "v_T1", "venue_post", **tag))
            slopes.append(slope_row(k, f"{venue} {kind}: venue's later move P(O+)/P(T1)-1 on g(T1)", "g_T1", "venue_post", **tag))
            slopes.append(slope_row(k, f"{venue} {kind}: venue's later move, fixed effects", "v_T1", "venue_post", fe=True, **tag))
            if kind == "weekend":
                slopes.append(slope_row(k, f"{venue} {kind}: gap on v(T0)", "v_T0", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: gap on g(T0)", "g_T0", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: gap on v(T0), fixed effects", "v_T0", fe=True, **tag))
                slopes.append(slope_row(k.filter(pl.col("core")), f"{venue} {kind}: gap on v(T0), core subset", "v_T0", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: venue's later move P(O+)/P(T0)-1 on v(T0)", "v_T0", "venue_post_T0", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: venue's move T0 to T1, P(T1)/P(T0)-1, on v(T0)", "v_T0", "venue_T0_T1", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: venue's move across the Sunday reopening P(R+)/P(R-)-1 on v(R-)", "v_Rm", "venue_reopen", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: venue's move across the Sunday reopening, fixed effects", "v_Rm", "venue_reopen", fe=True, **tag))
                slopes.append(slope_row(k.filter(pl.col("core")), f"{venue} {kind}: venue's move across the Sunday reopening, core subset", "v_Rm", "venue_reopen", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: gap on v(R-)", "v_Rm", **tag))
                slopes.append(slope_row(k, f"{venue} {kind}: gap on v(R+)", "v_Rp", **tag))
                for X in (0.01, 0.02):
                    d = k.filter(pl.col("v_T1").is_finite() & pl.col("gap").is_finite() & (pl.col("v_T1").abs() > X))
                    if d.height:
                        ratio = (d["gap"] / d["v_T1"]).to_numpy()
                        capture.append(dict(venue=venue, min_abs_venue_move=X, n=d.height, weekends=d["cl"].n_unique(), median_ratio=float(np.median(ratio)), mean_ratio=float(np.mean(ratio)),
                                            share_ratio_below_0=float(np.mean(ratio < 0)), share_ratio_0_to_1=float(np.mean((ratio >= 0) & (ratio <= 1))), share_ratio_above_1=float(np.mean(ratio > 1))))
                hv = halves(k)
                for half in ("sel", "eval"):
                    slopes.append(slope_row(k.filter(pl.col("cl").is_in(hv[half].tolist())), f"{venue} {kind}: gap on v(T1), {'earlier' if half == 'sel' else 'later'} half", "v_T1", placebo=False, **tag))
            if venue == "hyperliquid":
                slopes.append(slope_row(k.filter(pl.col("rth_only")), f"{venue} {kind}: gap on v(T1), contracts priced externally in regular hours only", "v_T1", placebo=False, **tag))
                slopes.append(slope_row(k.filter(~pl.col("rth_only")), f"{venue} {kind}: gap on v(T1), contracts with 24/5 external pricing", "v_T1", placebo=False, **tag))
            pc = per_contract(k, "v_T1")
            if pc.height:
                percon.append(pc.with_columns(venue=pl.lit(venue), kind=pl.lit(kind)))
                summary[f"per_contract_{venue}_{kind}"] = dict(contracts=pc.height, median_slope=float(pc["slope"].median()), q25=float(pc["slope"].quantile(0.25)),
                                                               q75=float(pc["slope"].quantile(0.75)), share_positive=float((pc["slope"] > 0).mean()), share_below_1=float((pc["slope"] < 1).mean()))
    idx_rows, idxp = index_rows(pl.read_parquet(RES / "panel_idx.parquet"))
    slopes += idx_rows

    # ---- Test 3 (and 4): rule
    grids, sels, senss, placebos = [], [], [], []
    fams = [("hl_weekend", hl, "hyperliquid", "weekend", GRID(["T1", "T0"]), False), ("hl_weekend_reopen", hl, "hyperliquid", "weekend", REOPEN, False),
            ("hl_night", hl, "hyperliquid", "night", GRID(["T1"]), False),
            ("bn_weekend", bn, "binance", "weekend", GRID(["T1", "T0"]), False), ("bn_weekend_reopen", bn, "binance", "weekend", REOPEN, False),
            ("bn_weekend_long_only", bn, "binance", "weekend", GRID(["T1", "T0"]) + REOPEN, True), ("bn_night", bn, "binance", "night", GRID(["T1"]), False)]
    keep = {}
    for fam, df, venue, kind, configs, lo in fams:
        k = df.filter(pl.col("kind") == kind)
        if k.height == 0:
            continue
        g, s, se, cands, hv = run_family(k, venue, kind, configs, fam, long_only=lo)
        grids.append(g); sels.append(s); senss.append(se); keep[fam] = (k, cands, hv)
        if not lo:
            placebos.append(placebo_family(cands, hv, fam))
        summary[f"{fam}_clusters"] = dict(all=int(len(hv["all"])), sel=int(len(hv["sel"])), eval=int(len(hv["eval"])),
                                          first=str(k["d0"].min()), last=str(k["d0"].max()), split_at=str(k.filter(pl.col("cl") == int(hv["eval"][0]))["d0"].min()) if len(hv["eval"]) else None,
                                          contracts=int(k["coin"].n_unique()), observations=int(k.height))
    grid = pl.concat(grids); sel = pl.concat(sels); sens = pl.concat(senss); plac = pl.concat(placebos)
    # Benjamini-Hochberg across the 10 weekend configurations of a venue together (8 + the 2 reopening ones)
    sel = sel.with_columns(eval_p_bh10=pl.lit(None, dtype=pl.Float64))
    for venue_fams in (("hl_weekend", "hl_weekend_reopen"), ("bn_weekend", "bn_weekend_reopen")):
        m = sel["family"].is_in(list(venue_fams)).to_numpy()
        adj = np.full(sel.height, np.nan); adj[m] = st.bh(sel["eval_p_net_le_0"].to_numpy()[m])
        sel = sel.with_columns(eval_p_bh10=pl.when(pl.Series(m)).then(pl.Series(adj)).otherwise(pl.col("eval_p_bh10")))

    # variants of the primary configuration (hl weekend, entry T1, neutral, exit O+): selected X and side
    prim = sel.filter((pl.col("family") == "hl_weekend") & (pl.col("entry") == "T1") & (pl.col("signal") == "neutral") & (pl.col("exit") == "Oplus")).row(0, named=True)
    k, cands, hv = keep["hl_weekend"]
    variants = []

    def variant(label, c, extra_mask=None):
        for half in ("sel", "eval", "all"):
            m = np.isin(c["cl"], hv[half])
            if extra_mask is not None:
                m &= extra_mask(c)
            kk = cell(c, prim["X"], prim["side"], m)
            s_ = st.cell_stats(kk["net"] * BPS, c["cl"][kk["take"]], hv[half])
            variants.append(dict(variant=label, X=prim["X"], side=prim["side"], half=half, trades=s_["trades"], clusters_with_trades=s_["clusters_with_trades"],
                                 gross_bps=float(np.mean(kk["gross"]) * BPS) if len(kk["gross"]) else np.nan, net_bps=s_["mean"], net_lo=s_["lo"], net_hi=s_["hi"], p_net_le_0=s_["p_le_0"]))
    variant("primary (entry T1, market-neutral with xyz:XYZ100, exit O+)", cands[("T1", "neutral", "Oplus")])
    variant("hedge with xyz:SP500", candidates(k, "hyperliquid", "T1", "Oplus", "neutral", hedge="J"))
    variant("entry delayed one hour (signal at T1, entry at T1 + 1h)", candidates(k, "hyperliquid", "T1p1", "Oplus", "neutral", sig_time="T1"))
    variant("core subset", cands[("T1", "neutral", "Oplus")], lambda c: np.isin(c["coin"], [f"xyz:{t}" for t in CORE]))
    variant("entry-hour volume of at least 100,000 USD (100 times a 1,000 USD trade)", cands[("T1", "neutral", "Oplus")], lambda c: c["ntl"] >= 1e5)

    # ---- premium, extended sample
    prem, med_rate = premium_by_regime(off)
    ext_rows, ext_trades, _ = extended(pl.read_parquet(RES / "panel_hl2h.parquet"), (prim["X"], prim["side"]), med_rate)
    slopes += ext_rows
    summary["median_hourly_funding_rate"] = med_rate

    # ---- Test 2
    basis = basis_rows(hl, "hyperliquid") + basis_rows(bn, "binance")
    breg = basis_regressions(hl, "hyperliquid") + basis_regressions(bn, "binance")

    # ---- H1 side by side
    S = pl.DataFrame(slopes, infer_schema_length=None)
    h1 = []
    for venue in ("hyperliquid", "binance"):
        for lab, key in (("PRIMARY: gap on v(T1), all contracts", f"{venue} weekend PRIMARY: gap on v(T1), all contracts"), ("gap on v(T0)", f"{venue} weekend: gap on v(T0)"),
                         ("gap on v(T1), fixed effects", f"{venue} weekend: gap on v(T1), fixed effects (stock-specific part)"), ("gap on v(T1), core subset", f"{venue} weekend: gap on v(T1), core subset"),
                         ("venue's later move T1 to open hour, on v(T1) [give-back = -slope]", f"{venue} weekend: venue's later move P(O+)/P(T1)-1 on v(T1)"),
                         ("venue's later move T0 to open hour, on v(T0) [give-back = -slope]", f"{venue} weekend: venue's later move P(O+)/P(T0)-1 on v(T0)"),
                         ("venue's later move across the Sunday reopening, on v(R-) [give-back = -slope]", f"{venue} weekend: venue's move across the Sunday reopening P(R+)/P(R-)-1 on v(R-)"),
                         ("gap on v(R-)", f"{venue} weekend: gap on v(R-)"), ("gap on v(R+)", f"{venue} weekend: gap on v(R+)")):
            r = S.filter(pl.col("label") == key)
            if r.height == 0 or r["slope"][0] is None:
                continue
            r = r.row(0, named=True)
            direct = "later move" in lab
            gb, glo, ghi = (-r["slope"], -r["hi"], -r["lo"]) if direct else (r["give_back"], r["give_back_lo"], r["give_back_hi"])
            verdict = "" if direct else ("SUPPORTED (interval below 1)" + (", economically small (slope >= 0.9)" if r["slope"] >= 0.9 else "") if r["hi"] < 1 else "REJECTED (interval contains 1 or lies above)")
            h1.append(dict(venue=venue, test=lab, weekends=r["clusters"], n=r["n"], slope=r["slope"], lo=r["lo"], hi=r["hi"], r2=r["r2"], give_back_share=gb, give_back_lo=glo, give_back_hi=ghi, h1_verdict=verdict))
    H1 = pl.DataFrame(h1)

    # ---- write
    S.write_csv(RES / "test1_slopes.csv")
    if percon:
        pl.concat(percon).write_csv(RES / "test1_per_contract.csv")
    pl.DataFrame(basis).write_csv(RES / "test2_basis.csv")
    pl.DataFrame(breg, infer_schema_length=None).write_csv(RES / "test2_regressions.csv")
    grid.write_csv(RES / "test3_grid.csv"); sel.write_csv(RES / "test3_selected.csv"); sens.write_csv(RES / "test3_costs.csv"); plac.write_csv(RES / "test3_placebo.csv")
    pl.DataFrame(variants).write_csv(RES / "test3_variants.csv")
    if ext_trades:
        pl.DataFrame(ext_trades).write_csv(RES / "test3_extended.csv")
    prem.write_csv(RES / "premium_by_regime.csv"); H1.write_csv(RES / "h1.csv")
    if capture:
        pl.DataFrame(capture).write_csv(RES / "test1_capture_ratio.csv")
    summary["primary_rule"] = prim
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    with pl.Config(tbl_rows=80, tbl_cols=30, tbl_width_chars=260, fmt_str_lengths=90, float_precision=3):
        print(H1)
        print(sel.select("family", "entry", "signal", "exit", "X", "side", "sel_trades", "sel_net_bps", "eval_trades", "eval_clusters_with_trades", "eval_gross_bps", "eval_cost_bps", "eval_net_bps", "eval_lo", "eval_hi", "eval_p_net_le_0", "eval_p_bh", "eval_p_bh10"))


if __name__ == "__main__":
    main()
