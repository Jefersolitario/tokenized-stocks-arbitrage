"""Statistics used by the study: OLS with cluster bootstrap, permutation placebos, trade-cell statistics.

One cluster = one closure (one weekend or one night). Every random draw uses the fixed seed in common.SEED.
"""
import numpy as np
from scipy.stats import t as tdist

from common import SEED

B_BOOT = 10_000
N_PERM = 2_000


def _codes(cl):
    u, inv = np.unique(np.asarray(cl), return_inverse=True)
    return u, inv


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum(); sxy = ((x - mx) * (y - my)).sum(); syy = ((y - my) ** 2).sum()
    b = sxy / sxx
    return my - b * mx, b, (sxy * sxy) / (sxx * syy) if syy > 0 else np.nan


def demean_within(v, cl):
    u, inv = _codes(cl)
    v = np.asarray(v, float)
    return v - (np.bincount(inv, weights=v) / np.bincount(inv))[inv]


def ols_cluster_boot(x, y, cl, fe=False, B=B_BOOT, seed=SEED):
    """Slope of y on x with a percentile interval from resampling whole clusters.

    fe=True: both sides demeaned within the cluster first (cluster fixed effects), slope without intercept.
    Returns dict(n, clusters, slope, lo, hi, intercept, r2, boot_share_below_1).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); cl = np.asarray(cl)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], cl[ok]
    if len(x) < 3 or len(np.unique(cl)) < 3:
        return dict(n=int(len(x)), clusters=int(len(np.unique(cl))), slope=np.nan, lo=np.nan, hi=np.nan, intercept=np.nan, r2=np.nan, boot_share_below_1=np.nan)
    if fe:
        x, y = demean_within(x, cl), demean_within(y, cl)
    u, inv = _codes(cl)
    f = lambda v: np.bincount(inv, weights=v, minlength=len(u))
    S = np.stack([f(np.ones_like(x)), f(x), f(y), f(x * x), f(x * y)], axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    T = S[idx].sum(axis=1)
    n, sx, sy, sxx, sxy = T.T
    with np.errstate(divide="ignore", invalid="ignore"):
        bb = sxy / sxx if fe else (n * sxy - sx * sy) / (n * sxx - sx * sx)
    if fe:
        b = (x * y).sum() / (x * x).sum(); a = 0.0
        r2 = (x * y).sum() ** 2 / ((x * x).sum() * (y * y).sum())
    else:
        a, b, r2 = ols(x, y)
    lo, hi = np.nanpercentile(bb, [2.5, 97.5])
    return dict(n=int(len(x)), clusters=int(len(u)), slope=float(b), lo=float(lo), hi=float(hi), intercept=float(a), r2=float(r2),
                boot_share_below_1=float(np.nanmean(bb < 1.0)))


def to_matrix(cl, unit, values):
    """Pivot long data to a (clusters x units) matrix with NaN for missing. Returns (matrix, cluster ids, unit ids)."""
    uc, ic = _codes(cl); uu, iu = _codes(unit)
    M = np.full((len(uc), len(uu)), np.nan)
    M[ic, iu] = np.asarray(values, float)
    return M, uc, uu


def _slope_mat(X, Y):
    ok = np.isfinite(X) & np.isfinite(Y)
    if ok.sum() < 3:
        return np.nan
    x, y = X[ok], Y[ok]
    return ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()


def slope_placebo(cl, unit, x, y, n_perm=N_PERM, seed=SEED):
    """Placebo for a slope: the x vectors are reassigned to other clusters (unit by unit matched).

    Returns dict(real, perm_mean, perm_p2_5, perm_p97_5, share_perm_ge_real, lag1) where lag1 uses the previous cluster's x.
    """
    ok = np.isfinite(np.asarray(x, float)) & np.isfinite(np.asarray(y, float))
    cl, unit, x, y = np.asarray(cl)[ok], np.asarray(unit)[ok], np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    X, uc, _ = to_matrix(cl, unit, x); Y, _, _ = to_matrix(cl, unit, y)
    real = _slope_mat(X, Y)
    rng = np.random.default_rng(seed)
    perm = np.array([_slope_mat(X[rng.permutation(len(uc))], Y) for _ in range(n_perm)])
    lag = _slope_mat(X[:-1], Y[1:]) if len(uc) > 3 else np.nan
    return dict(real=float(real), perm_mean=float(np.nanmean(perm)), perm_p2_5=float(np.nanpercentile(perm, 2.5)),
                perm_p97_5=float(np.nanpercentile(perm, 97.5)), share_perm_ge_real=float(np.nanmean(perm >= real)), lag1=float(lag))


def cell_stats(pnl, cl, all_clusters, B=B_BOOT, seed=SEED):
    """Mean PnL per trade with a cluster-bootstrap interval. `all_clusters` lists every cluster of the sample half,
    including those without a trade (they are resampled too). Returns a dict; values in the units of pnl."""
    pnl = np.asarray(pnl, float); cl = np.asarray(cl)
    allc = np.unique(np.asarray(all_clusters))
    pos = np.searchsorted(allc, cl)
    S = np.bincount(pos, weights=pnl, minlength=len(allc)); n = np.bincount(pos, minlength=len(allc)).astype(float)
    out = dict(trades=int(len(pnl)), clusters_with_trades=int((n > 0).sum()), clusters=int(len(allc)))
    if len(pnl) == 0:
        return dict(out, mean=np.nan, lo=np.nan, hi=np.nan, p_le_0=np.nan, median=np.nan, hit=np.nan, t_cluster=np.nan,
                    wk_mean=np.nan, wk_lo=np.nan, wk_hi=np.nan, wk_t=np.nan, wk_p_le_0=np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(allc), size=(B, len(allc)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mb = S[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.nanpercentile(mb, [2.5, 97.5])
    t = S.sum() / np.sqrt((S ** 2).sum()) if (S ** 2).sum() > 0 else np.nan
    # second, more conservative view with few clusters: one number per cluster (its mean PnL per trade), Student t across clusters
    mw = S[n > 0] / n[n > 0]; k = len(mw)
    if k >= 3 and np.std(mw, ddof=1) > 0:
        wt = mw.mean() / (np.std(mw, ddof=1) / np.sqrt(k)); wp = float(tdist.sf(wt, k - 1))
        half = tdist.ppf(0.975, k - 1) * np.std(mw, ddof=1) / np.sqrt(k)
        wk = dict(wk_mean=float(mw.mean()), wk_lo=float(mw.mean() - half), wk_hi=float(mw.mean() + half), wk_t=float(wt), wk_p_le_0=wp)
    else:
        wk = dict(wk_mean=float(mw.mean()) if k else np.nan, wk_lo=np.nan, wk_hi=np.nan, wk_t=np.nan, wk_p_le_0=np.nan)
    return dict(out, mean=float(pnl.mean()), lo=float(lo), hi=float(hi), p_le_0=float(np.nanmean(mb <= 0)), median=float(np.median(pnl)),
                hit=float((pnl > 0).mean()), t_cluster=float(t), **wk)


def cluster_t(P):
    """Cluster t of the mean of a (clusters x units) PnL matrix with NaN = no trade: sum of cluster sums / sqrt(sum of squares)."""
    S = np.nansum(P, axis=1)
    q = (S ** 2).sum()
    return S.sum() / np.sqrt(q) if q > 0 else np.nan


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(pvals, float); out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]; m = len(ok)
    adj = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def median_abs_boot(values, cl, B=2_000, seed=SEED):
    """Median absolute value with a cluster-bootstrap interval."""
    v = np.abs(np.asarray(values, float)); cl = np.asarray(cl)
    ok = np.isfinite(v); v, cl = v[ok], cl[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    u, inv = _codes(cl)
    groups = [v[inv == i] for i in range(len(u))]
    rng = np.random.default_rng(seed)
    meds = [np.median(np.concatenate([groups[j] for j in rng.integers(0, len(u), len(u))])) for _ in range(B)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)
