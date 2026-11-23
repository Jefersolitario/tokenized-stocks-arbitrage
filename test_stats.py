"""Checks of the statistics code on synthetic data with a known answer.   python test_stats.py"""
import numpy as np

import stats as st


def synth(seed, n_cl=30, n_u=60, slope=0.8):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_cl), n_u); unit = np.tile(np.arange(n_u), n_cl)
    common = np.repeat(rng.normal(0, 0.01, n_cl), n_u)                       # market-wide move per cluster
    x = common + rng.normal(0, 0.008, n_cl * n_u)
    y = slope * x + np.repeat(rng.normal(0, 0.004, n_cl), n_u) + rng.normal(0, 0.004, n_cl * n_u)
    return cl, unit, x, y


def main():
    cl, unit, x, y = synth(1)
    r = st.ols_cluster_boot(x, y, cl, B=2000)
    assert abs(r["slope"] - 0.8) < 0.1 and r["lo"] < r["slope"] < r["hi"], r
    a, b, r2 = st.ols(x, y); assert abs(b - r["slope"]) < 1e-12
    assert abs(np.polyfit(x, y, 1)[0] - b) < 1e-9
    rf = st.ols_cluster_boot(x, y, cl, fe=True, B=2000)
    assert abs(rf["slope"] - 0.8) < 0.05, rf
    # coverage of the cluster bootstrap interval for the true slope, 200 synthetic samples (should be near 95%)
    cover = np.mean([(lambda q: q["lo"] <= 0.8 <= q["hi"])(st.ols_cluster_boot(*(lambda c, u, xx, yy: (xx, yy, c))(*synth(s)), B=500, seed=s)) for s in range(200)])
    assert 0.85 <= cover <= 0.99, cover
    p = st.slope_placebo(cl, unit, x, y, n_perm=300)
    assert abs(p["real"] - b) < 1e-9 and abs(p["perm_mean"]) < 0.1 and p["share_perm_ge_real"] == 0.0, p
    # trade cells: mean of a known constant, clusters without trades are kept in the resampling
    pnl = np.full(40, 5.0); cs = st.cell_stats(pnl, np.repeat(np.arange(4), 10), np.arange(10), B=500)
    assert cs["mean"] == 5.0 and cs["lo"] == 5.0 and cs["hi"] == 5.0 and cs["clusters"] == 10 and cs["clusters_with_trades"] == 4, cs
    rng = np.random.default_rng(3); pn = rng.normal(0, 1, 3000); c3 = np.repeat(np.arange(30), 100)
    cs = st.cell_stats(pn, c3, np.arange(30), B=2000); assert cs["lo"] < 0 < cs["hi"], cs
    assert np.allclose(st.bh([0.01, 0.04, 0.03, np.nan]), [0.03, 0.04, 0.04, np.nan], equal_nan=True)
    P = np.array([[1.0, np.nan], [2.0, 1.0], [np.nan, np.nan]]); assert abs(st.cluster_t(P) - 4 / np.sqrt(10)) < 1e-12
    m, lo, hi = st.median_abs_boot([-1, 2, -3, 4], [0, 0, 1, 1], B=200); assert m == 2.5 and lo <= m <= hi
    print("stats checks passed; bootstrap coverage of the true slope in 200 synthetic samples:", round(float(cover), 3))


if __name__ == "__main__":
    main()
