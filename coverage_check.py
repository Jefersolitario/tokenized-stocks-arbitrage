"""How often the nominal 95% cluster-bootstrap interval covers the truth at this study's sample sizes (synthetic data).

    python coverage_check.py        -> results/coverage_check.csv

Slope: 14 / 29 / 115 clusters of 60 correlated units, true slope 0.8. Mean PnL: clusters with a common shock, a random
number of trades and some clusters without any trade, true mean 0. Seeds fixed.
"""
import numpy as np
import polars as pl

import stats as st
from common import RES
from test_stats import synth


def main():
    rows = []
    for n_cl in (14, 29, 115):
        cov = np.mean([(lambda q: q["lo"] <= 0.8 <= q["hi"])(st.ols_cluster_boot(*(lambda c, u, x, y: (x, y, c))(*synth(s, n_cl=n_cl)), B=500, seed=s)) for s in range(400)])
        rows.append(dict(statistic="slope", clusters=n_cl, simulations=400, coverage_of_nominal_95=float(cov)))
    rng = np.random.default_rng(0)
    for n_cl, p_trade in ((14, 0.6), (29, 0.6), (58, 0.8)):
        hits, n_sim = 0, 600
        for s in range(n_sim):
            pn, cl = [], []
            for c in range(n_cl):
                if rng.random() < p_trade:
                    k = rng.integers(1, 12); shock = rng.normal(0, 40)
                    pn += list(shock + rng.normal(0, 60, k)); cl += [c] * k
            if len(pn) < 3:
                continue
            r = st.cell_stats(np.array(pn), np.array(cl), np.arange(n_cl), B=500, seed=s)
            hits += r["lo"] <= 0 <= r["hi"]
        rows.append(dict(statistic="mean PnL per trade", clusters=n_cl, simulations=n_sim, coverage_of_nominal_95=hits / n_sim))
    RES.mkdir(exist_ok=True)
    out = pl.DataFrame(rows); out.write_csv(RES / "coverage_check.csv"); print(out)


if __name__ == "__main__":
    main()
