# tokenized-stocks-arbitrage

Research on the weekend prices of tokenized US stocks (Hyperliquid's `xyz` stock contracts and Binance's stock tokens)
against the cash market, and a paper test of the one trade the research left standing. No money, no keys, no orders:
everything here reads public market data.

A word on the name: this is not an arbitrage. Nothing forces these prices together by a set time. What is being tested
is a statistical fade with a hedge.

## What was found (22 Nov 2026)

The question was whether tokenized stocks drift over the weekend and converge to Monday's cash open.

| Claim | Verdict after independent review | Grade |
|---|---|---|
| The token converges to Monday's open, so fade the spread on Monday | Not supported. By Monday 08:00 UTC the cash open goes to where the token already is (slope 0.94, 95% interval 0.68 to 1.15, 29 weekends). Fading there loses 50 bps a trade. | strong that nothing is left by Monday |
| Part of the weekend drift reverses | Yes, but on Sunday evening New York time: a third to a half of it, mostly between 19:00 and 21:00. | strong that it exists, moderate on its size |
| Fading that reversal pays after real costs | Unknown. In trade prices: +73 bps a trade on 29 weekends, +52 on the later 14, shrinking. Forward estimate about +30 bps (90% range -15 to +65), about 65 USD a weekend at 1,000 USD a trade. | weak at 1,000 USD, none at 10,000 USD |

The pre-registration is `PREREG.md`; the result tables are in `results/`; `python report_numbers.py key` prints every
quoted number with the file it comes from.

## What runs now: the paper test

The backtest could not answer what an order really costs at 19:00 and 22:00 New York on a Sunday. So:

- **`record_quotes.py`** records public order books of the 86 stock and fund contracts and the two index contracts
  every Sunday from 17:00 to 22:30 New York time: books every 20 seconds around 18:00, 19:00, 20:05 and 22:00 and
  every minute otherwise, mid prices every 5 seconds, reference prices and funding every minute. Files go to
  `quotes/<date>/`.
- **`PAPER_RULE.md`** fixes the rule and the pass marks. It was written before the first recording, and later
  changes are dated addenda. In short: the two-leg fade (stock contract against beta times `xyz:XYZ100`) at a 1%
  threshold, entered at 19:00 and closed at 22:00 New York at recorded book prices. After 3 weekends it carries on
  only if a stock fill costs 15 bps or less on average; after 12, a real-money pilot is considered only above +25 bps
  a trade with a per-weekend t above 2.
- **`paper_eval.py <date>`** works out a recorded weekend under that rule and writes `results/paper/`.
- **`publish_dashboard.py`** publishes the backtest and the paper weekends for the portfolio dashboard, in its results
  contract, under `dashboard/` (strategy `sunday_fade_1k`, role `monitor`). Published weekends never change.

First weekend (22 Nov 2026): +58 bps a trade after costs on 30 trades, median +29, with one trade supplying 40% of
the total. One weekend is one observation.

### The schedule

`.github/workflows/weekend-quotes.yml` has a start time every half hour from Sunday 15:05 UTC to Monday 02:35 UTC,
because GitHub starts scheduled jobs late, by hours on some evenings. A run outside Sunday 16:45 to 22:30 New York
stops within seconds, and so does one that finds a recording already running. Expect about two dozen runs in the
Actions list each Sunday, all but one lasting a few seconds.

- **Start one by hand:** Actions, `weekend-quotes`, Run workflow; or change the number in `quotes/START` and push.
- **Stop it:** disable the `weekend-quotes` workflow in the Actions tab.
- A change to `record_quotes.py` runs a 150-second test and saves its output under `quotes/smoke/`.

## Layout

| Path | What |
|---|---|
| `PREREG.md` | The study's pre-registration and its three addenda |
| `fetch.py`, `fetch_config.json` | How the venue data was downloaded on 22 Nov 2026 |
| `data/` | Venue candles, funding and order-book snapshots as downloaded |
| `official/`, `official.py` | Official daily opens and closes, and how the file is built |
| `build_panel.py`, `analyze.py`, `posthoc*.py`, `spreads.py`, `audit.py`, `stats.py`, `common.py` | The study |
| `results/` | Every result table of the study; `results/paper/` holds the paper test |
| `report_numbers.py` | Prints the numbers quoted in the report |
| `test_stats.py`, `synth_test.py` | Checks of the statistics on synthetic data |
| `PAPER_RULE.md`, `record_quotes.py`, `paper_eval.py`, `quotes/` | The paper test |
| `publish_dashboard.py`, `dashboard/` | The series the portfolio dashboard reads |
| `HISTORY.md` | Where the commit history comes from, with old and new hashes |

## Reproduce the study

With Python 3.13 and `pip install -r requirements.txt`, from the committed data:

```
python build_panel.py && python spreads.py && python audit.py && python analyze.py && python posthoc.py
python this_weekend.py && python coverage_check.py && python posthoc_next_trade.py && python fee_history_check.py && python posthoc_breadth.py
python report_numbers.py key
python test_stats.py && python synth_test.py
```

Do not re-run `fetch.py` to check the study. The venue serves only its most recent 5,000 or so candles per interval,
so a new download replaces the earliest weekends of the sample with later ones.

## Data

- **Venue data:** public endpoints of Hyperliquid (`api.hyperliquid.xyz/info`) and Binance spot, no key. The sample
  holds only contracts that were still listed on 22 Nov 2026: the venue returns no history for delisted ones.
- **Official prices:** daily opens and closes from a public copy of Yahoo daily bars. `official/official_notes.txt`
  lists what each ticker covers.
- **Costs:** Hyperliquid's taker fee for these markets was 0.9 bps a fill in growth mode on 22 Nov 2026 (9 bps
  otherwise). The paper test's result depends on that staying so.

## History

This work was done on 22 and 23 Nov 2026 on a research branch of `Jefersolitario/All-In-tracker` and moved here with
its commit dates intact. `HISTORY.md` explains the move and maps the old hashes to the new ones.
