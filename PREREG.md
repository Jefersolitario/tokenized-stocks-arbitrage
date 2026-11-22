# Pre-registration: weekend prices of tokenized US stocks against the cash market

Written and committed on Sunday 22 Nov 2026, before any venue candle of this study was downloaded
(the first data commit on this branch comes after this file in the git history).
What I had seen when writing it: the brief's one anecdote (NVDA, weekend of 20 Nov 2026), the venue documentation,
and the published literature listed in the report. No outcome of the tests below.

## Question

Owner, verbatim: "Does the spread between hyper liquid or Binance tokenized stocks converge towards nasdaq price on
Monday or the spread is already closed?"

Two readings to separate:
- H_info ("already closed"): the venue's weekend move is information. The cash market opens at the venue's price, so
  there is nothing left to converge and fading the spread loses the costs.
- H_revert ("converges back"): the venue overshoots. Part of the weekend move reverses by Monday's open or close, so
  fading the spread on the venue earns money after costs.

## Data and clocks

- Venue prices: Hyperliquid builder dex `xyz`, trade candles (1h for the main sample, about the latest 5,000 hours;
  30m for the exact open time, about the latest 104 days; 2h for the extended history back to listing). Binance spot
  stock tokens (base asset = ticker + "B", quote USDT), 1h and 30m klines since listing.
- Official prices: Yahoo daily open and close (not adj_close) from the public copy alm0421/alm0421. Prices are
  un-adjusted for later splits with the file's split column (and SNDK 8-for-1, first split session 2026-11-20, per
  the brief). Any observation whose window lies within 7 calendar days of a split date of that ticker is dropped.
- Sessions: the dates in SPY's daily file are the session calendar. Open 09:30 and close 16:00 America/New_York
  converted to UTC per date (early closes 13:00 on 2025-11-28 and 2025-12-24).
- A "weekend" is a pair of consecutive sessions (d0, d1) with a Saturday and a Sunday between them (Monday or Friday
  holidays included: d1 is the reopening session). A "night" is a pair with d1 = d0 + 1 calendar day. Other
  closures (mid-week holidays) are in neither set.
- Venue price at time t, P(t): the close of the candle that ends exactly at t, required to exist and to contain at
  least one trade. Otherwise the observation is missing (never forward-filled).
- Times: F = cash close time on d0. T1 = 08:00 UTC on d1 (the brief's "last hourly close before Monday 08:00 UTC").
  T0 = 20:00 UTC on the calendar day before d1 (Sunday evening UTC, inside the venue's internal-pricing weekend
  session for a normal weekend). O+ = the end of the hourly candle that contains the cash open on d1 (open + 30
  minutes). C1 = cash close time on d1.
- Universe, point in time: every `xyz` contract whose name equals the ticker of a US-listed stock or ETF in the
  official data, in every weekend or night in which it has valid venue prices at the times a test needs, provided
  the venue price is within 30% of the official close at F (instrument and split check, uses only data known at F).
  Index contracts xyz:SP500 and xyz:XYZ100 are the market legs and are tested separately as time series against SPY
  and QQQ. "Core" subset reported alongside: NVDA, TSLA, AAPL, MSFT, GOOGL, AMZN, META, MU, SMH.
- The weekend of 20-23 Nov 2026 has no Monday yet and is excluded.

## Variables (per contract i and closure w)

- gap = Open(d1) / Close(d0) - 1 (official).
- v(T) = P(T) / P(F) - 1 (venue's own move since the cash close time).
- g(T) = P(T) / Close(d0) - 1 (the "spread" the owner sees: venue price against the last official close).
- m(T) = the same v for xyz:XYZ100 (market-wide move on the venue).
- beta_i = OLS slope of the stock's daily close-to-close return on QQQ's over the 60 sessions ending at d0
  (official data, known at F), clipped to [0, 3].
- e(T) = g(T) - beta_i * m(T) (stock-specific part of the spread).

## Test 1 - who moves to whom

Pooled OLS with intercept, gap on v(T1). Primary: all contracts, weekends, 1h sample.
Reported: slope, R2, RMSE and median absolute value of (gap - v) against the same for gap alone.
Pre-specified variants, all reported: (a) regressor g(T1) instead of v(T1); (b) T0 instead of T1; (c) weekend fixed
effects (both sides demeaned within the weekend: the stock-specific part); (d) index contracts alone as time series
(xyz:SP500 on SPY's gap, xyz:XYZ100 on QQQ's gap); (e) core subset; (f) per-contract slopes (distribution only);
(g) venue-only reversal regression: P(O+)/P(T1) - 1 on v(T1) and on g(T1); (h) extended history with 2h candles:
gap on g(T1), weekends before the 1h sample starts, as a separate earlier sample.
Reading: slope near 1 (95% interval contains 1, or slope >= 0.9) and a clearly smaller error than the Friday-close
benchmark = H_info. A 95% interval entirely below 1 = partial reversal (H_revert), size (1 - slope).

## Test 2 - leftover basis once cash is open

- B_open = P(open time of d1, 30m candle open) / Open(d1) - 1 where 30m candles exist; B_openhour = P(O+) / Open(d1) - 1
  (the brief's definition; it contains 30 minutes of real price movement, so it is an upper bound on the basis).
- B_close = P(C1) / Close(d1) - 1 on the reopening day; the same at F on d0.
- Reported: mean, median, median absolute value, 90th percentile of the absolute value, for Mondays (d1 of
  weekends) and for other days; the regression of (B_close - B_open) on B_open (slope -1 = the opening basis is
  gone by the close, 0 = it persists); the regression of B_open on g(T1) (does a large weekend spread leave a basis
  after the open?). Hyperliquid's hourly funding "premium" (perp against the venue's reference price) is reported by
  hour of the week as a synchronous cross-check that needs no official data.

## Test 3 - tradeable rule (weekends)

At entry time T (T1 and T0 are both tested), for each contract with a valid price:
- raw signal s = g(T); market-neutral signal s = e(T).
- If |s| > X (X in 1%, 2%, 3%): FADE = position -sign(s) on the venue at P(T). FOLLOW = the opposite position.
- Market-neutral version adds the hedge: +sign(s) * beta_i of xyz:XYZ100 for FADE (opposite for FOLLOW).
- Exits: at O+ (hour of the cash open) and, separately, at C1 (Monday's close time), at the venue's price.
- Gross PnL per trade in bps of the stock-leg notional. Net = gross - fees - half-spreads on every leg, entry and
  exit - funding actually paid on the perp legs between entry and exit.
- Selection on the earlier half of the weekends (first ceil(N/2) weekends of the 1h sample): for each of the 8
  configurations (entry T1/T0 x signal neutral/raw x exit O+/C1) pick the (X, side) with the highest mean net PnL
  per trade among cells with at least 8 trades (if no cell has 8 trades: X = 1%, FADE). Evaluation: once, on the
  later half, that one cell per configuration. The primary configuration is entry T1, market-neutral, exit O+.
- The full grid (2 entries x 2 signals x 3 thresholds x 2 exits x 2 sides) is reported for both halves.
- Variants reported for the primary configuration only: hedge with xyz:SP500 instead of xyz:XYZ100; entry delayed
  one hour (signal at T, entry at T + 1h); core subset; extended 2h sample before the 1h sample (entry T1, exit
  16:00 UTC on d1, thresholds and side as selected, funding assumed at the main sample's median hourly rate).
- Binance: the same rule on the spot tokens, signal against the same official close; market-neutral signal uses
  xyz:XYZ100's move and the hedge leg is the Hyperliquid contract. A spot token cannot be sold short without
  inventory, so results are shown for all trades and for the long-only subset.

## Test 4 - weekday nights

Tests 1 to 3 with nights instead of weekends, entry time T1 only (08:00 UTC on d1), halves split by night.

## Costs (primary case)

- Hyperliquid taker fee per side: the fee the venue documentation and the API metadata give for each contract
  (HIP-3 standard tier 0 or growth mode). If that cannot be verified: 4.5 bps (the brief's fallback).
  Sensitivities always shown: 4.5 bps and 9 bps per side, and a maker case (maker fee, no half-spread).
- Binance spot taker fee 10 bps per side (not verified for these pairs unless a schedule is found).
- Half-spread per side: measured from an order-book snapshot of each contract taken during this weekend (cost of a
  10,000 USD market order against the mid), floor 1 bp; 5 bps if no snapshot. Sensitivity: 2 and 10 bps.
- Funding: Hyperliquid's hourly funding history, summed over funding times in (entry, exit].
- Capacity: venue notional traded in the entry hour (candle ending at T), reported per trade.

## Inference

- Independent unit = one weekend (or one night). All intervals are percentile intervals from a cluster bootstrap
  over weekends/nights, 10,000 resamples, seed 20261122.
- Placebo 1 (shuffled dates): the signal vectors are reassigned to other weekends/nights at random (2,000
  permutations, seed 20261122), stock by stock matched; the real statistic is located in that distribution.
- Placebo 2 (lagged): the signal of the previous weekend/night is used for the current one.
- Multiple testing: Benjamini-Hochberg across the 8 evaluation-half tests of a family; and the best cell of the
  full grid is compared with the best cell of each placebo permutation (max-statistic).
- Verdict rule: H_revert is supported only if the selected FADE rule has a positive mean net PnL on the evaluation
  half with a 95% interval above zero and beats the 95th percentile of the placebo best-of-grid. H_info is
  supported if the Test 1 slope interval contains 1 (or slope >= 0.9) and no FADE cell survives costs out of sample.
  Grades: strong (evaluation-half interval excludes zero, placebo p < 0.05, same sign in both halves and in the
  extended sample), moderate (same sign everywhere, p < 0.10), weak (same sign, interval includes zero), none.

## Not possible / known limits before starting

- No intraday official prices: the cash price is known only at the open and the close.
- No historical order books: spreads come from one weekend snapshot; hourly candle closes are last trades, not mids.
- Hyperliquid serves about 5,000 candles per interval, so the hourly sample is about 7 months (about 29 weekends).
- Contracts delisted before today may be missing from the venue's list (survivorship cannot be fully checked).

## Addendum 1 - the owner's hypothesis H1 (added 22 Nov 2026, about 11:40 UTC)

Received from the coordinator while the full download was still running. At that moment the only venue data on
disk was the smoke run (xyz:NVDA, xyz:SP500, xyz:XYZ100, NVDAB, SPYB); it had been used for format checks and for
the clock audit (venue price against the official open and close at the session times). No slope, no reversal and
no PnL had been computed on any data.

H1 (owner, verbatim): "the price of tokenized stocks will converge to the Monday opening", i.e. the token gives
back part of its weekend move, rather than the stock opening where the token already was.

H1 is the hypothesis on trial. It is H_revert above; the alternative is H_info.

- Statistic: the Test 1 primary slope b (pooled OLS, official gap on the venue's own move v(T1), all contracts,
  weekends of the 1h sample), with its 95% percentile interval from the weekend-level bootstrap (10,000 resamples,
  seed 20261122).
- Verdict on H1, three-way:
  - SUPPORTED if the whole 95% interval is below 1.
  - REJECTED in favour of the alternative if the interval contains 1 or lies above it.
  - The size is reported separately from the significance: share of the weekend move given back = 1 - b, with the
    interval (1 - upper, 1 - lower). A slope whose interval is below 1 but which is 0.9 or more is reported as
    "H1 supported in sign, economically small"; whether it is worth trading is decided only by the fade rule.
- The same statistic is reported, as already pre-specified, for T0 (inside the internal weekend session), for the
  spread g instead of v, with weekend fixed effects, for the core subset, for the index contracts, for Binance and
  for the extended 2h sample. The verdict is read from the primary; the others say how far it generalises.
- Second, direct estimate of the give-back (variant (g) above): slope of the venue's own later move,
  P(O+)/P(T1) - 1, on v(T1). Give-back share = minus that slope. Needs no official price.
- Tradeable form: Monday's open is not known in advance, so H1 can only be traded as the FADE rule of Test 3.
  The report puts side by side: the H1 slope with its interval, the give-back share, and the net PnL per trade of
  the FADE rule (primary configuration: entry T1, market-neutral, exit O+; threshold chosen on the earlier half,
  evaluated once on the later half), plus the FADE cells at fixed X = 1%, 2%, 3% in both halves.
- H1 is called TRADEABLE only if the Test 3 verdict rule above is met (later-half interval above zero and beyond
  the 95th percentile of the placebo best-of-grid).

## Clarifications fixed before the full data arrived (no outcome seen)

- beta_i falls back to 1.0 when fewer than 30 daily returns exist in the 60-session window.
- Exact-open price for B_open: the open of the 30-minute candle that starts at the cash open (first trade at or
  after 09:30:00 New York). Also reported, as description only: the close of the 30-minute candle ending at the
  open (last trade before the open).
- Hyperliquid contracts delisted by the venue (ACN, SNAP, NU, MELI, STX, SNOW, SMCI) are included while they traded.
- Binance tokens: all spot symbols in the same exchange permission group as NVDABUSDT with quote USDT and a base
  asset ending in "B" (111 symbols on 22 Nov 2026), not only the 16 named in the brief.
- Fee facts found before the tests: xyz contracts have deployerFeeScale 1.0 and, for 133 of 147, growthMode
  enabled in the API metadata, which with the venue's fee formula gives a tier-0 taker fee of 0.9 bps (maker 0.3
  bps); 9 bps (maker 3 bps) where growth mode is off. That is the primary fee case per the rule above.

## Addendum 2 - the Sunday-night reopening (added 22 Nov 2026, about 11:50 UTC, before the full data arrived)

Reason: the venue documentation says most xyz stock contracts are priced from an external market from Sunday 20:05
New York time (overnight session), and from the order book alone between Friday 20:00 and Sunday 20:05. The one
published study of the venue (Blockworks Research, sponsored) measures the give-back at that reopening, not at
Monday's cash open. If H1 is true, the documentation suggests it should show there first. The tests above already
span that moment (entry T0, exit O+) but do not isolate it. No outcome had been computed when this was added.

- Two more clock times: R- = 19:00 New York time on the calendar day before d1 (one hour before the internal
  session ends); R+ = 22:00 New York time on the same day (about two hours after external pricing resumes).
- Test 1 additions: slope of the venue's own move across the reopening, P(R+)/P(R-) - 1, on v(R-) (give-back at
  the reopening = minus the slope); official gap on v(R-) and on v(R+).
- Test 3 addition, its own family ("reopening"): entry R-, exit R+, signals g(R-) and e(R-), X in 1%, 2%, 3%,
  FADE and FOLLOW, same costs, funding over (R-, R+], selection on the earlier half and one evaluation on the later
  half, same placebos. Benjamini-Hochberg is reported within this family (2 configurations) and across the 10
  weekend configurations together.
- Descriptive only (no test): for weekends with |v(T1)| above 1% and above 2%, the median of gap / v(T1)
  ("share of the venue's move that the cash open confirmed") and the shares of observations with a ratio below 0,
  between 0 and 1, and above 1.
