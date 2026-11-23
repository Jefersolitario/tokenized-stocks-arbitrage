# Paper test of the Sunday-evening fade: rule and pass marks

Fixed on 22 Nov 2026, before the first recording (the reopening of Sunday 22 Nov 2026). No money, no orders: the
recorder saves public order books, and the trades below are worked out afterwards from the recorded prices.

Any later change to this file is added at the end as a dated addendum, before the next recording, and never applied
to weekends already recorded. A change made after looking at results is labelled post hoc.

## Why

The weekend study (PREREG.md, results/) found that part of the weekend drift of Hyperliquid's tokenized-stock
contracts reverses between 19:00 and 22:00 New York time on Sunday. In trade prices the fade below earned +52 bps a
trade after assumed costs on the later 14 of 29 weekends. The independent review confirmed the numbers and put the
forward edge at about +30 bps a trade (90% range -15 to +65), with the open question being executable prices at
19:00 and 22:00, which no past data has. This test measures exactly that.

## What is recorded

`record_quotes.py`, every Sunday from 17:00-17:30 to 22:30 New York time, for the study's 86 stock and fund
contracts and the two index contracts on the `xyz` market: order books (every 20 s around 18:00, 19:00, 20:05 and
22:00, every 60 s otherwise), mid prices every 5 s, mark and reference prices and funding every 60 s, and 1-minute
and 1-hour trade candles. Weekends whose reopening is not on a Sunday evening (a market holiday on Monday) are not
recorded and are left out of the test.

## The rule

All times are New York time on the Sunday. "Friday" is the last cash session before the weekend.

1. **Hedge ratio.** `beta` = slope of the stock's daily close-to-close return on QQQ's over the 60 sessions ending
   on Friday, official closes only, clipped to 0 to 3, set to 1.0 when fewer than 30 returns exist. (As in the study.)
2. **Signal at 19:00:00.**
   - `spread` = stock contract's mid at 19:00:00 / Friday's official close of the stock - 1.
   - `index move` = xyz:XYZ100's mid at 19:00:00 / xyz:XYZ100's price at Friday's cash close (the close of the hourly
     trade candle ending at 16:00 on Friday) - 1.
   - `signal` = `spread` - `beta` x `index move`.
   - The mid at a time is the last recorded 5-second mid at or before that time, no older than 30 s.
3. **Trade** when |`signal`| is 1% or more.
   - `signal` above +1%: sell 1,000 USD of the stock contract and buy `beta` x 1,000 USD of xyz:XYZ100.
   - `signal` below -1%: buy 1,000 USD of the stock contract and sell `beta` x 1,000 USD of xyz:XYZ100.
4. **Entry** at 19:00:00 and **exit** at 22:00:00, each as a market order against the first book snapshot recorded
   at or after that time (no later than 60 s after it). The fill price of each leg is the average price of walking
   the recorded book levels for that leg's size. A contract with no usable book at entry is not traded; one with no
   usable book at exit is closed at the next usable snapshot.
5. **Costs.** The venue's taker fee on each of the four fills, as its fee schedule gives it on that day for these
   markets at the base tier (0.9 bps in growth mode on 22 Nov 2026, 9 bps otherwise), and the funding paid or received
   on both legs between entry and exit. The spread is already in the fill prices.
6. **Result** of a trade: profit of both legs after costs, in bps of the 1,000 USD stock leg.
7. A contract is skipped for a weekend when its Friday close is missing or a split takes effect over that weekend.

This is the **primary** cell and the only one that decides. Three more are reported beside it and decide nothing:
- the signal fixed at 18:00:00 instead of 19:00:00, same entry and exit (the reviewer's cleaner signal);
- the primary without trades whose stock-leg entry cost exceeds 20 bps;
- the primary at 10,000 USD for the stock leg.

## Pass marks

- **After 3 recorded weekends, a cost check.** Take every stock-leg fill of the primary cell (entries and exits).
  - Mean cost against the mid of 15 bps a fill or less: carry on.
  - More than 25 bps a fill: stop. The edge cannot pay.
  - In between: carry on to 6 weekends and check again with the same marks.
- **After 12 recorded weekends, the result.** Mean result per trade of the primary cell, and the t-statistic of the
  per-weekend means.
  - More than +25 bps a trade with t above 2: put a pilot with real money at 1,000 USD a trade to the owner.
  - Otherwise: stop.
- **Early stop.** After 6 or more weekends, a mean result at or below zero stops the test.

These marks are strict on purpose. If the true edge is 25 to 30 bps a trade, the 12-weekend mark is passed less than
half the time; an edge that small, at about 65 USD a weekend, is not worth running.

## Assumptions that are the owner's to change

Base fee tier with no discount and no front-end fee; taker execution on all four fills; 1,000 USD for the stock leg
(about 3,100 USD gross with the hedge on average); no margin or leverage limit modelled.

## Addendum 1 (23 Nov 2026, after the first recording)

1. **An early look, at the owner's request.** On 23 Nov the first weekend was worked out with `paper_eval.py`: the
   primary cell made +58.2 bps a trade on 30 trades (median +28.6; +36 without the best trade), and its stock fills
   cost 7.2 bps on average. This is one weekend and was looked at outside the schedule above. It changes no rule and no
   pass mark; the checks after 3 and after 12 weekends stand, and this weekend is the first of them.
2. **Friday closes that arrive late.** On 23 Nov the price source had no Friday close yet for 24 of the 86 contracts,
   so the rule skipped them. A weekend is worked out again once its official closes are complete, and that later
   figure is the one that counts. Until then those contracts are shown apart, with the venue's own price at Friday's
   cash close as a stand-in (`reference = venue`), which is not part of the rule. This was written after the stand-in
   figure had been seen (+114 bps on 14 trades); it follows from the rule's use of official closes and is not a choice
   between variants.
3. **Funding** is taken from the recorded funding rate just before each hourly payment while the position is open.
   The venue's payment history is not downloaded.
4. **Recorder changes from the second weekend on; the rule is unchanged.** It keeps 20 book levels a side instead of
   8 (on the first weekend 10 of 30 trades of the 10,000 USD cell could not be filled from 8 levels). It may start
   from 16:45 and waits for 17:00. It stands aside when a recording is already running. Its files carry the number of
   the run that wrote them.
5. **How a recording is started.** GitHub started the scheduled job four and a half hours late on 22 Nov, and that
   recording was started by hand at 17:41. The schedule on main now has a start time every half hour.
