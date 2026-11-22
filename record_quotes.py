"""Research only: records public order books of Hyperliquid's tokenized-stock contracts on Sunday evenings, New York time.

Why: the weekend study found that part of the weekend drift of these contracts reverses between 19:00 and 22:00
New York time on Sunday. Whether that can be traded depends on executable prices, which no backtest has; this script
saves them for the paper test in PAPER_RULE.md. Public market-data endpoint only: no key, no account, no orders.
Requests stay well below the venue's published limit (1,200 weight a minute). A refusal (403, 451) stops the run:
nothing is retried through another route.

Usage:
  python record_quotes.py              full run: only on a Sunday between 17:00 and 22:30 New York time, until 22:30
  python record_quotes.py --force      record from now until 22:30 New York today whatever the weekday
  python record_quotes.py --smoke 150  short test run of 150 seconds, any time
  python record_quotes.py --selftest   offline checks of the arithmetic, no network
  --git-push                           commit and push the finished files every 30 minutes and at the end

Writes research/weekend-basis/quotes/<New York date>/ :
  books_NNN.csv.gz    one row per contract and snapshot: best bid and ask, cost of market orders of 1,000 to 30,000 USD
                      against the mid, depth, and the first levels of the book. Every 20 s around 18:00, 19:00, 20:05
                      and 22:00 New York, every 60 s otherwise.
  mids_NNN.csv.gz     mid price of every contract every 5 s (one row per time, one column per contract)
  ctx_NNN.csv.gz      mark price, oracle (reference) price, funding, open interest and impact prices every 60 s
  candles_1m_NNN.csv.gz  1-minute trade candles of the recording window (written at the end)
  candles_1h_NNN.csv.gz  1-hour trade candles of the previous 110 hours (for the signal's reference prices)
  run_NNN.json        what was recorded, by which version of this file, and the error counts
"""
import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import gzip
import json
import pathlib
import subprocess
import threading
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

VERSION = "1.0"
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE / "quotes"
BRANCH = "claude/research-weekend-basis"
HL = "https://api.hyperliquid.xyz/info"
UA = {"User-Agent": "weekend-quotes-recorder/1.0 (python-urllib; public market data only; research)",
      "Accept": "application/json", "Content-Type": "application/json"}
NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
DEX = "xyz"

# The study's universe, fixed before the first recording: xyz contracts named after a US-listed stock or fund, and the
# two index contracts. Contracts the venue has delisted are skipped at run time; later listings are not added.
US_NAMES = """TSLA NVDA HOOD INTC PLTR COIN META AAPL MSFT ORCL GOOGL AMZN AMD MU SNDK MSTR CRCL NFLX COST LLY TSM RIVN
BABA USAR CRWV URNM GME EWY EWJ HIMS DKNG LITE XLE BX MRVL RKLB DRAM CBRS EWZ ZM EBAY ARM EWT SPCX ASML BB QNT DELL IBM
AVGO NOW NBIS WDC NOK SMH BE QCOM AMAT SHAZ SKHY KSTR GEV KORU LYTE NCLD SOXL MAGS IREN NET CRWD RDDT AAOI MRNA XBI BMNR
SNXX CVX TLT LRCX UMC TWST GLW TTWO MUU ICE PANW""".split()
INDEX_NAMES = ["XYZ100", "SP500"]

SIZES = (1_000, 3_000, 10_000, 30_000)          # USD of a market order: stock leg and hedge leg, small and large
LEVELS = 8                                      # book levels kept per side
START_NY, END_NY = (17, 0), (22, 30)            # a full run may start from 17:00 and always ends at 22:30 New York
KEY_WINDOWS = (((17, 55), (18, 10)), ((18, 55), (19, 10)), ((19, 55), (20, 15)), ((21, 55), (22, 10)))
FAST, SLOW = 20.0, 60.0                         # seconds between book snapshots inside and outside the key windows
MIDS_EVERY, CTX_EVERY, PUSH_EVERY = 5.0, 60.0, 1800.0
WORKERS = 4

stats = {"requests": 0, "errors": 0, "http_429": 0, "refused": 0, "book_rows": 0, "mid_rows": 0, "ctx_rows": 0,
         "cycles": 0, "notes": []}
state = {"penalty_until": 0.0, "slow_until": 0.0, "stop": False}
lock = threading.Lock()


def log(*a):
    print(dt.datetime.now(UTC).strftime("%H:%M:%SZ"), *a, flush=True)


def now_ms():
    return int(time.time() * 1000)


def post(body, timeout=10, attempts=2):
    """One info request. Returns the parsed answer or None. Backs off on HTTP 429 and stops the run on a refusal."""
    for k in range(attempts):
        if state["stop"]:
            return None
        wait = state["penalty_until"] - time.time()
        if wait > 0:
            time.sleep(min(wait, 60.0))
        try:
            req = urllib.request.Request(HL, data=json.dumps(body).encode(), headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode())
            with lock:
                stats["requests"] += 1
            return out
        except urllib.error.HTTPError as e:
            with lock:
                stats["requests"] += 1
                stats["errors"] += 1
                if e.code == 429:                           # too fast: pause everything, then go slower for a while
                    stats["http_429"] += 1
                    state["penalty_until"] = time.time() + 30.0
                    state["slow_until"] = time.time() + 300.0
                if e.code in (403, 418, 451):               # refused: do not insist
                    stats["refused"] += 1
                    state["stop"] = True
                    stats["notes"].append(f"refused with HTTP {e.code} on {body.get('type')}; run stopped")
            if e.code in (403, 418, 451):
                return None
            time.sleep(1.0 + k)
        except Exception:  # noqa: BLE001  (timeouts, resets, bad JSON)
            with lock:
                stats["errors"] += 1
            time.sleep(1.0 + k)
    return None


class PartWriter:
    """Gzip CSV written in numbered parts, so that finished parts can be committed while the run goes on."""

    def __init__(self, folder, stem, header):
        self.folder, self.stem, self.header = folder, stem, header
        self.lock = threading.Lock()
        have = [int(p.name[len(stem) + 1:len(stem) + 4]) for p in folder.glob(f"{stem}_[0-9][0-9][0-9].csv.gz")]
        self.part = max(have, default=-1)           # never overwrite the parts of an earlier run the same evening
        self.rows = 0
        self._open()

    def _open(self):
        self.part += 1
        self.rows = 0
        self.f = gzip.open(self.folder / f"{self.stem}_{self.part:03d}.csv.gz", "wt", encoding="utf-8",
                           newline="", compresslevel=6)
        self.w = csv.writer(self.f)
        self.w.writerow(self.header)

    def write(self, row):
        with self.lock:
            self.w.writerow(row)
            self.rows += 1

    @property
    def path(self):
        return self.folder / f"{self.stem}_{self.part:03d}.csv.gz"

    def rotate(self):
        with self.lock:
            if self.rows:
                self.f.close()
                self._open()

    def close(self):
        with self.lock:
            self.f.close()


def walk(levels, usd):
    """Average price of a market order of `usd` notional against one side of the book; None if the side is too thin."""
    left, qty = float(usd), 0.0
    for px, sz in levels:
        take = min(left, px * sz)
        qty += take / px
        left -= take
        if left <= 1e-9:
            return usd / qty
    return None


BOOK_HEADER = (["recv_ms", "book_ms", "coin", "bid", "ask", "bid_sz", "ask_sz", "n_bids", "n_asks"]
               + [f"{side}_{usd}_bps" for usd in SIZES for side in ("buy", "sell")]
               + ["bid_usd_25bps", "ask_usd_25bps", "bid_usd_all", "ask_usd_all", "bids", "asks"])


def book_row(coin, got, recv_ms):
    """One CSV row from an l2Book answer. Costs are in bps against the mid: positive = worse than the mid."""
    raw_b, raw_a = got["levels"][0], got["levels"][1]
    bids = [(float(x["px"]), float(x["sz"])) for x in raw_b]
    asks = [(float(x["px"]), float(x["sz"])) for x in raw_a]
    row = [recv_ms, got.get("time", ""), coin]
    if not bids or not asks:
        return row + [raw_b[0]["px"] if raw_b else "", raw_a[0]["px"] if raw_a else "",
                      raw_b[0]["sz"] if raw_b else "", raw_a[0]["sz"] if raw_a else "", len(bids), len(asks)] \
            + [""] * (2 * len(SIZES) + 4) \
            + ["|".join(f"{x['px']}:{x['sz']}" for x in raw_b[:LEVELS]), "|".join(f"{x['px']}:{x['sz']}" for x in raw_a[:LEVELS])]
    mid = (bids[0][0] + asks[0][0]) / 2.0
    row += [raw_b[0]["px"], raw_a[0]["px"], raw_b[0]["sz"], raw_a[0]["sz"], len(bids), len(asks)]
    for usd in SIZES:
        buy, sell = walk(asks, usd), walk(bids, usd)
        row += ["" if buy is None else f"{(buy / mid - 1.0) * 1e4:.2f}", "" if sell is None else f"{(1.0 - sell / mid) * 1e4:.2f}"]
    row += [f"{sum(p * s for p, s in bids if p >= mid * 0.9975):.0f}", f"{sum(p * s for p, s in asks if p <= mid * 1.0025):.0f}",
            f"{sum(p * s for p, s in bids):.0f}", f"{sum(p * s for p, s in asks):.0f}",
            "|".join(f"{x['px']}:{x['sz']}" for x in raw_b[:LEVELS]), "|".join(f"{x['px']}:{x['sz']}" for x in raw_a[:LEVELS])]
    return row


def in_key_window(t_ny):
    hm = (t_ny.hour, t_ny.minute)
    return any(a <= hm < b for a, b in KEY_WINDOWS)


def universe():
    """Wanted contracts that the venue lists now, index contracts first. Returns (coins, note)."""
    meta = post({"type": "meta", "dex": DEX}, timeout=20, attempts=3)
    uni = (meta or {}).get("universe", [])
    listed = {u.get("name"): u for u in uni if isinstance(u, dict)}
    coins, skipped = [], []
    for name in INDEX_NAMES + US_NAMES:
        coin = f"{DEX}:{name}"
        if coin in listed and not listed[coin].get("isDelisted"):
            coins.append(coin)
        else:
            skipped.append(name)
    return coins, {"listed_on_dex": len(listed), "recorded": len(coins), "not_listed_or_delisted": skipped}


def mids_loop(coins, w, end_ts):
    fails, nxt = 0, time.time()
    while time.time() < end_ts and not state["stop"]:
        got = post({"type": "allMids", "dex": DEX}, timeout=6, attempts=1)
        if isinstance(got, dict) and any(c in got for c in coins[:5]):
            w.write([now_ms()] + [got.get(c, "") for c in coins])
            with lock:
                stats["mid_rows"] += 1
            fails = 0
        else:
            fails += 1
            if fails == 60:                             # five minutes without an answer: stop asking
                with lock:
                    stats["notes"].append("allMids for the dex stopped answering; mids remain in ctx every minute")
                log("mids: no usable answer for 60 polls, stopping this stream")
                return
        nxt = max(nxt + MIDS_EVERY, time.time())
        time.sleep(max(0.0, nxt - time.time()))


def ctx_loop(coins, w, end_ts):
    want, nxt = set(coins), time.time()
    while time.time() < end_ts and not state["stop"]:
        got = post({"type": "metaAndAssetCtxs", "dex": DEX}, timeout=10, attempts=2)
        t = now_ms()
        try:
            names = [u["name"] for u in got[0]["universe"]]
            for name, c in zip(names, got[1]):
                if name in want:
                    imp = c.get("impactPxs") or ["", ""]
                    w.write([t, name, c.get("markPx", ""), c.get("oraclePx", ""), c.get("midPx", ""), c.get("funding", ""),
                             c.get("openInterest", ""), c.get("premium", ""), imp[0], imp[1], c.get("dayNtlVlm", "")])
                    with lock:
                        stats["ctx_rows"] += 1
        except Exception:  # noqa: BLE001
            with lock:
                stats["errors"] += 1
        nxt = max(nxt + CTX_EVERY, time.time())
        time.sleep(max(0.0, min(nxt, end_ts) - time.time()))


def book_cycle(coins, w, pool):
    futs = {pool.submit(post, {"type": "l2Book", "coin": c}, 8, 2): c for c in coins}
    n = 0
    for fut in cf.as_completed(futs):
        got = fut.result()
        if isinstance(got, dict) and isinstance(got.get("levels"), list) and len(got["levels"]) == 2:
            try:
                w.write(book_row(futs[fut], got, now_ms()))
                n += 1
            except Exception:  # noqa: BLE001
                with lock:
                    stats["errors"] += 1
    with lock:
        stats["book_rows"] += n
        stats["cycles"] += 1
    return n


def candles(coins, interval, start_ms, end_ms, path):
    n = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6) as f:
        w = csv.writer(f)
        w.writerow(["coin", "interval", "t", "T", "o", "h", "l", "c", "v", "n"])
        for coin in coins:
            if state["stop"]:
                break
            got = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ms,
                                                          "endTime": end_ms}}, timeout=20, attempts=3)
            for k in got if isinstance(got, list) else []:
                w.writerow([coin, interval, k.get("t"), k.get("T"), k.get("o"), k.get("h"), k.get("l"), k.get("c"), k.get("v"), k.get("n")])
                n += 1
            time.sleep(1.5)                             # about 25 weight a request: under 1,100 a minute
    return n


def git_push(message, still_open=()):
    """Commit the finished files and push them. Files still being written are left out until they are closed."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True)
    run("add", "--", ROOT.name, *[f":(exclude){p.relative_to(HERE).as_posix()}" for p in still_open])
    if run("diff", "--cached", "--quiet").returncode == 0:
        return True
    if run("commit", "-q", "-m", message).returncode != 0:
        log("git commit failed")
        return False
    for _ in range(3):
        run("pull", "-q", "--rebase", "--autostash", "origin", BRANCH)
        p = run("push", "-q", "origin", f"HEAD:{BRANCH}")
        if p.returncode == 0:
            return True
        time.sleep(5.0)
    log("git push failed:", p.stderr.strip()[-300:])
    return False


def selftest():
    asks = [(100.0, 5.0), (101.0, 10.0)]                # 500 USD at 100, then 1,010 USD at 101
    bids = [(99.0, 5.0), (98.0, 10.0)]
    assert walk(asks, 500) == 100.0
    assert abs(walk(asks, 1000) - 1000 / (5 + 500 / 101)) < 1e-12
    assert walk(asks, 2000) is None
    got = {"time": 1, "levels": [[{"px": "99", "sz": "5"}, {"px": "98", "sz": "10"}], [{"px": "100", "sz": "5"}, {"px": "101", "sz": "10"}]]}
    row = dict(zip(BOOK_HEADER, book_row("xyz:T", got, 2)))
    mid = 99.5
    assert row["bid"] == "99" and row["ask"] == "100" and row["n_bids"] == 2
    assert row["buy_1000_bps"] == f"{(walk(asks, 1000) / mid - 1) * 1e4:.2f}"
    assert row["sell_1000_bps"] == f"{(1 - walk(bids, 1000) / mid) * 1e4:.2f}"
    assert float(row["buy_1000_bps"]) > 50 and float(row["sell_1000_bps"]) > 50     # half-spread is 50.25 bps here
    assert row["buy_3000_bps"] == "" and row["sell_30000_bps"] == ""
    assert row["bids"] == "99:5|98:10" and row["asks"] == "100:5|101:10"
    assert len(book_row("xyz:T", {"levels": [[], []]}, 2)) == len(BOOK_HEADER)
    d = dt.datetime(2026, 11, 22, 18, 57, tzinfo=NY)
    assert in_key_window(d) and in_key_window(d.replace(hour=20, minute=14)) and not in_key_window(d.replace(hour=20, minute=15))
    assert not in_key_window(d.replace(hour=19, minute=30))
    assert len(set(US_NAMES)) == len(US_NAMES) == 86
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--git-push", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0

    now = dt.datetime.now(NY)
    if args.smoke:
        end = now + dt.timedelta(seconds=args.smoke)
        folder = ROOT / "smoke" / now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    else:
        start = now.replace(hour=START_NY[0], minute=START_NY[1], second=0, microsecond=0)
        end = now.replace(hour=END_NY[0], minute=END_NY[1], second=0, microsecond=0)
        if now >= end or not (args.force or (now.weekday() == 6 and now >= start)):
            log(f"New York time is {now:%a %H:%M}. A full run starts only on a Sunday between 17:00 and 22:30: nothing to record.")
            return 0
        end = min(end, now + dt.timedelta(hours=5, minutes=36))      # stay inside the job's time limit
        folder = ROOT / now.strftime("%Y-%m-%d")
    end_ts = end.timestamp()
    folder.mkdir(parents=True, exist_ok=True)

    coins, note = universe()
    if len(coins) < 3:
        log("could not read the venue's contract list; nothing recorded", note)
        return 1
    log(f"recording {len(coins)} of {len(INDEX_NAMES + US_NAMES)} wanted contracts until {end:%H:%M} New York into "
        f"{folder.relative_to(HERE)}; not listed now: {' '.join(note['not_listed_or_delisted']) or 'none'}")
    books = PartWriter(folder, "books", BOOK_HEADER)
    mids = PartWriter(folder, "mids", ["recv_ms"] + coins)
    ctx = PartWriter(folder, "ctx", ["recv_ms", "coin", "mark", "oracle", "mid", "funding", "open_interest", "premium",
                                     "impact_bid", "impact_ask", "day_usd_volume"])
    tag = f"{books.part:03d}"                    # first part number of this run: names its run file and its candles
    run_file = folder / f"run_{tag}.json"
    info = {"version": VERSION, "mode": "smoke" if args.smoke else "full", "started_utc": dt.datetime.now(UTC).isoformat(timespec="seconds"),
            "planned_end_utc": end.astimezone(UTC).isoformat(timespec="seconds"), "new_york_date": f"{now:%Y-%m-%d}",
            "coins": coins, "universe": note, "sizes_usd": SIZES, "levels": LEVELS, "key_windows_new_york": KEY_WINDOWS,
            "seconds_between_books": {"key_windows": FAST, "otherwise": SLOW}, "seconds_between_mids": MIDS_EVERY}

    def save_info(**more):
        with lock:
            snap = dict(stats)
        run_file.write_text(json.dumps(dict(info, **more, stats=snap), indent=1), encoding="utf-8")

    save_info()
    threads = [threading.Thread(target=mids_loop, args=(coins, mids, end_ts), daemon=True),
               threading.Thread(target=ctx_loop, args=(coins, ctx, end_ts), daemon=True)]
    for t in threads:
        t.start()
    t_start_ms, last_push, last_log = now_ms(), time.time(), time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while time.time() < end_ts and not state["stop"]:
            t0 = time.time()
            n = book_cycle(coins, books, pool)
            step = FAST if (args.smoke or in_key_window(dt.datetime.now(NY))) else SLOW
            if time.time() < state["slow_until"]:
                step *= 2.0
            if time.time() - last_log > 600 or stats["cycles"] == 1:
                last_log = time.time()
                log(f"cycle {stats['cycles']}: {n}/{len(coins)} books in {time.time() - t0:.1f}s; requests {stats['requests']}, "
                    f"errors {stats['errors']}, 429s {stats['http_429']}")
            if args.git_push and not args.smoke and time.time() - last_push > PUSH_EVERY:
                last_push = time.time()
                for w in (books, mids, ctx):
                    w.rotate()
                save_info()
                git_push(f"research(weekend-quotes): order books {info['new_york_date']}, part {books.part - 1}",
                         still_open=[w.path for w in (books, mids, ctx)])
            time.sleep(max(0.0, min(t0 + step, end_ts) - time.time()))
    for t in threads:
        t.join(timeout=15)
    for w in (books, mids, ctx):
        w.close()

    t_end_ms = now_ms()
    some = coins[:3] if args.smoke else coins
    n1 = candles(some, "1m", t_start_ms - 15 * 60_000, t_end_ms, folder / f"candles_1m_{tag}.csv.gz")
    n2 = candles(some, "1h", t_end_ms - 110 * 3_600_000, t_end_ms, folder / f"candles_1h_{tag}.csv.gz")
    save_info(ended_utc=dt.datetime.now(UTC).isoformat(timespec="seconds"), candles_1m=n1, candles_1h=n2)
    log(f"done: {stats['book_rows']} book rows in {stats['cycles']} cycles, {stats['mid_rows']} mid rows, {stats['ctx_rows']} context rows, "
        f"{n1} + {n2} candles; requests {stats['requests']}, errors {stats['errors']}, 429s {stats['http_429']}, refused {stats['refused']}")
    if args.git_push:
        git_push(f"research(weekend-quotes): {'smoke test' if args.smoke else 'order books'} {info['new_york_date']}, final")
    return 1 if (state["stop"] or stats["book_rows"] == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
