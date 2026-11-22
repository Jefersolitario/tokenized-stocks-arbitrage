"""Research only: public market-data candles for the weekend-basis study.

Runs in the research-weekend-basis workflow on its own branch. Public market-data endpoints only: no key, no account,
no orders. Requests are paced below the venues' published limits. A refusal (403, 451, persistent 429) is recorded and
the script moves on: nothing is retried through another route.

Usage: python fetch.py <phase>      phase = main | funding
Writes research/weekend-basis/data/*. Parameters are in fetch_config.json.
"""
import datetime as dt, gzip, json, pathlib, sys, time, urllib.error, urllib.parse, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data"
CFG = json.loads((HERE / "fetch_config.json").read_text(encoding="utf-8"))
UA = {"User-Agent": "research-candle-downloader/1.0 (python-urllib; public market data only)", "Accept": "application/json"}
HL = "https://api.hyperliquid.xyz/info"
BN = "https://data-api.binance.vision"
HL_BUDGET = float(CFG.get("hl_weight_per_minute", 900))   # published limit: 1200 per minute per address
HOUR = 3_600_000

# xyz contracts whose name is the ticker of a US-listed stock or ETF in the official daily data (matched offline),
# plus the two index contracts. Contracts that only share a name with a US ticker (GOLD, CL, JPY, SKHX) are left out.
US_NAMES = """TSLA NVDA HOOD INTC PLTR COIN META AAPL MSFT ORCL GOOGL AMZN AMD MU SNDK MSTR CRCL NFLX COST LLY TSM RIVN
BABA USAR CRWV URNM GME EWY EWJ HIMS DKNG LITE XLE BX MRVL RKLB DRAM CBRS EWZ ZM EBAY ARM EWT SPCX ASML BB QNT DELL IBM
AVGO NOW NBIS WDC NOK SMH BE QCOM AMAT SHAZ SKHY KSTR GEV KORU LYTE NCLD SOXL MAGS IREN NET CRWD RDDT AAOI MRNA XBI BMNR
SNXX CVX TLT LRCX UMC TWST GLW TTWO MUU ICE PANW""".split()
INDEX_NAMES = ["XYZ100", "SP500"]
# tickers that may exist as Binance stock tokens (base asset = ticker + "B")
BN_TICKERS = sorted(set(US_NAMES) | set("SPY QQQ CRM IWM DIA GLD SLV UBER SHOP ADBE JPM V MA WMT DIS BA KO PEP XOM UNH JNJ PFE "
                                        "BRKB NKE MCD SBUX PYPL SQ ABNB SNOW TSMC CSCO TXN ON MRNA LMT GS MS C BAC WFC".split()))

status = {"started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "config": CFG, "errors": [],
          "hl_requests": 0, "hl_weight": 0, "bn_requests": 0, "refused": []}
_stop = {"hl": False, "bn": False}


def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def http_json(url, body=None, note=""):
    """One request with polite retries. Returns (object or None, http status or error name)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = dict(UA, **({"Content-Type": "application/json"} if body is not None else {}))
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=60) as r:
                return json.loads(r.read().decode()), 200
        except urllib.error.HTTPError as e:
            last = e.code
            txt = e.read()[:200]
            if e.code in (403, 418, 451):                       # refused: do not insist
                status["errors"].append(f"{note or url}: HTTP {e.code} {txt!r}")
                status["refused"].append(f"{note or url}: HTTP {e.code}")
                return None, e.code
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(20 * (attempt + 1)); continue
            status["errors"].append(f"{note or url}: HTTP {e.code} {txt!r}")
            if e.code == 429:
                status["refused"].append(f"{note or url}: persistent HTTP 429")
            return None, e.code
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
            if attempt < 3:
                time.sleep(5 * (attempt + 1)); continue
            status["errors"].append(f"{note or url}: {type(e).__name__}: {e}")
    return None, last


def hl(body, note, items_per_weight=None, base_weight=20):
    """Hyperliquid info request, paced by the published weight rules."""
    if _stop["hl"]:
        return None
    got, code = http_json(HL, body, note)
    n = len(got) if isinstance(got, list) else 0
    w = base_weight + (n // items_per_weight if items_per_weight else 0)
    status["hl_requests"] += 1; status["hl_weight"] += w
    time.sleep(w * 60.0 / HL_BUDGET)
    if got is None and code in (403, 451, 429):
        _stop["hl"] = True
        log("Hyperliquid refused", code, "- stopping Hyperliquid requests for this phase")
    return got


def bn(path, params, note):
    if _stop["bn"]:
        return None
    got, code = http_json(f"{BN}{path}?{urllib.parse.urlencode(params)}" if params else f"{BN}{path}", note=note)
    status["bn_requests"] += 1
    time.sleep(0.25)
    if got is None and code in (403, 418, 451, 429):
        _stop["bn"] = True
        log("Binance refused", code, "- stopping Binance requests for this phase")
    return got


def write_parquet(path, cols, types):
    import pyarrow as pa, pyarrow.parquet as pq
    arrays = {k: pa.array(v, type=types[k]) for k, v in cols.items()}
    table = pa.table(arrays)
    f32 = [k for k, t in types.items() if t == pa.float32()]
    pq.write_table(table, path, compression="zstd", compression_level=19, use_dictionary=[k for k in cols if k not in f32],
                   column_encoding={k: "BYTE_STREAM_SPLIT" for k in f32} or None, data_page_size=1 << 20)
    log("wrote", path.name, table.num_rows, "rows", round(path.stat().st_size / 1e6, 2), "MB")


def write_json(path, obj, gz=False):
    txt = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write(txt)
    else:
        path.write_text(txt + "\n", encoding="utf-8")
    log("wrote", path.name, round(path.stat().st_size / 1e6, 3), "MB")


def now_ms():
    return int(time.time() * 1000)


def hl_universe():
    meta = hl({"type": "meta", "dex": "xyz"}, "hl meta xyz")
    uni = (meta or {}).get("universe", [])
    names = [u.get("name") for u in uni if isinstance(u, dict)]
    want = {f"xyz:{n}" for n in US_NAMES + INDEX_NAMES}
    coins = [n for n in names if n in want]
    coins.sort(key=lambda c: (c.split(":")[1] not in INDEX_NAMES, c))
    if CFG.get("mode") == "smoke":
        coins = [c for c in coins if c.split(":")[1] in ("XYZ100", "SP500", "NVDA")]
    return meta, names, coins


def hl_candles(coin, interval, start, end):
    """All candles the venue serves in [start, end]; pages forward if a response stops early."""
    step = {"30m": HOUR // 2, "1h": HOUR, "2h": 2 * HOUR, "4h": 4 * HOUR}[interval]
    out, cur = {}, start
    for _ in range(6):
        got = hl({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": cur, "endTime": end}},
                 f"hl candles {coin} {interval}", items_per_weight=60)
        if not got:
            break
        new = 0
        for c in got:
            if isinstance(c, dict) and "t" in c and c["t"] not in out:
                out[c["t"]] = c; new += 1
        last = max(out)
        if new == 0 or last >= end - 2 * step:
            break
        cur = last + 1
    return [out[k] for k in sorted(out)]


def books_hl(coins, tag):
    snap = {"taken_utc_ms_start": now_ms(), "tag": tag, "books": {}}
    for coin in coins:
        got = hl({"type": "l2Book", "coin": coin}, f"hl l2Book {coin}", base_weight=2)
        if isinstance(got, dict) and "levels" in got:
            snap["books"][coin] = {"time": got.get("time"), "bids": [[x["px"], x["sz"], x.get("n")] for x in got["levels"][0]],
                                   "asks": [[x["px"], x["sz"], x.get("n")] for x in got["levels"][1]]}
    snap["taken_utc_ms_end"] = now_ms()
    return snap


def phase_main():
    import pyarrow as pa
    OUT.mkdir(parents=True, exist_ok=True)
    t_now = now_ms()

    # ---------------- Hyperliquid metadata
    dexs = hl({"type": "perpDexs"}, "hl perpDexs")
    xyz = [d for d in (dexs or []) if isinstance(d, dict) and d.get("name") == "xyz"]
    meta, names, coins = hl_universe()
    ctx = hl({"type": "metaAndAssetCtxs", "dex": "xyz"}, "hl metaAndAssetCtxs xyz")
    write_json(OUT / "hl_meta_xyz.json", {"fetched_utc_ms": t_now, "perpDex": xyz, "meta": meta, "metaAndAssetCtxs": ctx})
    status["hl_universe_size"] = len(names); status["hl_coins"] = coins
    log("xyz universe", len(names), "selected", len(coins))
    if meta:
        log("meta sample:", json.dumps(meta["universe"][:2])[:600])

    # ---------------- Binance: which stock tokens exist, klines, books
    info = bn("/api/v3/exchangeInfo", None, "binance exchangeInfo")
    syms = (info or {}).get("symbols", [])
    status["binance_symbols_total"] = len(syms)

    def key(s):
        return json.dumps([s.get("permissionSets"), s.get("permissions")], sort_keys=True)
    ref = [s for s in syms if s["symbol"] in ("NVDABUSDT", "TSLABUSDT")]
    ref_keys = {key(s) for s in ref}
    same = [s for s in syms if key(s) in ref_keys]
    distinctive = 0 < len(same) <= 150
    stock = [s for s in syms if s.get("quoteAsset") == "USDT" and s.get("baseAsset", "").endswith("B")
             and (s["baseAsset"][:-1] in BN_TICKERS or (distinctive and key(s) in ref_keys))]
    status["binance_ref_group_size"] = len(same); status["binance_group_distinctive"] = distinctive
    status["binance_stock_symbols"] = [[s["symbol"], s["status"]] for s in stock]
    if syms:
        with gzip.open(OUT / "bn_symbols.csv.gz", "wt", encoding="utf-8", compresslevel=9) as f:
            f.write("symbol,base,quote,status\n")
            for s in syms:
                f.write(f"{s['symbol']},{s.get('baseAsset','')},{s.get('quoteAsset','')},{s.get('status','')}\n")
        write_json(OUT / "bn_exchange_info_stock.json", {"fetched_utc_ms": t_now, "serverTime": info.get("serverTime"), "symbols": stock})
    log("binance symbols", len(syms), "stock tokens", [s["symbol"] for s in stock])
    bn_syms = [s["symbol"] for s in stock]
    if CFG.get("mode") == "smoke":
        bn_syms = [s for s in bn_syms if s in ("NVDABUSDT", "SPYBUSDT")]
    bn_start = int(dt.datetime.strptime(CFG.get("binance_start", "2026-05-01"), "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    for iv in ("1h", "30m"):
        cols = {k: [] for k in ("symbol", "t", "o", "h", "l", "c", "v", "qv", "n")}
        for sym in bn_syms + ["USDCUSDT"]:
            cur, rows = bn_start, 0
            while cur < t_now:
                got = bn("/api/v3/klines", {"symbol": sym, "interval": iv, "startTime": cur, "endTime": t_now, "limit": 1000}, f"binance klines {sym} {iv}")
                if not got:
                    break
                for k in got:
                    cols["symbol"].append(sym); cols["t"].append(int(k[0]))
                    for name, j in (("o", 1), ("h", 2), ("l", 3), ("c", 4), ("v", 5), ("qv", 7)):
                        cols[name].append(float(k[j]))
                    cols["n"].append(int(k[8]))
                rows += len(got)
                if len(got) < 1000:
                    break
                cur = int(got[-1][0]) + 1
            status.setdefault("binance_rows", {})[f"{sym} {iv}"] = rows
        if cols["t"]:
            write_parquet(OUT / f"bn_klines_{iv}.parquet", cols, {"symbol": pa.string(), "t": pa.int64(), "o": pa.float64(), "h": pa.float64(),
                          "l": pa.float64(), "c": pa.float64(), "v": pa.float64(), "qv": pa.float64(), "n": pa.int32()})
    bsnap = {"taken_utc_ms_start": now_ms(), "books": {}}
    for sym in bn_syms:
        got = bn("/api/v3/depth", {"symbol": sym, "limit": 100}, f"binance depth {sym}")
        if isinstance(got, dict) and "bids" in got:
            bsnap["books"][sym] = {"lastUpdateId": got.get("lastUpdateId"), "bids": got["bids"], "asks": got["asks"]}
    bsnap["taken_utc_ms_end"] = now_ms()
    tick = bn("/api/v3/ticker/24hr", {"symbols": json.dumps(bn_syms, separators=(",", ":"))}, "binance 24hr ticker") if bn_syms else None
    bsnap["ticker24h"] = tick

    # ---------------- Hyperliquid: order books (weekend snapshot), candles, order books again
    snaps = [books_hl(coins, "before candles")]
    spec = [("1h", 5200, ("o", "h", "l", "c", "v", "n")), ("30m", 5200, ("o", "c", "v", "n")), ("2h", 5200, ("o", "c", "v", "n"))]
    for iv, back, fields in spec:
        step = {"30m": HOUR // 2, "1h": HOUR, "2h": 2 * HOUR}[iv]
        cols = {k: [] for k in ("coin", "t") + fields}
        for i, coin in enumerate(coins):
            got = hl_candles(coin, iv, t_now - back * step, t_now)
            kept = 0
            for c in got:
                if iv == "30m":                                   # keep only the half-hours around the cash open and close
                    hod = (c["t"] % 86_400_000) / HOUR
                    if not (12 <= hod < 16 or 19 <= hod < 22):
                        continue
                cols["coin"].append(coin); cols["t"].append(int(c["t"]))
                for k in fields:
                    cols[k].append(int(c.get("n", 0)) if k == "n" else float(c[k]))
                kept += 1
            status.setdefault("hl_rows", {})[f"{coin} {iv}"] = [len(got), kept, got[0]["t"] if got else None, got[-1]["t"] if got else None]
            if i < 2 and got:
                log("sample", coin, iv, json.dumps(got[0]), "...", json.dumps(got[-1]))
        if cols["t"]:
            types = {"coin": pa.string(), "t": pa.int64(), "n": pa.int32(), "v": pa.float64()}
            types.update({k: pa.float32() for k in ("o", "h", "l", "c") if k in fields})
            write_parquet(OUT / f"hl_candles_{iv}.parquet", {k: cols[k] for k in ("coin", "t") + fields}, {k: types[k] for k in ("coin", "t") + fields})
        log("done candles", iv, "requests so far", status["hl_requests"], "weight", status["hl_weight"])
    snaps.append(books_hl(coins, "after candles"))
    write_json(OUT / "hl_books.json.gz", snaps, gz=True)
    write_json(OUT / "bn_books.json.gz", bsnap, gz=True)


def phase_funding():
    import pyarrow as pa
    OUT.mkdir(parents=True, exist_ok=True)
    t_now = now_ms()
    _, _, coins = hl_universe()
    start = t_now - int(CFG.get("funding_hours_back", 5200)) * HOUR
    if CFG.get("mode") == "smoke":
        start = t_now - 30 * 24 * HOUR
    cols = {k: [] for k in ("coin", "t", "rate", "premium")}
    for coin in coins:
        cur, rows = start, 0
        for _ in range(60):
            got = hl({"type": "fundingHistory", "coin": coin, "startTime": cur, "endTime": t_now}, f"hl funding {coin}", items_per_weight=20)
            if not got:
                break
            for f in got:
                cols["coin"].append(coin); cols["t"].append(int(f["time"]))
                cols["rate"].append(float(f.get("fundingRate", "nan"))); cols["premium"].append(float(f.get("premium", "nan")))
            rows += len(got)
            last = int(got[-1]["time"])
            if len(got) < 500 or last >= t_now - HOUR:
                break
            cur = last + 1
        status.setdefault("funding_rows", {})[coin] = rows
    if cols["t"]:
        write_parquet(OUT / "hl_funding.parquet", cols, {"coin": pa.string(), "t": pa.int64(), "rate": pa.float64(), "premium": pa.float64()})
    write_json(OUT / "hl_books_after_funding.json.gz", [books_hl(coins, "after funding")], gz=True)


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "main"
    try:
        {"main": phase_main, "funding": phase_funding}[phase]()
    finally:
        status["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        status["errors"] = status["errors"][:300]
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"status_{phase}.json").write_text(json.dumps(status, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        log("phase", phase, "hl requests", status["hl_requests"], "hl weight", status["hl_weight"], "binance requests", status["bn_requests"],
            "errors", len(status["errors"]), "refused", status["refused"][:5])
        for e in status["errors"][:20]:
            log("  ", e[:240])
    return 0


if __name__ == "__main__":
    sys.exit(main())
