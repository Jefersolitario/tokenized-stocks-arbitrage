"""Builds the official daily open/close extract used by the study from a local copy of public Yahoo daily bars.

Usage (the source directory is third-party data: read only, parsed as text, nothing in it is executed):
    python -I official.py /home/claude/alm0421/alm0421/data/prices official/daily.csv.gz

Yahoo's open and close are adjusted backwards for splits; the venues' prices are not. `open` and `close` written here
are un-adjusted to the prices as traded on the day (file value x product of the ratios of all later splits).
`open_file` and `close_file` keep the file's values (used only for return-based betas).
"""
import csv, gzip, pathlib, sys

START = "2025-06-01"
TICKERS = """TSLA NVDA HOOD INTC PLTR COIN META AAPL MSFT ORCL GOOGL AMZN AMD MU SNDK MSTR CRCL NFLX COST LLY TSM RIVN
BABA USAR CRWV URNM GME EWY EWJ HIMS DKNG LITE XLE BX MRVL RKLB DRAM CBRS EWZ ZM EBAY ARM EWT SPCX ASML BB QNT DELL IBM
AVGO NOW NBIS WDC NOK SMH BE QCOM AMAT SHAZ SKHY KSTR GEV KORU LYTE NCLD SOXL MAGS IREN NET CRWD RDDT AAOI MRNA XBI BMNR
SNXX CVX TLT LRCX UMC TWST GLW TTWO MUU ICE PANW SPY QQQ ^GSPC ^NDX CRM""".split()
TICKERS += "ACN SNAP NU MELI STX SNOW SMCI".split()          # US-stock contracts Hyperliquid has delisted
TICKERS += """ADBE ALAB AMC APP ASTS AXTI BSP COHR CRDO CYPH DJT EWZS FLNC FWDI GS HPE INTW IWM JPM MDB MVLL PDD PENG PYPL
SECZ SKDD SKUU SOXS SQQQ TMF TQQQ USDE UVXY VST WEN BRK-B""".split()   # further Binance stock tokens
# splits the file does not carry (source: the brief; venue notice: positions settled at the 19 Nov 2026 close, 8-for-1)
EXTRA_SPLITS = {"SNDK": [("2026-11-20", 8.0)]}


def main(src, dst, extra_tickers=()):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows_out, notes, splits_out = [], [], []
    for t in sorted(set(TICKERS) | set(extra_tickers)):
        f = src / f"{t}.csv"
        if not f.is_file():
            notes.append(f"{t}: no file"); continue
        with open(f, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["date"] >= START]
        splits = [(r["date"], float(r["split"])) for r in rows if r.get("split") not in ("", "0", "0.0", "1", "1.0", None)]
        splits += [s for s in EXTRA_SPLITS.get(t, []) if s[0] not in {d for d, _ in splits}]
        for r in rows:
            try:
                o, c = float(r["open"]), float(r["close"])
            except ValueError:
                notes.append(f"{t} {r['date']}: unreadable price"); continue
            if not (o > 0 and c > 0):
                notes.append(f"{t} {r['date']}: non-positive price"); continue
            fac = 1.0
            for d, ratio in splits:
                if d > r["date"]:
                    fac *= ratio
            rows_out.append((t, r["date"], round(o * fac, 6), round(c * fac, 6), o, c, fac, r.get("dividend") or "0", r.get("split") or "0"))
        if splits:
            notes.append(f"{t}: splits {splits}")
            splits_out += [(t, d, ratio) for d, ratio in splits]
        if rows:
            notes.append(f"{t}: {rows[0]['date']} to {rows[-1]['date']} ({len(rows)} rows)")
    opener = gzip.open if dst.suffix == ".gz" else open
    with opener(dst, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("ticker", "date", "open", "close", "open_file", "close_file", "split_factor", "dividend", "split"))
        w.writerows(rows_out)
    with open(dst.parent / "splits.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("ticker", "date", "ratio")); w.writerows(splits_out)
    (dst.parent / "official_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(len(rows_out), "rows,", len({r[0] for r in rows_out}), "tickers ->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
