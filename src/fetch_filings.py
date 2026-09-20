import json, time, pathlib, requests

# EDGAR requires a real User-Agent with contact info, or it returns 403
HEADERS = {"User-Agent": "Shreyash shreyash@example.com"}

COMPANIES = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "META":  "0001326801",
}

OUT = pathlib.Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)


def latest_10k(cik: str):
    """Return (accession_no, primary_doc, filing_date) of the most recent 10-K."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    recent = requests.get(url, headers=HEADERS, timeout=30).json()["filings"]["recent"]

    for form, acc, doc, date in zip(recent["form"], recent["accessionNumber"],
                                    recent["primaryDocument"], recent["filingDate"]):
        if form == "10-K":                      # exact match — skips "10-K/A" amendments
            return acc.replace("-", ""), doc, date
    raise ValueError(f"no 10-K found for CIK {cik}")


for ticker, cik in COMPANIES.items():
    acc, doc, date = latest_10k(cik)
    cik_int = int(cik)                          # archive path uses the un-padded CIK
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"

    html = requests.get(url, headers=HEADERS, timeout=60).text
    path = OUT / f"{ticker}_10K_{date[:4]}.htm"
    path.write_text(html, encoding="utf-8")

    print(f"{ticker:6} {date}  {len(html)/1_000_000:5.1f} MB  -> {path.name}")
    time.sleep(0.5)                             # EDGAR rate limit: max 10 req/sec