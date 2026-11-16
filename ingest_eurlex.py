"""Fetch the consolidated REACH Regulation (EC) No 1907/2006 from EUR-Lex.

Used for two things the ECHA list pages do not carry in their tables:
the "Exempted (categories of) uses" column of Annex XIV, and the
substance tables of Annex XVII Appendices 1 to 6 (CMR substances that
entries 28 to 30 refer to).

EUR-Lex answers plain HTTP requests, so no browser is needed.
"""
import datetime, json, pathlib, sys
import requests

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
# Latest consolidated version at retrieval time. ECHA's own Annex XVII
# appendix links point to the same consolidation (02006R1907-20260622).
CELEX = "02006R1907-20260622"
URL = f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{CELEX}"


def main():
    RAW.mkdir(exist_ok=True)
    meta_path = RAW / "fetch_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    r = requests.get(URL, timeout=120, headers={"User-Agent": "open-research track 25 data build"})
    rec = {"source_page": URL, "celex": CELEX, "http_status": r.status_code,
           "retrieved": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    if r.ok and "ANNEX XIV" in r.text:
        out = RAW / "reach_consolidated.html"
        out.write_bytes(r.content)
        rec.update(status="ok", files=[f"raw/{out.name}"], bytes=len(r.content))
    else:
        rec.update(status=f"gap: HTTP {r.status_code} or Annex XIV not found in response")
    meta["reach_consolidated_eurlex"] = rec
    meta_path.write_text(json.dumps(meta, indent=1))
    print(rec)
    return 0 if rec["status"] == "ok" else 1

if __name__ == "__main__":
    sys.exit(main())
