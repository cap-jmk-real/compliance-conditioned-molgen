"""Parse Annex XIV exemptions and Annex XVII Appendices 1-6 from EUR-Lex.

Outputs data/annex_xiv_exemptions.csv and data/annex_xvii_appendix_substances.csv.
"""
import csv, pathlib, re, warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
HERE = pathlib.Path(__file__).resolve().parent
RAW, DATA = HERE / "raw", HERE / "data"
CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
EC_RE = re.compile(r"\b\d{3}-\d{3}-\d\b")
INDEX_RE = re.compile(r"^\d{3}-\d{3}-\d{2}-[\dX]$")
NO_EXEMPTION = {"", "—", "-", "–"}


def txt(el):
    return re.sub(r"\s+", " ", el.get_text(" ")).replace("\xa0", " ").strip()


def cas_valid(cas):
    """CAS check digit test."""
    digits = cas.replace("-", "")
    body, check = digits[:-1], int(digits[-1])
    return sum(int(d) * (i + 1) for i, d in enumerate(reversed(body))) % 10 == check


def main():
    soup = BeautifulSoup((RAW / "reach_consolidated.html").read_text(encoding="utf-8"), "lxml")
    tables = soup.find_all("table")
    DATA.mkdir(exist_ok=True)

    # Annex XIV: the table whose header has "Exempted (categories of) uses".
    xiv = next(t for t in tables if "Exempted (categories of) uses" in txt(t) and "Sunset" in txt(t))
    out = []
    for tr in xiv.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        m = re.match(r"^(\d+)\.", txt(tds[0]))
        if not m:
            continue
        exempt = txt(tds[5])
        transitional = txt(tds[3]) + " " + txt(tds[4])
        out.append({"entry_no": int(m.group(1)), "substance_text": txt(tds[1])[:300],
                    "exempted_uses": exempt[:500], "has_exemption": exempt not in NO_EXEMPTION,
                    "has_use_specific_derogation": "derogation" in transitional.lower()})
    with open(DATA / "annex_xiv_exemptions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
    print("Annex XIV rows", len(out), "with exemptions", sum(r["has_exemption"] for r in out),
          "with use-specific derogations", sum(r["has_use_specific_derogation"] for r in out))

    # Annex XVII appendices: tables headed Substances | Index No | EC No | CAS No | Notes,
    # labelled by the nearest preceding "Appendix N" heading.
    rows = []
    for t in tables:
        hdr = [txt(c) for c in t.find("tr").find_all(["td", "th"])]
        if hdr[:4] != ["Substances", "Index No", "EC No", "CAS No"]:
            continue
        head = t.find_previous(string=re.compile(r"^\s*Appendix \d+\s*$"))
        app = int(re.search(r"\d+", head).group())
        if app > 6:
            continue
        for tr in t.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 4 or not INDEX_RE.match(txt(tds[1])):
                continue
            cas_all = CAS_RE.findall(txt(tds[3]))
            rows.append({"appendix": app, "index_no": txt(tds[1]), "name": txt(tds[0])[:400],
                         "ec": "; ".join(EC_RE.findall(txt(tds[2]))),
                         "cas": "; ".join(c for c in cas_all if cas_valid(c)),
                         "cas_failed_checksum": "; ".join(c for c in cas_all if not cas_valid(c))})
    with open(DATA / "annex_xvii_appendix_substances.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    for a in sorted({r["appendix"] for r in rows}):
        print("Appendix", a, sum(r["appendix"] == a for r in rows), "index entries")

if __name__ == "__main__":
    main()
