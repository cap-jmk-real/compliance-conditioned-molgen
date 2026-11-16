"""Parse cached ECHA list pages (raw/*.html) into one row per list entry.

Output: data/entries.csv (one row per list entry) and data/members.csv
(one row per member substance shown inside a group entry).
"""
import csv, json, pathlib, re
from bs4 import BeautifulSoup

HERE = pathlib.Path(__file__).resolve().parent
RAW, DATA = HERE / "raw", HERE / "data"
LISTS = ["candidate_list", "annex_xiv_authorisation_list", "annex_xvii_restriction_list"]


def clean(x):
    x = re.sub(r"\s+", " ", x or "").strip()
    return "" if x in ("-", "") else x


def parse_file(path):
    soup = BeautifulSoup(path.read_text(), "lxml")
    table = max(soup.find_all("table"), key=lambda t: len(t.find_all("tr")))
    rows = table.find_all("tr")
    header = [clean(th.get_text(" ")) .replace(" expand / collapse", "") for th in rows[0].find_all(["th", "td"])]
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        cells = {h: td for h, td in zip(header, tds)}
        namecell = cells["Substance name"]
        head = namecell.find("div", class_="multiFieldTable")
        name = clean(head.get_text(" ")) if head else clean(namecell.get_text(" "))
        link = head.find("a") if head else None
        members = []
        for g in namecell.find_all("div", class_="groupSubstance"):
            a = g.find("a")
            vals = [clean(v.get_text()) for v in g.find_all("span", class_="groupEcCasValue")]
            mname = clean(a.get_text(" ")) if a else clean(g.contents[0] if g.contents else "")
            if not mname:
                mname = clean(re.split(r"EC No\.:", g.get_text(" "))[0])
            members.append({"name": mname, "ec": vals[0] if vals else "", "cas": vals[1] if len(vals) > 1 else ""})
        if not name and not members:
            continue  # empty spacer row at the end of the table
        rec = {"name": name,
               "ec": clean(cells["EC No."].get_text(" ")),
               "cas": clean(cells["CAS No."].get_text(" ")),
               "echa_substance_url": link["href"] if link and link.has_attr("href") else "",
               "members": members}
        for k in ("Entry No.", "Entry no."):
            if k in cells:
                rec["entry_no"] = clean(cells[k].get_text(" "))
        if "Date of inclusion" in cells:
            rec["date_of_inclusion"] = clean(cells["Date of inclusion"].get_text(" "))
        if "Reason for inclusion" in cells:
            rec["reason"] = clean(cells["Reason for inclusion"].get_text(" "))
        if "Intrinsic property" in cells:
            rec["reason"] = clean(cells["Intrinsic property"].get_text(" "))
        if "Conditions" in cells:
            a = cells["Conditions"].find("a")
            rec["conditions_doc"] = a["href"] if a else ""
        yield rec


def main():
    DATA.mkdir(exist_ok=True)
    entries, members = [], []
    for lst in LISTS:
        files = sorted(RAW.glob(f"{lst}_p*.html"), key=lambda p: int(re.search(r"_p(\d+)", p.name).group(1)))
        n = 0
        for f in files:
            for rec in parse_file(f):
                n += 1
                eid = f"{lst}:{n:03d}"
                for i, m in enumerate(rec.pop("members"), 1):
                    members.append({"entry_id": eid, "member_no": i, **m})
                entries.append({"entry_id": eid, "list": lst, "row": n, **rec,
                                "n_members_shown": sum(1 for m in members if m["entry_id"] == eid)})
        print(lst, n, "entries")
    cols = ["entry_id", "list", "row", "entry_no", "name", "ec", "cas", "n_members_shown", "reason",
            "date_of_inclusion", "conditions_doc", "echa_substance_url"]
    with open(DATA / "entries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore"); w.writeheader(); w.writerows(entries)
    with open(DATA / "members.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["entry_id", "member_no", "name", "ec", "cas"]); w.writeheader(); w.writerows(members)
    print(len(members), "members")

if __name__ == "__main__":
    main()
