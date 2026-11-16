"""Classify every list entry and write the coverage report.

Inputs: data/entries.csv, data/members.csv, data/annex_xvii_appendix_substances.csv,
data/annex_xiv_exemptions.csv, data/resolution.csv, data/structures.csv.
Outputs: data/entry_classification.csv and COVERAGE.md.

Every entry gets exactly one structural class, decided by the first rule
that fires (the rule name is written to the 'class_rule' column so each
decision can be audited):

  hazard_class_defined   no CAS/EC and the entry is defined by a CLP hazard
                         classification or by reference to an appendix
  variable_composition   the name marks a UVCB, reaction mass/product,
                         mineral fibre, polymer, ethoxylate, oil/pitch/tar
                         or a chain-length range of alkyl groups
  group (members)        ECHA lists members and at least one of them does
                         not resolve to exactly one structure
  one_structure          all identifiers of the entry (its own CAS, else its
                         name, plus any member substances ECHA lists) resolve
                         to exactly one InChIKey
  group                  the name carries a family word (FAMILY_RE: salts,
                         compounds, isomers, branched, covering ...), checked
                         before one_structure because PubChem often maps a
                         family CAS to one arbitrary representative. Also
                         group: the entry covers several structures: ECHA lists
                         members that resolve to more than one InChIKey, or
                         the name is a family name (salts, compounds, esters,
                         isomers, 'covering', 'and its', chain ranges ...)
  multiple_structures    a single-substance entry whose CAS or name maps to
                         more than one InChIKey in PubChem
  no_structure           nothing resolved in PubChem

The conditional-scope flag is separate from the structural class.
"""
import collections, csv, pathlib, re
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

HAZARD_RE = re.compile(r"classified as|hazard class|fulfilling the criteria|listed in appendix|"
                       r"substances in tattoo|carcinogen category|mutagen category|reproductive toxicant", re.I)
VARIABLE_RE = re.compile(
    r"reaction mass|reaction products?|oligomeri[sz]ation|\bUVCB\b|distillates?|\boils?\b|\bpitch\b|\btar\b|"
    r"creosote|fibres?\b|residues?\b|extracts?\b|paraffins?\b|alkanes?, C\d|microparticles|polymer|"
    r"ethoxylated|\(petroleum\)|\bcoal\b|asbestos|derivs\.|C\d+-\d+-alkyl|C\d+-C?\d+.{0,20}alkyl|"
    r"alkyl.{0,30}C\d+-C?\d+|-rich|mixture of|alkylation products|complex combination|combination of|"
    r"\bnaphtha\b|\bgases\b|gas oils?|kerosine|ligroine|sludges|slimes|\(coal\)|\(coal tar\)", re.I)
# Family words in the entry name. These override a single PubChem hit,
# because PubChem often maps a family CAS (e.g. "dipentyl ester, branched")
# to one arbitrary representative.
FAMILY_RE = re.compile(
    r"\bsalts?\b|compounds|isomers|derivatives?|related substances|homologues|branched|\bcovering\b|"
    r"and any of|releasers|\bPFAS\b|PFCAs|\bPAH\b|polychlorinated|polybromo|azocolourants|azodyes|"
    r"diisocyanates|\bacids generated\b|chromates\b|\boligomers\b|\bcarbonates\b|\bsulphates\b|\bsulfates\b|"
    r"with the exception|\bincluding\b", re.I)
GROUP_RE = re.compile(
    r"\bsalts?\b|compounds|\besters\b|isomers|derivatives?|related substances|homologues|branched|"
    r"\bcovering\b|\[\d\]|;\s*\S|and any of|releasers|\bentry \d+|\bPFAS\b|PFCAs|\bPAH\b|polychlorinated|polybromo|"
    r"azocolourants|azodyes|diisocyanates|C\d+-C\d+|\bacids generated\b|chromates|\boligomers\b|"
    r"including|with the exception", re.I)
# Element words in a name that must show up in the structure. A miss means
# PubChem returned an unrelated record for the CAS number.
NAME_ELEMENTS = {r"nickel": "Ni", r"\blead\b|plumb": "Pb", r"cadmium": "Cd", r"chromium|chromate": "Cr",
                 r"cobalt": "Co", r"arsen": "As", r"mercur": "Hg", r"stann|\btin\b": "Sn", r"barium": "Ba",
                 r"\bboron|borate|boric": "B", r"zinc": "Zn", r"antimon": "Sb", r"beryll": "Be", r"titan": "Ti",
                 r"zircon": "Zr", r"alumin": "Al", r"sodium": "Na", r"potassium": "K", r"strontium": "Sr",
                 r"silic|silox|silane": "Si", r"fluoro|fluorid|fluorin": "F", r"chloro|chlorid": "Cl",
                 r"bromo|bromid": "Br", r"iodo|iodid": "I", r"phosph": "P", r"sulph|sulf|thio": "S"}
COMMON_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"}
LIST_LABEL = {
    "candidate_list": "Candidate List (SVHC)",
    "annex_xiv_authorisation_list": "REACH Annex XIV",
    "annex_xvii_restriction_list": "REACH Annex XVII",
    "annex_xvii_appendix_1": "Annex XVII App. 1 (carc. 1A)",
    "annex_xvii_appendix_2": "Annex XVII App. 2 (carc. 1B)",
    "annex_xvii_appendix_4": "Annex XVII App. 4 (muta. 1B)",
    "annex_xvii_appendix_5": "Annex XVII App. 5 (repr. 1A)",
    "annex_xvii_appendix_6": "Annex XVII App. 6 (repr. 1B)",
}


def read(name):
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def missing_name_elements(name, elems):
    low = name.lower()
    return sorted({sym for pat, sym in NAME_ELEMENTS.items() if re.search(pat, low) and sym not in elems})


def structure_flags(smiles, pubchem_key, name=""):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"rdkit_parses": False}
    elems = {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()}
    return {"rdkit_parses": True, "name_elements_missing": ";".join(missing_name_elements(name, elems)),
            "single_component": "." not in smiles,
            "common_organic_elements": "C" in elems and elems <= COMMON_ELEMENTS,
            "rdkit_inchikey_matches": Chem.MolToInchiKey(mol) == pubchem_key}


def main():
    res = {r["item_id"]: r for r in read("resolution.csv")}
    struct = {r["InChIKey"]: r for r in read("structures.csv")}
    members = collections.defaultdict(list)
    for m in read("members.csv"):
        members[m["entry_id"]].append(f"{m['entry_id']}/m{int(m['member_no']):03d}")
    xiv_cond = {int(r["entry_no"]): (r["has_exemption"] == "True" or r["has_use_specific_derogation"] == "True")
                for r in read("annex_xiv_exemptions.csv")}

    entries = []
    for e in read("entries.csv"):
        entries.append({"entry_id": e["entry_id"], "list": e["list"], "entry_no": e["entry_no"], "name": e["name"],
                        "cas": e["cas"], "ec": e["ec"], "member_items": members[e["entry_id"]],
                        "conditions_doc": e["conditions_doc"]})
    for a in read("annex_xvii_appendix_substances.csv"):
        entries.append({"entry_id": f"annex_xvii_appendix_{a['appendix']}:{a['index_no']}",
                        "list": f"annex_xvii_appendix_{a['appendix']}", "entry_no": a["index_no"], "name": a["name"],
                        "cas": a["cas"], "ec": a["ec"], "member_items": [], "conditions_doc": ""})

    out = []
    for e in entries:
        r = res.get(e["entry_id"], {})
        own_keys = set(filter(None, r.get("inchikeys", "").split(";")))
        mem_keys = set()
        for mid in e["member_items"]:
            mem_keys |= set(filter(None, res.get(mid, {}).get("inchikeys", "").split(";")))
        mem_resolved = sum(1 for mid in e["member_items"] if res.get(mid, {}).get("n_inchikeys") == "1")
        all_keys = own_keys | mem_keys
        has_id = bool(e["cas"] or e["ec"])
        name = e["name"]

        if not has_id and not e["member_items"] and HAZARD_RE.search(name):
            cls, rule = "hazard_class_defined", "HAZARD_RE on name, no identifiers"
        elif HAZARD_RE.search(name) and not own_keys and not mem_keys:
            cls, rule = "hazard_class_defined", "HAZARD_RE on name, nothing resolved"
        elif VARIABLE_RE.search(name):
            cls, rule = "variable_composition", f"VARIABLE_RE: {VARIABLE_RE.search(name).group(0)}"
        elif FAMILY_RE.search(name):
            cls, rule = "group", f"FAMILY_RE: {FAMILY_RE.search(name).group(0)}"
        elif e["member_items"] and mem_resolved < len(e["member_items"]):
            cls, rule = "group", f"{len(e['member_items']) - mem_resolved} of {len(e['member_items'])} listed members unresolved"
        elif len(all_keys) == 1:
            cls, rule = "one_structure", "all identifiers give one InChIKey"
        elif e["member_items"] and len(mem_keys) > 1:
            cls, rule = "group", "ECHA lists members resolving to >1 InChIKey"
        elif GROUP_RE.search(name) or (not has_id and e["member_items"]):
            cls, rule = "group", f"GROUP_RE: {GROUP_RE.search(name).group(0)}" if GROUP_RE.search(name) else "members listed, no own identifier"
        elif len(all_keys) > 1:
            cls, rule = "multiple_structures", f"{len(all_keys)} InChIKeys from CAS/name"
        else:
            cls, rule = "no_structure", "nothing resolved in PubChem"

        lst = e["list"]
        if lst == "annex_xvii_restriction_list":
            conditional = bool(e["conditions_doc"])
        elif lst == "annex_xiv_authorisation_list":
            try:
                conditional = xiv_cond.get(int(e["entry_no"]), False)
            except ValueError:
                conditional = False
        else:
            conditional = False

        rec = {"entry_id": e["entry_id"], "list": lst, "entry_no": e["entry_no"], "name": name[:300],
               "cas": e["cas"], "ec": e["ec"], "class": cls, "class_rule": rule,
               "route": r.get("route", ""), "cas_name_agreement": r.get("cas_name_agreement", ""),
               "n_members_listed": len(e["member_items"]), "n_members_one_structure": mem_resolved,
               "n_inchikeys_total": len(all_keys), "inchikey": next(iter(all_keys)) if cls == "one_structure" else "",
               "conditional_scope": conditional}
        if cls == "one_structure":
            s = struct.get(rec["inchikey"], {})
            rec["smiles"] = s.get("SMILES", "")
            rec.update(structure_flags(rec["smiles"], rec["inchikey"], name))
        rec["usable"] = bool(cls == "one_structure" and rec.get("single_component") and rec.get("common_organic_elements")
                             and rec["cas_name_agreement"] != "conflict" and not rec.get("name_elements_missing"))
        rec["cross_checked"] = rec["usable"] and rec["cas_name_agreement"] in ("same", "same_skeleton")
        out.append(rec)

    cols = ["entry_id", "list", "entry_no", "name", "cas", "ec", "class", "class_rule", "route",
            "cas_name_agreement", "n_members_listed", "n_members_one_structure", "n_inchikeys_total",
            "inchikey", "smiles", "rdkit_parses", "single_component", "common_organic_elements",
            "rdkit_inchikey_matches", "name_elements_missing", "usable", "cross_checked", "conditional_scope"]
    with open(DATA / "entry_classification.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(out)
    write_report(out)


NOTES = """## How to read these tables

Classes are exclusive and set in this order: defined by hazard class, variable composition, group by family word,
group because a listed member does not resolve, one structure, group by other signals, several InChIKeys, not found.
See the module docstring of `classify_coverage.py` for the exact rules.

"One structure" means every identifier PubChem returned for the entry maps to a single InChIKey. "Usable" adds four
conditions: the structure is one component (no salts or counter-ions), contains carbon and only
C, H, N, O, S, P, F, Cl, Br, I, B, Si, the CAS lookup and the name lookup do not point at different structures, and
no element named in the substance name is missing from the structure. "Cross-checked" means the CAS and the name
lookup both returned the same InChIKey (or the same first InChIKey block).

Conditional scope:

- Annex XVII: an entry counts as conditional when ECHA links a conditions-of-restriction document for it. All 79 do.
  Every Annex XVII restriction is written as conditions (uses, concentration limits, product types, dates), so the
  legal effect of every entry depends on more than the structure.
- Annex XIV: counted from the EUR-Lex table. No entry has text in the "Exempted (categories of) uses" column. Five
  entries (4 to 7, the phthalates DEHP, BBP, DBP, DIBP, and 42, 4-(1,1,3,3-tetramethylbutyl)phenol ethoxylated) carry use-specific derogations in their transitional
  arrangements. The rest apply to all uses, subject to authorisation.
- Candidate List and the Annex XVII appendices: n/a. Candidate List obligations (REACH Articles 7, 31, 33) use one
  threshold for every entry (0.1 % w/w in articles), so there is no entry-specific condition to count. The appendices
  only list substances. The conditions sit in entries 28 to 30.

Known weak points:

- Several InChIKeys: most cases are one substance listed with its hydrate CAS numbers (cadmium sulphate, sodium
  dichromate, hydrazine, disodium tetraborate). ECHA states that each Candidate List entry covers anhydrous and
  hydrated forms. These are kept separate here and not counted as one structure.
- PubChem sometimes maps a CAS number to an unrelated or arbitrary record. Manual reads found three such cases while
  the rules were being written, and each led to a rule: "dipentyl ester, branched" (CAS 84777-06-0) maps to a
  ring-alkylated phthalate, fixed by the family-word rule. Nickel barium titanium primrose priderite (CAS 68610-24-2)
  maps to a dichloroquinazolinedione, caught by the element check. Annex XVII "Entry 9" resolved through only one
  of its listed members, fixed by the listed-members rule. After the fixes I read a random sample of 40 usable
  structures and all usable structures that were not cross-checked, and found no further errors. That is a reading
  of names against SMILES, not a formal validation. Inorganic substances are often represented as loose ions with the
  wrong stoichiometry (for example nickel arsenate). They are excluded from "usable" anyway.
- The variable-composition and group rules are keyword rules on names. I read every non-single entry of the three
  main lists and a sample of the appendix entries. The appendices are dominated by petroleum and coal-tar streams,
  which is why so few of their entries are single structures.
"""


def write_report(out):
    classes = ["one_structure", "multiple_structures", "group", "variable_composition",
               "hazard_class_defined", "no_structure"]
    lines = ["# Structure-resolution coverage", "",
             "Generated by `classify_coverage.py`. Every number is a count over the cached list data and the",
             "cached PubChem lookups. The class of each entry and the rule that set it are in",
             "`data/entry_classification.csv`.", "",
             "## Entries per class", "",
             "| List | Entries | One structure | Several InChIKeys | Group | Variable composition (UVCB etc.) | Defined by hazard class | Not found | Conditional scope |",
             "|---|---|---|---|---|---|---|---|---|"]
    by_list = collections.defaultdict(list)
    for r in out:
        by_list[r["list"]].append(r)
    for lst in LIST_LABEL:
        rows = by_list.get(lst, [])
        if not rows:
            continue
        c = collections.Counter(r["class"] for r in rows)
        n = len(rows)
        cells = [f"{c[k]} ({100 * c[k] / n:.0f}%)" for k in classes]
        cond = sum(r["conditional_scope"] for r in rows)
        cond_cell = f"{cond}" if lst in ("annex_xvii_restriction_list", "annex_xiv_authorisation_list") else "n/a"
        lines.append(f"| {LIST_LABEL[lst]} | {n} | " + " | ".join(cells) + f" | {cond_cell} |")

    lines += ["", "## Entries that resolve to one structure, by usability", "",
              "| List | One structure | RDKit parses | Single component | Organic, common elements | Single-component and organic | CAS and name disagree | Element named but absent | Usable | Usable and CAS/name cross-checked |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for lst in LIST_LABEL:
        rows = [r for r in by_list.get(lst, []) if r["class"] == "one_structure"]
        if not rows:
            continue
        p = sum(bool(r.get("rdkit_parses")) for r in rows)
        sc = sum(bool(r.get("single_component")) for r in rows)
        org = sum(bool(r.get("common_organic_elements")) for r in rows)
        both = sum(bool(r.get("single_component")) and bool(r.get("common_organic_elements")) for r in rows)
        chk = sum(r["cas_name_agreement"] != "" for r in rows)
        bad = sum(r["cas_name_agreement"] == "conflict" for r in rows)
        use = sum(r["usable"] for r in rows)
        xc = sum(r["cross_checked"] for r in rows)
        miss = sum(bool(r.get("name_elements_missing")) for r in rows)
        n_all = len(by_list[lst])
        lines.append(f"| {LIST_LABEL[lst]} | {len(rows)} | {p} | {sc} | {org} | {both} | {bad} | {miss} | "
                     f"{use} ({100 * use / n_all:.0f}% of entries) | {xc} ({100 * xc / n_all:.0f}%) |")

    lines += ["", "## Unique structures", ""]
    allkeys = {r["inchikey"] for r in out if r["class"] == "one_structure"}
    usable = {r["inchikey"] for r in out if r["usable"]}
    xchk = {r["inchikey"] for r in out if r["cross_checked"]}
    lines.append(f"Across all lists, {len(allkeys)} distinct InChIKeys come from one-structure entries, "
                 f"{len(usable)} of them usable (single-component organic molecule, common elements, no CAS/name "
                 f"disagreement, no named element missing), and {len(xchk)} of those confirmed by both CAS and name.")
    lines += ["", "## Group entries with listed members", "",
              "| List | Group entries | With members listed by ECHA | Members listed | Members resolving to one structure |",
              "|---|---|---|---|---|"]
    for lst in LIST_LABEL:
        g = [r for r in by_list.get(lst, []) if r["class"] in ("group", "variable_composition")]
        if not g:
            continue
        wm = [r for r in g if r["n_members_listed"]]
        lines.append(f"| {LIST_LABEL[lst]} | {sum(r['class'] == 'group' for r in g)} | {len(wm)} | "
                     f"{sum(r['n_members_listed'] for r in wm)} | {sum(r['n_members_one_structure'] for r in wm)} |")
    lines += ["", NOTES]
    (HERE / "COVERAGE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
