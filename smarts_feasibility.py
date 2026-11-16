"""Sort the non-single-structure entries of the three main lists by whether
a SMARTS substructure rule could express them in principle.

The labels come from keyword rules on the entry name (first match wins) and
were then checked by reading every name in the output. They are a reading
of the entry text, not a legal interpretation.

  element_rule        "X compounds", "X and its compounds", organotin: an
                      element (or element-carbon bond) pattern. Chromium VI
                      is 'partial' because SMARTS has no oxidation state.
  parent_plus_forms   one parent structure plus salts, isomers, stereoisomers,
                      hydrates: parent skeleton with charges/stereo ignored
  family_pattern      a functional family (PFAS, PFCAs, phthalate esters,
                      diisocyanates, polyhalogenated biphenyls/terphenyls,
                      bromoacetic esters, siloxanes, PAH ...). PAH is
                      'partial': the entries list named PAH, a ring pattern
                      would over-match.
  partial             expressible only in part: azo dyes (the rule is about
                      amines released on reductive cleavage), 'related
                      substances' or releasers (defined by degradation)
  single_not_in_pubchem  one defined substance that PubChem did not return
  not_structural      UVCB, reaction masses, minerals and fibres, polymers,
                      ethoxylates of variable length, hazard-class entries

Output: data/group_smarts_feasibility.csv and a count table on stdout.
"""
import collections, csv, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
MAIN = ("candidate_list", "annex_xiv_authorisation_list", "annex_xvii_restriction_list")
RULES = [  # (label, regex on the name), first match wins
    ("single_not_in_pubchem", r"Reactive Brown 51|^Sodium peroxometaborate$"),
    ("family_pattern", r"tridecafluorooctyl\)silanetriol"),
    ("element_rule", r"inorganic ammonium salts"),
    ("not_structural", r"\(C10-C13\)-alkyl|classified as|hazard class|fulfilling the criteria|tattoo|reaction mass|reaction products|"
                       r"oligomeri|pitch|anthracene oil|creosote|fibres|asbestos|microparticles|ethoxylat|paraffin|"
                       r"alkanes, C|alkylation products|derivs\.|C\d+-\d+-alkyl|C\d+-\d+-branched|terphenyl, hydrogenated|"
                       r"pyrochlore|lead-doped|basic|mixture of isomers|Fatty acids"),
    ("partial", r"azocolourants|azodyes|related substances|releasers|chromium vi|acids generated"),
    ("partial_pah", r"polycyclic|\bPAH\b"),
    ("element_rule", r"compounds|organostannic"),
    ("family_pattern", r"\bPFAS\b|PFCAs|phthalates|benzenedicarboxylic acid|diisocyanate|polychlorinated|polybromo|"
                       r"esters of bromoacetic|siloxane|D4\)|isocyanate"),
    ("parent_plus_forms", r"salts?\b|isomers|stereoisomers|covering|branched and linear|\[1\]|;|perborate|"
                          r"anhydride|hexabromocyclododecane|dechlorane|sulphate|dichromate|tetraborate|hydrazine|"
                          r"entry \d+|lead (carbonates|sulphates)|nonylphenol|heptylphenol|ugilec|octabromo"),
]


def main():
    with open(DATA / "entry_classification.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["list"] in MAIN and r["class"] != "one_structure"]
    out = []
    for r in rows:
        label = "unlabelled"
        for lab, pat in RULES:
            if re.search(pat, r["name"], re.I):
                label = lab
                break
        out.append({"entry_id": r["entry_id"], "list": r["list"], "entry_no": r["entry_no"],
                    "name": r["name"][:200], "structural_class": r["class"], "smarts_feasibility": label})
    with open(DATA / "group_smarts_feasibility.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
    labels = ["single_not_in_pubchem", "element_rule", "parent_plus_forms", "family_pattern", "partial", "partial_pah", "not_structural", "unlabelled"]
    print("| List | Entries not resolving to one structure | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 2) + "|")
    for lst in MAIN:
        c = collections.Counter(o["smarts_feasibility"] for o in out if o["list"] == lst)
        print(f"| {lst} | {sum(c.values())} | " + " | ".join(str(c[l]) for l in labels) + " |")

if __name__ == "__main__":
    main()
