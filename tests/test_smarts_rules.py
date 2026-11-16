import pathlib, sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from smarts_rules import RULES, rule  # noqa: E402

CASES = {
    "ortho_phthalate_diester": {
        "match": {
            "DEHP": "CCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCC",
            "DBP": "CCCCOC(=O)c1ccccc1C(=O)OCCCC",
            "BBP": "CCCCOC(=O)c1ccccc1C(=O)OCc1ccccc1",
            "DIBP": "CC(C)COC(=O)c1ccccc1C(=O)OCC(C)C",
        },
        "no_match": {
            "DEHT (terephthalate)": "CCCCC(CC)COC(=O)c1ccc(cc1)C(=O)OCC(CC)CCCC",
            "MEHP (monoester)": "CCCCC(CC)COC(=O)c1ccccc1C(=O)O",
            "phthalic anhydride": "O=C1OC(=O)c2ccccc12",
            "phthalic acid": "OC(=O)c1ccccc1C(=O)O",
        },
    },
    "pfas_oecd_2021": {
        "match": {
            "PFOA": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "PFHxS": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "trifluoroacetic acid": "OC(=O)C(F)(F)F",
            "HFC-134a": "FCC(F)(F)F",
        },
        "no_match": {
            "fluorobenzene": "Fc1ccccc1",
            "difluoromethane": "FCF",
            "chlorodifluoromethane": "FC(F)Cl",
            "vinylidene fluoride": "C=C(F)F",
        },
    },
    "pfca_c9_c14": {
        "match": {
            "PFNA (C9)": "OC(=O)" + "C(F)(F)" * 7 + "C(F)(F)F",
            "PFDA (C10)": "OC(=O)" + "C(F)(F)" * 8 + "C(F)(F)F",
            "PFTeDA (C14)": "OC(=O)" + "C(F)(F)" * 12 + "C(F)(F)F",
            "PFDA sodium salt": "[O-]C(=O)" + "C(F)(F)" * 8 + "C(F)(F)F.[Na+]",
            "branched C10": "OC(=O)" + "C(F)(F)" * 7 + "C(F)(C(F)(F)F)C(F)(F)F",
        },
        "no_match": {
            "PFOA (C8)": "OC(=O)" + "C(F)(F)" * 6 + "C(F)(F)F",
            "C15 PFCA": "OC(=O)" + "C(F)(F)" * 13 + "C(F)(F)F",
            "PFDS (sulfonic acid)": "OS(=O)(=O)" + "C(F)(F)" * 9 + "C(F)(F)F",
            "decanoic acid": "OC(=O)CCCCCCCCC",
        },
    },
    "organotin": {
        "match": {
            "tributyltin chloride": "CCCC[Sn](Cl)(CCCC)CCCC",
            "dibutyltin dilaurate": "CCCCCCCCCCCC(=O)O[Sn](CCCC)(CCCC)OC(=O)CCCCCCCCCCC",
        },
        "no_match": {
            "tin(II) chloride": "Cl[Sn]Cl",
            "tin dioxide": "O=[Sn]=O",
        },
    },
    "benzidine_and_salts": {
        "match": {
            "benzidine": "Nc1ccc(cc1)-c1ccc(N)cc1",
            "benzidine dihydrochloride": "[NH3+]c1ccc(cc1)-c1ccc([NH3+])cc1.[Cl-].[Cl-]",
        },
        "no_match": {
            "4-aminobiphenyl": "Nc1ccc(cc1)-c1ccccc1",
            "aniline": "Nc1ccccc1",
            "N,N,N',N'-tetramethylbenzidine": "CN(C)c1ccc(cc1)-c1ccc(N(C)C)cc1",
        },
    },
}


def test_every_rule_has_cases():
    assert {r.rule_id for r in RULES} == set(CASES)


@pytest.mark.parametrize("rid,label,smi", [(r, l, s) for r, c in CASES.items() for l, s in c["match"].items()])
def test_matches(rid, label, smi):
    assert rule(rid).matches(smi), f"{rid} should match {label}"


@pytest.mark.parametrize("rid,label,smi", [(r, l, s) for r, c in CASES.items() for l, s in c["no_match"].items()])
def test_does_not_match(rid, label, smi):
    assert not rule(rid).matches(smi), f"{rid} should not match {label}"
