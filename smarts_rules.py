"""Worked examples: list group entries written as SMARTS substructure rules.

Each rule is a hand-written reading of one group entry. A match means
"this structure falls inside the chemical family the entry names". It does
not mean the legal restriction applies: concentration limits, uses,
product types and derogations are outside what a structure can encode.

The rules are deliberately few and commented. They are examples of what
can be expressed, not a complete compliance rule set.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
from rdkit import Chem


@dataclass
class Rule:
    rule_id: str
    covers: str              # list entries the rule is meant to express
    smarts: str
    note: str
    extra: Optional[Callable[[Chem.Mol], bool]] = None  # checks SMARTS cannot express
    _patt: Chem.Mol = field(default=None, init=False, repr=False)

    def pattern(self):
        if self._patt is None:
            self._patt = Chem.MolFromSmarts(self.smarts)
            if self._patt is None:
                raise ValueError(f"bad SMARTS in {self.rule_id}")
        return self._patt

    def matches(self, mol_or_smiles):
        mol = Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else mol_or_smiles
        if mol is None:
            raise ValueError("unparsable structure")
        if not mol.HasSubstructMatch(self.pattern()):
            return False
        return self.extra(mol) if self.extra else True


def _longest_perfluoro_acid_chain(mol):
    """Carbons in the longest chain that runs from a carboxyl carbon through
    fully fluorinated sp3 carbons (carboxyl carbon included). Branches count
    as part of the chain only along the longest path."""
    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    best = 0

    def perfluoro(a):
        return (a.GetSymbol() == "C" and a.GetHybridization() == Chem.HybridizationType.SP3
                and a.GetTotalNumHs() == 0
                and all(n.GetSymbol() in ("C", "F") for n in a.GetNeighbors()))

    def walk(idx, seen):
        longest = 0
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            if n.GetIdx() not in seen and perfluoro(n):
                longest = max(longest, 1 + walk(n.GetIdx(), seen | {n.GetIdx()}))
        return longest

    for (c, _o, _oh) in mol.GetSubstructMatches(acid):
        best = max(best, 1 + walk(c, {c}))
    return best


RULES = [
    Rule(
        "ortho_phthalate_diester",
        "Annex XVII entries 51 and 52 (named phthalates); Annex XIV entries 4-7; "
        "Candidate List phthalate entries",
        # benzene-1,2-dicarboxylic acid with both carboxyls esterified by a carbon
        # [#6] on the ester oxygen accepts alkyl, branched alkyl, benzyl or aryl
        "c1ccc(C(=O)O[#6])c(C(=O)O[#6])c1",
        "Captures the whole ortho-phthalate diester family. The legal entries name "
        "specific esters (DEHP, DBP, BBP, DIBP, DIDP, DINP, DNOP ...), so the rule is "
        "wider than any single entry. Terephthalates and monoesters do not match.",
    ),
    Rule(
        "pfas_oecd_2021",
        "Annex XVII entry 82 (PFAS) and the PFAS idea behind entries 68, 73 and 79",
        # OECD (2021) definition: at least one fully fluorinated methyl or methylene
        # carbon, i.e. sp3 carbon with no H, Cl, Br or I and at least two F
        "[CX4;H0;!$([CX4][Cl,Br,I])](F)(F)",
        "Follows the OECD 2021 structural definition. It is very broad: CF3-containing "
        "drugs and pesticides and trifluoroacetic acid match. The Annex XVII PFAS "
        "entry is limited to firefighting foams, which no structure rule can express.",
    ),
    Rule(
        "pfca_c9_c14",
        "Annex XVII entry 68 (C9-C14 PFCAs, their salts and related substances)",
        # carboxylic acid or carboxylate carbon followed by at least eight
        # perfluorinated sp3 carbons, i.e. at least C9 in total
        "[OX2H1,OX1-]C(=O)[CX4](F)(F)[CX4](F)(F)[CX4](F)(F)[CX4](F)(F)"
        "[CX4](F)(F)[CX4](F)(F)[CX4](F)(F)[CX4](F)",
        "SMARTS gives the lower bound (C9). The upper bound (C14) and branched chains "
        "need a chain-length count in code, done by _longest_perfluoro_acid_chain. "
        "The 'related substances' (precursors that degrade to these acids) are not covered.",
        extra=lambda m: 9 <= _longest_perfluoro_acid_chain(m) <= 14,
    ),
    Rule(
        "organotin",
        "Annex XVII entry 20 (organostannic compounds)",
        "[Sn]~[#6]",  # at least one tin-carbon bond
        "Element plus bond rule. Inorganic tin salts and oxides do not match.",
    ),
    Rule(
        "benzidine_and_salts",
        "Annex XVII entry 13 (benzidine and its salts)",
        # biphenyl-4,4'-diamine with neutral or protonated amines (salts)
        "[NX3H2,NX4H3+]c1ccc(-c2ccc([NX3H2,NX4H3+])cc2)cc1",
        "Matches the free base and its ammonium salts. Substituted benzidines "
        "(3,3'-dichlorobenzidine etc.) also match because SMARTS ring atoms are not "
        "required to be unsubstituted. Add H1 constraints on ring carbons to exclude them.",
    ),
]


def rule(rule_id):
    return next(r for r in RULES if r.rule_id == rule_id)
