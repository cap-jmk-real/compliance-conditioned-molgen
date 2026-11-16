"""Shared helpers for the UV-Vis steered-generation experiment.

Everything here is ordinary materials and dye chemistry. The target property is the
first UV-Vis absorption maximum in nm. The structural rules from smarts_rules.py are
used only as a screening metric: a molecule "passes" the screen when it matches none
of the flagged structural families. The rules are never a generation target and never
a reward. No synthesis, no reactions, no route prediction.
"""
from __future__ import annotations

import os
import sys
import pathlib

import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

# Make the repo root importable so smarts_rules resolves from any working directory.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smarts_rules  # noqa: E402


def canonical_smiles(smiles: str):
    """Return a canonical SMILES string or None if the structure does not parse."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Morgan (ECFP-like) bit fingerprint as a float32 numpy array, or None."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def rule_hits(smiles_or_mol):
    """List of rule ids from smarts_rules.py that this molecule matches.

    A match means the structure falls inside a flagged structural family. It does
    not decide whether any legal restriction applies.
    """
    mol = Chem.MolFromSmiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    if mol is None:
        return None
    hits = []
    for rule in smarts_rules.RULES:
        try:
            if rule.matches(mol):
                hits.append(rule.rule_id)
        except Exception:
            # A rule that cannot evaluate on this molecule is treated as no match.
            continue
    return hits


def passes_screen(smiles_or_mol):
    """True when the molecule matches none of the flagged structural families."""
    hits = rule_hits(smiles_or_mol)
    if hits is None:
        return None
    return len(hits) == 0


def in_band(value_nm: float, band_lo: float, band_hi: float) -> bool:
    return band_lo <= value_nm <= band_hi
