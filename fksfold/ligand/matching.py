from __future__ import annotations

from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import rdFMCS


def match_ligands_atom_to_large_mol_atom(ligand_mols: list[Chem.Mol], large_mol: Chem.Mol) -> list[list[tuple[int, int]]]:
    collection: list[list[tuple[int, int]]] = []
    for ligand_mol in ligand_mols:
        mcs_result = rdFMCS.FindMCS([ligand_mol, large_mol])
        mcs_smarts = mcs_result.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        mcs_match = ligand_mol.GetSubstructMatch(mcs_mol)
        mcs_match_large_mol = large_mol.GetSubstructMatch(mcs_mol)
        collection.append(list(zip(mcs_match, mcs_match_large_mol)))
    return collection

