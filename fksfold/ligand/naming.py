from __future__ import annotations

from collections import defaultdict
from typing import List
from rdkit import Chem


def assign_chai_lab_atom_names_to_mol(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.AddHs(mol)
    element_counter: dict[str, int] = defaultdict(int)
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        element_counter[elem] += 1
        atom.SetProp("name", f"{elem.upper()}{element_counter[elem]}")
    mol = Chem.RemoveHs(mol)
    return mol


def get_rdkit_index_to_atom_name_map_smiles(mol: Chem.Mol) -> dict[int, str]:
    mol = assign_chai_lab_atom_names_to_mol(mol)
    return {atom.GetIdx(): atom.GetProp("name") for atom in mol.GetAtoms()}


def get_rdkit_index_to_atom_name_map_pdb(mol: Chem.Mol) -> dict[int, str]:
    return {atom.GetIdx(): atom.GetPDBResidueInfo().GetName().strip() for atom in mol.GetAtoms()}

