from __future__ import annotations

from functools import lru_cache
from typing import List
from rdkit import Chem

from .io import biopandas_extract_ligand_and_write_to_pdb, get_ligand_mol_from_pdb
from .naming import assign_chai_lab_atom_names_to_mol, get_rdkit_index_to_atom_name_map_smiles, get_rdkit_index_to_atom_name_map_pdb
from .matching import match_ligands_atom_to_large_mol_atom


@lru_cache(maxsize=16)
def get_ligand_atom_name_mapping_from_ligand_and_chai_lab(cif_file: str, smiles: str):
    df, ligand_res_names, output_pdb = biopandas_extract_ligand_and_write_to_pdb(cif_file)
    ref_smiles = {
        "G74": "CN1C=C(C=N1)C1=CN=C(N)C2=C1SC=C2C1=CC2=C(C=C1)N(CC2)C(=O)CC1=CC=CC=C1",
        "9BW": "C[C@H](NC(=O)[C@@H]1C[C@@H](O)CN1C(=O)[C@@H](N)C(C)(C)C)C1=CC=C(C=C1)C1=C(C)N=CS1",
        "A1B": "ClC1=C(C=CC=C1C1=CC=CC=C1)C1CCC(=O)NC1=O",
        "LVY": "C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N",
    }
    small_mols = [get_ligand_mol_from_pdb(pdb, ref_smiles.get(ligand_name)) for ligand_name, pdb in zip(ligand_res_names, output_pdb)]
    large_mol = Chem.MolFromSmiles(smiles)
    large_mol = assign_chai_lab_atom_names_to_mol(large_mol)
    large_mol_atom_name_mapping = get_rdkit_index_to_atom_name_map_smiles(large_mol)
    match_result = match_ligands_atom_to_large_mol_atom(small_mols, large_mol)
    ligand_atom_mapping = []
    for ligand_name, ligand_mol, match in zip(ligand_res_names, small_mols, match_result):
        ligand_idx_to_atom_name_mapping = get_rdkit_index_to_atom_name_map_pdb(ligand_mol)
        for ligand_atom_idx, large_mol_atom_idx in match:
            ligand_atom_name = ligand_idx_to_atom_name_mapping[ligand_atom_idx]
            large_mol_atom_name = large_mol_atom_name_mapping[large_mol_atom_idx]
            ligand_atom_mapping.append(((ligand_name, ligand_atom_name), large_mol_atom_name))
    return tuple(ligand_atom_mapping)

