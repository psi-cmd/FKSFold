"""Utility to reproduce Chai-Lab ligand atom naming from SMILES.

给定一个 SMILES 字符串，返回与 `RefConformerGenerator.generate` 完全一致的
atom name 列表（如 C1、C2、N1 等）。

"""

from __future__ import annotations

from collections import defaultdict
from typing import List
import sys

from rdkit import Chem
from io import StringIO
from rdkit.Chem import AllChem, rdFMCS, rdmolops

import io
import pandas as pd
from biopandas.mmcif import PandasMmcif
from .utils import ProteinDFUtils

def assign_chai_lab_atom_names_to_mol(mol: Chem.Mol) -> List[str]:
    """Return atom names following the Chai-Lab convention.

    The algorithm matches `RefConformerGenerator.generate`:
      1. Parse SMILES with RDKit to obtain canonical heavy-atom order.
      2. Add explicit hydrogens (these do **not** affect heavy-atom indices).
      3. Assign names element+counter (C1, C2, …) in current atom order.
      4. Remove hydrogens; the remaining heavy atoms retain the names.
    """

    # Step 2: add/remove hydrogens to mirror original pipeline
    mol = Chem.AddHs(mol)

    element_counter: dict[str, int] = defaultdict(int)
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        element_counter[elem] += 1
        # Upper-case names to match upstream code
        atom.SetProp("name", f"{elem.upper()}{element_counter[elem]}")

    # Remove hydrogens – heavy atoms keep their properties and order
    mol = Chem.RemoveHs(mol)
    return mol

def get_rdkit_index_to_atom_name_map_smiles(mol: Chem.Mol) -> dict[int, str]:
    """the mol should be generated from smiles identical to chai-lab input"""
    assign_chai_lab_atom_names_to_mol(mol)
    return {atom.GetIdx(): atom.GetProp("name") for atom in mol.GetAtoms()}

def get_rdkit_index_to_atom_name_map_pdb(mol: Chem.Mol) -> dict[int, str]:
    """the mol should be generated from pdb file"""
    return {atom.GetIdx(): atom.GetPDBResidueInfo().GetName().strip() for atom in mol.GetAtoms()}

def match_ligands_atom_to_large_mol_atom(ligand_mol: list[Chem.Mol], large_mol: Chem.Mol) -> dict[int, int]:
    collection = []
    for ligand_mol in ligand_mol:
        mcs_result = rdFMCS.FindMCS([ligand_mol, large_mol],)
        mcs_smarts = mcs_result.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)

        # 获取 MCS 的匹配
        mcs_match = ligand_mol.GetSubstructMatch(mcs_mol)
        mcs_match_large_mol = large_mol.GetSubstructMatch(mcs_mol)
        collection.append(list(zip(mcs_match, mcs_match_large_mol)))
    
    def has_overlap(collection: list[list[tuple[int, int]]]) -> bool:
        for i in range(len(collection)):
            for j in range(i + 1, len(collection)):
                if set(list(zip(*collection[i]))[1]) & set(list(zip(*collection[j]))[1]):
                    return True
        return False
    
    if has_overlap(collection):
        print("Warning: Ligands mapping to large mol overlapped")
    return collection

def get_ligand_mol_from_pdb(pdb_file: str|io.IOBase, ref_smiles: str | None = None) -> Chem.Mol:
    if isinstance(pdb_file, str):
        mol = Chem.MolFromPDBFile(pdb_file)
    else:
        mol = Chem.MolFromPDB(pdb_file.read())
    if ref_smiles is not None:
        ref_structure = Chem.MolFromSmiles(ref_smiles)
        ref_structure = Chem.RemoveAllHs(ref_structure)
        AllChem.AssignBondOrdersFromTemplate(ref_structure, mol)
    return mol

def get_ligand_mol_from_sdf_large_structure(sdf_file: str|io.IOBase) -> Chem.Mol:
    if isinstance(sdf_file, str):
        return Chem.MolFromPDBFile(sdf_file)
    else:
        return Chem.MolFromPDB(sdf_file.read())
    
def biopandas_extract_ligand_and_write_to_pdb(cif_file: str, ligand_res_names: list[str] | None = None, output_pdbs: list[str] | None = None) -> pd.DataFrame:
    cif = PandasMmcif().read_mmcif(cif_file)
    df = cif.df["ATOM"].copy()
    if ligand_res_names is None:
        ligand_res_names = ProteinDFUtils.get_ligand_res_names(df)
    if output_pdbs is not None:
        assert len(ligand_res_names) == len(output_pdbs), "Number of ligand residue names and output PDB files must match."
    else:
        output_pdbs = [f"{i}_lig.pdb" for i in ligand_res_names]

    # 1. 读取 mmCIF 并筛选配体原子
    cif.df["ATOM"] = cif.df["HETATM"].copy()
    for i, name in enumerate(ligand_res_names):
        ligand_atoms = df[df.label_comp_id == name]
        if ligand_atoms.empty:
            raise ValueError("No ligand atoms found for given residue names.")
        ligand_atoms.loc[:, "atom_number"] = range(1, len(ligand_atoms) + 1)
        cif.df["HETATM"] = ligand_atoms
        pdb = cif.convert_to_pandas_pdb()
        pdb.df["HETATM"]["atom_number"] = range(1, len(pdb.df["HETATM"]) + 1)
        pdb.to_pdb(output_pdbs[i])
    return df[df.label_comp_id.isin(ligand_res_names)], ligand_res_names, output_pdbs

def get_ligand_atom_name_mapping_from_ligand_and_chai_lab(cif_file: str, smiles: str) -> dict[str, str]:
    df, ligand_res_names, output_pdb = biopandas_extract_ligand_and_write_to_pdb(cif_file)
    ref_smiles = {
        "G74": "CN1C=C(C=N1)C1=CN=C(N)C2=C1SC=C2C1=CC2=C(C=C1)N(CC2)C(=O)CC1=CC=CC=C1",
        "9BW": "C[C@H](NC(=O)[C@@H]1C[C@@H](O)CN1C(=O)[C@@H](N)C(C)(C)C)C1=CC=C(C=C1)C1=C(C)N=CS1"
    }
    small_mols = [get_ligand_mol_from_pdb(pdb, ref_smiles[ligand_name]) for ligand_name, pdb in zip(ligand_res_names, output_pdb)]
    large_mol = Chem.MolFromSmiles(smiles)
    large_mol = assign_chai_lab_atom_names_to_mol(large_mol)
    large_mol_atom_name_mapping = get_rdkit_index_to_atom_name_map_smiles(large_mol)
    match_result = match_ligands_atom_to_large_mol_atom(small_mols, large_mol)
    ligand_atom_mapping = []
    for ligand_name, ligand_mol, match_result in zip(ligand_res_names, small_mols, match_result):
        # get atom name mapping from ligand_mol to large_mol
        ligand_idx_to_atom_name_mapping = get_rdkit_index_to_atom_name_map_pdb(ligand_mol)
        for ligand_atom_idx, large_mol_atom_idx in match_result:
            ligand_atom_name = ligand_idx_to_atom_name_mapping[ligand_atom_idx]
            large_mol_atom_name = large_mol_atom_name_mapping[large_mol_atom_idx]
            ligand_atom_mapping.append(((ligand_name, ligand_atom_name), large_mol_atom_name))
    return ligand_atom_mapping

if __name__ == "__main__":
    import sys
    df, ligand_res_names, output_pdb = biopandas_extract_ligand_and_write_to_pdb(sys.argv[1])
    small_mols = [get_ligand_mol_from_pdb(pdb) for pdb in output_pdb]
    smiles = sys.argv[2]
    large_mol = Chem.MolFromSmiles(smiles)
    large_mol = assign_chai_lab_atom_names_to_mol(large_mol)
    large_mol_atom_name_mapping = get_rdkit_index_to_atom_name_map_smiles(large_mol)
    match_result = match_ligands_atom_to_large_mol_atom(small_mols, large_mol)
    for ligand_name, ligand_mol, match_result in zip(ligand_res_names, small_mols, match_result):
        # get atom name mapping from ligand_mol to large_mol
        ligand_atom_name_mapping = get_rdkit_index_to_atom_name_map_pdb(ligand_mol)
        for ligand_atom_idx, large_mol_atom_idx in match_result:
            ligand_atom_name = ligand_atom_name_mapping[ligand_atom_idx]
            large_mol_atom_name = large_mol_atom_name_mapping[large_mol_atom_idx]
            print(f"{ligand_name} {ligand_atom_name} {large_mol_atom_name}")