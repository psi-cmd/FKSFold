from __future__ import annotations

import io as _io
from typing import List, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.mmcif import PandasMmcif

from ..constants import INVALID_LIGAND_RES_NAMES, three2one


def biopandas_extract_ligand_and_write_to_pdb(
    cif_file: str, ligand_res_names: list[str] | None = None, output_pdbs: list[str] | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cif = PandasMmcif().read_mmcif(cif_file)
    # Only consider HETATM as ligand candidates; exclude water/ions and standard amino acids
    het_df = cif.df["HETATM"].copy()
    # Reset HETATM/ATOM in the working copy to avoid writing protein atoms later
    cif.df["HETATM"] = cif.df["HETATM"].drop(cif.df["HETATM"].index)
    cif.df["ATOM"] = cif.df["HETATM"].copy()
    if ligand_res_names is None:
        invalid_names = set(INVALID_LIGAND_RES_NAMES) | set(three2one.keys())
        residue_name_candidates = het_df[~het_df.label_comp_id.isin(invalid_names)].label_comp_id.unique().tolist()
        ligand_res_names = [name for name in residue_name_candidates if name not in invalid_names]
    if output_pdbs is not None:
        assert len(ligand_res_names) == len(output_pdbs), "Number of ligand residue names and output PDB files must match."
    else:
        output_pdbs = [f"{i}_lig.pdb" for i in ligand_res_names]

    for i, name in enumerate(ligand_res_names):
        ligand_atoms = het_df[het_df.label_comp_id == name].copy()
        if ligand_atoms.empty:
            raise ValueError("No ligand atoms found for given residue names.")
        if len(name) > 3:
            ligand_atoms.label_comp_id = name[:3]
            ligand_atoms.auth_comp_id = name[:3]
            ligand_res_names[i] = name[:3]
            output_pdbs[i] = f"{name[:3]}_lig.pdb"
        ligand_atoms.loc[:, "atom_number"] = range(1, len(ligand_atoms) + 1)
        ligand_atoms = ligand_atoms[ligand_atoms.type_symbol != "H"]

        cif.df["HETATM"] = ligand_atoms
        pdb = cif.convert_to_pandas_pdb()
        pdb.df["HETATM"]["atom_number"] = range(1, len(pdb.df["HETATM"]) + 1)
        pdb.to_pdb(output_pdbs[i])
    return het_df[het_df.label_comp_id.isin(ligand_res_names)], ligand_res_names, output_pdbs


def get_ligand_mol_from_pdb(pdb_file: str | _io.IOBase, ref_smiles: str | None = None) -> Chem.Mol:
    if isinstance(pdb_file, str):
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=True)
    else:
        pdb_block = pdb_file.read()
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)
    mol = Chem.RemoveAllHs(mol)
    if ref_smiles is not None:
        ref_structure = Chem.MolFromSmiles(ref_smiles)
        ref_structure = Chem.RemoveAllHs(ref_structure)
        try:
            AllChem.AssignBondOrdersFromTemplate(ref_structure, mol)
        except ValueError:
            pass
    return mol

