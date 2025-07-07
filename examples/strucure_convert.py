from biopandas.mmcif import PandasMmcif
#open another process to run this script
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

def biopandas_to_pdb(cif_file: str, ligand_res_names: list[str], output_pdb: list[str]) -> pd.DataFrame:
    assert len(ligand_res_names) == len(output_pdb), "Number of ligand residue names and output PDB files must match."
    # 1. 读取 mmCIF 并筛选配体原子
    cif = PandasMmcif().read_mmcif(cif_file)
    df = cif.df["ATOM"].copy()
    cif.df["ATOM"] = cif.df["HETATM"].copy()
    for i, name in enumerate(ligand_res_names):
        ligand_atoms = df[df.label_comp_id == name]
        if ligand_atoms.empty:
            raise ValueError("No ligand atoms found for given residue names.")
        ligand_atoms["atom_number"] = range(1, len(ligand_atoms) + 1)
        cif.df["HETATM"] = ligand_atoms
        pdb = cif.convert_to_pandas_pdb()
        pdb.df["HETATM"]["atom_number"] = range(1, len(pdb.df["HETATM"]) + 1)
        pdb.to_pdb(output_pdb[i])
    return df[df.label_comp_id.isin(ligand_res_names)]
        

def rdkit_read_sdf(sdf_file: str):
    mol = Chem.SDMolSupplier(sdf_file)[0]
    Chem.SanitizeMol(mol)
    return mol



if __name__ == "__main__":
    ligand_res_names = ["9BW"]
    ligand_df = biopandas_to_pdb("center_10_macro_2.cif", ligand_res_names, [f"{i}_lig.pdb" for i in ligand_res_names])
    mol = Chem.MolFromPDBFile("9BW_lig.pdb")
    mol_protac = Chem.MolFromSmiles("CC(=O)N[C@H](C(=O)N1C[C@@H](C[C@H]1C(=O)NCc2ccc(c3c2cccc3)c4cc[nH]n4)O)C(C)(C)C")
    # 使用 rdFMCS 找到最大公共子结构
    mcs_result = rdFMCS.FindMCS([mol, mol_protac],)
    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)

    # 获取 MCS 的匹配
    mcs_match = mol.GetSubstructMatch(mcs_mol)
    mcs_match_protac = mol_protac.GetSubstructMatch(mcs_mol)
    print("MCS match with atom mapping:", list(zip(mcs_match, mcs_match_protac)))
