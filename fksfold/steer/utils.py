from chai_lab.utils.tensor_utils import tensorcode_to_string
import requests
from Bio import pairwise2

from .constants import three2one, INVALID_LIGAND_RES_NAMES


INVALID_LIGAND_RES_NAMES = INVALID_LIGAND_RES_NAMES

def build_restype_mapping(struct_ctx):
    """
    返回 {int_id: 'ALA', ...} 字典；
    如果一个 int_id 在不同 residue 上出现冲突，会抛异常。
    """
    ints = struct_ctx.token_residue_type # (N,)
    names = struct_ctx.token_residue_name # (N, 8)
    mapping = {}
    for i, n in zip(ints.tolist(), names):
        name3 = tensorcode_to_string(n)
        name3 = name3.strip() # 去掉右侧 padding
        if i not in mapping:
            mapping[i] = name3
        elif mapping[i] != name3:
            raise ValueError(f"编号 {i} 同时映射到 {mapping[i]} 和 {name3}")
    return mapping

from biopandas.mmcif import PandasMmcif
from Bio.Data.IUPACData import protein_letters_3to1
import pandas as pd
import numpy as np
import difflib
from scipy.spatial import cKDTree

three2one = three2one

class ProteinDFUtils:
    def __init__(self, pdbid, cif_file=None):
        self.pdbid = pdbid
        if cif_file is not None:
            self.cif = PandasMmcif().read_mmcif(cif_file)
        else:
            self.cif = PandasMmcif()
        

    @staticmethod
    def get_protein_chain_ids(cif_df):
        filtered_df = cif_df[cif_df.label_comp_id.isin(three2one.keys())]
        return filtered_df.label_asym_id.unique().tolist()

    @staticmethod
    def get_ligand_chain_ids(cif_df):
        return cif_df[~cif_df.label_comp_id.isin(three2one.keys())].label_asym_id.unique().tolist()
    
    @staticmethod
    def get_ligand_res_names(cif_df):
        residue_name_candidates = cif_df[~cif_df.label_comp_id.isin(three2one.keys())].label_comp_id.unique().tolist()
        return [name for name in residue_name_candidates if name not in INVALID_LIGAND_RES_NAMES]

    @staticmethod
    def get_chain_atoms(chain_id, cif_df):
        return cif_df.query(f"label_asym_id == '{chain_id}'")

    @staticmethod
    def get_chain_res_seqs(chain_id, cif_df, return_seq_id: bool = False):
        three_letters = cif_df.query(f"label_asym_id == '{chain_id}'") \
                .sort_values("label_seq_id").groupby("label_seq_id")["label_comp_id"] \
                .unique().explode().tolist()
        if not return_seq_id:
            return [three2one[three_letter] for three_letter in three_letters if three_letter in three2one]
        corresponding_seq_id = cif_df.query(f"label_asym_id == '{chain_id}'") \
            .sort_values("label_seq_id").groupby("label_seq_id")["label_seq_id"] \
            .unique().explode().tolist()
        return [three2one[three_letter] for three_letter in three_letters if three_letter in three2one], corresponding_seq_id

    @staticmethod
    def match_chains(df1, df2):
        matched_chains = []
        for chain_id_1 in ProteinDFUtils.get_protein_chain_ids(df1):
            diff_result = [
                (chain_id_2, pairwise2.align.globalxx("".join(ProteinDFUtils.get_chain_res_seqs(chain_id_1, df1)), 
                                                      "".join(ProteinDFUtils.get_chain_res_seqs(chain_id_2, df2)))[0].score)
                for chain_id_2 in ProteinDFUtils.get_protein_chain_ids(df2)
            ]
            diff_result.sort(key=lambda x: x[1], reverse=True)
            matched_chains.append((chain_id_1, diff_result[0][0]))
        print(f"matched chains: {matched_chains}")
        return matched_chains
    
    @staticmethod
    def match_ligand_atoms(df1, df2) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
    

    @staticmethod
    def update_ligand_atom_name(
        df_with_ligand: pd.DataFrame,
        ligand_atom_mapping: list[tuple[tuple[str, str], str]] | None,
        updated_ligand_name: str,
    ) -> pd.DataFrame:
        """Harmonize residue and atom names for ligand atoms in the reference dataframe.
        Parameters
        ----------
        df_with_ligand : pd.DataFrame
            DataFrame that contains only ligand atoms extracted from the reference structure.
        ligand_atom_mapping : list | None
            Mapping produced by `mol_utils.get_ligand_atom_name_mapping_from_ligand_and_chai_lab`.
            Each element looks like `((ligand_res_name, ligand_atom_name), large_mol_atom_name)`.
            When *None*, no renaming is performed and the original DataFrame is returned.
        updated_ligand_name : str
            The residue name (comp_id) used for the ligand in the predicted structure. All
            corresponding rows in the reference DataFrame will be replaced with this value.
        """

        if ligand_atom_mapping is None:
            return df_with_ligand

        for (ligand_name, ligand_atom_name), large_mol_atom_name in ligand_atom_mapping:
            mask = (
                (df_with_ligand["label_comp_id"] == ligand_name)
                & (df_with_ligand["label_atom_id"] == ligand_atom_name)
            )
            # 一次性更新 comp_id 与 atom_name，避免第一次更新 comp_id 导致第二次匹配失败
            df_with_ligand.loc[mask, ["label_comp_id", "label_atom_id"]] = [
                updated_ligand_name,
                large_mol_atom_name,
            ]

        return df_with_ligand

    @staticmethod
    def align_two_chains(chain_1, chain_2):
        chain_1.loc[:, "label_seq_id"] = chain_1["label_seq_id"].astype(int) - chain_1["label_seq_id"].astype(int).min()
        chain_2.loc[:, "label_seq_id"] = chain_2["label_seq_id"].astype(int) - chain_2["label_seq_id"].astype(int).min()
        max_common_atoms = 0
        
        if len(chain_1) > len(chain_2):
            chain_2 = chain_2.copy()
        else:
            chain_1 = chain_1.copy()
        
        for i in range(abs(len(chain_1) - len(chain_2)) + 1):
            set1 = set(zip(chain_1["label_seq_id"], chain_1["label_atom_id"]))
            set2 = set(zip(chain_2["label_seq_id"], chain_2["label_atom_id"]))
            common_set = set1 & set2
            if len(common_set) > max_common_atoms:
                max_common_atoms = len(common_set)
                best_i = i
            if len(chain_1) > len(chain_2):
                chain_2.loc[:, "label_seq_id"] += 1
            else:
                chain_1.loc[:, "label_seq_id"] += 1
            
        return best_i

    

def send_file_to_remote(file_path, url="http://psi-cmd.koishi.me:8000"):

    files = {'file': open(file_path, 'rb')}

    response = requests.post(url, files=files)

    print(response.status_code)
    print(response.text)