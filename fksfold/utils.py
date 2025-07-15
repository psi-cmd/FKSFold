from chai_lab.utils.tensor_utils import tensorcode_to_string
from typing import Tuple
import torch
import requests
from Bio import pairwise2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .config import global_config


INVALID_LIGAND_RES_NAMES = ["HOH", "ZN", "NA", "CL", "K", "MG", "CA", "MN", "FE", "CU"]

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

three2one = {k.upper(): v.upper() for k, v in protein_letters_3to1.items()}

class ProteinDFUtils:
    def __init__(self, pdbid, cif_file=None):
        self.pdbid = pdbid
        if cif_file is not None:
            self.cif = PandasMmcif().read_mmcif(cif_file)
        else:
            self.cif = PandasMmcif()
        

    def add_atom(self, chain_id, res_seq, atom_name, atom_pos):
        pass

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
    def get_chain_res_seqs(chain_id, cif_df):
        three_letters = cif_df.query(f"label_asym_id == '{chain_id}'") \
                .sort_values("label_seq_id").groupby("label_seq_id")["label_comp_id"] \
                .unique().explode().tolist()
        return [three2one[three_letter] for three_letter in three_letters if three_letter in three2one]

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
    def _kabsch_square_error_and_derivative(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Rotate coords2 onto coords1, return Σ‖P−Q_rot‖² and ∂/∂coords1."""
        assert coords1.shape == coords2.shape and coords1.shape[0] >= 3

        # 1. 去质心
        P = coords1 - coords1.mean(0, keepdims=True)
        Q = coords2 - coords2.mean(0, keepdims=True)

        # 2. Kabsch：Q → P
        H = Q.T @ P                   # 3×3
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt                    # 关键行：U · Vᵀ

        # 若出现反射，修正最后一列
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        # 3. 旋转并计算误差
        Q_rot = Q @ R
        diff  = P - Q_rot
        square_error = np.sum(diff ** 2)

        # 4. ∂/∂P
        grad_P = 2 * diff
        grad_coords1 = grad_P - grad_P.mean(0, keepdims=True)

        return square_error, grad_coords1
    
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

    bias_cache = {}

    @classmethod
    def calculate_square_error_between_matched_chains_and_derivative(
        cls,
        df_update,
        df_ref,
        total_atoms: int,
        ligand_atom_name_map: dict[str, dict[str, str]] | None = None,
        sigma_next: float | None = None,
        fk_sigma_threshold: float = 1.0,
        particle = None,
        **kwargs,
    ):
        # matched_chains = ProteinDFUtils.match_chains(df1, df2)
        matched_chains = ProteinDFUtils.match_chains(df_update, df_ref)
        # matched_chains = [(chain_id, chain_id) for chain_id in matched_chains]
        
        deriv_array = np.zeros((total_atoms, 3), dtype=np.float32)

        coords1_list, coords2_list, index_list = [], [], []

        for chain_id_1, chain_id_2 in matched_chains:
            chain_1 = df_update.query(f"label_asym_id == '{chain_id_1}'")
            chain_2 = df_ref.query(f"label_asym_id == '{chain_id_2}'")
            # reassign res_seq_id start from same value
            chain_1.loc[:, "label_seq_id"] = chain_1["label_seq_id"].astype(int) - chain_1["label_seq_id"].astype(int).min()
            chain_2.loc[:, "label_seq_id"] = chain_2["label_seq_id"].astype(int) - chain_2["label_seq_id"].astype(int).min()
            # if (chain_id_1, chain_id_2) in cls.bias_cache:
            #     best_i = cls.bias_cache[(chain_id_1, chain_id_2)]
            # else:
            #     best_i = ProteinDFUtils.align_two_chains(chain_1, chain_2)
            #     cls.bias_cache[(chain_id_1, chain_id_2)] = best_i
            # if len(chain_1) > len(chain_2):
            #     chain_2.loc[:, "label_seq_id"] += best_i
            # else:
            #     chain_1.loc[:, "label_seq_id"] += best_i
            # match atoms by res_seq_id and atom_name, and calculate rmsd
            chain_1_atoms = chain_1[["label_seq_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z", "atom_index"]].copy()
            chain_2_atoms = chain_2[["label_seq_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"]].copy()

            chain_1_atoms.columns = ["res_seq_id", "label_atom_id", "Cartn_x1", "Cartn_y1", "Cartn_z1", "atom_index"]
            chain_2_atoms.columns = ["res_seq_id", "label_atom_id", "Cartn_x2", "Cartn_y2", "Cartn_z2"]

            chain_merged = chain_1_atoms.merge(chain_2_atoms, on=["res_seq_id", "label_atom_id"], how="inner")

            if chain_merged.empty:
                continue  # no matched atoms

            coords1_list.append(
                chain_merged[["Cartn_x1", "Cartn_y1", "Cartn_z1"]].to_numpy(dtype=float)
            )
            coords2_list.append(
                chain_merged[["Cartn_x2", "Cartn_y2", "Cartn_z2"]].to_numpy(dtype=float)
            )
            index_list.extend(chain_merged["atom_index"].to_numpy(dtype=int))

        # --------------------
        # 额外加入配体（非蛋白）原子的 RMSD 计算
        # --------------------
        df_update_lig = df_update[~df_update.label_comp_id.isin(three2one.keys())][
            ["label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z", "atom_index"]
        ].copy()

        df_ref_lig = df_ref[~df_ref.label_comp_id.isin(three2one.keys())][
            ["label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"]
        ].copy()

        updated_ligand_name = df_update_lig.label_comp_id.unique().tolist()
        assert len(updated_ligand_name) == 1, "Only one ligand is supported"
        updated_ligand_name = updated_ligand_name[0]
        
        # 如果提供映射，利用 update_ligand_atom_name 统一配体 comp_id 与 atom_name
        df_ref_lig = ProteinDFUtils.update_ligand_atom_name(df_ref_lig, ligand_atom_name_map, updated_ligand_name=updated_ligand_name)

        # 重命名列，准备 merge
        if not df_update_lig.empty and not df_ref_lig.empty:
            df_update_lig.columns = [
                "label_comp_id",
                "label_atom_id",
                "Cartn_x1",
                "Cartn_y1",
                "Cartn_z1",
                "atom_index",
            ]
            df_ref_lig.columns = [
                "label_comp_id",
                "label_atom_id",
                "Cartn_x2",
                "Cartn_y2",
                "Cartn_z2",
            ]

            lig_merged = df_update_lig.merge(
                df_ref_lig, on=["label_comp_id", "label_atom_id"], how="inner"
            )

            if not lig_merged.empty:
                coords1_list.append(
                    lig_merged[["Cartn_x1", "Cartn_y1", "Cartn_z1"]].to_numpy(dtype=float)
                )
                coords2_list.append(
                    lig_merged[["Cartn_x2", "Cartn_y2", "Cartn_z2"]].to_numpy(dtype=float)
                )
                index_list.extend(lig_merged["atom_index"].to_numpy(dtype=int))

        if len(coords1_list) == 0:
            # no overlap
            return 1e3, deriv_array  # large rmsd

        coords1_all = np.concatenate(coords1_list, axis=0)
        coords2_all = np.concatenate(coords2_list, axis=0)

        se_total, grad_all = ProteinDFUtils._kabsch_square_error_and_derivative(coords1_all, coords2_all)

        # map gradients back
        deriv_array[np.array(index_list, dtype=int)] = grad_all.astype(np.float32)

        # STEERING STRATEGY
        deriv_array = torch.from_numpy(deriv_array)  # shape (N,3)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deriv_array = deriv_array.to(device).float()

        if sigma_next < global_config["rmsd_sigma_threshold"]:
            # progress = (fk_sigma_threshold - sigma_next) / fk_sigma_threshold
            # protein_lr_rmsd  = kwargs["protein_lr_max"] * progress.clamp(0,1)     # lr_max 先设 0.5
            # ligand_lr_rmsd  = kwargs["ligand_lr_max"] * progress.clamp(0,1)     # lr_max 先设 0.5
            # n_matched_atoms = (deriv_array.norm(dim=1) > 0).sum().item()
            # protein_lr_rmsd *= n_matched_atoms
            # ligand_lr_rmsd *= n_matched_atoms

            ligand_indices = df_update_lig["atom_index"].to_numpy(dtype=int)
            all_indices = np.arange(deriv_array.shape[0])
            is_ligand = np.isin(all_indices, ligand_indices)  # shape: (N_atoms,)
            # deriv_array[~is_ligand] = protein_lr_rmsd * deriv_array[~is_ligand]
            # deriv_array[is_ligand] = ligand_lr_rmsd * deriv_array[is_ligand]
            deriv_array[~is_ligand] = global_config["protein_lr_max"] * deriv_array[~is_ligand]
            deriv_array[is_ligand] = global_config["ligand_lr_max"] * deriv_array[is_ligand]
            # deriv_array *= n_matched_atoms

        return se_total, deriv_array.unsqueeze(0), df_update_lig["atom_index"].to_numpy(dtype=int)

def send_file_to_remote(file_path, url="http://psi-cmd.koishi.me:8000"):

    files = {'file': open(file_path, 'rb')}

    response = requests.post(url, files=files)

    print(response.status_code)
    print(response.text)