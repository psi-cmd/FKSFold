from chai_lab.utils.tensor_utils import tensorcode_to_string
from typing import Tuple
import torch



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
        return cif_df[~cif_df.label_comp_id.isin(three2one.keys())].label_comp_id.unique().tolist()

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
                (chain_id_2, difflib.SequenceMatcher(None, ProteinDFUtils.get_chain_res_seqs(chain_id_1, df1), ProteinDFUtils.get_chain_res_seqs(chain_id_2, df2)).ratio())
                for chain_id_2 in ProteinDFUtils.get_protein_chain_ids(df2)
            ]
            diff_result.sort(key=lambda x: x[1], reverse=True)
            matched_chains.append((chain_id_1, diff_result[0][0]))
        return matched_chains
    
    @staticmethod
    def match_ligand_atoms(df1, df2) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
    
    @staticmethod
    def _kabsch_rmsd_and_derivative(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute RMSD after optimal superposition AND its gradient w.r.t coords1 (moving set)."""
        assert coords1.shape == coords2.shape and coords1.shape[0] >= 3, "Need at least 3 matched atoms"

        # Center the coordinate sets
        P = coords1 - coords1.mean(axis=0, keepdims=True)  # moving
        Q = coords2 - coords2.mean(axis=0, keepdims=True)  # reference

        # Kabsch alignment (rotate Q onto P)
        H = P.T @ Q
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        Q_rot = (R @ Q.T).T

        diff = P - Q_rot                        # shape (N,3)
        N    = coords1.shape[0]
        rmsd = np.sqrt((diff ** 2).sum() / N)

        # Gradient w.r.t P (coords1):  diff / (N * rmsd)
        grad_P = diff / (N * rmsd + 1e-8)

        # Centering: P = coords1 - mean_P  => d/dcoords1 = grad_P - mean(grad_P)
        grad_coords1 = grad_P - grad_P.mean(axis=0, keepdims=True)

        return rmsd, grad_coords1
    
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
    def calculate_rmsd_between_matched_chains_and_derivative(
        df_update,
        df_ref,
        total_atoms: int,
        ligand_atom_name_map: dict[str, dict[str, str]] | None = None,
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

        rmsd_total, grad_all = ProteinDFUtils._kabsch_rmsd_and_derivative(coords1_all, coords2_all)

        # map gradients back
        deriv_array[np.array(index_list, dtype=int)] = grad_all.astype(np.float32)

        return rmsd_total, deriv_array, df_update_lig["atom_index"].to_numpy(dtype=int)
