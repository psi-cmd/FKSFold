from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from biopandas.mmcif import PandasMmcif
from biopandas.pdb import PandasPdb

from chai_lab.utils.tensor_utils import tensorcode_to_string
from chai_lab.data.io.cif_utils import _tensor_to_atom_names, get_chain_letter

from .utils import ProteinDFUtils
from .constants import three2one
from .seq_align import map_indices
from .ligand.mapping import get_ligand_atom_name_mapping_from_ligand_and_chai_lab
from ..config import global_config


def _clean_ref_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["type_symbol"] != "H"]
    df = df[df["label_comp_id"] != "HOH"]
    df = df[df["label_alt_id"].isna() | (df["label_alt_id"] == "A")]
    return df


def _predicted_atoms_to_df(inputs: dict, atom_pos: torch.Tensor) -> pd.DataFrame:
    """Build a DataFrame aligned with biopandas columns for current batch=0."""
    sc = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    asym_id = sc["token_asym_id"][0]
    res_idx = sc["token_residue_index"][0]
    res_name3 = sc["token_residue_name"][0]
    atom_token = sc["atom_token_index"][0]
    atom_name_chr = sc["atom_ref_name_chars"][0]
    exists_mask = sc["atom_exists_mask"][0]

    res_name3_str = [tensorcode_to_string(x).strip() for x in res_name3]
    atom_names = _tensor_to_atom_names(atom_name_chr)
    chain_letters = [get_chain_letter(int(i)) if i > 0 else "UNK" for i in asym_id]

    result: List[list] = []
    for a_idx in torch.where(exists_mask)[0].tolist():
        t_idx = atom_token[a_idx].item()
        key = [
            chain_letters[t_idx],
            int(res_idx[t_idx].item()) + 1,
            res_name3_str[t_idx].strip(),
            atom_names[a_idx].strip(),
        ]
        # positions on CPU for DataFrame; keep float
        result.append(key + atom_pos[0, a_idx].detach().cpu().tolist() + [a_idx])

    return pd.DataFrame(
        result,
        columns=[
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "label_atom_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "atom_index",
        ],
    )


def _torch_kabsch_square_error_and_derivative(
    coords1: torch.Tensor, coords2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate coords2 onto coords1 (optimal superposition), return sum of squared errors and d/dcoords1.

    Args:
        coords1: (N, 3) tensor on target device
        coords2: (N, 3) tensor on target device
    Returns:
        square_error: scalar tensor
        grad_coords1: (N, 3) tensor
    """
    assert coords1.shape == coords2.shape and coords1.shape[-1] == 3
    P = coords1 - coords1.mean(dim=0, keepdim=True)
    Q = coords2 - coords2.mean(dim=0, keepdim=True)

    H = Q.T @ P
    U, _, Vh = torch.linalg.svd(H)
    R = U @ Vh
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1, :] *= -1
        R = U @ Vh

    Q_rot = Q @ R
    diff = P - Q_rot
    se = (diff * diff).sum()

    grad_P = 2.0 * diff
    grad_coords1 = grad_P - grad_P.mean(dim=0, keepdim=True)
    return se, grad_coords1


class MSESteer:
    """MSE steering helper.

    - Initialize with a reference structure file path (mmCIF or PDB)
    - Optionally provide a FASTA file to derive ligand atom mapping only once
    - Expose compute() to return current MSE and derivative on device
    - Provide normalize_to_delta() to align gradient norm with diffusion delta
    """

    def __init__(
        self,
        ref_structure_file: str | Path,
        fasta_file: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.ref_structure_file = str(ref_structure_file)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Load and clean reference atoms DataFrame (protein + ligand)
        if self.ref_structure_file.endswith(".cif"):
            ref_df = PandasMmcif().read_mmcif(self.ref_structure_file).df["ATOM"]
        elif self.ref_structure_file.endswith(".pdb"):
            ref_df = PandasPdb().read_pdb(self.ref_structure_file).df["ATOM"]
        else:
            raise ValueError(f"Unsupported file type: {self.ref_structure_file}")
        self.ref_df = _clean_ref_df(ref_df)

        # Precompute ligand subset and KD-tree for interface selection
        self.ref_lig_df = self.ref_df[~self.ref_df.label_comp_id.isin(three2one.keys())][
            ["label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"]
        ].copy()
        self._lig_kdtree = None
        if not self.ref_lig_df.empty:
            lig_coords = self.ref_lig_df[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(dtype=float)
            if lig_coords.shape[0] >= 1:
                self._lig_kdtree = cKDTree(lig_coords)

        # Optional: cache ligand atom renaming mapping
        self._ligand_atom_mapping = None
        if fasta_file is not None:
            smiles = self._get_molecularglue_smiles(fasta_file)
            try:
                self._ligand_atom_mapping = get_ligand_atom_name_mapping_from_ligand_and_chai_lab(
                    self.ref_structure_file, smiles
                )
            except Exception:
                # Fallback: no mapping; steering still works for protein-only
                self._ligand_atom_mapping = None

    @staticmethod
    def _get_molecularglue_smiles(fasta_file: str) -> str:
        with open(fasta_file, "r") as f:
            for line in f:
                if line.startswith(">ligand"):
                    return f.readline().strip()
        raise ValueError("No smiles found in fasta file")

    @torch.no_grad()
    def compute(
        self,
        inputs: dict,
        atom_pos: torch.Tensor,
        *,
        sigma_next: float | None = None,
        fk_sigma_threshold: float = 1.0,
    ) -> Tuple[float, torch.Tensor, np.ndarray]:
        """Compute MSE and derivative for current atom coordinates.

        Returns:
            (square_error, derivative[1,N,3] on device, ligand_indices_in_pred)
        """
        # Build predicted atoms df
        pred_df = _predicted_atoms_to_df(inputs, atom_pos)
        total_atoms = atom_pos.shape[1]

        # Chain matching and atom merges
        matched_chains = ProteinDFUtils.match_chains(pred_df, self.ref_df)

        deriv_array = np.zeros((total_atoms, 3), dtype=np.float32)
        coords1_list: List[np.ndarray] = []
        coords2_list: List[np.ndarray] = []
        index_list: List[int] = []

        for chain_id_1, chain_id_2 in matched_chains:
            chain_1 = pred_df.query(f"label_asym_id == '{chain_id_1}'")
            chain_2 = self.ref_df.query(f"label_asym_id == '{chain_id_2}'")

            chain_1 = chain_1.copy()
            chain_2 = chain_2.copy()
            chain_1.loc[:, "label_seq_id"] = chain_1["label_seq_id"].astype(int) - chain_1["label_seq_id"].astype(int).min()
            chain_2.loc[:, "label_seq_id"] = chain_2["label_seq_id"].astype(int) - chain_2["label_seq_id"].astype(int).min()

            unique_ids1 = sorted(chain_1.label_seq_id.unique())
            pos2seq1 = {pos: seqid for pos, seqid in enumerate(unique_ids1)}
            unique_ids2 = sorted(chain_2.label_seq_id.unique())
            pos2seq2 = {pos: seqid for pos, seqid in enumerate(unique_ids2)}

            chain_1_seq = "".join(ProteinDFUtils.get_chain_res_seqs(chain_id_1, pred_df))
            chain_2_seq = "".join(ProteinDFUtils.get_chain_res_seqs(chain_id_2, self.ref_df))
            mapping = map_indices(chain_1_seq, chain_2_seq)
            seqid_map = [(pos2seq1[i], pos2seq2[j]) for i, j in mapping]

            if len(seqid_map) == 0:
                continue

            chain_1_seqid, chain_2_seqid = zip(*seqid_map)
            chain_2_filtered = chain_2[chain_2["label_seq_id"].isin(chain_2_seqid)].copy()
            seqid_map_reversed = dict(zip(chain_2_seqid, chain_1_seqid))
            chain_2_filtered["label_seq_id"] = chain_2_filtered["label_seq_id"].map(seqid_map_reversed)
            chain_2 = chain_2_filtered

            chain_1_atoms = chain_1[["label_seq_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z", "atom_index"]].copy()
            chain_2_atoms = chain_2[["label_seq_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"]].copy()

            chain_1_atoms.columns = ["res_seq_id", "label_atom_id", "Cartn_x1", "Cartn_y1", "Cartn_z1", "atom_index"]
            chain_2_atoms.columns = ["res_seq_id", "label_atom_id", "Cartn_x2", "Cartn_y2", "Cartn_z2"]

            chain_merged = chain_1_atoms.merge(chain_2_atoms, on=["res_seq_id", "label_atom_id"], how="inner")
            if chain_merged.empty:
                continue

            coords1_list.append(chain_merged[["Cartn_x1", "Cartn_y1", "Cartn_z1"]].to_numpy(dtype=float))
            coords2_list.append(chain_merged[["Cartn_x2", "Cartn_y2", "Cartn_z2"]].to_numpy(dtype=float))
            index_list.extend(chain_merged["atom_index"].to_numpy(dtype=int))

        # Ligand atoms merge
        df_update_lig = pred_df[~pred_df.label_comp_id.isin(three2one.keys())][
            ["label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z", "atom_index"]
        ].copy()
        df_ref_lig = self.ref_df[~self.ref_df.label_comp_id.isin(three2one.keys())][
            ["label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"]
        ].copy()

        ligand_indices_pred = df_update_lig["atom_index"].to_numpy(dtype=int)

        if not df_update_lig.empty and not df_ref_lig.empty:
            # updated ligand name for mapping
            updated_ligand_name = df_update_lig.label_comp_id.unique().tolist()
            if len(updated_ligand_name) == 1:
                updated_ligand_name = updated_ligand_name[0]
                df_ref_lig = ProteinDFUtils.update_ligand_atom_name(
                    df_ref_lig, self._ligand_atom_mapping, updated_ligand_name=updated_ligand_name
                )

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

            lig_merged = df_update_lig.merge(df_ref_lig, on=["label_comp_id", "label_atom_id"], how="inner")
            if not lig_merged.empty:
                coords1_list.append(lig_merged[["Cartn_x1", "Cartn_y1", "Cartn_z1"]].to_numpy(dtype=float))
                coords2_list.append(lig_merged[["Cartn_x2", "Cartn_y2", "Cartn_z2"]].to_numpy(dtype=float))
                index_list.extend(lig_merged["atom_index"].to_numpy(dtype=int))

        if len(coords1_list) == 0:
            return 1e3, torch.zeros((1, total_atoms, 3), device=self.device, dtype=torch.float32), ligand_indices_pred

        coords1_all = np.concatenate(coords1_list, axis=0)
        coords2_all = np.concatenate(coords2_list, axis=0)

        # Interface-specific selection (optional)
        interfacial_radius = global_config.get("interfacial_radius", 0)
        use_interface = interfacial_radius is not None and interfacial_radius > 0 and self._lig_kdtree is not None

        if use_interface:
            distances, _ = self._lig_kdtree.query(coords2_all)
            mask = distances <= interfacial_radius
            if np.count_nonzero(mask) >= 3:
                coords1_all = coords1_all[mask]
                coords2_all = coords2_all[mask]

        # Torch Kabsch on device
        t_coords1 = torch.from_numpy(coords1_all).to(self.device, dtype=torch.float32)
        t_coords2 = torch.from_numpy(coords2_all).to(self.device, dtype=torch.float32)
        se_total, grad_sel = _torch_kabsch_square_error_and_derivative(t_coords1, t_coords2)

        # If used interface mask, need to expand back to all matched atoms
        if use_interface:
            grad_all_np = np.zeros_like(np.asarray(coords1_all if not use_interface else np.concatenate(coords1_list, axis=0)))
            # recompute mask to place gradients properly relative to the matched subset
            distances, _ = self._lig_kdtree.query(np.concatenate(coords2_list, axis=0))
            mask = distances <= interfacial_radius
            if np.count_nonzero(mask) >= 3:
                grad_all_np[mask] = grad_sel.detach().cpu().numpy()
                grad_all = torch.from_numpy(grad_all_np).to(self.device, dtype=torch.float32)
            else:
                grad_all = grad_sel
        else:
            grad_all = grad_sel

        # Map gradients back to full atom array
        deriv_array = torch.zeros((total_atoms, 3), device=self.device, dtype=torch.float32)
        deriv_array[torch.as_tensor(np.array(index_list, dtype=int), device=self.device)] = grad_all.to(torch.float32)

        ligand_indices = torch.as_tensor(ligand_indices_pred, device=self.device)
        all_indices = torch.arange(deriv_array.shape[0], device=self.device)
        is_ligand = torch.isin(all_indices, ligand_indices)
        deriv_array[~is_ligand] = global_config.get("protein_lr_max", 1.0) * deriv_array[~is_ligand]
        deriv_array[is_ligand] = global_config.get("ligand_lr_max", 1.0) * deriv_array[is_ligand]

        return float(se_total.detach().cpu().item()), deriv_array.unsqueeze(0), ligand_indices_pred

    @staticmethod
    def normalize_to_delta(g_raw: torch.Tensor, d_i: torch.Tensor) -> torch.Tensor:
        """Normalize gradient norm to match delta norm (per VecRectified FFJORD-style steering).

        Args:
            g_raw: (1, N, 3) or (N, 3)
            d_i: (1, N, 3) or (N, 3)
        Returns:
            normalized gradient with same shape as g_raw
        """
        if g_raw.dim() == 3:
            g_flat = g_raw.reshape(g_raw.shape[0], -1)
            d_flat = d_i.reshape(d_i.shape[0], -1)
            d_norm = torch.linalg.norm(d_flat, dim=1, keepdim=True).unsqueeze(1)
            g_norm = torch.linalg.norm(g_flat, dim=1, keepdim=True).unsqueeze(1)
            return g_raw / (g_norm + 1e-8) * d_norm
        else:
            g_flat = g_raw.reshape(1, -1)
            d_flat = d_i.reshape(1, -1)
            d_norm = torch.linalg.norm(d_flat, dim=1, keepdim=True).unsqueeze(1)
            g_norm = torch.linalg.norm(g_flat, dim=1, keepdim=True).unsqueeze(1)
            return g_raw / (g_norm + 1e-8) * d_norm


# Backward compatibility alias
MSESteerer = MSESteer

