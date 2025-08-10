#! /usr/bin/env python3

from biopandas.mmcif import PandasMmcif
from seq_align import map_indices
# --- Minimal re-implementation of required utilities from fksfold.utils ---
from Bio.Data.IUPACData import protein_letters_3to1
from Bio import pairwise2
import numpy as np

INVALID_LIGAND_RES_NAMES = [
    "HOH", "ZN", "NA", "CL", "K", "MG", "CA", "MN", "FE", "CU",
]

three2one = {k.upper(): v.upper() for k, v in protein_letters_3to1.items()}


class ProteinDFUtils:
    """Subset of fksfold.utils.ProteinDFUtils needed by this script.

    Only the static methods referenced below are implemented so that this
    script can run without installing the full *fksfold* package.
    """

    # ------------- Basic chain utilities -------------
    @staticmethod
    def get_protein_chain_ids(cif_df):
        """Return chain IDs that belong to protein residues (based on 3-letter codes).

        Parameters
        ----------
        cif_df : pd.DataFrame
            Combined ATOM/HETATM dataframe from *biopandas* where each row
            represents an atom.
        """
        filtered_df = cif_df[cif_df.label_comp_id.isin(three2one.keys())]
        return filtered_df.label_asym_id.unique().tolist()

    # ------------- Ligand helpers -------------
    @staticmethod
    def get_ligand_res_names(cif_df):
        """Return residue names considered as ligands (non-protein, non-metals)."""
        residue_name_candidates = cif_df[~cif_df.label_comp_id.isin(three2one.keys())].label_comp_id.unique().tolist()
        return [name for name in residue_name_candidates if name not in INVALID_LIGAND_RES_NAMES]

    # ------------- Sequence extraction -------------
    @staticmethod
    def get_chain_res_seqs(chain_id, cif_df, return_seq_id: bool = False):
        """Extract residue sequence for a given chain.

        Returns either
        --------
        seq : List[str]
            List of 1-letter residue codes.
        OR (when *return_seq_id* is True)
        seq, seq_ids : Tuple[List[str], List[int]]
            Sequence and corresponding residue IDs.
        """
        # Collect unique residue names in order of *label_seq_id*
        three_letters = (
            cif_df.query(f"label_asym_id == '{chain_id}'")
            .sort_values("label_seq_id")
            .groupby("label_seq_id")["label_comp_id"]
            .unique()
            .explode()
            .tolist()
        )

        if not return_seq_id:
            return [three2one[tl] for tl in three_letters if tl in three2one]

        corresponding_seq_id = (
            cif_df.query(f"label_asym_id == '{chain_id}'")
            .sort_values("label_seq_id")
            .groupby("label_seq_id")["label_seq_id"]
            .unique()
            .explode()
            .tolist()
        )
        return [three2one[tl] for tl in three_letters if tl in three2one], corresponding_seq_id

    # ------------- Chain matching -------------
    @staticmethod
    def match_chains(df1, df2):
        """Match chains between two structures based on global sequence alignment."""
        matched_chains = []
        for chain_id_1 in ProteinDFUtils.get_protein_chain_ids(df1):
            diff_result = [
                (
                    chain_id_2,
                    pairwise2.align.globalxx(
                        "".join(ProteinDFUtils.get_chain_res_seqs(chain_id_1, df1)),
                        "".join(ProteinDFUtils.get_chain_res_seqs(chain_id_2, df2)),
                    )[0].score,
                )
                for chain_id_2 in ProteinDFUtils.get_protein_chain_ids(df2)
            ]
            diff_result.sort(key=lambda x: x[1], reverse=True)
            matched_chains.append((chain_id_1, diff_result[0][0]))

        print(f"matched chains: {matched_chains}")
        return matched_chains

    # ------------- Kabsch algorithm -------------
    @staticmethod
    def kabsch(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Return coordinate differences after optimal superposition (coords2→coords1)."""
        assert coords1.shape == coords2.shape and coords1.shape[0] >= 3

        P = coords1 - coords1.mean(0, keepdims=True)
        Q = coords2 - coords2.mean(0, keepdims=True)
        H = Q.T @ P
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        Q_rot = Q @ R
        diff = P - Q_rot
        return diff, R

# ---------------------------------------------------------------------------
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np


def main(target_file: str, ref_file: str, around_ang: float = 5.0):
    # get molecule from cif file, no solvent
    assert target_file.endswith(".cif") and ref_file.endswith(".cif")
    parser = PandasMmcif()
    target_struct = parser.read_mmcif(target_file)
    target_df = pd.concat([target_struct.df["ATOM"], target_struct.df["HETATM"]], ignore_index=True)
    ref_struct = parser.read_mmcif(ref_file)
    ref_df = pd.concat([ref_struct.df["ATOM"], ref_struct.df["HETATM"]], ignore_index=True)

    # remove Hs
    target_df = target_df[target_df.type_symbol != "H"]
    ref_df = ref_df[ref_df.type_symbol != "H"]

    residue_map = get_residue_mapping(target_df, ref_df)
    interface_residues_id_in_ref = get_interface_residues_id(ref_df, around_ang)

    target_coords = np.empty((0, 3), dtype=float)
    ref_coords = np.empty((0, 3), dtype=float)
    for ref_residue_id in interface_residues_id_in_ref:
        target_residue_id = residue_map[ref_residue_id]
        target_atoms = target_df[(target_df.label_asym_id == target_residue_id[0]) & (target_df.label_seq_id == target_residue_id[1])]
        ref_atoms = ref_df[(ref_df.label_asym_id == ref_residue_id[0]) & (ref_df.label_seq_id == ref_residue_id[1])]
        assert len(target_atoms) == len(ref_atoms), f"target_atoms: {target_atoms}, ref_atoms: {ref_atoms}"
        target_coords = np.concatenate([target_coords, target_atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].values])
        ref_coords = np.concatenate([ref_coords, ref_atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].values])
    rmsd = align_coord_and_calculate_rmsd(target_coords, ref_coords)
    print(rmsd)

def get_residue_mapping(target_df: pd.DataFrame, ref_df: pd.DataFrame) -> dict:
    def _mapping_no_overlap(mapping: list[tuple[int, int]]) -> dict:
        return len(mapping) == len(set(list(zip(*mapping))[0])) == len(set(list(zip(*mapping))[1]))

    chain_id_mapping = ProteinDFUtils.match_chains(target_df, ref_df)
    assert _mapping_no_overlap(chain_id_mapping)
    residue_maps = {}
    for target_chain_id, ref_chain_id in chain_id_mapping:
        # get map indices of target_df and ref_df
        target_chain_seq, target_seq_id = ProteinDFUtils.get_chain_res_seqs(target_chain_id, target_df, return_seq_id=True)
        target_chain_seq = "".join(target_chain_seq)
        ref_chain_seq, ref_seq_id = ProteinDFUtils.get_chain_res_seqs(ref_chain_id, ref_df, return_seq_id=True)
        ref_chain_seq = "".join(ref_chain_seq)
        residue_map = map_indices(target_chain_seq, ref_chain_seq)
        seq_id_map = [(target_seq_id[i], ref_seq_id[j]) for i, j in residue_map]
        assert _mapping_no_overlap(seq_id_map)
        for target_seq_id, ref_seq_id in seq_id_map:
            residue_maps[(ref_chain_id, ref_seq_id)] = (target_chain_id, target_seq_id)
    return residue_maps

def get_interface_residues_id(df: pd.DataFrame, around_ang: float = 5.0) -> list[tuple[int, int]]:
    interface_atoms = get_interface_atoms(df, around_ang)
    # Group by both chain_id and seq_id, then aggregate unique residue names
    interface_seq_id = interface_atoms.groupby(["label_asym_id", "label_seq_id"]).agg({"label_comp_id": "unique"}).reset_index()
    return [(chain_id, seq_id) for chain_id, seq_id in zip(interface_seq_id["label_asym_id"], interface_seq_id["label_seq_id"])]

def get_interface_atoms(df: pd.DataFrame, around_ang: float = 5.0) -> pd.DataFrame:
    ligand_res_names = ProteinDFUtils.get_ligand_res_names(df)
    assert len(ligand_res_names) > 0
    print(str(len(ligand_res_names)) + " Ligands found: " + ", ".join(ligand_res_names))

    ligand_atoms = df[df.label_comp_id.isin(ligand_res_names)]
    not_ligand_atoms = df[~df.label_comp_id.isin(ligand_res_names)]
    # ckdtree for ligand atoms
    tree = cKDTree(ligand_atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].values)
    # get atoms around ligand atoms
    ligand_atoms_around = tree.query_ball_point(not_ligand_atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].values, around_ang)
    mask = [len(neighbors) > 0 for neighbors in ligand_atoms_around]
    # get atoms around ligand atoms
    interface_atoms = not_ligand_atoms[mask]
    return interface_atoms


def align_coord_and_calculate_rmsd(target_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    assert target_coords.shape == ref_coords.shape, f"target_coords.shape: {target_coords.shape}, ref_coords.shape: {ref_coords.shape}"

    diff, _ = ProteinDFUtils.kabsch(target_coords, ref_coords)
    rmsd = np.sqrt(np.sum(diff ** 2) / target_coords.shape[0])
    return rmsd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align interface residues between two CIF files.")
    parser.add_argument("target_file", type=str, help="Path to the target CIF file")
    parser.add_argument("ref_file", type=str, help="Path to the reference CIF file")
    parser.add_argument("--around", type=float, default=5.0, help="Distance threshold for interface atoms")
    args = parser.parse_args()

    main(args.target_file, args.ref_file, args.around)