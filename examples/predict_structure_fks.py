import tempfile
import logging
import shutil
from pathlib import Path
import numpy as np
import sys
import os
import uuid
from itertools import product

# add parent directory before path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fksfold.chai_fks import run_inference

logging.basicConfig(level=logging.INFO)  # control verbosity

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

# tmp_dir = Path(tempfile.mkdtemp())


class ConfigScheduler:
    def __init__(self):
        self.configs = []

        self.create_config()

    def create_config(self):
        param_grid = {
            "protein_lr_max": [0.6],
            "ligand_lr_max": [0.6],
            "resampling_interval": [1],
            "fk_sigma_threshold": [2],
            "rmsd_sigma_threshold": [10],
            "lambda_weight": [12.0, 15.0, 18.0],
        }

        for params in product(*param_grid.values()):
            self.configs.append(dict(zip(param_grid.keys(), params)))


    def __iter__(self):
        for config in self.configs:
            yield config

    @staticmethod
    def param_dict_format(config):
        return f"{config['protein_lr_max']}_{config['ligand_lr_max']}_{config['resampling_interval']}_{config['fk_sigma_threshold']}_{config['rmsd_sigma_threshold']}_{config['lambda_weight']}"

    def save_progress(self):
        import pickle
        with open("progress.pkl", "wb") as f:
            pickle.dump(self.configs, f)

    def load_progress(self):
        import pickle
        with open("progress.pkl", "rb") as f:
            self.configs = pickle.load(f)

scheduler = ConfigScheduler()
if os.path.exists("progress.pkl"):
    scheduler.load_progress()
# FKS version: Score=0.9383
# if you want to use ft steering:
from fksfold.config import update_global_config
for config in scheduler:
    random_str = str(uuid.uuid4())
    tmp_dir = Path(f"./result/tmp_{random_str}")
    os.makedirs(tmp_dir, exist_ok=True)

    fasta_file = sys.argv[1]

    with open(fasta_file, "r") as f:
        fasta_context = f.read().strip()
        fasta_path = tmp_dir / Path(fasta_file).name
        fasta_path.write_text(fasta_context)

    update_global_config(**config)
    output_dir = tmp_dir / f"outputs_{ConfigScheduler.param_dict_format(config)}"
    os.makedirs(output_dir, exist_ok=True)

    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        # constraint_path="./path_to_contact.restraints",
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        num_particles=4,  # number of diffusion paths
        resampling_interval=config["resampling_interval"],  # diffusion path length
        lambda_weight=config["lambda_weight"],  # lower this to, say 2.0, to make it more random
        potential_type="vanilla",  # "diff" or "max" or "vanilla"
        fk_sigma_threshold=config["fk_sigma_threshold"],
        num_trunk_samples=1,
        seed=42,
        device="cuda:0",
        use_esm_embeddings=True,
        low_memory=False,
        use_msa_server=False,
        ref_structure_file=sys.argv[2],
        # rmsd_strength=float(sys.argv[3]),  # from 0 to 1, how strong the RMSD force is
        protein_lr_max=config["protein_lr_max"],
        ligand_lr_max=config["ligand_lr_max"],
        # save_intermediate=True,
    )

    scheduler.save_progress()

cif_paths = candidates.cif_paths
scores = [rd.aggregate_score for rd in candidates.ranking_data]

# Load pTM, ipTM, pLDDTs and clash scores
scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))


