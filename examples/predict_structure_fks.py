import logging
from pathlib import Path
import numpy as np
import sys
import os
import uuid
from itertools import product

import multiprocessing as mp
from multi_gpu import gpu_map

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
        self.param_grid = {}

        self.create_config()

    def create_config(self):
        # try import param from config.py
        try:
            from config import param_grid
            self.param_grid = param_grid
        except ImportError:
            self.param_grid = {
                "protein_lr_max": [0.6],
                "ligand_lr_max": [0],
                "resampling_interval": [5],
                "fk_sigma_threshold": [0],
                "rmsd_sigma_threshold": [1, 2, 4, 6, 8],
                "lambda_weight": [10.0],
                "ita": [0.15, 0.17, 0.19, 0.21, 0.23, 0.25],
            }

        for params in product(*self.param_grid.values()):
            self.configs.append(dict(zip(self.param_grid.keys(), params)))


    def __iter__(self):
        for config in self.configs:
            yield config

    def param_dict_format(self, config):
        return "_".join([f"{v}" for k, v in config.items() if k in self.param_grid.keys()])

    def save_progress(self):
        # multiprocess safe
        with mp.Lock():
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
print(scheduler.param_grid)
@gpu_map
def run(config):
    from fksfold.config import update_global_config

    random_str = str(uuid.uuid4())
    tmp_dir = Path(f"./result/tmp_{random_str}")
    os.makedirs(tmp_dir, exist_ok=True)

    fasta_file = sys.argv[1]

    with open(fasta_file, "r") as f:
        fasta_context = f.read().strip()
        fasta_path = tmp_dir / Path(fasta_file).name
        fasta_path.write_text(fasta_context)

    update_global_config(**config)
    output_dir = tmp_dir / f"outputs_{scheduler.param_dict_format(config)}"
    os.makedirs(output_dir, exist_ok=True)

    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        # constraint_path="./path_to_contact.restraints",
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        num_particles=1,  # number of diffusion paths
        resampling_interval=config["resampling_interval"],  # diffusion path length
        lambda_weight=config["lambda_weight"],  # lower this to, say 2.0, to make it more random
        potential_type="vanilla",  # "diff" or "max" or "vanilla"
        fk_sigma_threshold=config["fk_sigma_threshold"],
        num_trunk_samples=1,
        seed=None,
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

def if_port_is_open(host, port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


if __name__ == "__main__":
    # if not if_port_is_open("psi-cmd.koishi.me", 8000):
    #     print("upload server is not open, please check if the server is running")
    #     exit()
    run(scheduler.configs)
# cif_paths = candidates.cif_paths
# scores = [rd.aggregate_score for rd in candidates.ranking_data]

# # Load pTM, ipTM, pLDDTs and clash scores
# scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))


