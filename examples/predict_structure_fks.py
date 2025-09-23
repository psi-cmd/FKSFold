import logging
from pathlib import Path
import numpy as np
import sys
import os
import uuid
from itertools import product
import re

# import pyplot and set backend to agg
import matplotlib
matplotlib.use("Agg")

import multiprocessing as mp
# from multi_gpu import gpu_map, gpu_map_debug

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
        # try import param from config.py
        try:
            from config import param_grid
        except ImportError:
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

    def _fmt(self, v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def param_dict_format(self, config):
        params = "_".join([f"{self._fmt(config[k])}" for k in self.logging_params])
        if "smiles" in config:
            params += f"_{config['smiles'][0]}"
        return params

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

import ray
# Local Ray runtime with Tune for hyper-parameter optimisation
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import optuna
# Use local Ray; ignore repeated inits when notebook re-runs
ray.init(ignore_reinit_error=True, _temp_dir="/tmp/ray")


# The core folding routine (was remote before, now runs locally and returns a score)
def run(config, fasta_file, cif_file, target_file, param_format_str: str = None):
    
    from fksfold.config import update_global_config

    random_str = str(uuid.uuid4())
    tmp_dir = Path(f"./result/tmp_{random_str}")
    os.makedirs(tmp_dir, exist_ok=True)

    with open(fasta_file, "r") as f:
        fasta_context = f.read().strip()
        fasta_path = tmp_dir / Path(fasta_file).name
        fasta_path.write_text(fasta_context)

    update_global_config(**config)
    output_dir = tmp_dir / f"outputs_{param_format_str or scheduler.param_dict_format(config)}"
    os.makedirs(output_dir, exist_ok=True)

    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        # constraint_path="./path_to_contact.restraints",
        num_trunk_recycles=3,
        num_diffn_timesteps=220,
        num_particles=config["num_particles"],  # number of diffusion paths
        resampling_interval=config["resampling_interval"],  # diffusion path length
        lambda_weight=config["lambda_weight"],  # lower this to, say 2.0, to make it more random
        potential_type="vanilla",  # "diff" or "max" or "vanilla"
        fk_sigma_threshold=config["fk_sigma_threshold"],
        num_trunk_samples=1,
        seed=config["seed"],
        device="cuda:0",
        use_esm_embeddings=True,
        low_memory=False,
        use_msa_server=False,
        ref_structure_file=cif_file,
        # rmsd_strength=float(sys.argv[3]),  # from 0 to 1, how strong the RMSD force is
        protein_lr_max=config["protein_lr_max"],
        ligand_lr_max=config["ligand_lr_max"],
        # save_intermediate=True,
    )

    # save optimisation progress and report score
    scheduler.save_progress()
    # score = calculate_rmsd_with_usalign(candidates.cif_paths[0], target_file)
    score = 0.0
    return score

def calculate_rmsd_with_usalign(cif_path, ref_cif_path):
    # get output from subprocess
    output = subprocess.run(["USalign", cif_path, ref_cif_path], capture_output=True, text=True)
    # extract rmsd from output
    rmsd = re.search(r"RMSD=\s*(\d*?\.\d*)", output.stdout)
    return float(rmsd.group(1))

def fasta_sub_ligand(fasta_base:Path, smiles: tuple[str, str]):
    regex = re.compile(r"^>ligand\|(.*)\n(.*)\n", re.MULTILINE)
    with open(fasta_base, "r") as f:
        fasta_context = f.read().strip()
    fasta_context = regex.sub(rf">ligand|{smiles[0]}\n{smiles[1]}\n", fasta_context)
    new_fasta_path = fasta_base.parent / f"{fasta_base.stem}_{smiles[0]}.fasta"
    with open(new_fasta_path, "w") as f:
        f.write(fasta_context)
    return new_fasta_path

# -----------------------------------------------------------
# Ray Tune objective function
# -----------------------------------------------------------

def run_trial(trial_config):
    """Objective for Ray Tune hyper-parameter optimisation."""
    from fksfold.config import update_global_config

    # Constant parameters that are not part of the search space
    base_cfg = {
        "protein_lr_max": 1,  # disable diffusion steering now
        "ligand_lr_max": 0,  # this is for protac, matching warhead automatically
        "fk_sigma_threshold": 0,
        "lambda_weight": 10.0,
        "ref_file": "9nfr_clean.cif",
        "rmsd_diffusion_steering_threshold": 51.28504910256902,
        "ita": 0.8910603983681669,
        "rmsd_cutoff": 1.0223982331084525,
        "rmsd_sigma_threshold": 0,
        "resampling_interval": 5,
        "num_particles": 1,
    }

    cfg = {**base_cfg, **trial_config}
    update_global_config(**cfg)

    proj_dir = Path(__file__).resolve().parent.parent
    # fasta_file = proj_dir / "examples" / "Ripk1_VHL.fasta"
    fasta_file = proj_dir / "examples" / "glue_example.fasta"
    fasta_file = fasta_sub_ligand(fasta_file, cfg["smiles"])
    cif_file = proj_dir / "examples" / cfg["ref_file"]
    target_file = proj_dir / "examples" / "9nfr_clean.cif"
    param_format_str = "_".join([f"{trial_config[k]}" if k != "smiles" else f"{trial_config[k][0]}" for k in trial_config.keys()])
    print(param_format_str)
    score = run(cfg, str(fasta_file), str(cif_file), str(target_file), param_format_str)
    print(f"Score: {score}")
    tune.report({"score": score})



if __name__ == "__main__":

    # Search space for the three hyper-parameters
    import glob
    # ref_files = glob.glob("../state3_L42/center_*.cif")
    search_space = {
        "seed": tune.randint(0, 1000000),
        # "seed": tune.choice([None]),
        "smiles": tune.choice([
            # ("CPD1", "COC1=CSC=C1C1=CN(N=N1)C1CCC(=O)NC1=O"),
            # ("CPD2", "CSC1=CC=C(C=C1)C1(CCOC1)NC(=O)NC1=CC2=C(C=C1)C(=O)N(C2)C1CCC(=O)NC1=O"),
            # ("CPD3", "FC1=CC(=CC=C1N1CCN(CC2=CC=C(COC3=CC=CC4=C3CN(C3CCC(=O)NC3=O)C4=O)C=C2)CC1)C#N"),
            # ("CPD4", "O=C(NC1=CC2=C(C=C1)C(=O)N(C2)C1CCC(=O)NC1=O)N1CC2(C1)CCC1=C2C=CC=C1"),
            # ("CPD5", "[H]N(CC1=CC2=NC=C(N2C=C1)N1CCC(=O)NC1=O)C(=O)[C@@H]1CC2=C(CN1S(=O)(=O)CC1=CC=CC=C1)C=CC=C2")
            ("RMS-246", "CN1C=CC(COC2=CC=C(C=C2)C2=CC=CC(N3CCC(=O)NC3=O)=C2Cl)=N1"),
            # ("VAV1-013", "CN1C=CC(COC2=CC=C(C=C2)C2=CC=CN([C@H]3CCC(=O)NC3=O)C2=O)=N1")
            ("A1BYX", "CN1C=CC(COC2=CC=C(C=C2)C2=CC=CC([C@H]3CCC(=O)NC3=O)=C2Cl)=N1")
        ])
    }

    random_sampler = optuna.samplers.RandomSampler(seed=2025)
    # grid_sampler = optuna.samplers.GridSampler(search_space)

    algo = OptunaSearch(metric="score", mode="min", sampler=random_sampler)
    scheduler_asha = ASHAScheduler(metric="score", mode="min")

    analysis = tune.run(
        run_trial,
        name="RMS-246 and A1BYX",
        search_alg=algo,
        scheduler=scheduler_asha,
        config=search_space,
        resources_per_trial={"cpu": 8, "gpu": 1},
        resume="AUTO",
        num_samples=100,
    )
    best_config = analysis.get_best_config(metric="score", mode="min")

    print("Best hyper-parameters found:", best_config)

# cif_paths = candidates.cif_paths
# scores = [rd.aggregate_score for rd in candidates.ranking_data]

# # Load pTM, ipTM, pLDDTs and clash scores
# scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))


