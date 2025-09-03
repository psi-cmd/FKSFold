import os
import uuid
from pathlib import Path
import re
import subprocess

from fksfold.chai_fks import run_inference

# cif_file = Path(__file__).resolve().parent / ".." / "state3_L42" / "center_14_macro_2.cif"
# fasta_file = Path(__file__).resolve().parent / "Ripk1_VHL.fasta"
# target_file = Path(__file__).resolve().parent / "9nfr_clean.cif"

cif_file = Path(__file__).resolve().parent / "state1.cif"
fasta_file = Path(__file__).resolve().parent / "glue_example.fasta"

def param_dict_format(config):
    logging_params = ["rmsd_sigma_threshold", "ita", "rmsd_cutoff"]
    return "_".join([f"{v}" for k, v in config.items() if k in logging_params]) + "_" + cif_file.name


def run(config):
    from fksfold.config import update_global_config

    random_str = str(uuid.uuid4())
    tmp_dir = Path(f"/tmp/result/tmp_{random_str}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    with open(str(fasta_file), "r") as f:
        fasta_context = f.read().strip()
        fasta_path = tmp_dir / Path(fasta_file).name
        fasta_path.write_text(fasta_context)

    update_global_config(**config)
    output_dir = tmp_dir / f"outputs_{param_dict_format(config)}"
    os.makedirs(output_dir, exist_ok=True)

    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        # constraint_path="./path_to_contact.restraints",
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        num_particles=config["num_particles"],  # number of diffusion paths
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
        ref_structure_file=str(cif_file),
        # rmsd_strength=float(sys.argv[3]),  # from 0 to 1, how strong the RMSD force is
        protein_lr_max=config["protein_lr_max"],
        ligand_lr_max=config["ligand_lr_max"],
        save_intermediate=True,
    )

    # score = calculate_rmsd_with_usalign(candidates.cif_paths[0], str(target_file))
    # print(f"Score: {score}")
    # return score

def calculate_rmsd_with_usalign(cif_path, ref_cif_path):
    # get output from subprocess
    output = subprocess.run(["USalign", cif_path, ref_cif_path], capture_output=True, text=True)
    # extract rmsd from output
    rmsd = re.search(r"RMSD=\s*(\d*?\.\d*)", output.stdout)
    return float(rmsd.group(1))

if __name__ == "__main__":
    import numpy as np
    config = {
        "rmsd_diffusion_steering_threshold": 51.28504910256902,
        "ita": 0.8910603983681669,
        "rmsd_cutoff": 1.0223982331084525,
        "fk_sigma_threshold": 0,
        "resampling_interval": 5,
        "lambda_weight": 1.0,
        "protein_lr_max": 1,
        "ligand_lr_max": 0,
        "seed": None,
        "num_particles": 1,
        "ref_file": "9nfr_clean.cif",
        "rmsd_sigma_threshold": 0,
        "interfacial_radius": 0,
    }
    run(config)


