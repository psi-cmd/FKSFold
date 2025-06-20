import tempfile
import logging
import shutil
from pathlib import Path
import numpy as np
import sys
import os

# add parent directory before path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fksfold.chai_fks import run_inference

logging.basicConfig(level=logging.INFO)  # control verbosity

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

tmp_dir = Path(tempfile.mkdtemp())

fasta_file = sys.argv[1]

with open(fasta_file, 'r') as f:
    fasta_context = f.read().strip()
    fasta_path = tmp_dir / Path(fasta_file).name
    fasta_path.write_text(fasta_context)


# FKS version: Score=0.9383
output_dir = tmp_dir / "outputs"
# if you want to use ft steering:
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # constraint_path="./path_to_contact.restraints",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    num_particles=4,         # number of diffusion paths
    resampling_interval=5,  # diffusion path length
    lambda_weight=10.0,      # lower this to, say 2.0, to make it more random
    potential_type="vanilla",   # "diff" or "max" or "vanilla"
    # fk_sigma_threshold=float("inf"),
    num_trunk_samples=1,
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    low_memory=False,
    use_msa_server=True,
)

cif_paths = candidates.cif_paths
scores = [rd.aggregate_score for rd in candidates.ranking_data]

# Load pTM, ipTM, pLDDTs and clash scores
scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))
