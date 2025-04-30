import tempfile
import logging
import shutil
from pathlib import Path
import numpy as np

from fksfold.chai_fks import run_inference

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

tmp_dir = Path(tempfile.mkdtemp())

fasta_name = "8cyo"
example_fasta = """
>protein|8cyo-protein
MKKGHHHHHHGAISLISALVRAHVDSNPAMTSLDYSRFQANPDYQMSGDDTQHIQQFYDLLTGSMEIIRGWAEKIPGFADLPKADQDLLFESAFLELFVLRLAYRSNPVEGKLIFCNGVVLHRLQCVRGFGEWIDSIVEFSSNLQNMNIDISAFSCIAALAMVTERHGLKEPKRVEELQNKIVNTLKDHVTFNNGGLNRPNYLSKLLGKLPELRTLCTQGLQRIFYLKLEDLVPPPAIIDKLFLDTLPF
>ligand|8cyo-ligand
c1cc(c(cc1OCC(=O)NCCS)Cl)Cl
""".strip()
fasta_path = tmp_dir / "example.fasta"
fasta_path.write_text(example_fasta)


output_dir = tmp_dir / "outputs"
# if you want to use ft steering:
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # constraint_path="./path_to_contact.restraints",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    num_particles=3,         # number of diffusion paths
    resampling_interval=10,  # diffusion path length
    lambda_weight=10.0,      # lower this to, say 2.0, to make it more random
    potential_type="diff",   # "diff" or "max" or "vanilla"
    fk_sigma_threshold=float("inf"),
    num_trunk_samples=1,
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    low_memory=True,
)
cif_paths = candidates.cif_paths
scores = [rd.aggregate_score for rd in candidates.ranking_data]

# Load pTM, ipTM, pLDDTs and clash scores
scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))
