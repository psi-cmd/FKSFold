from __future__ import annotations

from Bio.Data.IUPACData import protein_letters_3to1

# Mapping of three-letter to one-letter amino acid codes (upper-cased)
three2one = {k.upper(): v.upper() for k, v in protein_letters_3to1.items()}

# Ligand residue names to be treated as invalid/non-ligand for our purposes
INVALID_LIGAND_RES_NAMES = [
    "HOH", "ZN", "NA", "CL", "K", "MG", "CA", "MN", "FE", "CU"
]

