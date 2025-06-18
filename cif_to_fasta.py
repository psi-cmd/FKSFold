import gemmi
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from gemmi import EntityType

from Bio.Data import IUPACData

import sys
import requests

aa_map = dict(IUPACData.protein_letters_3to1)
aa_map["CYM"] = "C"
aa_map["Cym"] = "C"

def cif_to_fasta_and_smiles(cif_str, fasta_file):
    doc = gemmi.cif.read_string(cif_str)
    structure = gemmi.make_structure_from_block(doc.sole_block())

    seq_records = []
    ligands = []
    metals = []

    # 1. 蛋白链
    for chain in structure[0]:
        seq = ""
        if chain[0].entity_type == EntityType.Polymer:
            for res in chain:
                resname = res.name.capitalize()
                aa = aa_map.get(resname, "X")
                seq += aa
            record = SeqRecord(Seq(seq),
                                id=f'protein|Chain_{chain.name}',
                                description=f'Sequence extracted from chain {chain.name}')
            seq_records.append(record)
        elif chain[0].entity_type == EntityType.NonPolymer:
            for res in chain:
                if len(res) == 1:
                    metals.append(res[0])
                else:
                    ligands.append(res)
        elif chain[0].entity_type == EntityType.Water:
            pass
        else:
            breakpoint()
    # 配体SMILES
    for lig in ligands:
        chem_comp_id = lig.name.strip()
        response = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{chem_comp_id}")
        response.raise_for_status()
        chem_comp = response.json()
        smiles = chem_comp['rcsb_chem_comp_descriptor']['smilesstereo']
        ligid = lig.name.strip()
        record = SeqRecord(Seq(smiles),
                           id=f'ligand|{ligid}',
                           description=f'Ligand SMILES from {ligid}')
        seq_records.append(record)

    # 金属离子
    for metal in metals:
        charge = "+2"
        smiles = f"[{metal.name[:2].capitalize()}{charge}]"
        record = SeqRecord(Seq(smiles),
                           id=f'ligand|{metal.name}',
                           description=f'Metal ion {metal.name}')
        seq_records.append(record)
    # 写入FASTA
    SeqIO.write(seq_records, fasta_file, 'fasta-2line')



if __name__ == "__main__":
    if sys.argv[1].startswith("http"):
        response = requests.get(sys.argv[1])
        response.raise_for_status()
        cif_str = response.text
    else:
        with open(sys.argv[1], 'r') as f:
            cif_str = f.read()
    cif_to_fasta_and_smiles(cif_str, 'output.fasta')
