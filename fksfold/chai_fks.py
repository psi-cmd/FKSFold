# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

# Modifications Copyright (c) 2025 YDS Pharmatech.
# This file is derived from software originally developed by Chai Discovery, Inc.
# and has been modified by YDS Pharmatech.

import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import warnings
from enum import Enum

import numpy as np
from sympy import assuming
import torch
import torch.export
from einops import einsum, rearrange, repeat
from torch import Tensor
from tqdm import tqdm

from chai_lab.data.collate.collate import Collate
from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
from chai_lab.data.dataset.all_atom_feature_context import (
    MAX_MSA_DEPTH,
    MAX_NUM_TEMPLATES,
    AllAtomFeatureContext,
)
from chai_lab.data.dataset.msas.utils import (
    subsample_and_reorder_msa_feats_n_mask,
)
from chai_lab.data.features.generators.token_bond import TokenBondRestraint
from chai_lab.data.io.cif_utils import get_chain_letter, save_to_cif
from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
from chai_lab.model.utils import center_random_augmentation
from chai_lab.ranking.frames import get_frames_and_mask
from chai_lab.ranking.rank import SampleRanking, get_scores, rank
from chai_lab.utils.plot import plot_msa
from chai_lab.utils.tensor_utils import move_data_to_device, set_seed, und_self

from chai_lab.chai1 import (
    feature_factory,
    DiffusionConfig,
    StructureCandidates,  # plddt, pae, pde
    make_all_atom_feature_context,
    load_exported,
    raise_if_too_many_tokens,
    raise_if_too_many_templates,
    raise_if_msa_too_deep,
)
import chai_lab.ranking.ptm as ptm

from .utils import build_restype_mapping

from biopandas.mmcif import PandasMmcif
from biopandas.pdb import PandasPdb
from .config import global_config
from .utils import send_file_to_remote
# %%
# Inference logic
@torch.no_grad()
def run_inference(
    fasta_file: Path,
    *,
    output_dir: Path,
    # Configuration for ESM, MSA, constraints, and templates
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    # Parameters controlling how we do inference
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    # num_diffn_samples: int = 5,
    num_trunk_samples: int = 1,
    # Diffusion inference time scaling
    num_particles: int = 2,
    resampling_interval: int = 5,
    lambda_weight: float = 10.0,
    potential_type: str = "vanilla",
    fk_sigma_threshold: float = 1.0,
    # Misc
    seed: int | None = None,
    device: str | None = None,
    low_memory: bool = True,
    **kwargs
) -> StructureCandidates:
    # Check for deprecated num_diffn_samples in kwargs
    kwargs["fasta_file"] = fasta_file
    if 'num_diffn_samples' in kwargs:
        warnings.warn(
            "'num_diffn_samples' value is ignored in the fks implementation. "
            "Use 'num_trunk_samples' to control the number of independent folding runs.",
            DeprecationWarning
        )
        kwargs.pop('num_diffn_samples') # Remove it from kwargs

    assert num_trunk_samples > 0
    if output_dir.exists():
        assert not any(
            output_dir.iterdir()
        ), f"Output directory {output_dir} is not empty."

    torch_device = torch.device(device if device is not None else "cuda:0")

    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_file,
        output_dir=output_dir,
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        templates_path=template_hits_path,
        esm_device=torch_device,
    )

    all_candidates: list[StructureCandidates] = []
    for trunk_idx in range(num_trunk_samples):
        logging.info(f"Trunk sample {trunk_idx + 1}/{num_trunk_samples}")
        cand = run_folding_on_context(
            feature_context,
            output_dir=(
                output_dir / f"trunk_{trunk_idx}"
                if num_trunk_samples > 1
                else output_dir
            ),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            # num_diffn_samples=num_diffn_samples,
            recycle_msa_subsample=recycle_msa_subsample,
            # diffusion inference time scaling
            num_particles=num_particles,
            resampling_interval=resampling_interval,
            lambda_weight=lambda_weight,
            potential_type=potential_type,
            fk_sigma_threshold=fk_sigma_threshold,
            # misc
            seed=seed + trunk_idx if seed is not None else None,
            device=torch_device,
            low_memory=low_memory,
            **kwargs
        )
        all_candidates.append(cand)
    return StructureCandidates.concat(all_candidates)


def _bin_centers(min_bin: float, max_bin: float, no_bins: int) -> Tensor:
    return torch.linspace(min_bin, max_bin, 2 * no_bins + 1)[1::2]


class PotentialType(Enum):
    VANILLA = "vanilla"  # vanilla method
    DIFF = "diff"  # diff method
    MAX = "max"  # max method
    
    @classmethod
    def from_str(cls, value: str) -> "PotentialType":
        """Convert string to PotentialType"""
        value = value.lower()
        if value == "default":
            value = "vanilla"
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"Invalid potential type: {value}. Must be one of {valid_values}")


@dataclass 
class ParticleState:
    atom_pos: Tensor
    plddt: Tensor | None = None
    interface_ptm: Tensor | None = None
    avg_interface_ptm: float | None = None
    historical_ptm: float | None = None 
    rmsd: float | None = float("inf")
    rmsd_derivative: Tensor | None = None


class ParticleFilter:
    def __init__(
        self,
        num_particles: int,
        resampling_interval: int = 5,
        lambda_weight: float = 10.0,
        potential_type: PotentialType = PotentialType.VANILLA,
        fk_sigma_threshold: float = 1.0,
        device: torch.device = torch.device("cuda")
    ):
        self.num_particles = num_particles
        self.resampling_interval = resampling_interval
        self.lambda_weight = lambda_weight
        self.potential_type = potential_type
        self.restraint_sigma_threshold = fk_sigma_threshold
        self.device = device
        self.particles: list[ParticleState] = []
        
    def initialize_particles(self, batch_size: int, num_atoms: int, sigma: float, device: torch.device):
        """Initialize particles with different random noise"""
        self.particles = []
        for _ in range(self.num_particles):
            initial_atom_pos = sigma * torch.randn(
                batch_size, num_atoms, 3, device=device
            )
            particle = ParticleState(atom_pos=initial_atom_pos)
            self.particles.append(particle)
            
    def should_resample(self, step_idx: int, sigma_next: float) -> bool:
        return (step_idx > 0 and 
                step_idx % self.resampling_interval == 0 and 
                (sigma_next < self.restraint_sigma_threshold or sigma_next < global_config["rmsd_sigma_threshold"] ))
    
    def resample(self) -> None:
        """Resample particles based on their scores"""
        if not all(p.avg_interface_ptm is not None for p in self.particles):
            return
        if global_config["rmsd_sigma_threshold"] == 0 and global_config["fk_sigma_threshold"] == 0:
            return
        
        current_scores = torch.zeros(len(self.particles), device=self.device)
        # Get current scores
        if global_config["current_sigma"] < global_config["rmsd_sigma_threshold"]:
            rmsd_scores = torch.tensor([-p.rmsd / 20 for p in self.particles], device=self.device)
            print("rmsd_scores:", rmsd_scores)
            current_scores += rmsd_scores
        if global_config["current_sigma"] < global_config["fk_sigma_threshold"]:
            ptm_scores = torch.tensor([p.avg_interface_ptm for p in self.particles], device=self.device)
            print("ptm_scores:", ptm_scores)
            current_scores += ptm_scores
        print("current_scores:", current_scores)
        
        # Get historical scores (if exists)
        historical_scores = torch.tensor([
            p.historical_ptm if p.historical_ptm is not None else float('0') 
            for p in self.particles
        ], device=self.device)

        # Calculate weights based on different potential types
        if self.potential_type == PotentialType.VANILLA:
            # Vanilla method
            weights = torch.exp(self.lambda_weight * current_scores)
        elif self.potential_type == PotentialType.DIFF:
            # Diff method - make sure diffs are not all 0
            diffs = current_scores - historical_scores
            weights = torch.exp(self.lambda_weight * diffs + 1e-6)
        elif self.potential_type == PotentialType.MAX:
            # Max method
            weights = torch.exp(self.lambda_weight * torch.max(current_scores, historical_scores))
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")
        weights = weights.clamp(min=1e-6)
        
        # Calculate sampling probabilities
        probs = weights / weights.sum()
        
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logging.warning(f"Invalid probabilities detected: {probs}")
            probs = torch.ones_like(probs) / len(probs)
        
        # Multinomial sampling with lambda weight
        indices = torch.multinomial(probs, num_samples=self.num_particles, replacement=True)
        
        # Create new particle list based on "indices" sampling
        new_particles = []
        for idx in indices:
            # Create a new particle with copied state
            new_particle = ParticleState(
                atom_pos=self.particles[idx].atom_pos.clone(),
                plddt=self.particles[idx].plddt.clone() if self.particles[idx].plddt is not None else None,
                interface_ptm=self.particles[idx].interface_ptm.clone() if self.particles[idx].interface_ptm is not None else None,
                avg_interface_ptm=self.particles[idx].avg_interface_ptm,
                historical_ptm=self.particles[idx].avg_interface_ptm  # update historical PTM to current PTM
            )
            new_particles.append(new_particle)
            
        self.particles = new_particles

    def get_best_particle(self) -> ParticleState:
        """Return the particle with highest score"""
        best_idx = max(range(len(self.particles)), 
                      key=lambda i: self.particles[i].avg_interface_ptm if self.particles[i].avg_interface_ptm is not None else float('-inf'))
        return self.particles[best_idx]


@torch.no_grad()
def run_folding_on_context(
    feature_context: AllAtomFeatureContext,
    *,
    output_dir: Path,
    # expose some params for easy tweaking
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    # diffusion inference time scaling
    num_particles: int = 2,  # HACK 6: diffusion inference time scaling, change to > 1
    resampling_interval: int = 5,  # For path resampling
    lambda_weight: float = 10.0,  # For path resampling
    potential_type: str = "vanilla",
    fk_sigma_threshold: float = 1.0,
    # all diffusion samples come from the same trunk - num_diffn_samples removed
    seed: int | None = None,
    device: torch.device | None = None,
    low_memory: bool,
    ref_structure_file: Path,
    rmsd_strength: float = 0.0,
    **kwargs,
) -> StructureCandidates:
    """
    Function for in-depth explorations.
    User completely controls folding inputs.
    """

    if kwargs.get("save_intermediate", False):
        warnings.warn("Saving intermediate results is deprecated and needs to be reimplemented.", DeprecationWarning)

    # Set seed
    if seed is not None:
        set_seed([seed])

    if device is None:
        device = torch.device("cuda:0")

    # Clear memory
    torch.cuda.empty_cache()

    if ref_structure_file.endswith(".cif"):
        ref_df = PandasMmcif().read_mmcif(ref_structure_file).df["ATOM"]
        ref_df = clean_df(ref_df)
    elif ref_structure_file.endswith(".pdb"):
        ref_df = PandasPdb().read_pdb(ref_structure_file).df["ATOM"]
        ref_df = clean_df(ref_df)
    else:
        raise ValueError(f"Unsupported file type: {ref_structure_file}")

    ##
    ## Validate inputs
    ##

    n_actual_tokens = feature_context.structure_context.num_tokens
    raise_if_too_many_tokens(n_actual_tokens)
    raise_if_too_many_templates(feature_context.template_context.num_templates)
    raise_if_msa_too_deep(feature_context.msa_context.depth)
    # NOTE: profile MSA used only for statistics; no depth check
    feature_context.structure_context.report_bonds()

    ##
    ## Prepare batch
    ##

    # Collate inputs into batch
    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )

    feature_contexts = [feature_context]
    batch_size = len(feature_contexts)
    batch = collator(feature_contexts)

    if not low_memory:
        batch = move_data_to_device(batch, device=device)

    # Get features and inputs from batch
    features = {name: feature for name, feature in batch["features"].items()}
    inputs = batch["inputs"]
    block_indices_h = inputs["block_atom_pair_q_idces"]
    block_indices_w = inputs["block_atom_pair_kv_idces"]
    atom_single_mask = inputs["atom_exists_mask"]
    atom_token_indices = inputs["atom_token_index"].long()
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    token_reference_atom_index = inputs["token_ref_atom_index"]
    atom_within_token_index = inputs["atom_within_token_index"]
    msa_mask = inputs["msa_mask"]
    template_input_masks = und_self(
        inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
    )
    block_atom_pair_mask = inputs["block_atom_pair_mask"]
    ##
    ## Load exported models
    ##

    _, _, model_size = msa_mask.shape
    assert model_size in AVAILABLE_MODEL_SIZES

    feature_embedding = load_exported("feature_embedding.pt", device)
    bond_loss_input_proj = load_exported("bond_loss_input_proj.pt", device)
    token_input_embedder = load_exported("token_embedder.pt", device)
    trunk = load_exported("trunk.pt", device)
    diffusion_module = load_exported("diffusion_module.pt", device)
    confidence_head = load_exported("confidence_head.pt", device)

    ##
    ## Run the features through the feature embedder
    ##

    embedded_features = feature_embedding.forward(
        crop_size=model_size,
        move_to_device=device,
        return_on_cpu=low_memory,
        **features,
    )
    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
        "TOKEN_PAIR"
    ].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
        "ATOM"
    ].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = (
        embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    )
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    ##
    ## Bond feature generator
    ## Separate from other feature embeddings due to export limitations
    ##

    bond_ft_gen = TokenBondRestraint()
    bond_ft = bond_ft_gen.generate(batch=batch).data
    trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(
        return_on_cpu=low_memory,
        move_to_device=device,
        crop_size=model_size,
        input=bond_ft,
    ).chunk(2, dim=-1)
    token_pair_input_feats += trunk_bond_feat
    token_pair_structure_input_feats += structure_bond_feat

    ##
    ## Run the inputs through the token input embedder
    ##

    token_input_embedder_outputs: tuple[Tensor, ...] = token_input_embedder.forward(
        return_on_cpu=low_memory,
        move_to_device=device,
        token_single_input_feats=token_single_input_feats,
        token_pair_input_feats=token_pair_input_feats,
        atom_single_input_feats=atom_single_input_feats,
        block_atom_pair_feat=block_atom_pair_input_feats,
        block_atom_pair_mask=block_atom_pair_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_single_mask=atom_single_mask,
        atom_token_indices=atom_token_indices,
        crop_size=model_size,
    )
    token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = (
        token_input_embedder_outputs
    )

    ##
    ## Run the input representations through the trunk
    ##

    # Recycle the representations by feeding the output back into the trunk as input for
    # the subsequent recycle
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr
    for _ in tqdm(range(num_trunk_recycles), desc="Trunk recycles"):
        subsampled_msa_input_feats, subsampled_msa_mask = None, None
        if recycle_msa_subsample > 0:
            subsampled_msa_input_feats, subsampled_msa_mask = (
                subsample_and_reorder_msa_feats_n_mask(
                    msa_input_feats,
                    msa_mask,
                )
            )
        (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
            move_to_device=device,
            token_single_trunk_initial_repr=token_single_initial_repr,
            token_pair_trunk_initial_repr=token_pair_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,  # recycled
            token_pair_trunk_repr=token_pair_trunk_repr,  # recycled
            msa_input_feats=(
                subsampled_msa_input_feats
                if subsampled_msa_input_feats is not None
                else msa_input_feats
            ),
            msa_mask=(
                subsampled_msa_mask if subsampled_msa_mask is not None else msa_mask
            ),
            template_input_feats=template_input_feats,
            template_input_masks=template_input_masks,
            token_single_mask=token_single_mask,
            token_pair_mask=token_pair_mask,
            crop_size=model_size,
        )
    # We won't be using the trunk anymore; remove it from memory
    del trunk
    torch.cuda.empty_cache()

    ##
    ## Denoise the trunk representation by passing it through the diffusion module
    ##

    atom_single_mask = atom_single_mask.to(device)

    static_diffusion_inputs = dict(
        token_single_initial_repr=token_single_structure_input.float(),
        token_pair_initial_repr=token_pair_structure_input_feats.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_feats=atom_single_structure_input_feats.float(),
        atom_block_pair_input_feats=block_atom_pair_structure_input_feats.float(),
        atom_single_mask=atom_single_mask,
        atom_block_pair_mask=block_atom_pair_mask,
        token_single_mask=token_single_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_token_indices=atom_token_indices,
    )
    static_diffusion_inputs = move_data_to_device(
        static_diffusion_inputs, device=device
    )

    def _denoise(atom_pos: Tensor, sigma: Tensor, ds: int = 1) -> Tensor:
        # verified manually that ds dimension can be arbitrary in diff module
        atom_noised_coords = rearrange(
            atom_pos, "(b ds) ... -> b ds ...", ds=ds
        ).contiguous()
        noise_sigma = repeat(sigma, " -> b ds", b=batch_size, ds=ds)
        return diffusion_module.forward(
            atom_noised_coords=atom_noised_coords.float(),
            noise_sigma=noise_sigma.float(),
            crop_size=model_size,
            **static_diffusion_inputs,
        )

    inference_noise_schedule = InferenceNoiseSchedule(
        s_max=DiffusionConfig.S_tmax,
        s_min=4e-4,
        p=7.0,  # HACK 3: increase p to get better sampling
        sigma_data=DiffusionConfig.sigma_data,
    )
    sigmas = inference_noise_schedule.get_schedule(
        device=device, num_timesteps=num_diffn_timesteps
    )
    gammas = torch.where(
        (sigmas >= DiffusionConfig.S_tmin) & (sigmas <= DiffusionConfig.S_tmax),
        min(DiffusionConfig.S_churn / num_diffn_timesteps, math.sqrt(2) - 1),
        0.0,
    )

    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

    # Initialize particle filter with potential type
    particle_filter = ParticleFilter(
        num_particles=num_particles,
        resampling_interval=resampling_interval,
        lambda_weight=lambda_weight,
        potential_type=PotentialType.from_str(potential_type),
        fk_sigma_threshold=fk_sigma_threshold,
        device=device
    )

    # Initial atom positions for the particle filter
    _, num_atoms = atom_single_mask.shape
    particle_filter.initialize_particles(
        batch_size=batch_size,
        num_atoms=num_atoms,
        sigma=sigmas[0],
        device=device
    )

    # Main Diffusion Loop
    for step_idx, (sigma_curr, sigma_next, gamma_curr) in tqdm(
        enumerate(sigmas_and_gammas),
        desc="Diffusion steps",
        total=len(sigmas_and_gammas)
    ):
        # Process each particle
        for particle_idx, particle in enumerate(particle_filter.particles):
            if seed is not None:
                # Ensure different seeds for each particle at each step
                torch.manual_seed(seed + step_idx * num_particles + particle_idx)

            # Center coords - operates on single particle's pos [batch=1, atoms, 3]
            atom_pos_candidate = center_random_augmentation(
                particle.atom_pos.clone(),
                atom_single_mask=atom_single_mask,
            )

            # Alg 2. lines 4-6
            noise = DiffusionConfig.S_noise * torch.randn(
                atom_pos_candidate.shape, device=atom_pos_candidate.device
            )
            sigma_hat = sigma_curr + gamma_curr * sigma_curr
            atom_pos_noise = (sigma_hat**2 - sigma_curr**2).clamp_min(1e-6).sqrt()
            atom_pos_hat = atom_pos_candidate + noise * atom_pos_noise

            # Lines 7-8
            denoised_pos = _denoise(
                atom_pos=atom_pos_hat,
                sigma=sigma_hat,
                ds=1,
            )
            d_i = (atom_pos_hat - denoised_pos) / sigma_hat
            atom_pos_candidate = atom_pos_hat + (sigma_next - sigma_hat) * d_i

            if global_config is not None:
                global_config["current_sigma"] = sigma_next
            # Lines 9-11
            if sigma_next != 0 and DiffusionConfig.second_order:  # second order update
                denoised_pos = _denoise(
                    atom_pos_candidate,
                    sigma=sigma_next,
                    ds=1,
                )
                d_i_prime = (atom_pos_candidate - denoised_pos) / sigma_next
                atom_pos_candidate = atom_pos_candidate + (sigma_next - sigma_hat) * ((d_i_prime + d_i) / 2)

            if sigma_next < global_config["rmsd_sigma_threshold"]:
                particle.rmsd, particle.rmsd_derivative, ligand_index = get_rmsd_and_derivative(inputs, particle.atom_pos, ref_df, kwargs["fasta_file"], ref_structure_file,
                                                                                                sigma_next=sigma_next, fk_sigma_threshold=fk_sigma_threshold, protein_lr_max=kwargs["protein_lr_max"],
                                                                                                ligand_lr_max=kwargs["ligand_lr_max"], particle=particle)
                # atom_pos_candidate = atom_pos_candidate - particle.rmsd_derivative.to(device).float()
                # print(f"original diffusion step: { ((sigma_next - sigma_hat) * d_i)[0, ligand_index, :]}")
                # print(f"RMSD force: {particle.rmsd_derivative[0, ligand_index, :]}")
            particle.atom_pos = atom_pos_candidate

            if "save_intermediate" in kwargs and kwargs["save_intermediate"]:
                cif_out_path = output_dir.joinpath(f"pred.model_idx_{step_idx}_particle_{particle_idx}.cif")
                save_to_cif(
                    coords=particle.atom_pos.to(device="cpu"),
                    bfactors=None,
                    output_batch=move_data_to_device(inputs, torch.device("cpu")),
                    write_path=cif_out_path,
                    # Set asym names to be A, B, C, ...
                    asym_entity_names={
                        i: get_chain_letter(i)
                        for i in range(1, len(feature_context.chains) + 1)
                    },
                )

        # Check if we need to resample every resampling_interval steps
        if particle_filter.should_resample(step_idx, sigma_next):
            # Calculate scores for all particles
            for particle in particle_filter.particles:
                with torch.inference_mode():
                    temp_confidence = confidence_head.forward(
                        move_to_device=device,
                        token_single_input_repr=token_single_initial_repr,
                        token_single_trunk_repr=token_single_trunk_repr,
                        token_pair_trunk_repr=token_pair_trunk_repr,
                        token_single_mask=token_single_mask,
                        atom_single_mask=atom_single_mask,
                        atom_coords=particle.atom_pos,
                        token_reference_atom_index=token_reference_atom_index,
                        atom_token_index=atom_token_indices,
                        atom_within_token_index=atom_within_token_index,
                        crop_size=model_size,
                    )

                    # Calculate scores
                    temp_plddt = einsum(
                        temp_confidence[2].float().softmax(dim=-1),
                        _bin_centers(0, 1, temp_confidence[2].shape[-1]).to(device),
                        "b a d, d -> b a"
                    )
                    
                    _, valid_frames_mask = get_frames_and_mask(
                        particle.atom_pos,
                        inputs["token_asym_id"].to(device),
                        inputs["token_residue_index"].to(device),
                        inputs["token_backbone_frame_mask"].to(device),
                        inputs["token_centre_atom_index"].to(device),
                        inputs["token_exists_mask"].to(device),
                        atom_single_mask,
                        inputs["token_backbone_frame_index"].to(device),
                        atom_token_indices,
                    )

                    ptm_scores = ptm.get_scores(
                        pae_logits=temp_confidence[0].float(),
                        token_exists_mask=token_single_mask.to(device),
                        valid_frames_mask=valid_frames_mask.to(device),
                        bin_centers=_bin_centers(0.0, 32.0, 64).to(device),
                        token_asym_id=inputs["token_asym_id"].to(device),
                    )

                    # Update particle scores
                    particle.plddt = temp_plddt.detach()
                    particle.interface_ptm = ptm_scores.interface_ptm.detach()
                    particle.avg_interface_ptm = ptm_scores.interface_ptm.mean().item()
                    

            # Perform resampling
            particle_filter.resample()
            torch.cuda.empty_cache()

    # Use the best particle for final output
    best_particle = particle_filter.get_best_particle()
    atom_pos = best_particle.atom_pos.detach().clone()
    # atom_pos = center_random_augmentation(
    #     atom_pos,
    #     atom_single_mask=atom_single_mask,
    # )

    # We won't be running diffusion anymore
    del diffusion_module, static_diffusion_inputs, particle_filter
    torch.cuda.empty_cache()

    ##
    ## Run the confidence model
    ##

    confidence_outputs: list[tuple[Tensor, ...]] = [
        confidence_head.forward(
            move_to_device=device,
            token_single_input_repr=token_single_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            token_single_mask=token_single_mask,
            atom_single_mask=atom_single_mask,
            atom_coords=atom_pos[ds : ds + 1],
            token_reference_atom_index=token_reference_atom_index,
            atom_token_index=atom_token_indices,
            atom_within_token_index=atom_within_token_index,
            crop_size=model_size,
        )
        for ds in range(1)
    ]

    pae_logits, pde_logits, plddt_logits = [
        torch.cat(single_sample, dim=0)
        for single_sample in zip(*confidence_outputs, strict=True)
    ]

    assert atom_pos.shape[0] == 1
    assert pae_logits.shape[0] == 1

    def softmax_einsum_and_cpu(
        logits: Tensor, bin_mean: Tensor, pattern: str
    ) -> Tensor:
        # utility to compute score from bin logits
        res = einsum(
            logits.float().softmax(dim=-1), bin_mean.to(logits.device), pattern
        )
        return res.to(device="cpu")

    token_mask_1d = rearrange(token_single_mask, "1 b -> b")

    pae_scores = softmax_einsum_and_cpu(
        pae_logits[:, token_mask_1d, :, :][:, :, token_mask_1d, :],
        _bin_centers(0.0, 32.0, 64),
        "b n1 n2 d, d -> b n1 n2",
    )

    pde_scores = softmax_einsum_and_cpu(
        pde_logits[:, token_mask_1d, :, :][:, :, token_mask_1d, :],
        _bin_centers(0.0, 32.0, 64),
        "b n1 n2 d, d -> b n1 n2",
    )

    plddt_scores_atom = softmax_einsum_and_cpu(
        plddt_logits,
        _bin_centers(0, 1, plddt_logits.shape[-1]),
        "b a d, d -> b a",
    )

    # converting per-atom plddt to per-token
    [mask] = atom_single_mask.cpu()
    [indices] = atom_token_indices.cpu()

    def avg_per_token_1d(x):
        n = torch.bincount(indices[mask], weights=x[mask])
        d = torch.bincount(indices[mask]).clamp(min=1)
        return n / d

    plddt_scores = torch.stack([avg_per_token_1d(x) for x in plddt_scores_atom])

    ##
    ## Write the outputs
    ##
    # Move data to the CPU so we don't hit GPU memory limits
    inputs = move_data_to_device(inputs, torch.device("cpu"))
    atom_pos = atom_pos.cpu()
    plddt_logits = plddt_logits.cpu()
    pae_logits = pae_logits.cpu()

    # Plot coverage of tokens by MSA, save plot
    output_dir.mkdir(parents=True, exist_ok=True)

    if feature_context.msa_context.mask.any():
        msa_plot_path = plot_msa(
            input_tokens=feature_context.structure_context.token_residue_type,
            msa_tokens=feature_context.msa_context.tokens,
            out_fname=output_dir / "msa_depth.pdf",
        )
    else:
        msa_plot_path = None

    cif_paths: list[Path] = []
    ranking_data: list[SampleRanking] = []

    for idx in range(1):
        ##
        ## Compute ranking scores
        ##

        _, valid_frames_mask = get_frames_and_mask(
            atom_pos[idx : idx + 1],
            inputs["token_asym_id"],
            inputs["token_residue_index"],
            inputs["token_backbone_frame_mask"],
            inputs["token_centre_atom_index"],
            inputs["token_exists_mask"],
            inputs["atom_exists_mask"],
            inputs["token_backbone_frame_index"],
            inputs["atom_token_index"],
        )

        ranking_outputs: SampleRanking = rank(
            atom_pos[idx : idx + 1],
            atom_mask=inputs["atom_exists_mask"],
            atom_token_index=inputs["atom_token_index"],
            token_exists_mask=inputs["token_exists_mask"],
            token_asym_id=inputs["token_asym_id"],
            token_entity_type=inputs["token_entity_type"],
            token_valid_frames_mask=valid_frames_mask,
            lddt_logits=plddt_logits[idx : idx + 1],
            lddt_bin_centers=_bin_centers(0, 1, plddt_logits.shape[-1]).to(
                plddt_logits.device
            ),
            pae_logits=pae_logits[idx : idx + 1],
            pae_bin_centers=_bin_centers(0.0, 32.0, 64).to(pae_logits.device),
        )

        ranking_data.append(ranking_outputs)

        ##
        ## Write output files
        ##

        cif_out_path = output_dir.joinpath("..", f"pred_{param_dict_format(global_config)}.cif")
        aggregate_score = ranking_outputs.aggregate_score.item()
        print(f"Score={aggregate_score:.4f}, writing output to {cif_out_path}")

        # use 0-100 scale for pLDDT in pdb outputs
        scaled_plddt_scores_per_atom = 100 * plddt_scores_atom[idx : idx + 1]

        save_to_cif(
            coords=atom_pos[idx : idx + 1],
            bfactors=scaled_plddt_scores_per_atom,
            output_batch=inputs,
            write_path=cif_out_path,
            # Set asym names to be A, B, C, ...
            asym_entity_names={
                i: get_chain_letter(i)
                for i in range(1, len(feature_context.chains) + 1)
            },
        )
        cif_paths.append(cif_out_path)
        send_file_to_remote(cif_out_path)

        scores_out_path = output_dir.joinpath(f"scores.model_idx_{idx}.npz")

        np.savez(scores_out_path, **get_scores(ranking_outputs))

    return StructureCandidates(
        cif_paths=cif_paths,
        ranking_data=ranking_data,
        msa_coverage_plot_path=msa_plot_path,
        pae=pae_scores,
        pde=pde_scores,
        plddt=plddt_scores,
    )


import torch
from chai_lab.utils.tensor_utils import tensorcode_to_string
from chai_lab.data.io.cif_utils import _tensor_to_atom_names, get_chain_letter
from fksfold.utils import ProteinDFUtils
import pandas as pd

def predicted_atoms_to_df(inputs: dict, atom_pos: torch.Tensor):
    """
    Return { (chain, resSeq, atomName): coord } for current batch (batch=0)
    """
    sc = move_data_to_device(inputs, torch.device("cpu"))
    # 取 batch=0
    asym_id      = sc["token_asym_id"][0]          # (N_res,)
    res_idx      = sc["token_residue_index"][0]    # (N_res,)
    res_name3    = sc["token_residue_name"][0]     # (N_res,8)
    atom_token   = sc["atom_token_index"][0]       # (N_atom,)
    atom_name_chr= sc["atom_ref_name_chars"][0]    # (N_atom,4)
    exists_mask  = sc["atom_exists_mask"][0]       # (N_atom,)

    # 预先解码
    res_name3_str = [tensorcode_to_string(x).strip() for x in res_name3]
    # Ensure the tensor is on CPU to avoid device-related issues during conversion
    atom_names  = _tensor_to_atom_names(atom_name_chr.cpu())
    chain_letters = [get_chain_letter(int(i)) if i > 0 else "UNK" for i in asym_id]

    # 建表
    result = []
    for a_idx in torch.where(exists_mask)[0].tolist():
        t_idx   = atom_token[a_idx].item()
        key = [
            chain_letters[t_idx],              # 'A'
            int(res_idx[t_idx].item()) + 1,    
            res_name3_str[t_idx].strip(),
            atom_names[a_idx].strip(),         # 'CA', 'N', ...
        ]
        result.append(key + atom_pos[0, a_idx].cpu().tolist() + [a_idx])

    
    return pd.DataFrame(result, columns=["label_asym_id", "label_seq_id", "label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z", "atom_index"])

def get_rmsd_and_derivative(inputs, atom_pos, ref_atoms_df, fasta_file, ref_structure_file, **kwargs):
    predicted_atoms_df = predicted_atoms_to_df(inputs, atom_pos)
    total_atoms = atom_pos.shape[1]
    ligand_atom_name_mapping = get_ligand_atom_name_mapping(ref_structure_file, get_molecularglue_smiles(fasta_file))
    rmsd, rmsd_derivative, ligand_index = ProteinDFUtils.calculate_rmsd_between_matched_chains_and_derivative(predicted_atoms_df, ref_atoms_df, total_atoms, ligand_atom_name_mapping, **kwargs)
    return rmsd, rmsd_derivative, ligand_index

def clean_df(df):
    # remove Hydrogen atoms
    df = df[df["type_symbol"] != "H"]
    # remove water molecules
    df = df[df["label_comp_id"] != "HOH"]
    # remove alternative locations
    df = df[df["label_alt_id"].isna() | (df["label_alt_id"] == "A")]
    return df

from .mol_utils import get_ligand_atom_name_mapping_from_ligand_and_chai_lab

import functools

def only_exec_once(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.executed:
            wrapper.executed = True
            wrapper.result = func(*args, **kwargs)
            return wrapper.result
        else:
            return wrapper.result
    wrapper.executed = False
    return wrapper

@only_exec_once
def get_ligand_atom_name_mapping(cif_file: str, smiles: str) -> dict[str, str]:
    return get_ligand_atom_name_mapping_from_ligand_and_chai_lab(cif_file, smiles)

def get_molecularglue_smiles(fasta_file: str) -> str:
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">ligand"):
                return f.readline().strip()
    raise ValueError("No smiles found in fasta file")

# output parameter
def param_dict_format(config):
    return "_".join([f"{v}" for k, v in config.items()])