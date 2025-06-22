import pandas as pd
import numpy as np
import os  
import torch
import re
import gc
from torch.utils.data import Dataset, DataLoader
from numba import njit, prange
import h5py
from torch.utils.data import Sampler
from torch.nn import functional as F
from .inference import Inference


class Processing():
    def __init__(self, GeneralConfig, AEConfig=None, EMConfig=None):
        self.exponential = torch.tensor()
        
        self.abundances_min = torch.tensor(GeneralConfig.abundances_lower_clipping, dtype=torch.float32, device=GeneralConfig.device)
        self.abundances_max = torch.tensor(GeneralConfig.abundances_upper_clipping, dtype=torch.float32, device=GeneralConfig.device)
        
        if AEConfig is not None:
            latents_minmax = np.load(AEConfig.latents_minmax_path)
            self.components_min = torch.tensor(latents_minmax[0], dtype=torch.float32, device=GeneralConfig.device)
            self.components_max = torch.tensor(latents_minmax[1], dtype=torch.float32, device=GeneralConfig.device)
                
        self.physical_parameter_ranges= GeneralConfig.physical_parameter_ranges

    ### PreProcessing Functions
    
    def physical_parameter_scaling(
        self,
        physical_parameters: np.ndarray
        ):
        """
        Preprocesses the dataset by minmax scaling the latent components to (0, 1) and scaling the physical parameters.
        """
        np.log10(
            physical_parameters,
            out=physical_parameters
        )
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)
            
            physical_parameters[:, i] = (physical_parameters[:, i] - log_param_min) / (log_param_max - log_param_min)


    def abundances_scaling(
        self,
        abundances: np.ndarray,
        ):
        """
        Abundances are log10'd and then minmax scaled between (0, 1) for easier training.
        """
        np.log10(abundances, out=abundances)
        np.subtract(abundances, self.abundances_min, out=abundances)
        np.divide(abundances, (self.abundances_max - self.abundances_min), out=abundances)


    def latent_components_scaling(
        self,
        components: torch.Tensor,
        ):
        """
        Scales latent components from encoder to be between (0, 1) for easier emulator training.
        """
        
        return (components - self.components_min) / (self.components_max - self.components_min)


    ### PostProcessing Functions

    def inverse_physical_parameter_scaling(
        self,
        physical_parameters: np.array
        ):
        """
        Reverses the preprocessing of the dataset by applying inverse min-max scaling and exponentiation
        to recover the original physical parameter values. Operates in-place.
        """        
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)
            
            physical_parameters[:, i] = physical_parameters[:, i] * (log_param_max - log_param_min) + log_param_min
        
        np.power(10, physical_parameters, out=physical_parameters)

    @staticmethod
    @torch.jit.script
    def jit_inverse_abundances_scaling(
        abundances: torch.Tensor,
        min_: torch.Tensor,
        max_: torch.Tensor,
        exponential_: torch.Tensor
        ):
        log_abundances = abundances * (max_ - min_) + min_
        abundances = torch.exp(exponential_ * log_abundances)
        return abundances


    def inverse_abundances_scaling(
        self,
        abundances
        ):
        if isinstance(abundances, torch.Tensor):
            abundances = self.jit_inverse_abundances_scaling(
                abundances,
                self.abundances_min,
                self.abundances_max,
                self.exponential,
            )
        else:
            ab_min_np = self.abundances_min.cpu().numpy()
            ab_max_np = self.abundances_max.cpu().numpy()
            exponential_np = self.exponential.cpu().numpy()
            
            np.multiply(abundances, (ab_max_np - ab_min_np), out=abundances)
            np.add(abundances, ab_min_np, out=abundances)
            np.exp(exponential_np * abundances, out=abundances)
            
    @staticmethod
    @torch.jit.script
    def jit_inverse_latent_component_scaling(
        scaled_components: torch.Tensor,
        min_: torch.Tensor,
        max_: torch.Tensor
    ):
        return scaled_components * (max_ - min_) + min_


    def inverse_latent_components_scaling(
        self,
        scaled_components: torch.Tensor, 
        ):
        """
        Scaled latent components are inverse transformed and can then be used directly in the decoder.
        """
        return self.jit_inverse_latent_component_scaling(
            scaled_components,
            self.components_min,
            self.components_max
        )

    def save_latents_minmax(
        DatasetConfig,
        AEConfig,
        dataset_t: torch.Tensor,
        inference_functions: Inference,
        ):
        min_, max_ = float('inf'), float('-inf')

        with torch.no_grad():
            for i in range(0, len(dataset_t), AEConfig.batch_size):
                batch = dataset_t[i:i + AEConfig.batch_size].to(AEConfig.device)
                encoded = inference_functions.encode(batch).cpu()
                min_ = min(min_, encoded.min().item())
                max_ = max(max_, encoded.max().item())

        minmax_np = np.array([min_, max_], dtype=np.float32)
        print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")
        np.save(AEConfig.latents_minmax_path, minmax_np)


@njit
def calculate_emulator_indices(
    dataset_np: np.ndarray,
    window_size: int = 16,
    ):
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)
    
    total_seqs = 0
    for group in model_groups:
        n = len(group)
        total_seqs += n - window_size + 1
    
    sequences = np.full((total_seqs, window_size), -1, dtype=np.int32)
    
    seq_idx = 0
    for group in model_groups:
        indices = group[:, 0]
        n = len(indices)
        for start_idx in range(n - window_size + 1):
            sequences[seq_idx, :] = indices[start_idx:start_idx + window_size]
            seq_idx += 1
    
    return sequences


def preprocessing_emulator_dataset(
    GeneralConfig,
    EMConfig,
    dataset_np: np.array,
    processing_functions: Processing,
    inference_functions: Inference,
    ):
    """
    Generates index pairs for training.
    Generates latent components using autoencoder for the dataset.
    Scales physical parameters
    """
    num_species = GeneralConfig.num_species
    num_phys = GeneralConfig.num_phys
    num_metadata = GeneralConfig.num_metadata
    
    #dataset_np[:, 0] = np.arange(len(dataset_np))
    new_column = np.arange(len(dataset_np))
    dataset_np = np.insert(dataset_np, 0, new_column, axis=1) 
    
    processing_functions.physical_parameter_scaling(dataset_np[:, num_metadata:num_metadata+num_phys])
    processing_functions.abundances_scaling(dataset_np[:, -num_species:])
    
    latent_components = inference_functions.encode(dataset_np[:, num_metadata+1+num_phys:])
    encoded_dataset_np = np.hstack((dataset_np, latent_components), dtype=np.float32)
    
    index_pairs_np = calculate_emulator_indices(encoded_dataset_np, EMConfig.window_size)
    
    perm = np.random.permutation(len(index_pairs_np))
    index_pairs_shuffled_np = index_pairs_np[perm]

    encoded_t = torch.from_numpy(encoded_dataset_np).float()
    index_pairs_shuffled_t = torch.from_numpy(index_pairs_shuffled_np).int()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return (encoded_t, index_pairs_shuffled_t)