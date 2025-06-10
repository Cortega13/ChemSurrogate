import numpy as np
import os
from joblib import load
import torch
import importlib.resources as pkg_resources
from ChemSurrogate import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetConfig:
    working_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # The path to the root folder of the project.
    num_timesteps_per_model = 296   # Duration that each model runs for. Multiply by timestep_duration to get total evolution time.
    timestep_duration = 92.9       # In years
    num_metadata = 2
    num_physical_parameters = 4
    num_species = 333
    physical_parameter_ranges = {
        "Density":  (68481, 1284211415),       # H nuclei per cm^3. Limits arbitrarily chosen.
        "radfield": (1e-2, 26),     # Habing field. Limits arbitrarily chosen.
        "Av":       (1e-1, 6914),    # Magnitudes. Limits arbitrarily choisen.
        "gasTemp":  (13, 133),      # Kelvin. Grain reactions are too complex under 10 K. Ice mostly sublimates at 150 K and UCLChem sets it as a strict constraint.
    }
    abundances_lower_clipping = np.float32(1e-20)   # Abundances are arbitrarily clipped to 1e-20 since anything lower is insignificant.
    abundances_upper_clipping = np.float32(1)       # All abundances are relative to number of Hydrogen nuclei. Maximum abundance is all hydrogen in elemental form.

    initial_abundances_path = os.path.join(working_path, "utils/initial_abundances.npy")
    initial_abundances = np.load(initial_abundances_path)
    
    stoichiometric_matrix_path = os.path.join(working_path, "utils/stoichiometric_matrix.npy")
    stoichiometric_matrix = np.load(stoichiometric_matrix_path)
    
    metadata = ["Tracer", "Time"]
    physical_parameters = list(physical_parameter_ranges.keys())
    species_path = os.path.join(working_path, "utils/species.txt")
    species = np.loadtxt(species_path, dtype=str, delimiter=" ", comments=None).tolist()
    
    dataset_path = os.path.join(working_path, "data/grav_collapse_clean.h5")

class AEConfig:
    columns = DatasetConfig.species
    num_columns = len(columns)
    component_scalers_path = os.path.join(DatasetConfig.working_path, "utils/component_scalers.npy")
    # Model Config
    input_dim = DatasetConfig.num_species
    output_dim = DatasetConfig.num_species
    hidden_dims = (160, 80)
    latent_dim = 14
    
    # Hyperparameters Config
    lr = 1e-3
    lr_decay = 0.5
    lr_decay_patience = 10
    betas = (0.99, 0.999)
    weight_decay = 1e-4
    exponential_coefficient = 20
    conservation_weight = 1e2
    batch_size = 8*8192
    stagnant_epoch_patience = 20
    gradient_clipping = 2
    dropout_decay_patience = 8
    dropout_reduction_factor = 0.05
    dropout = 0.2
    noise = 0.1
    shuffle_chunk_size = 1
    save_model = True
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "models/autoencoder.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "models/autoencoder.pth")

    
class EMConfig:
    columns = DatasetConfig.metadata + DatasetConfig.physical_parameters + DatasetConfig.species
    num_columns = len(columns)
    # Model Config
    input_dim = DatasetConfig.num_physical_parameters + AEConfig.latent_dim
    hidden_dim = 16
    output_dim = AEConfig.latent_dim
    num_blocks = 8
    window_size = 48
    
    # Hyperparameters Config
    lr = 1e-3
    lr_decay = 0.5
    lr_decay_patience = 10
    betas = (0.99, 0.999)
    weight_decay = 1e-4
    loss_scaling_factor = 1e-3
    exponential_coefficient = 20
    alpha = 1e3
    batch_size = 2048
    stagnant_epoch_patience = 20
    gradient_clipping = 2
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "models/lstm.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "models/lstm.pth")
    dropout_decay_patience = 8
    dropout_reduction_factor = 0.05
    dropout = 0.0
    save_model = True
    shuffle = True
    shuffle_chunk_size = 1


class PredefinedTensors:
    ab_min = torch.tensor(np.log10(DatasetConfig.abundances_lower_clipping), device=device).float()
    ab_max = torch.tensor(np.log10(DatasetConfig.abundances_upper_clipping), device=device).float()

    component_scalers_path = os.path.join(DatasetConfig.working_path, "utils/component_scalers.npy")
    ae_min, ae_max = np.load(component_scalers_path)
    ae_min = torch.tensor(ae_min, device=device).float()
    ae_max = torch.tensor(ae_max, device=device).float()
    
    stoichiometric_matrix = torch.tensor(DatasetConfig.stoichiometric_matrix, device=device).float().contiguous()
    
    exponential = torch.log(torch.tensor(10, device=device).float())

    EM_loss_scaling_factor = torch.tensor(EMConfig.loss_scaling_factor, device=device).float()
    
    AE_exponential_coefficient = torch.tensor(AEConfig.exponential_coefficient, device=device).float()
    EM_exponential_coefficient = torch.tensor(EMConfig.exponential_coefficient, device=device).float()
    
    AE_conservation_weight = torch.tensor(AEConfig.conservation_weight, device=device).float()
    
    EM_alpha = torch.tensor(EMConfig.alpha, device=device).float()
    
    mace_max_abundance = torch.tensor(0.85, device=device).float()
    mace_factor = torch.tensor(468/335, device=device).float()