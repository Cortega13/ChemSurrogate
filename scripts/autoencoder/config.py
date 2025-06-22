import numpy as np
import os
import torch

class GeneralConfig:
    working_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # The path to the root folder of the project.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(working_path, "data/grav_collapse_clean.h5")

    num_timesteps_per_model = 296   # Duration that each model runs for. Multiply by timestep_duration to get total evolution time.
    timestep_duration = 92.9       # In years
    physical_parameter_ranges = {
        "Density":  (68481, 1284211415),       # H nuclei per cm^3.
        "radfield": (1e-2, 26),     # Habing field.
        "Av":       (1e-1, 6914),    # Magnitudes.
        "gasTemp":  (13, 133),      # Kelvin.
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
    
    num_metadata = len(metadata)
    num_physical_parameters = len(physical_parameters)
    num_species = len(species)
    
class AEConfig:
    columns = DatasetConfig.species
    num_columns = len(columns)
    latents_minmax_path = os.path.join(DatasetConfig.working_path, "utils/latents_minmax.npy")
    
    # Model Config
    input_dim = DatasetConfig.num_species # input_dim = output_dim
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
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "weights/autoencoder.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "weights/autoencoder.pth")