import pandas as pd
import numpy as np
import os  
import torch
import re
import gc
from torch.utils.data import Dataset, DataLoader
from numba import njit, prange
from .nn import Autoencoder
from .configs import DatasetConfig, AEConfig, EMConfig, PredefinedTensors
import h5py
from torch.utils.data import Sampler
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_datasets(
    columns: list
    ):
    """
    Datasets are loaded from hdf5 files, filtered to only contain the columns of interest, and converted to np arrays for speed.
    """
    training_dataset = pd.read_hdf(
        DatasetConfig.dataset_path, 
        "train", 
        start=0, 
        #stop=5000,
        #stop=1500000
        ).astype(np.float32)
    validation_dataset = pd.read_hdf(
        DatasetConfig.dataset_path, 
        "val",
        start=0,
        #stop=5000,
        #stop=1500000
        ).astype(np.float32)
        
    training_np = training_dataset[columns].to_numpy(copy=False)
    validation_np = validation_dataset[columns].to_numpy(copy=False)

    np.clip(
        training_np[:, -DatasetConfig.num_species:], 
        DatasetConfig.abundances_lower_clipping, 
        DatasetConfig.abundances_upper_clipping, 
        out=training_np[:, -DatasetConfig.num_species:]
    )

    np.clip(
        validation_np[:, -DatasetConfig.num_species:], 
        DatasetConfig.abundances_lower_clipping, 
        DatasetConfig.abundances_upper_clipping,
        out=validation_np[:, -DatasetConfig.num_species:]
    )
    
    print
    
    del training_dataset, validation_dataset
    gc.collect()
    return training_np, validation_np


def generate_stoichiometric_matrix():
    """
    Generates a stoichiometric matrix for the elements in the dataset.
    An unscaled vector of the species multiplied by this matrix will give the elemental abundances, which are conserved.
    Additionally tracks BULK and SURFACE stoichiometric.
    """
    elements = ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]
    stoichiometric_matrix = np.zeros((len(elements), DatasetConfig.num_species))
    modified_species = [s.replace("BULK_", "").replace("SURF_", "") for s in DatasetConfig.species]
    
    elements_patterns = {
        'H': re.compile(r'H(?!E)(\d*)'),
        'HE': re.compile(r'HE(\d*)'),
        'C': re.compile(r'C(?!L)(\d*)'),
        'N': re.compile(r'N(\d*)'),
        'O': re.compile(r'O(\d*)'),
        'S': re.compile(r'S(?!I)(\d*)'),
        'SI': re.compile(r'SI(\d*)'),
        'MG': re.compile(r'MG(\d*)'),
        'CL': re.compile(r'CL(\d*)'),
    }

    for element, pattern in elements_patterns.items():
        elem_index = elements.index(element)
        for i, species in enumerate(modified_species):
            match = pattern.search(species)
            if match and species not in ["SURFACE", "BULK"]:
                multiplier = int(match.group(1)) if match.group(1) else 1
                stoichiometric_matrix[elem_index, i] = multiplier
        
    return stoichiometric_matrix.T


def calculate_component_scalers(
    dataset_t: torch.Tensor,
    encoding_batch_size: int = 32*8192
):
    
    ae = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dims=AEConfig.hidden_dims,
    ).to("cuda")
    ae.load_state_dict(torch.load(AEConfig.save_model_path))
    ae.eval()
    
    min_, max_ = None, None
    
    with torch.no_grad():
        for batch_start in range(0, len(dataset_t), encoding_batch_size):
            batch_end = min(batch_start + encoding_batch_size, len(dataset_t))
            batch = dataset_t[batch_start:batch_end]
            batch_tensor = batch.to("cuda")
            encoded_batch = ae.encode(batch_tensor).cpu()
            
            batch_min = torch.min(encoded_batch).item()
            batch_max = torch.max(encoded_batch).item()
            
            if min_ is None:
                min_, max_ = batch_min, batch_max
            else:
                min_ = min(min_, batch_min)
                max_ = max(max_, batch_max)
    
    scalers_np = np.array([min_, max_], dtype=np.float32)
    return scalers_np


def abundances_scaling(
    abundances: np.ndarray, 
    min_: torch.Tensor = PredefinedTensors.ab_min.cpu().numpy(), 
    max_: torch.Tensor = PredefinedTensors.ab_max.cpu().numpy(),
    ):
    """
    Abundances are log10'd and then minmax scaled between (0, 1) for easier training.
    """
    np.log10(abundances, out=abundances)
    np.subtract(abundances, min_, out=abundances)
    np.divide(abundances, (max_ - min_), out=abundances)


def inverse_abundances_scaling_np(
    abundances: np.array, 
    min_: np.array = PredefinedTensors.ab_min.cpu().numpy(), 
    max_: np.array = PredefinedTensors.ab_max.cpu().numpy(),
    exponent: np.array = PredefinedTensors.exponential.cpu(),
    ):
    """
    Scaled abundances are inverse transformed and then exponentiated.
    """
    np.multiply(abundances, (max_ - min_), out=abundances)
    np.add(abundances, min_, out=abundances)
    np.exp(exponent * abundances, out=abundances)
    

@torch.jit.script
def inverse_abundances_scaling(
    scaled_abundances: torch.Tensor, 
    min_: torch.Tensor = PredefinedTensors.ab_min, 
    max_: torch.Tensor = PredefinedTensors.ab_max,
    exponent: torch.Tensor = PredefinedTensors.exponential,
    ):
    """
    Scaled abundances are inverse transformed and then exponentiated.
    """
    log_abundances = scaled_abundances * (max_ - min_) + min_
    abundances = torch.exp(exponent * log_abundances)
    return abundances


def latent_components_scaling(
    components: torch.Tensor, 
    min_: torch.Tensor = PredefinedTensors.ae_min.cpu(), 
    max_: torch.Tensor = PredefinedTensors.ae_max.cpu(),
    ):
    """
    Scales latent components from encoder to be between (0, 1) for easier emulator training.
    """
    
    return (components - min_) / (max_ - min_)


@torch.jit.script
def inverse_latent_components_scaling(
    scaled_components: torch.Tensor, 
    min_: torch.Tensor = PredefinedTensors.ae_min, 
    max_: torch.Tensor = PredefinedTensors.ae_max,
    ):
    """
    Scaled latent components are inverse transformed and can then be used directly in the decoder.
    """
    return scaled_components * (max_ - min_) + min_


@torch.jit.script
def stoichiometric_matrix_mult(
    tensor: torch.Tensor,
    stoichiometric_matrix: torch.Tensor = PredefinedTensors.stoichiometric_matrix,
    ):
    """
    Given a tensor of abundances, this function calculates the elemental abundances.
    """
    return torch.matmul(tensor, stoichiometric_matrix)


@torch.jit.script
def calculate_conservation_loss(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor
    ):
    """
    Given the actual and predicted abundances, this function calculates a loss between the elemental abundances of both.
    """
    unscaled_tensor1 = inverse_abundances_scaling(tensor1)
    unscaled_tensor2 = inverse_abundances_scaling(tensor2)
    
    elemental_abundances1 = torch.abs(stoichiometric_matrix_mult(unscaled_tensor1))
    elemental_abundances2 = torch.abs(stoichiometric_matrix_mult(unscaled_tensor2))

    log_elemental_abundances1 = torch.log10(elemental_abundances1)
    log_elemental_abundances2 = torch.log10(elemental_abundances2)
    
    diff = torch.abs(log_elemental_abundances2 - log_elemental_abundances1)
    
    return torch.sum(diff) / tensor1.size(0)


#@torch.jit.script
def autoencoder_loss_function(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    exponential: torch.Tensor = PredefinedTensors.exponential,
    exponential_coefficient: torch.Tensor = PredefinedTensors.AE_exponential_coefficient,
    conservation_weight: torch.Tensor = PredefinedTensors.AE_conservation_weight,
    ):
    """
    This is the custom loss function for the autoencoder. It's a combination of the reconstruction loss and the conservation loss.
    """
    
    elementwise_loss = torch.abs(outputs - targets)
    elementwise_loss = torch.exp(exponential_coefficient * exponential * elementwise_loss) - 1
    elementwise_loss = torch.sum(elementwise_loss) / targets.size(0)
    
    conservation_error = conservation_weight * calculate_conservation_loss(outputs, targets)    
    
    total_loss = (elementwise_loss +  conservation_error) * 1e-3
    
    print(f"Recon: {elementwise_loss.detach():.3e} | Cons: {conservation_error.detach():.3e} | Total: {total_loss.detach():.3e}")
    return total_loss


#@torch.jit.script
def emulator_training_loss_function(
    outputs,
    targets,
    alpha: torch.Tensor = PredefinedTensors.EM_alpha, 
    loss_scaling_factor: torch.Tensor = PredefinedTensors.EM_loss_scaling_factor,
    exponential: torch.Tensor = PredefinedTensors.exponential,
    exponential_coefficient: torch.Tensor = PredefinedTensors.EM_exponential_coefficient,
    ):
    """
    This is the custom loss function for the emulator. It's a combination of the predictive loss and the conservation loss.
    """
    elementwise_loss = torch.abs(outputs - targets)
    exp_elementwise_loss = torch.exp(exponential_coefficient * exponential * elementwise_loss) - 1
    sum_elementwise_loss = torch.sum(exp_elementwise_loss) / targets.size(0)
    
    conservation_error = calculate_conservation_loss(outputs, targets)
    
    total_loss = sum_elementwise_loss + alpha*conservation_error
    total_loss = total_loss * loss_scaling_factor
    print(f"Recon: {sum_elementwise_loss:.3e} | Cons: {alpha*conservation_error:.3e} | Total: {total_loss:.3e}")
    return total_loss


@torch.jit.script
def validation_loss_function(
    outputs, 
    targets, 
    ):
    unscaled_outputs = inverse_abundances_scaling(outputs)
    unscaled_targets = inverse_abundances_scaling(targets)
    
    loss = (torch.abs(unscaled_targets - unscaled_outputs) / unscaled_targets)
    
    return torch.sum(loss, dim=0)


class EmulatorSequenceDataset(Dataset):
    def __init__(
        self,
        data_matrix: torch.Tensor,
        data_indices: torch.Tensor,
    ):
        self.data_matrix = data_matrix
        self.data_indices = data_indices
        self.num_tracers = np.unique(data_matrix[:, 1]).shape[0]
        self.num_datapoints = len(data_indices)
        self.num_metadata = DatasetConfig.num_metadata
        self.num_physical_parameters = DatasetConfig.num_physical_parameters
        self.num_species = DatasetConfig.num_species
        self.num_components = AEConfig.latent_dim
        self.num_timesteps = DatasetConfig.num_timesteps_per_model

        data_matrix_size = self.data_matrix.nbytes / (1024 ** 2)
        indices_matrix_size = self.data_indices.nbytes / (1024 ** 2)

        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")
        print(f"Indices_matrix Memory usage: {indices_matrix_size:.3f} MB")
        
        print(f"Dataset Size: {len(data_indices)}\n")


    def __len__(self):
        return self.num_datapoints


    def __getitems__(self, indices: list):
        indices = torch.tensor(indices, dtype=torch.long)
        
        data_indices = self.data_indices[indices]
        
        rows = self.data_matrix[data_indices]
        
        physical_parameters = rows[:, :-1, 1+self.num_metadata: 1+self.num_metadata+self.num_physical_parameters]
        features = rows[:, :-1, -self.num_components:]
        targets = rows[:, 1:, 1+self.num_metadata+self.num_physical_parameters:-self.num_components]
        
        return physical_parameters, features, targets


class AutoencoderDataset(Dataset):
    def __init__(
        self,
        data_matrix: torch.Tensor,
    ):
        self.data_matrix = data_matrix
        data_matrix_size = self.data_matrix.nbytes / (1024 ** 2)
        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")


    def __len__(self):
        return len(self.data_matrix)


    def __getitems__(
        self,
        indices: list
        ):
        
        indices = torch.tensor(indices, dtype=torch.long)
        features = self.data_matrix[indices]
        return features, 1


class ChunkedShuffleSampler(Sampler):
    def __init__(self, data_size: int, chunk_size: int, seed: int = 13):
        super().__init__()
        self.data_size = int(data_size)
        self.chunk_size = int(chunk_size)
        self.base_seed = seed
        self.epoch = 0
        
        self.chunks = []
        start = 0
        while start < self.data_size:
            end = min(start + self.chunk_size, self.data_size)
            self.chunks.append((start, end))
            start = end
        
        self.generator = torch.Generator()
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        
        chunk_indices = torch.randperm(len(self.chunks), generator=g)
        
        for i, chunk_idx in enumerate(chunk_indices):
            chunk_seed = self.base_seed + self.epoch * 10000 + i
            g.manual_seed(chunk_seed)
            
            start, end = self.chunks[chunk_idx]
            length = end - start
            
            chunk_perm = torch.randperm(length, generator=g)
            chunk_perm += start
            
            yield from chunk_perm.tolist()
    
    def __len__(self):
        return self.data_size

@njit
def calculate_autoencoder_index_pairs(
    dataset_np: np.ndarray
    ):
    """
    Given the dataset, this function calculates consecutive timestep pairs
    for the emulator training (one timestep apart).
    Format: (time1, time2)
    Example Pairs:
    (0, 1)
    (1, 2)
    (2, 3)
    """
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)
    
    total_pairs = 0
    for group in model_groups:
        n = len(group[:, 0])
        total_pairs += max(0, n)
    
    index_pairs = np.zeros((total_pairs, 2), dtype=np.int32)
    
    index = 0
    for group in model_groups:
        sub_array = group[:, 0]
        n = len(sub_array)
        
        for i in prange(n):
            index_pairs[index, 0] = sub_array[i]
            if i + 1 < n:
                index_pairs[index, 1] = sub_array[i + 1]
            else:
                index_pairs[index, 1] = sub_array[i]
            index += 1
    
    return index_pairs

@njit
def calculate_emulator_indices_sequential(
    dataset_np: np.ndarray,
    window_size: int = 12,
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


def physical_parameter_scaling(
    physical_parameters: np.ndarray
    ):
    """
    Preprocesses the dataset by minmax scaling the latent components to (0, 1) and scaling the physical parameters.
    """
    np.log10(
        physical_parameters,
        out=physical_parameters
    )
    for i, parameter in enumerate(DatasetConfig.physical_parameter_ranges):
        param_min, param_max = DatasetConfig.physical_parameter_ranges[parameter]
        log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)
        
        physical_parameters[:, i] = (physical_parameters[:, i] - log_param_min) / (log_param_max - log_param_min)


def inverse_physical_parameter_scaling(
    physical_parameters: np.array
    ):
    """
    Reverses the preprocessing of the dataset by applying inverse min-max scaling and exponentiation
    to recover the original physical parameter values. Operates in-place.
    """        
    for i, parameter in enumerate(DatasetConfig.physical_parameter_ranges):
        param_min, param_max = DatasetConfig.physical_parameter_ranges[parameter]
        log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)
        
        physical_parameters[:, i] = physical_parameters[:, i] * (log_param_max - log_param_min) + log_param_min
    
    np.power(10, physical_parameters, out=physical_parameters)


def encode_dataset( 
    dataset_np: np.ndarray | torch.Tensor,
    encoding_batch_size: int = 12*8192
    ):
    dataset_t = torch.from_numpy(dataset_np)
    
    ae = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dims=AEConfig.hidden_dims,
    ).to("cuda")
    ae.load_state_dict(torch.load(AEConfig.save_model_path))
    ae.eval()
    
    num_samples = len(dataset_t)
    latent_dim = AEConfig.latent_dim
    encoded_dataset = torch.zeros((num_samples, latent_dim))
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, encoding_batch_size):
            batch_end = min(batch_start + encoding_batch_size, num_samples)
            batch_tensor = dataset_t[batch_start:batch_end].to("cuda")
            encoded_dataset[batch_start:batch_end] = ae.encode(batch_tensor).cpu()
    
    scaled_components = latent_components_scaling(encoded_dataset)
    return scaled_components


def prepare_autoencoder_dataset(
    dataset_np: np.array
    ):
    num_species = DatasetConfig.num_species
    
    dataset_np[:, 0] = np.arange(len(dataset_np))
    
    abundances_scaling(dataset_np[:, -num_species:])
    
    index_pairs_np = calculate_autoencoder_index_pairs(dataset_np)
    
    perm = np.random.permutation(len(index_pairs_np))
    index_pairs_shuffled_np = index_pairs_np[perm]
    
    dataset_t = torch.from_numpy(dataset_np).float()
    index_pairs_shuffled_t = torch.from_numpy(index_pairs_shuffled_np).int()
    
    return (dataset_t, index_pairs_shuffled_t)


def prepare_emulator_dataset(
    dataset_np: np.array,
    window_size: int = EMConfig.window_size
    ):
    """
    Generates index pairs for training.
    Generates latent components using autoencoder for the dataset.
    Scales physical parameters
    """
    num_species = DatasetConfig.num_species
    num_params = DatasetConfig.num_physical_parameters
    num_metadata = DatasetConfig.num_metadata
    
    #dataset_np[:, 0] = np.arange(len(dataset_np))
    new_column = np.arange(len(dataset_np))
    dataset_np = np.insert(dataset_np, 0, new_column, axis=1) 
    
    physical_parameter_scaling(dataset_np[:, num_metadata+1:num_metadata+1+num_params])
    abundances_scaling(dataset_np[:, -num_species:])
    
    latent_components = encode_dataset(dataset_np[:, num_metadata+1+num_params:])
    encoded_dataset_np = np.hstack((dataset_np, latent_components), dtype=np.float32)
    
    index_pairs_np = calculate_emulator_indices_sequential(encoded_dataset_np, window_size)
    
    perm = np.random.permutation(len(index_pairs_np))
    index_pairs_shuffled_np = index_pairs_np[perm]

    encoded_t = torch.from_numpy(encoded_dataset_np).float()
    index_pairs_shuffled_t = torch.from_numpy(index_pairs_shuffled_np).int()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return (encoded_t, index_pairs_shuffled_t)


def save_tensors_to_hdf5(
    tensors: torch.Tensor, 
    category: str
    ):
    dataset, indices = tensors
    with h5py.File(f"data/{category}.h5", "w") as f:
        f.create_dataset("dataset", data=dataset.numpy(), dtype=np.float32)
        f.create_dataset("indices", data=indices.numpy(), dtype=np.int32)


def load_tensors_from_hdf5(
    category: str
    ):
    dataset_path = os.path.join(DatasetConfig.working_path, f"data/{category}.h5")
    with h5py.File(dataset_path, "r") as f:
        dataset = f["dataset"][:]
        indices = f["indices"][:]
    dataset = torch.from_numpy(dataset).float()
    indices = torch.from_numpy(indices).int()
    return dataset, indices


def collate_function(batch):
    if len(batch) == 2:
        features, targets = batch
        return features, targets
    else:
        physical_parameters, features, targets = batch
        return physical_parameters, features, targets


def tensor_to_dataloader(
    training_config,
    torchDataset: Dataset,
    ):
    data_size = len(torchDataset)
    multiplier = training_config.shuffle_chunk_size
    sampler = ChunkedShuffleSampler(data_size, chunk_size=multiplier * data_size)
    dataloader = DataLoader(
        torchDataset,
        batch_size=training_config.batch_size,
        pin_memory=True,
        num_workers=0,
        in_order=False,
        sampler=sampler,
        collate_fn=collate_function
    )
    return dataloader


def reconstruct_emulated_outputs(encoded_inputs, emulated_outputs):
    """
    Adds the time and physical parameter columns to the latent components.
    """
    num_physical_parameters = DatasetConfig.num_physical_parameters    
    reconstructed_emulated_outputs = torch.cat((encoded_inputs[:, :1+num_physical_parameters], emulated_outputs), dim=1)
    return reconstructed_emulated_outputs


def baseAvtoAv(
    physical_parameters: np.array,
    ):
    """
    This conversion is used internally in UCLCHEM. Our dataset has Av, although the dataset was generated using baseAv.
    """
    baseAv_idx = 2
    density_idx = 0
    multiplier = 0.0000964375
    additive = np.multiply(multiplier, physical_parameters[:, density_idx])
    np.add(physical_parameters[:, baseAv_idx], additive, out=physical_parameters[:, baseAv_idx])


### Inferencing Functions
def encoder_inferencing(autoencoder, inputs, batch_size=4*8192):
    num_inputs = inputs.size(0)
        
    encoded_features = []
    for batch_start in range(0, num_inputs, batch_size):
        batch_end = min(batch_start + batch_size, num_inputs)
        batch = inputs[batch_start:batch_end]
        batch = batch.to(device)
        batch_encoded = autoencoder.encode(batch)
        encoded_features.append(batch_encoded)
    encoded_features = torch.cat(encoded_features, dim=0)
    
    return encoded_features


def decoder_inferencing(autoencoder, emulated_features, batch_size=4*8192):
    decoded_features = []
    for batch_start in range(0, len(emulated_features), batch_size):
        batch_end = min(batch_start + batch_size, len(emulated_features))
        batch = emulated_features[batch_start:batch_end]
        batch = batch.to(device)
        batch_decoded = autoencoder.decode(batch)
        decoded_features.append(batch_decoded)
    decoded_features = torch.cat(decoded_features, dim=0)

    decoded_features = inverse_abundances_scaling(decoded_features)
    
    return decoded_features


def emulator_inferencing(emulator, encoded_inputs, scale_components=True, batch_size=4*8192):
    num_physical_parameters = DatasetConfig.num_physical_parameters
    
    if scale_components:
        encoded_inputs[:, 1+num_physical_parameters:] = latent_components_scaling(encoded_inputs[:, 1+num_physical_parameters:])

    emulated_outputs = []
    for batch_start in range(0, len(encoded_inputs), batch_size):
        batch_end = min(batch_start + batch_size, len(encoded_inputs))
        batch = encoded_inputs[batch_start:batch_end].to(device)
        batch_outputs = emulator(batch)
        emulated_outputs.append(batch_outputs)
    
    emulated_outputs = torch.cat(emulated_outputs, dim=0)
    emulated_outputs = inverse_latent_components_scaling(emulated_outputs)
    
    return emulated_outputs
