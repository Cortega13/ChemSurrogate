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

class CSVtoHDF5:
    """
    Reads the UCLChem CSV output files (1 csv per model) and compresses them into a single HDF5 file.
    """
    def __init__(self, work_path, data_folder_name="grid_folder", output_filename="uclchem_rawdata_v6.h5"):
        self.work_path = work_path
        self.data_folder_path = os.path.join(work_path, data_folder_name)
        self.h5_store_path = os.path.join(work_path, output_filename)
        self.batch_size = 4192
        self.h5_store = pd.HDFStore(self.h5_store_path)

    @staticmethod
    def rename_columns(columns):
        """Renames columns to remove problematic characters and renames them to be more readable."""
        name_mapping = {
            'radfield': 'Radfield',
            '@H2COH': '@H3CO',
            'H2COH': 'H3CO',
            'point': 'Model',
            'H2CSH+': 'H3CS+',
            'SISH+': 'HSIS+',
            'E-': 'E_minus',
            'HOSO+': 'HSO2+',
            'H2COH+': 'H3CO+',
            'OCSH+': 'HOCS+',
            '#H2COH': '#H3CO',
        }
        columns = [col.strip() for col in columns]
        columns = [name_mapping[col] if col in name_mapping else col for col in columns]
        columns = [col.replace('#', 'SURF_')
                  .replace('+', 'Plus')
                  .replace('@', 'BULK_') for col in columns]
        return columns
    

    def process_and_store_data(self):
        """Reads CSV files, processes them, and stores them in an HDF5 file."""
        files_list = os.listdir(self.data_folder_path)
        batch_data = []
        global_index = 0
        
        for i, file in enumerate(files_list):
            if i % 100 == 0:
                print(f"Currently on Model: {i}")
            
            file_path = os.path.join(self.data_folder_path, file)
            single_model_data = pd.read_csv(file_path)
            single_model_data["Model"] = i
            
            row_count = len(single_model_data)
            single_model_data["Index"] = range(global_index, global_index + row_count)
            global_index += row_count
            
            single_model_data.columns = self.rename_columns(single_model_data.columns)
            
            single_model_data = single_model_data.drop(columns=["zeta", "point", "dustTemp", "SURFACE", "BULK"], errors='ignore')
            single_model_data = single_model_data.astype(np.float32)
            single_model_data["Model"] = single_model_data["Model"].astype(int)
            single_model_data = single_model_data.drop(index=1, errors='ignore')
            
            batch_data.append(single_model_data)
            
            if (i + 1) % self.batch_size == 0 or (i + 1) == len(files_list):
                combined_data = pd.concat(batch_data)
                self.h5_store.append('models', combined_data, format='table')
                batch_data = []
        
        print("Raw Data Saving Completed")
        self.h5_store.close()


    def run(self):
        """Executes the full compression process."""
        self.process_and_store_data()


class DatasetCleaner:
    """
    Confirms that timesteps are consistently 1kyr and clips abundances and physical parameters to preferred ranges.
    """
    def __init__(self, config):
        self.config = config
        self.working_path = config.working_path
        self.raw_filename = config.raw_filename
        self.df = None
    
    def load_data(self):
        self.df = pd.read_hdf(os.path.join(self.working_path, self.raw_filename), "models", start=0, dtype=np.float32)
        self.df = self.df.astype(np.float32)
        self.df.reset_index(drop=True, inplace=True)
        self.df.sort_values(by=["Model", "Time"], inplace=True)
        if "Index" not in self.df.columns:
            self.df['Index'] = range(len(self.df))
        print("-=+=- Dataset Loaded -=+=-")
        print(f"Original Total Dataset Size: {len(self.df)}")
    
    def clip_data(self):
        self.df = self.df.clip(lower=self.config.lower_clipping_threshold)
        for param, (min_val, max_val) in self.config.physical_parameter_ranges.items():
            if param in self.df.columns:
                self.df = self.df[(self.df[param] > min_val) & (self.df[param] < max_val)]
        self.df.infer_objects(copy=False)
        print("-=+=- Dataset Clipped by Threshold and Physical Parameter Ranges -=+=-")
    
    @staticmethod
    def filter_constant_timesteps(df, timestep=1000):
        df['diffs'] = df['Time'].diff().fillna(timestep)
        df['is_new_group'] = df['diffs'] != timestep
        df['temp_group'] = df['is_new_group'].cumsum()
        
        group_sizes = df.groupby('temp_group').size()
        max_group = group_sizes.idxmax()
        
        group_indices = df[df['temp_group'] == max_group].index
        start_index = group_indices[0]
        end_index = group_indices[-1]
        filtered_df = df.loc[start_index:end_index].drop(columns=['diffs', 'is_new_group', 'temp_group'])
        return filtered_df
    
    def process_data(self):
        df_constant_dt = (
            self.df.groupby('Model', group_keys=False)
            .apply(lambda group: self.filter_constant_timesteps(group.assign(Model=group.name)))
            .reset_index(drop=True)
        )
                
        print(f"Total Dataset Size: {len(df_constant_dt)} | Percentage: {len(df_constant_dt) / len(self.df) * 100:.2f}%")
        
        df_constant_dt.reset_index(drop=True, inplace=True)
        
        self.save_data(df_constant_dt, f"{self.config.data_category}.h5")
    
    def save_data(self, df, filename):
        df.to_hdf(os.path.join(self.working_path, filename), key="models", mode="a")
        print(f"-=+=- Data Successfully Saved: {filename} -=+=-")
    
    def run(self):
        self.load_data()
        self.clip_data()
        self.process_data()


def load_datasets(
    columns: list
    ):
    """
    Datasets are loaded from hdf5 files, filtered to only contain the columns of interest, and converted to np arrays for speed.
    """
    training_dataset = pd.read_hdf(
        DatasetConfig.training_dataset_path, 
        "models", 
        start=0, 
        #stop=5000,
        #stop=1500000
        ).astype(np.float32)
    validation_dataset = pd.read_hdf(
        DatasetConfig.validation_dataset_path, 
        "models",
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


@torch.jit.script
def calculate_structural_loss(
    targets: torch.Tensor,
    latent_components: torch.Tensor,
    num_anchors: int = AEConfig.num_anchors,
    eps: torch.Tensor = torch.tensor(1e-8).to("cuda").float(),
    device: torch.device = device,
):
    """
    Anchor-based latent structure loss. 
    Random points are sampled from regular and latent space.
    Distances of all points to these anchors are calculated and normalized.
    These normalized distances are compared to calculate the loss.
    """
    batch_size = targets.size(0)
    
    anchor_indices = torch.randperm(batch_size, device=device)[:num_anchors]
    
    target_anchors = torch.index_select(targets, 0, anchor_indices)
    latent_anchors = torch.index_select(latent_components, 0, anchor_indices)
    
    target_dists = torch.cdist(targets, target_anchors, p=2.0)
    
    latent_dists = torch.cdist(latent_components, latent_anchors, p=2.0)
    
    target_dists_max = torch.max(target_dists)
    latent_dists_max = torch.max(latent_dists)
    
    target_dists_normalized = target_dists / (target_dists_max + eps)
    latent_dists_normalized = latent_dists / (latent_dists_max + eps)
    
    return torch.mean((target_dists_normalized - latent_dists_normalized) ** 2)

@torch.jit.script
def temporal_smoothness_loss(outputs, outputsT1):
    return F.mse_loss(outputs, outputsT1)

#@torch.jit.script
def autoencoder_loss_function(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    latent_components: torch.Tensor,
    exponential: torch.Tensor = PredefinedTensors.exponential,
    exponential_coefficient: torch.Tensor = PredefinedTensors.AE_exponential_coefficient,
    conservation_weight: torch.Tensor = PredefinedTensors.AE_conservation_weight,
    structural_weight: torch.Tensor = PredefinedTensors.AE_structural_weight,
    ):
    """
    This is the custom loss function for the autoencoder. It's a combination of the reconstruction loss and the conservation loss.
    """
    
    elementwise_loss = torch.abs(outputs - targets)
    elementwise_loss = torch.exp(exponential_coefficient * exponential * elementwise_loss) - 1
    elementwise_loss = torch.sum(elementwise_loss) / targets.size(0)
    
    conservation_error = conservation_weight * calculate_conservation_loss(outputs, targets)
    
    structural_loss = structural_weight * calculate_structural_loss(
        targets, 
        latent_components
        )
    
    
    total_loss = (elementwise_loss +  conservation_error + structural_loss) * 1e-3
    
    print(f"Recon: {elementwise_loss.detach():.3e} | Cons: {conservation_error.detach():.3e} | Structure: {structural_loss.detach():.3e} | Total: {total_loss.detach():.3e}")
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


class EmulatorTimestepDataset(Dataset):
    def __init__(
        self,
        data_matrix: torch.Tensor,
        index_pairs: torch.Tensor,
    ):
        self.data_matrix = data_matrix
        self.index_pairs = index_pairs
        self.num_metadata = DatasetConfig.num_metadata
        self.num_physical_parameters = DatasetConfig.num_physical_parameters
        self.num_species = DatasetConfig.num_species
        self.num_components = AEConfig.latent_dim
        self.num_timesteps = DatasetConfig.num_timesteps_per_model

        data_matrix_size = self.data_matrix.nbytes / (1024 ** 2)
        index_pairs_size = self.index_pairs.nbytes / (1024 ** 2)

        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")
        print(f"Index_pairs Memory usage: {index_pairs_size:.2f} MB\n")
        
        print(f"Dataset Size: {len(self.data_matrix)} | Index Pairs: {len(self.index_pairs)}\n")


    def __len__(self):
        return len(self.index_pairs)


    def __getitems__(
        self,
        indices: list
        ):
        
        indices = torch.tensor(indices, dtype=torch.long)
        pairs = self.index_pairs[indices]
        rows = self.data_matrix[pairs[:, :-1]]
        
        features = rows[:, 0, :]
        targets = rows[:, 1, :]
        
        left_index = self.num_metadata
        right_index = left_index + self.num_physical_parameters
        physical_parameters = features[:, left_index:right_index]
        encoded_components = features[:, -self.num_components:]
        
        timesteps = (pairs[:, 2].unsqueeze(1).float() / self.num_timesteps).float()
        features = torch.cat((timesteps, physical_parameters, encoded_components), dim=1)
        
        left_index = self.num_metadata + self.num_physical_parameters
        right_index = left_index + self.num_species
        targets = targets[:, left_index:right_index]
        
        return features, targets


class EmulatorSequenceDataset(Dataset):
    def __init__(
        self,
        data_matrix: torch.Tensor,
    ):
        self.data_matrix = data_matrix
        self.num_models = np.unique(data_matrix[:, 1]).shape[0]
        self.num_metadata = DatasetConfig.num_metadata
        self.num_physical_parameters = DatasetConfig.num_physical_parameters
        self.num_species = DatasetConfig.num_species
        self.num_components = AEConfig.latent_dim
        self.num_timesteps = DatasetConfig.num_timesteps_per_model

        data_matrix_size = self.data_matrix.nbytes / (1024 ** 2)

        print(f"Data_matrix Memory usage: {data_matrix_size:.3f} MB")
        
        print(f"Dataset Size: {len(self.data_matrix)}\n")


    def __len__(self):
        return self.num_models


    def __getitems__(self, indices: list):
        indices = torch.tensor(indices, dtype=torch.long)
        start_indices = indices * self.num_timesteps
        
        features = self.data_matrix[start_indices]
        cols_to_keep = list(range(3, 7)) + list(range(features.shape[1] - 12, features.shape[1]))
        features = features[:, cols_to_keep]
        

        target_indices = [torch.arange(start + 1, start + self.num_timesteps) for start in start_indices]
        target_indices = torch.cat(target_indices)
        left_index = self.num_metadata + self.num_physical_parameters
        right_index = left_index + self.num_species
        targets = self.data_matrix[:, left_index:right_index][target_indices].reshape(len(indices), self.num_timesteps - 1, -1)[:, 10:30, :]
        
        return features, targets


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
def calculate_emulator_index_pairs(
    dataset_np: np.ndarray
    ):
    """
    Given the dataset, this function calculates all timestep pairs for the emulator training.
    Format: (time1, time2, timestep)
    Example Pairs:
    (0, 1, 1)
    (0, 2, 2)
    (1, 2, 1)
    (1, 3, 2)
    """
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)
    total_pairs = 0
    for group in model_groups:
        n = len(group[:, 0])
        total_pairs += (n * (n - 1)) // 2
    
    index_pairs = np.zeros((total_pairs, 3), dtype=np.int32)
    index = 0
    for group in model_groups:
        sub_array = group[:, 0]
        n = len(sub_array)
        for i in prange(n):
            feature_index = sub_array[i]
            for j in prange(i + 1, n):
                target_index = sub_array[j]
                timestep = target_index - feature_index
                index_pairs[index, 0] = feature_index
                index_pairs[index, 1] = target_index
                index_pairs[index, 2] = timestep
                index += 1
    return index_pairs

@njit
def calculate_emulator_index_pairs_sequential(
    dataset_np: np.ndarray
    ):
    """
    Returns a 2D array with sequences padded with -1 values.
    For each model with indices [0,1,2,...,n]:
    - First sequence: [0,1,2,...,n]
    - Second sequence: [1,2,3,...,n]
    - And so on, with the last sequence being [n-1, n]
    Padding value of -1 indicates end of valid data.
    """
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)
    
    total_seqs = 0
    max_seq_len = 0
    for group in model_groups:
        n = len(group)
        total_seqs += n - 1
        max_seq_len = max(max_seq_len, n)
    
    sequences = np.full((total_seqs, max_seq_len), -1, dtype=np.int32)
    
    seq_idx = 0
    for group in model_groups:
        indices = group[:, 0]
        n = len(indices)
        
        for start_idx in range(n-1):
            seq_len = n - start_idx
            sequences[seq_idx, :seq_len] = indices[start_idx:]
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
    sequential_time: bool = False
    ):
    """
    Generates index pairs for training.
    Generates latent components using autoencoder for the dataset.
    Scales physical parameters
    """
    num_species = DatasetConfig.num_species
    num_params = DatasetConfig.num_physical_parameters
    num_metadata = DatasetConfig.num_metadata
    
    dataset_np[:, 0] = np.arange(len(dataset_np))
    
    physical_parameter_scaling(dataset_np[:, num_metadata:num_metadata+num_params])
    abundances_scaling(dataset_np[:, -num_species:])
    
    latent_components = encode_dataset(dataset_np[:, num_metadata+num_params:])
    encoded_dataset_np = np.hstack((dataset_np, latent_components), dtype=np.float32)
    
    if sequential_time:
        index_pairs_np = calculate_emulator_index_pairs_sequential(encoded_dataset_np)
    else:
        index_pairs_np = calculate_emulator_index_pairs(encoded_dataset_np)
    
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
    features, targets = batch  
    return features, targets


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
def encoder_inferencing(autoencoder, inputs, batch_size=8192):
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


def decoder_inferencing(autoencoder, emulated_features, batch_size=8192):
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


def emulator_inferencing(emulator, encoded_inputs, scale_components=True, batch_size=8192):
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
