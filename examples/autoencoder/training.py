#from data_generation import DataGenerator
from ChemSurrogate import data_processing as dp
import gc
import torch
import numpy as np
from ChemSurrogate.trainer import (
    AutoencoderTrainer, 
    load_autoencoder_objects,
)
from ChemSurrogate.configs import (
    DatasetConfig,
    AEConfig,
)

if __name__ == "__main__":
    # conservation_matrix_path = DatasetConfig.working_path + "/utils/conservation_matrix.npy"
    # conservation_matrix = dp.generate_stoichiometric_matrix()
    # np.save(conservation_matrix_path, conservation_matrix)
    # print(f"Stochiometry Matrix: {conservation_matrix} | Shape: {conservation_matrix.shape}")
    
    training_np, validation_np = dp.load_datasets(AEConfig.columns)
    dp.abundances_scaling(training_np)
    dp.abundances_scaling(validation_np)
    training_dataset = torch.from_numpy(training_np)
    validation_dataset = torch.from_numpy(validation_np)
    
    # training_Dataset = dp.AutoencoderDataset(training_dataset)
    # validation_Dataset = dp.AutoencoderDataset(validation_dataset)

    # training_dataloader = dp.tensor_to_dataloader(AEConfig, training_Dataset)
    # validation_dataloader = dp.tensor_to_dataloader(AEConfig, validation_Dataset)

    # autoencoder, optimizer, scheduler = load_autoencoder_objects()
    
    # autoencoder_trainer = AutoencoderTrainer(
    #     autoencoder,
    #     optimizer,
    #     scheduler,
    #     training_dataloader,
    #     validation_dataloader
    #     )
    
    # autoencoder_trainer.train()
    
    total_dataset = torch.vstack((training_dataset, validation_dataset))
    
    component_scalers = dp.calculate_component_scalers(total_dataset)
    print(f"Component Scalers (Min-Max of Latent Space): {component_scalers}")
    
    component_scalers_path = DatasetConfig.working_path + "/utils/component_scalers.npy"
    np.save(component_scalers_path, component_scalers)