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
    # conservation_matrix = dp.generate_conservation_matrix()
    # np.save(conservation_matrix_path, conservation_matrix)
    # print(f"Stochiometry Matrix: {conservation_matrix}")
    
    training_np, validation_np = dp.load_datasets(AEConfig.columns)

    dp.abundances_scaling(training_np)
    dp.abundances_scaling(validation_np)
    
    training_t = torch.from_numpy(training_np).to(torch.float32)
    validation_t = torch.from_numpy(validation_np).to(torch.float32)
    
    training_Dataset = dp.AutoencoderDataset(training_t)
    validation_Dataset = dp.AutoencoderDataset(validation_t)
    
    del training_np, validation_np, training_t, validation_t
    gc.collect()
    
    training_dataloader = dp.tensor_to_dataloader(AEConfig, training_Dataset, is_emulator=False)
    validation_dataloader = dp.tensor_to_dataloader(AEConfig, validation_Dataset, is_emulator=False)
    
    autoencoder, optimizer, scheduler = load_autoencoder_objects(final_training_phase=False)
    
    autoencoder_trainer = AutoencoderTrainer(
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
        )
    
    autoencoder_trainer.train()
    
    # total_dataset = torch.vstack((training_t, validation_t))
    # del training_np, validation_np
    
    # component_scalers = dp.calculate_component_scalers(total_dataset)
    # print(f"Component Scalers (Min-Max of Latent Space): {component_scalers}")
    
    # component_scalers_path = DatasetConfig.working_path + "/utils/component_scalers.npy"
    # np.save(component_scalers_path, component_scalers)