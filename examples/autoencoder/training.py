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
    
    # training_np, validation_np = dp.load_datasets(AEConfig.columns)
    # training_dataset = dp.prepare_autoencoder_dataset(training_np)
    # validation_dataset = dp.prepare_autoencoder_dataset(validation_np)
    # dp.save_tensors_to_hdf5(training_dataset, category="training_ae")
    # dp.save_tensors_to_hdf5(validation_dataset, category="validation_ae")
    
    training_dataset, training_indices = dp.load_tensors_from_hdf5(category="training_ae")
    validation_dataset, validation_indices = dp.load_tensors_from_hdf5(category="validation_ae")

    training_Dataset = dp.AutoencoderRowRetrievalDataset(training_dataset, training_indices)
    validation_Dataset = dp.AutoencoderRowRetrievalDataset(validation_dataset, validation_indices)
    del training_dataset, validation_dataset, training_indices, validation_indices
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