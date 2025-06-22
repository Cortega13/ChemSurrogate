import torch
import os
import numpy as np
from AstroChemNet import data_processing as dp
from AstroChemNet import data_loading as dl
from AstroChemNet.inference import Inference
from AstroChemNet.trainer import (
    AutoencoderTrainer, 
    load_objects,
)
from .config import (
    GeneralConfig,
    AEConfig,
)
from .nn import (
    Autoencoder
)

def load_autoencoder(Config):
    ae = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dims=AEConfig.hidden_dims,
        noise=AEConfig.noise,
        dropout=AEConfig.dropout,
    ).to(Config.device)
    if os.path.exists(AEConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        ae.load_state_dict(torch.load(AEConfig.pretrained_model_path))


if __name__ == "__main__":
    stoichiometric_matrix_path = GeneralConfig.working_path + "/utils/stoichiometric_matrix.npy"
    stoichiometric_matrix = dp.generate_stoichiometric_matrix()
    np.save(stoichiometric_matrix_path, stoichiometric_matrix)
    print(f"Stochiometry Matrix: {stoichiometric_matrix} | Shape: {stoichiometric_matrix.shape}")
   
    # processing_functions = dp.Processing(GeneralConfig, AEConfig)
    
    # training_np, validation_np = dl.load_datasets(AEConfig.columns)
    
    # processing_functions.abundances_scaling(training_np)
    # processing_functions.abundances_scaling(validation_np)
    # training_dataset = torch.from_numpy(training_np)
    # validation_dataset = torch.from_numpy(validation_np)
            
    # training_Dataset = dl.AutoencoderDataset(training_dataset)
    # validation_Dataset = dl.AutoencoderDataset(validation_dataset)
    
    # training_dataloader = dl.tensor_to_dataloader(AEConfig, training_Dataset)
    # validation_dataloader = dl.tensor_to_dataloader(AEConfig, validation_Dataset)

    # autoencoder = load_autoencoder(AEConfig)
    # optimizer, scheduler = load_objects(autoencoder)
    
    # autoencoder_trainer = AutoencoderTrainer(
    #     autoencoder,
    #     optimizer,
    #     scheduler,
    #     training_dataloader,
    #     validation_dataloader
    #     )
    
    # autoencoder_trainer.train()
    
    # total_dataset = torch.vstack((training_dataset, validation_dataset))
    # inference_functions = Inference(processing_functions, autoencoder)
    # processing_functions.save_latents_minmax(GeneralConfig, total_dataset)