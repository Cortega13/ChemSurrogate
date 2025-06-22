import torch
import os

from AstroChemNet import data_processing as dp
from AstroChemNet import data_loading as dl
from AstroChemNet.inference import (
    Inference
)
from AstroChemNet.trainer import (
    EmulatorTrainerSequential,
    load_objects
)
from .config import (
    AEConfig,
    EMConfig
)
from .nn import (
    Autoencoder,
    Emulator
)

def load_autoencoder(Config):
    autoencoder = Autoencoder(
        input_dim=Config.input_dim,
        latent_dim=Config.latent_dim,
        hidden_dims=Config.hidden_dims,
        noise=Config.noise,
        dropout=Config.dropout,
    ).to(Config.device)
    if os.path.exists(Config.pretrained_model_path):
        print("Loading Pretrained Model")
        autoencoder.load_state_dict(torch.load(Config.pretrained_model_path))
    return autoencoder


def load_emulator(Config):
    emulator = Emulator(
    ).to(Config.device)
    if os.path.exists(Config.pretrained_model_path):
        print("Loading Pretrained Model")
        emulator.load_state_dict(torch.load(Config.pretrained_model_path))
    return emulator


if __name__ == "__main__":
    training_np, validation_np = dl.load_datasets(EMConfig.columns)
    
    processing_functions = dp.Processing(
        AEConfig,
        EMConfig
    )
    inference_functions = Inference(AEConfig)
    training_dataset = dp.preprocessing_emulator_dataset(training_np, processing_functions)
    validation_dataset = dp.preprocessing_emulator_dataset(validation_np, processing_functions)
    
    dl.save_tensors_to_hdf5(training_dataset, category="training_seq")
    dl.save_tensors_to_hdf5(validation_dataset, category="validation_seq")

    training_dataset, training_indices = dl.load_tensors_from_hdf5(category="training_seq")
    validation_dataset, validation_indices = dl.load_tensors_from_hdf5(category="validation_seq")
    training_Dataset = dl.EmulatorSequenceDataset(training_dataset, training_indices)
    validation_Dataset = dl.EmulatorSequenceDataset(validation_dataset, validation_indices)
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dl.tensor_to_dataloader(EMConfig, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(EMConfig, validation_Dataset)
    
    
    autoencoder = load_autoencoder(AEConfig)
    emulator = load_emulator(EMConfig)
    optimizer, scheduler = load_objects(emulator)
    
    emulator_trainer = EmulatorTrainerSequential(
        emulator,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
        )
    emulator_trainer.train()