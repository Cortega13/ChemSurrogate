from ChemSurrogate import data_processing as dp
from ChemSurrogate.trainer import (
    EmulatorTrainer,
    load_skipcon_emulator_objects
)
from ChemSurrogate.configs import (
    EMConfig
)

if __name__ == "__main__":
    # training_np, validation_np = dp.load_datasets(EMConfig.columns)
    
    # training_dataset = dp.prepare_emulator_dataset(training_np)
    # validation_dataset = dp.prepare_emulator_dataset(validation_np)
        
    # =
    # dp.save_tensors_to_hdf5(validation_dataset, category="validation")

    training_dataset, training_indices = dp.load_tensors_from_hdf5(category="training")
    validation_dataset, validation_indices = dp.load_tensors_from_hdf5(category="validation")
    
    training_Dataset = dp.EmulatorTimestepDataset(training_dataset, training_indices)
    validation_Dataset = dp.EmulatorTimestepDataset(validation_dataset, validation_indices)
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dp.tensor_to_dataloader(EMConfig, training_Dataset)
    validation_dataloader = dp.tensor_to_dataloader(EMConfig, validation_Dataset)
    
    emulator, autoencoder, optimizer, scheduler = load_skipcon_emulator_objects()
    emulator_trainer = EmulatorTrainer(
        emulator,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
        )
    emulator_trainer.train()