from ChemSurrogate import data_processing as dp
from ChemSurrogate.trainer import (
    EmulatorTrainerSequential,
    load_iterative_emulator_objects
)
from ChemSurrogate.configs import (
    EMConfig
)

if __name__ == "__main__":
    # training_np, validation_np = dp.load_datasets(EMConfig.columns)
    
    # training_dataset = dp.prepare_emulator_dataset(training_np)
    # validation_dataset = dp.prepare_emulator_dataset(validation_np)
    
    # dp.save_tensors_to_hdf5(training_dataset, category="training_seq")
    # dp.save_tensors_to_hdf5(validation_dataset, category="validation_seq")

    training_dataset, training_indices = dp.load_tensors_from_hdf5(category="training_seq")
    validation_dataset, validation_indices = dp.load_tensors_from_hdf5(category="validation_seq")
    training_Dataset = dp.EmulatorSequenceDataset(training_dataset, training_indices)
    validation_Dataset = dp.EmulatorSequenceDataset(validation_dataset, validation_indices)
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dp.tensor_to_dataloader(EMConfig, training_Dataset)
    validation_dataloader = dp.tensor_to_dataloader(EMConfig, validation_Dataset)
    
    autoencoder, emulator, optimizer, scheduler = load_iterative_emulator_objects()
    emulator_trainer = EmulatorTrainerSequential(
        emulator,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
        )
    emulator_trainer.train()