from ChemSurrogate import data_processing as dp
from ChemSurrogate.trainer import (
    EmulatorTrainerSequential,
    load_iterative_emulator_objects
)
from ChemSurrogate.configs import (
    EMConfig
)

if __name__ == "__main__":
    training_np, validation_np = dp.load_datasets(EMConfig.columns)
    
    training_dataset = dp.prepare_emulator_dataset(training_np, sequential_time=True)
    validation_dataset = dp.prepare_emulator_dataset(validation_np, sequential_time=True)
    
    t_ds, t_inds = training_dataset
    v_ds, t_inds = validation_dataset
    
    t_ds = t_ds[:, :-12]
    v_ds = v_ds[:, :-12]
    
    training_dataset = (t_ds, t_inds)
    validation_dataset = (v_ds, t_inds)
    
    print(training_dataset[0].shape)
    print(validation_dataset[0].shape)
    dp.save_tensors_to_hdf5(training_dataset, category="training_seq")
    dp.save_tensors_to_hdf5(validation_dataset, category="validation_seq")

    # training_dataset, training_indices = dp.load_tensors_from_hdf5(category="training")
    # validation_dataset, validation_indices = dp.load_tensors_from_hdf5(category="validation")
    
    # training_Dataset = dp.EmulatorSequenceDataset(training_dataset)
    # validation_Dataset = dp.EmulatorSequenceDataset(validation_dataset)
    # del training_dataset, validation_dataset, training_indices, validation_indices

    # training_dataloader = dp.tensor_to_dataloader(EMConfig, training_Dataset)
    # validation_dataloader = dp.tensor_to_dataloader(EMConfig, validation_Dataset)
    
    # emulator, autoencoder, optimizer, scheduler = load_iterative_emulator_objects(final_training_phase=False)
    # emulator_trainer = EmulatorTrainerSequential(
    #     emulator,
    #     autoencoder,
    #     optimizer,
    #     scheduler,
    #     training_dataloader,
    #     validation_dataloader
    #     )
    # emulator_trainer.train()