import torch
import numpy as np

class Inference():
    def __init__(self, GeneralConfig, processing_functions, autoencoder=None, emulator=None):
        self.device = GeneralConfig.device
        self.autoencoder = autoencoder
        self.emulator = emulator
            
        self.inverse_abundances_scaling = processing_functions.inverse_abundances_scaling
        
        self.latent_components_scaling = processing_functions.latent_components_scaling
        self.inverse_latent_components_scaling = processing_functions.inverse_latent_components_scaling

    def convert_to_tensor(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float().to(self.device)
        elif not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        return inputs
    
    def encode(self, abundances):
        with torch.no_grad():
            abundances = self.convert_to_tensor(abundances)
            latents = self.autoencoder.encode(abundances)
            return latents


    def decode(self, latents):
        with torch.no_grad():
            latents = self.convert_to_tensor(latents)
            scaled_abundances = self.autoencoder.decode(latents)
            abundances = self.inverse_abundances_scaling(scaled_abundances)
            return abundances


    def latent_emulate(self, phys, latents):
        with torch.no_grad():
            phys = self.convert_to_tensor(phys)
            latents = self.convert_to_tensor(latents)
            scaled_latents = self.latent_components_scaling(latents)
            scaled_evolved_latents = self.emulator(scaled_latents)
            evolved_latents = self.inverse_latent_components_scaling(scaled_evolved_latents)
            return evolved_latents

    def emulate(self, phys, abundances):
        encoded_abundances = self.encode(abundances)
        evolved_latents = self.latent_emulate(phys, encoded_abundances)
        evolved_abundances = self.decode(evolved_latents)
        return evolved_abundances