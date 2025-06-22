import torch

class LossFunctions():
    def __init__(self, constants):
        self.stoichiometric_matrix = constants.stoichiometric_matrix

    @staticmethod
    @torch.jit.script
    def jit_elemental_conservation(
        tensor1: torch.Tensor, 
        tensor2: torch.Tensor,
        stoichiometric_matrix: torch.Tensor,
        inverse_abundances_scaling: callable,
        ):
        """
        Given the actual and predicted abundances, this function calculates a loss between the elemental abundances of both.
        """
        unscaled_tensor1 = inverse_abundances_scaling(tensor1)
        unscaled_tensor2 = inverse_abundances_scaling(tensor2)
        
        elemental_abundances1 = torch.abs(torch.matmul(unscaled_tensor1, stoichiometric_matrix))
        elemental_abundances2 = torch.abs(torch.matmul(unscaled_tensor2, stoichiometric_matrix))

        log_elemental_abundances1 = torch.log10(elemental_abundances1)
        log_elemental_abundances2 = torch.log10(elemental_abundances2)
        
        diff = torch.abs(log_elemental_abundances2 - log_elemental_abundances1)
        
        return torch.sum(diff) / tensor1.size(0)


    @staticmethod
    @torch.jit.script
    def jit_elementwise_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        exponential: torch.Tensor,
        power_weight: torch.Tensor,
        ):
        elementwise_loss = torch.abs(outputs - targets)
        elementwise_loss = torch.exp(power_weight * exponential * elementwise_loss) - 1
        elementwise_loss = torch.sum(elementwise_loss) / targets.size(0)
        return elementwise_loss


    @staticmethod
    @torch.jit.script
    def jit_relative_error(
        outputs, 
        targets,
        inverse_abundances_scaling: callable,
        ):
        unscaled_outputs = inverse_abundances_scaling(outputs)
        unscaled_targets = inverse_abundances_scaling(targets)
        
        loss = (torch.abs(unscaled_targets - unscaled_outputs) / unscaled_targets)
        
        return torch.sum(loss, dim=0)


    def training(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        ):
        """
        This is the custom loss function for the autoencoder. It's a combination of the reconstruction loss and the conservation loss.
        """
        
        
        elementwise_loss = self.jit_elementwise_loss(outputs, targets, self.exponential, self.power_weight)
        
        conservation_error = self.conservation_weight * self.jit_elemental_conservation(outputs, targets, self.stoichiometric_matrix, self.inverse_abundances_scaling)  
        
        total_loss = 1e-3 * (elementwise_loss + conservation_error)
        
        print(f"Recon: {elementwise_loss.detach():.3e} | Cons: {conservation_error.detach():.3e} | Total: {total_loss.detach():.3e}")
        return total_loss


    def validation(self, outputs, targets):
        """
        This is the custom loss function for the autoencoder. It's a combination of the reconstruction loss and the conservation loss.
        """
        loss = self.jit_relative_error(outputs, targets, self.inverse_abundances_scaling)
        
        return loss
