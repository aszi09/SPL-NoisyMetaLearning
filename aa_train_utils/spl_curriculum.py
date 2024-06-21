
from functools import partial
import jax
import jax.numpy as jnp

import netket as nk
from torch.utils.data import DataLoader, Subset

@partial(jax.jit, static_argnums=0)
def per_sample_loss(part_func, key_loss,X, y, x_test, y_test):
    return part_func(X, y, x_test, y_test, rngs={'default': key_loss})

class SPL_curriculum:
    def __init__(self, start_rate, growth_epochs, dataset, batch_size, rng, chunk_size=128):

        self.batch_size = batch_size
        self.start_rate = start_rate
        self.growth_epochs= growth_epochs 
        self.dataset = dataset
        self.rng = rng
        self.weight_log = []
        self.epoch_losses_log = []
        self.chunk_size = chunk_size



    def data_curriculum(self, loss_partial, epoch, num_context_samples):
        """ Use the model to calculate the loss for the whole dataset, 
        and then use the loss to calculate the SPL weights for the dataset
        based on the current schedule
        """

        # Calculating the expansion on the dataset based on the current schedule with linear increase.
        
        # Over how many epochs should the data_rate increase from start_rate to 1.0
        if(epoch == 0):
            data_rate = self.start_rate

        else:

            data_rate =  min(1.0 , self.start_rate + (1.0 - self.start_rate) / self.growth_epochs * epoch)

        
        curr_data_size = int(data_rate * self.dataset.__len__())

        # dont do unecessary sorting and loss calculation if we are already full dataset size
        if curr_data_size == self.dataset.__len__():
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        

        losses = self.calculate_difficulty_ordering(loss_partial, num_context_samples)
        sorted_indices = jnp.argsort(losses)[:curr_data_size]
        self.weight_log.append(sorted_indices)
        self.epoch_losses_log.append(losses)

        return DataLoader(Subset(self.dataset, sorted_indices), batch_size=self.batch_size, shuffle=True, drop_last=True)  # Maybe shuffle? 
        # calculate the loss over the dataset with the current model and params
    
    def calculate_difficulty_ordering(self,loss_partial, num_context_samples):

        """ Calculate the difficulty of the dataset based on the model and params
        """
        # currently problem since self.dataset._get_data() returns tuple of 4 arrays, but we need to return a tuple of 4 arrays for each sample, so we need to do it in a different way
        

        self.rng, key_model = jax.random.split(self.rng) # might be problematic to always use the self.rng? if we reset it it should be reproducible i think
        
        key_losses = jax.random.split(key_model, self.dataset.__len__())
        chunked_loss_f =nk.jax.vmap_chunked(partial(per_sample_loss, loss_partial), in_axes=(0,0,0,0,0), chunk_size=self.chunk_size)

        xs , ys =self.dataset._get_data() 
        X, x_test = jnp.split(xs, indices_or_sections=(num_context_samples, ), axis=1)
        y, y_test = jnp.split(ys, indices_or_sections=(num_context_samples, ), axis=1)
        losses = chunked_loss_f(key_losses, X, y, x_test,y_test) 
        return losses 





