import jax
from jax import lax, random, numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state, orbax_utils
import torch
import optax
import orbax.checkpoint as ocp
from typing import Any
import matplotlib.pyplot as plt

def numpy_normalize(x, mean, std):
    x = x.resize(size=(128, 128))
    x = np.array(x, dtype=jnp.float32) / 255.
    return (x - mean) / std

def numpy_collate(batch):
    transposed_data = list(zip(*batch))
    imgs = np.stack(transposed_data[0])
    labels = np.stack(transposed_data[1])
    return imgs, labels

def create_train_state(model, rng, learning_rate):
    """ Instanciate the state outside of the training loop """
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))['params']
    opti = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params, tx=opti)
    
def numpy_to_torch(array):
    array = jax.device_get(array)
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor
    
class TrainStateEMA(train_state.TrainState):
    batch_stats: Any

def create_train_state_EMA(model, rng, learning_rate):
    """ Instanciate the state outside of the training loop """
    variables = model.init(rng, jnp.ones([1, 32, 32, 3]), training=True, mutable=True)
    opti = optax.adam(learning_rate)
    return TrainStateEMA.create(apply_fn=model.apply, params=variables['params'],
                                batch_stats=variables['batch_stats'], tx=opti)
    
def plot_reconstruction(original, reconstruction, step=0):
    fig = plt.figure(figsize=(20, 6))
    for i in range(10):
        ax = plt.subplot(3, 10, i + 1)
        img = (original[0][i+10].reshape((128, 128, 3)) * 0.25) + 0.5
        ax.imshow(img)
        ax.axis('off')
        ax.set_title("Original", fontweight="bold")
        ax = plt.subplot(3, 10, i + 11)
        img = (reconstruction[i+10].reshape((128, 128, 3)) * 0.25) + 0.5
        ax.imshow(img)
        ax.axis('off')
        ax.set_title("GD Rec", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f'./results/celeba_reconstruction_{step}.png')