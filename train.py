import jax
import numpy as np
from flax.training import orbax_utils
import optax
import orbax.checkpoint as ocp
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CelebA
from tqdm import tqdm

import utils
from models.vae import VQVAE, VQVAE_EMA

# Globales
batch_size = 256
workers = 8
learning_rate = 0.001
epochs = 100
seed = 1234
beta = 0.25
gamma = 0.9
num_embeddings = 512
latent_dim = 64
writer = SummaryWriter("./logs/standard_loss")
use_ema = False
# Calculated in the jupyter notebook
img_mean = (0.5, 0.5, 0.5)
img_std = (0.25, 0.25, 0.25)

@jax.jit
def train_step(state, batch):
    """ Train for a single step """
    def loss_fn(params):
        x_recon, perplexity, codebook_loss, commitment_loss = state.apply_fn({'params': params}, batch[0])
        recon_loss = optax.squared_error(predictions=x_recon, targets=batch[0]).mean()
        metrics = {"perplexity": perplexity, "recon_loss": recon_loss,
                   "codebook_loss": codebook_loss, "commitment_loss": commitment_loss}
        return recon_loss + codebook_loss + commitment_loss, metrics

    # Update parameters with gradient descent
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics

@jax.jit
def train_step_EMA(state, batch):
    """ Train for a single step """
    def loss_fn(params, batch_stats):
        (x_recon, perplexity, commitment_loss), _ = state.apply_fn({'params': params,
                                                                    'batch_stats': batch_stats},
                                                                    batch[0], training=True,
                                                                    mutable=True)
        recon_loss = optax.squared_error(predictions=x_recon, targets=batch[0]).mean()
        metrics = {"perplexity": perplexity, "recon_loss": recon_loss,
                   "commitment_loss": commitment_loss}
        return recon_loss + commitment_loss, metrics

    # Update parameters with gradient descent
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state.batch_stats)
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics

@jax.jit
def eval_step(state, batch):
    """ Computes the metric on the test batch (code already included in train_step for train batch) """
    x_recon, perplexity, codebook_loss, commitment_loss = state.apply_fn({'params': state.params}, batch[0])
    recon_loss = optax.l2_loss(predictions=x_recon, targets=batch[0]).mean()
    metrics = {"perplexity": perplexity, "recon_loss": recon_loss,
               "codebook_loss": codebook_loss, "commitment_loss": commitment_loss}
    return x_recon, recon_loss + codebook_loss + commitment_loss, metrics

@jax.jit
def eval_step_EMA(state, batch):
    """ Computes the metric on the test batch (code already included in train_step for train batch) """
    x_recon, perplexity, commitment_loss = state.apply_fn({'params': state.params,
                                                           'batch_stats': state.batch_stats},
                                                          batch[0], training=False)
    recon_loss = optax.l2_loss(predictions=x_recon, targets=batch[0]).mean()
    metrics = {"perplexity": perplexity, "recon_loss": recon_loss,
               "commitment_loss": commitment_loss}
    return x_recon, recon_loss + commitment_loss, metrics

def main():
    trainset = CelebA(root="./data", split='train', download=True,
                       transform=lambda x: utils.numpy_normalize(x, img_mean, img_std))
    testset = CelebA(root="./data", split='test', download=True,
                       transform=lambda x: utils.numpy_normalize(x, img_mean, img_std))
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=workers,
                             collate_fn=utils.numpy_collate, drop_last=True)
    testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=workers,
                            collate_fn=utils.numpy_collate, drop_last=True)

    # Model initialization
    if use_ema:
        init_rng = jax.random.PRNGKey(seed)
        init_rng, codebook_rng = jax.random.split(init_rng)
        model = VQVAE_EMA(num_embeddings=num_embeddings, latent_dim=latent_dim, rng=codebook_rng,
                        beta=beta, gamma=gamma)
        state = utils.create_train_state_EMA(model, init_rng, learning_rate)
    else:
        model = VQVAE(num_embeddings=num_embeddings, latent_dim=latent_dim, beta=beta)
        init_rng = jax.random.PRNGKey(seed)
        state = utils.create_train_state(model, init_rng, learning_rate)
    del init_rng

    # Training loop
    epoch = tqdm(range(epochs))
    step = 0
    for e in epoch:
        loss_train, loss_test, perplexity_train, perplexity_test = [], [], [], []
        recon_loss, codebook_loss, commitment_loss = [], [], []
        torch.manual_seed(seed)
        for batch in trainloader:
            if use_ema:
                state, loss, metrics = train_step_EMA(state, batch)
            else:
                state, loss, metrics = train_step(state, batch)
            loss_train.append(loss)
            perplexity_train.append(metrics["perplexity"].item())
            recon_loss.append(metrics["recon_loss"].item())
            codebook_loss.append(metrics["codebook_loss"].item())
            commitment_loss.append(metrics["commitment_loss"].item())
        writer.add_scalars('losses_train', {'recon': np.mean(recon_loss),
                                            'codebook': np.mean(codebook_loss),
                                            'commitment': np.mean(commitment_loss)}, step)

        # Compute metrics on the test set after each training epoch
        test_state = state
        for batch in testloader:
            if use_ema:
                img_reconstruction, loss, metrics = eval_step_EMA(test_state, batch)
            else:
                img_reconstruction, loss, metrics = eval_step(test_state, batch)
            loss_test.append(loss)
            perplexity_test.append(metrics["perplexity"].item())
        
        utils.plot_reconstruction(batch, img_reconstruction, step=step)
            
        writer.add_scalars('loss', {'train': np.mean(loss_train),
                                    'test': np.mean(loss_test)}, e)
        writer.add_scalars('perplexity', {'train': np.mean(perplexity_train),
                                          'test': np.mean(perplexity_test)}, e)
        epoch.set_description(f"Epoch: {e+1}/{epochs} - Train Loss: {np.mean(loss_train):.4f} - Test loss: {np.mean(loss_test):.4f}")
    
    # Save model
    ckpt = {'model': state}
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(f"./vqvae_std_lr{learning_rate}_e{epochs}", ckpt, save_args=save_args)


if __name__ == "__main__":
    main()
