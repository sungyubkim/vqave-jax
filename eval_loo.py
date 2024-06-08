import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils
import optax
import orbax.checkpoint as ocp
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets import CelebA
from tqdm import tqdm
from copy import deepcopy

import utils
from models.vae import VQVAE, VQVAE_EMA
from omegaconf import OmegaConf
from argparse import ArgumentParser

@jax.jit
def eval_step(state, batch):
    x_recon, _, _, _ = state.apply_fn({'params': state.params}, batch[0])
    recon_loss = optax.l2_loss(predictions=x_recon, targets=batch[0]).sum(axis=(1, 2, 3))
    return x_recon, recon_loss

def eval_dataset(state, testloader):
    max_losses = []
    max_reconstructions = []
    original_samples = []

    for batch in testloader:
        x_recon, loss = eval_step(state, batch)
        # Get the indices of the largest losses in the current batch
        sorted_indices = np.argsort(loss)[::-1]
        batch_top_indices = sorted_indices[:16]

        for idx in batch_top_indices:
            if len(max_losses) < 16:
                max_losses.append(loss[idx])
                max_reconstructions.append(x_recon[idx])
                original_samples.append(batch[0][idx])
            else:
                # Find the smallest loss in the current top 16 losses
                min_loss_idx = np.argmin(max_losses)
                if loss[idx] > max_losses[min_loss_idx]:
                    # Replace the smallest loss with the current loss
                    max_losses[min_loss_idx] = loss[idx]
                    max_reconstructions[min_loss_idx] = x_recon[idx]
                    original_samples[min_loss_idx] = batch[0][idx]

    # Convert to numpy arrays for returning
    max_losses = np.array(max_losses)
    max_reconstructions = np.array(max_reconstructions)
    original_samples = np.array(original_samples)

    # Sort the final top losses and reconstructions
    sorted_final_indices = np.argsort(max_losses)[::-1]
    top_losses = max_losses[sorted_final_indices]
    top_reconstructions = max_reconstructions[sorted_final_indices]
    top_originals = original_samples[sorted_final_indices]
    
    # concatenate the top 8 original samples and reconstructions
    top_reconstructions = jnp.concatenate([top_originals, top_reconstructions], axis=0)

    return top_losses.sum(), top_reconstructions

def drop_codebook(loo_state, state, idx):
    # drop the idx column only
    new_params = deepcopy(state.params)
    new_params['VectorQuantizer_0']['codebook'] = jnp.delete(state.params['VectorQuantizer_0']['codebook'], idx, axis=1)
    loo_state = loo_state.replace(params=new_params)
    return loo_state

def main(args):
    conf = OmegaConf.load(args.config)
    writer = SummaryWriter('./logs/loo_effect')
    model_name = "vqvae_ema" if conf.use_ema else "vqvae"
    
    # Data loading
    testset = CelebA(root="~/data", split='test', download=True,
                       transform=lambda x: utils.numpy_normalize(x, conf.img_mean, conf.img_std))
    testloader = DataLoader(testset, conf.batch_size, shuffle=False, num_workers=conf.num_workers,
                            collate_fn=utils.numpy_collate, drop_last=True)

    # Model initialization
    model = VQVAE(num_embeddings=conf.num_embeddings, latent_dim=conf.latent_dim, beta=conf.beta)
    init_rng = jax.random.PRNGKey(conf.seed)
    state = utils.create_train_state(model, init_rng, conf.learning_rate)
    loo_model = VQVAE(num_embeddings=conf.num_embeddings-1, latent_dim=conf.latent_dim, beta=conf.beta)
    loo_state = utils.create_train_state(loo_model, init_rng, conf.learning_rate)
    
    # load state from checkpoint
    ckpt = {'model': state}
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    state = orbax_checkpointer.restore(
        f"/home/sungyub/vqave-jax/results/{model_name}_std_lr{conf.learning_rate}_e{conf.num_epochs}", 
        ckpt,
        )['model']
    
    original_loss_test, original_image_reconstruction = eval_dataset(state, testloader)
    original_image_reconstruction = utils.numpy_to_torch(original_image_reconstruction)
    img_grid = torchvision.utils.make_grid(original_image_reconstruction, nrow=8, normalize=True, pad_value=0.9, value_range=(-2, 2))
    writer.add_image('original_reconstructions', img_grid, 0)
    
    indices = tqdm(range(conf.num_embeddings))
    for idx in indices:
        loo_state = drop_codebook(loo_state, state, idx)
        loo_loss_test, loo_img_reconstruction = eval_dataset(loo_state, testloader)
        writer.add_scalars('loo_effect', {'test': loo_loss_test - original_loss_test}, idx)
        loo_img_reconstruction = utils.numpy_to_torch(loo_img_reconstruction)
        img_grid = torchvision.utils.make_grid(loo_img_reconstruction, nrow=8, normalize=True, pad_value=0.9, value_range=(-2, 2))
        writer.add_image('loo_reconstructions', img_grid, idx)
        indices.set_description(f"idx: {idx}, loo_loss: {loo_loss_test}, original_loss: {original_loss_test}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args)
