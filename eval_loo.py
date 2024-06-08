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
    """ Computes the metric on the test batch (code already included in train_step for train batch) """
    x_recon, perplexity, codebook_loss, commitment_loss = state.apply_fn({'params': state.params}, batch[0])
    recon_loss = optax.l2_loss(predictions=x_recon, targets=batch[0]).mean()
    metrics = {"perplexity": perplexity, "recon_loss": recon_loss,
               "codebook_loss": codebook_loss, "commitment_loss": commitment_loss}
    return x_recon, recon_loss + codebook_loss + commitment_loss, metrics

def eval_dataset(state, testloader):
    loss_test, perplexity_test = [], []
    for batch in testloader:
        img_reconstruction, loss, metrics = eval_step(state, batch)
        loss_test.append(loss)
        perplexity_test.append(metrics["perplexity"].item())
    return np.mean(loss_test), img_reconstruction

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
    
    indices = tqdm(range(conf.num_embeddings))
    for idx in indices:
        loo_state = drop_codebook(loo_state, state, idx)
        loo_loss_test, loo_img_reconstruction = eval_dataset(loo_state, testloader)
        writer.add_scalars('loo_effect', {'test': loo_loss_test - original_loss_test}, idx)
        loo_img_reconstruction = utils.numpy_to_torch(loo_img_reconstruction[:16])
        img_grid = torchvision.utils.make_grid(loo_img_reconstruction, nrow=4, normalize=True, pad_value=0.9)
        writer.add_image('sample_cat', img_grid, idx)
        indices.set_description(f"idx: {idx}, loo_loss: {loo_loss_test}, original_loss: {original_loss_test}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args)
