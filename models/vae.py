from jax.typing import ArrayLike
from flax import linen as nn

from models.modules import Encoder, Decoder
from models.vector_qunatizer import VectorQuantizer, VectorQuantizerEMA

class VQVAE(nn.Module):
    latent_dim: int
    num_embeddings: int
    beta: float

    @nn.compact
    def __call__(self, x):
        ze = Encoder(latent_dim=self.latent_dim, num_embeddings=self.num_embeddings)(x)
        zq, perplexity, codebook_loss, commitment_loss = VectorQuantizer(num_embeddings=self.num_embeddings,embedding_dim=self.latent_dim, beta=self.beta)(ze)
        reconstructions = Decoder(latent_dim=self.latent_dim)(zq)
        return reconstructions, perplexity, codebook_loss, commitment_loss
    
class VQVAE_EMA(nn.Module):
    latent_dim: int
    num_embeddings: int
    rng: ArrayLike
    beta: float
    gamma: float
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, training):
        ze = Encoder(latent_dim=self.latent_dim, num_embeddings=self.num_embeddings)(x)
        zq, perplexity, loss = VectorQuantizerEMA(num_embeddings=self.num_embeddings,
                                                   embedding_dim=self.latent_dim, rng=self.rng,
                                                   beta=self.beta, gamma=self.gamma, epsilon=self.epsilon)(ze, training)
        reconstructions = Decoder(latent_dim=self.latent_dim)(zq)
        return reconstructions, perplexity, loss
