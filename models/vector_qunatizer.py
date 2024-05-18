import jax
from jax import lax, numpy as jnp
from flax import linen as nn

from typing import Any
from jax.typing import ArrayLike

class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    beta: float

    @nn.compact
    def __call__(self, inputs):
        codebook = self.param('codebook', nn.initializers.lecun_uniform(),
                                   (self.embedding_dim, self.num_embeddings))
        flat_inputs = jnp.reshape(inputs, (-1, self.embedding_dim))
        distances = (jnp.sum(jnp.square(flat_inputs), 1, keepdims=True) -
                     2 * jnp.matmul(flat_inputs, codebook) +
                     jnp.sum(jnp.square(codebook), 0, keepdims=True))
        encoding_indices = jnp.argmin(distances, 1)
        flat_quantized = jnp.take(codebook, encoding_indices, axis=1).swapaxes(1, 0)
        quantized = jnp.reshape(flat_quantized, inputs.shape)

        # Losses computation
        codebook_loss = jnp.mean(jnp.square(quantized - jax.lax.stop_gradient(inputs)))
        commitment_loss = self.beta * jnp.mean(jnp.square(jax.lax.stop_gradient(quantized) - inputs))

        # Straight Through Estimator : returns the value of the quantized latent space
        # and multiplies gradient by 1 in chain rule, as input = output
        # - i.e. gradient from the decoder passed directly to the encoder in backprop phase
        ste = inputs + jax.lax.stop_gradient(quantized - inputs)

        # Perplexity computation
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings, dtype=distances.dtype)
        avg_probs = jnp.mean(encodings, 0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        return ste, perplexity, codebook_loss, commitment_loss
    
class EMA(nn.Module):
    # inspired from https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/moving_averages.py
    decay: float
    shape: list
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, value, update_stats):
        hidden = self.variable('batch_stats', 'hidden',
                                    lambda shape: jnp.zeros(shape, dtype=self.dtype),
                                    self.shape)
        average = self.variable('batch_stats', 'average',
                                     lambda shape: jnp.zeros(shape, dtype=self.dtype),
                                     self.shape)
        counter = self.variable('batch_stats', 'counter',
                                     lambda shape: jnp.zeros(shape, dtype=jnp.int32),
                                     ())
        
        counter = counter.value + 1
        decay = jax.lax.convert_element_type(self.decay, value.dtype)
        one = jnp.ones([], value.dtype)
        hidden = hidden.value * decay + value * (one - decay)
        average = hidden
        # Apply zero-debiasing
        average /= (one - jnp.power(decay, counter))
        if update_stats:
            counter.value = counter
            hidden.value = hidden
            average.value = average
        return average


class VectorQuantizerEMA(nn.Module):
    # Method described in the appendix of the original article https://arxiv.org/pdf/1711.00937.pdf
    # Codebook loss replaced by EMA update of the codebook
    num_embeddings: int
    embedding_dim: int
    rng: ArrayLike
    beta: float
    gamma: float
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, inputs, training: bool):
        codebook = self.variable('batch_stats', 'codebook',
                                      nn.initializers.lecun_uniform(), self.rng,
                                      (self.embedding_dim, self.num_embeddings))
        ema_cluster_size = EMA(self.gamma, (self.num_embeddings))
        ema_dw = EMA(self.gamma, (self.embedding_dim, self.num_embeddings))
        
        flat_inputs = jnp.reshape(inputs, (-1, self.embedding_dim))
        distances = (jnp.sum(jnp.square(flat_inputs), 1, keepdims=True) -
                     2 * jnp.matmul(flat_inputs, codebook.value) +
                     jnp.sum(jnp.square(codebook.value), 0, keepdims=True))
        encoding_indices = jnp.argmin(distances, 1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings, dtype=distances.dtype)
        flat_quantized = jnp.take(codebook.value, encoding_indices, axis=1).swapaxes(1, 0)
        quantized = jnp.reshape(flat_quantized, inputs.shape)
        loss = self.beta * jnp.mean(jnp.square(jax.lax.stop_gradient(quantized) - inputs))

        # Update the codebook with EMA only if the model is in training mode
        if training:
            # Number of closest inputs for each embedding in the codebook (size: num_embeddings)
            cluster_size = jnp.sum(encodings, axis=0)
            updated_ema_cluster_size = ema_cluster_size(cluster_size, update_stats=True)
            # Sum of inputs within clusters (equation (7))
            dw = jnp.matmul(flat_inputs.T, encodings)
            updated_ema_dw = ema_dw(dw, update_stats=True)
            # Laplace smoothing of cluster size / nb of elements
            n = jnp.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
            normalised_updated_ema_w = updated_ema_dw / jnp.reshape(updated_ema_cluster_size, [1, -1])

            self.codebook.value = normalised_updated_ema_w

        # Straight Through Estimator : returns the value of the quantized latent space
        # and multiplies gradient by 1 in chain rule, as input = output
        # - i.e. gradient from the decoder passed directly to the encoder in backprop phase
        ste = inputs + jax.lax.stop_gradient(quantized - inputs)

        # Perplexity computation
        avg_probs = jnp.mean(encodings, 0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        return ste, perplexity, loss