import jax
from flax import linen as nn

class ResnetBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        h = nn.swish(nn.GroupNorm()(x))
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=1)(h)
        h = nn.swish(nn.GroupNorm()(h))
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=1)(h)
        return x + h

class Encoder(nn.Module):
    latent_dim: int
    num_embeddings: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(self.latent_dim // 2, kernel_size=(4, 4), strides=(2, 2), padding=1)(x))
        x = nn.relu(nn.Conv(self.latent_dim, kernel_size=(4, 4), strides=(2, 2), padding=1)(x))
        x = nn.Conv(self.latent_dim, kernel_size=(3, 3), strides=(1, 1), padding=1)(x)
        x = ResnetBlock(self.latent_dim)(x)
        x = ResnetBlock(self.latent_dim)(x)
        return x

class Upsample(nn.Module):
    out_channels: int
    upfactor: int

    @nn.compact
    def __call__(self, x):
        batch, height, width, channels = x.shape
        h = jax.image.resize(x,shape=(batch, height * self.upfactor, width * self.upfactor, channels), method="bilinear")
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=1)(h)
        return h

class Decoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = ResnetBlock(self.latent_dim)(x)
        x = ResnetBlock(self.latent_dim)(x)
        x = nn.relu(Upsample(out_channels=self.latent_dim // 2, upfactor=2)(x))
        x = Upsample(out_channels=3, upfactor=2)(x)
        return x