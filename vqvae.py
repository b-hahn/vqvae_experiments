import haiku as hk
import jax
from jax.interpreters.batching import batch, batch_subtrace
import jax.numpy as jnp


class ResidualStack(hk.Module):

    def __init__(self, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int, name: str = None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = hk.Conv2D(output_channels=num_residual_hiddens,
                              kernel_shape=(3, 3),
                              stride=(1, 1),
                              name="res3x3_%d" % i)
            conv1 = hk.Conv2D(
                output_channels=num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def __call__(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(jax.nn.relu(h))
            conv1_out = conv1(jax.nn.relu(conv3_out))
            h += conv1_out
        return jax.nn.relu(h)  # Resnet V1 style


class Encoder(hk.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = hk.Conv2D(
            output_channels=self._num_hiddens // 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")
        self._enc_2 = hk.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")
        self._enc_3 = hk.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

    def __call__(self, x):
        h = jax.nn.relu(self._enc_1(x))
        h = jax.nn.relu(self._enc_2(h))
        h = jax.nn.relu(self._enc_3(h))
        return self._residual_stack(h)


class Decoder(hk.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = hk.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        self._dec_2 = hk.Conv2DTranspose(
            output_channels=self._num_hiddens // 2,
            # output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")
        self._dec_3 = hk.Conv2DTranspose(
            output_channels=3,
            # output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")

    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = jax.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)
        return x_recon


class VQVAEModel(hk.Module):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1,
                 data_variance, name=None):
        super().__init__(name=name)
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    def __call__(self, inputs, is_training):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z, is_training=is_training)
        x_recon = self._decoder(vq_output['quantize'])
        recon_error = jnp.mean((x_recon - inputs) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
            'z': z,
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'vq_output': vq_output,
        }

    def sample(self, key):
        batch_size = 32
        random_vector = jax.random.normal(key,
                                          shape=(batch_size, 32, 32, self._vqvae.embedding_dim))
                                        #   shape=(batch_size, self._vqvae.num_embeddings, self._vqvae.embedding_dim))
        vq_output = self._vqvae(random_vector, is_training=False)
        sample = self._decoder(vq_output['quantize'])
        return sample

    def get_code(self, input: jnp.ndarray):
        z = self._pre_vq_conv1(self._encoder(input))
        vq_output = self._vqvae(z, is_training=False)
        return vq_output
