from dataclasses import dataclass
from typing import Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from vqvae import VQVAEModel, Encoder, Decoder


@dataclass
class Config:
    # Set hyper-parameters.
    batch_size: int = 32
    image_size: int = 32

    # 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
    num_training_updates: int = 100000

    num_hiddens: int = 128
    num_residual_hiddens: int = 32
    num_residual_layers: int = 2
    # These hyper-parameters define the size of the model (number of parameters and layers).
    # The hyper-parameters in the paper were (For ImageNet):
    # batch_size = 128
    # image_size = 128
    # num_hiddens = 128
    # num_residual_hiddens = 32
    # num_residual_layers = 2

    # This value is not that important, usually 64 works.
    # This will not change the capacity in the information-bottleneck.
    embedding_dim: int = 64

    # The higher this value, the higher the capacity in the information bottleneck.
    num_embeddings: int = 512

    # commitment_cost should be set appropriately. It's often useful to try a couple
    # of values. It mostly depends on the scale of the reconstruction cost
    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
    # commitment_cost should also be multiplied with the same amount.
    commitment_cost: float = 0.25

    # Use EMA updates for the codebook (instead of the Adam optimizer).
    # This typically converges faster, and makes the model less dependent on choice
    # of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
    # developed afterwards). See Appendix of the paper for more details.
    vq_use_ema: bool = True

    # This is only used for EMA updates.
    decay: float = 0.99

    learning_rate: float = 3e-4


def cast_and_normalise_images(data_dict: Dict):
    """Convert images to floating point with the range [-0.5, 0.5]"""
    data_dict['image'] = (tf.cast(data_dict['image'], tf.float32) / 255.0) - 0.5
    return data_dict


class Trainer:
    def __init__(self):
        self.cfg = Config()

    @jax.jit
    def train_step(self, params, state, opt_state, data):
        def adapt_forward(params, state, data):
            # Pack model output and state together.
            model_output, state = self.forward.apply(params, state, None, data, is_training=True)
            loss = model_output['loss']
            return loss, (model_output, state)

        grads, (model_output, state) = (
            jax.grad(adapt_forward, has_aux=True)(params, state, data))

        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, state, opt_state, model_output

    def forward(self, data, is_training):
        encoder = Encoder(self.cfg.num_hiddens, self.cfg.num_residual_layers, self.cfg.num_residual_hiddens)
        decoder = Decoder(self.cfg.num_hiddens, self.cfg.num_residual_layers, self.cfg.num_residual_hiddens)
        pre_vq_conv1 = hk.Conv2D(output_channels=self.cfg.embedding_dim, kernel_shape=(1, 1), stride=(1, 1), name="to_vq")

        if vq_use_ema:
            vq_vae = hk.nets.VectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                decay=decay)
        else:
            vq_vae = hk.nets.VectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost)

        model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                        data_variance=train_data_variance)

        return model(data['image'], is_training)

    def train(self):
        # # Data Loading.
        train_dataset = tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(train_data_dict)
            .map(cast_and_normalise_images)
            .shuffle(10000)
            .repeat(-1)  # repeat indefinitely
            .batch(batch_size, drop_remainder=True)
            .prefetch(-1))
        valid_dataset = tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(valid_data_dict)
            .map(cast_and_normalise_images)
            .repeat(1)  # 1 epoch
            .batch(batch_size)
            .prefetch(-1))

        # Build modules.
        self.forward = hk.transform_with_state(self.forward)
        self.optimizer = optax.adam(learning_rate)

        train_losses = []
        train_recon_errors = []
        train_perplexities = []
        train_vqvae_loss = []

        rng = jax.random.PRNGKey(42)
        train_dataset_iter = iter(train_dataset)
        params, state = self.forward.init(rng, next(train_dataset_iter), is_training=True)
        opt_state = self.optimizer.init(params)

        for step in range(1, num_training_updates + 1):
            data = next(train_dataset_iter)
            params, state, opt_state, train_results = (
                self.train_step(params, state, opt_state, data))

            train_results = jax.device_get(train_results)
            train_losses.append(train_results['loss'])
            train_recon_errors.append(train_results['recon_error'])
            train_perplexities.append(train_results['vq_output']['perplexity'])
            train_vqvae_loss.append(train_results['vq_output']['loss'])

            if step % 100 == 0:
                print(f'[Step {step}/{num_training_updates}] ' +
                    ('train loss: %f ' % np.mean(train_losses[-100:])) +
                    ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                    ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
                    ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))


        train_batch = next(iter(train_dataset))
        valid_batch = next(iter(valid_dataset))

        # Put data through the model with is_training=False, so that in the case of
        # using EMA the codebook is not updated.
        train_reconstructions = self.forward.apply(params, state, rng, train_batch, is_training=False)[0]['x_recon']
        valid_reconstructions = self.forward.apply(params, state, rng, valid_batch, is_training=False)[0]['x_recon']

if __name__=='__main__':
    trainer = Trainer()
    trainer.train()