import datetime
from functools import partial
from pathlib import Path
from typing import Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from config import Config
from vqvae import VQVAEModel, Encoder, Decoder
import dataloader
import visualization


def cast_and_normalise_images(data_dict: Dict):
    """Convert images to floating point with the range [-0.5, 0.5]"""
    data_dict['image'] = (tf.cast(data_dict['image'], tf.float32) / 255.0) - 0.5
    return data_dict

def cast_and_normalise_images_swisstopo(data: jnp.ndarray):
    """Convert images to floating point with the range [-0.5, 0.5]"""
    data = (tf.cast(data, tf.float32) / 255.0) - 0.5
    return data


class Trainer:
    def __init__(self):
        self.cfg = Config()

    @partial(jax.jit, static_argnums=(0,))
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

        if self.cfg.vq_use_ema:
            vq_vae = hk.nets.VectorQuantizerEMA(
                embedding_dim=self.cfg.embedding_dim,
                num_embeddings=self.cfg.num_embeddings,
                commitment_cost=self.cfg.commitment_cost,
                decay=self.cfg.decay)
        else:
            vq_vae = hk.nets.VectorQuantizer(
                embedding_dim=self.cfg.embedding_dim,
                num_embeddings=self.cfg.num_embeddings,
                commitment_cost=self.cfg.commitment_cost)

        model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                        data_variance=self.train_data_variance)

        # return model(data['image'], is_training)
        return model(data, is_training)

    def train(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_dir = Path(f"run_{timestamp}")
        run_dir.mkdir(exist_ok=True)
        # # Data Loading.
        # train_data_dict = dataloader.get_cifar_dataset(split='train')
        # train_dataset = tfds.as_numpy(
        #     tf.data.Dataset.from_tensor_slices(train_data_dict)
        #     .map(cast_and_normalise_images)
        #     .shuffle(10000)
        #     .repeat(-1)  # repeat indefinitely
        #     .batch(self.cfg.batch_size, drop_remainder=True)
        #     .prefetch(-1))
        # train_dataset = tfds.as_numpy(
        #     dataloader.get_swisstopo_dataset()
        #     .map(cast_and_normalise_images)
        #     .shuffle(10000)
        #     .repeat(-1)  # repeat indefinitely
        #     .batch(self.cfg.batch_size, drop_remainder=True)
        #     .prefetch(-1))
        # valid_data_dict = dataloader.get_cifar_dataset(split='val')
        # valid_dataset = tfds.as_numpy(
        #     tf.data.Dataset.from_tensor_slices(valid_data_dict)
        #     .map(cast_and_normalise_images)
        #     .repeat(1)  # 1 epoch
        #     .batch(self.cfg.batch_size)
        #     .prefetch(-1))
        # self.train_data_variance = np.var(train_data_dict['image'] / 255.0)
        train_dataset = dataloader.get_swisstopo_dataset(split='train', img_w=128,
                                                         img_h=128).map(cast_and_normalise_images_swisstopo).repeat(-1)
        valid_dataset = dataloader.get_swisstopo_dataset(split='val', img_w=128,
                                                         img_h=128).map(cast_and_normalise_images_swisstopo).repeat(1)
        # TODO: compute actual variance
        self.train_data_variance = 0.01

        # valid_data_dict = dataloader.get_swisstopo_dataset((split='val')
        # valid_dataset = tfds.as_numpy(
        #     tf.data.Dataset.from_tensor_slices(valid_data_dict)
        #     .map(cast_and_normalise_images)
        #     .repeat(1)  # 1 epoch
        #     .batch(self.cfg.batch_size)
        #     .prefetch(-1))

        # Build modules.
        self.forward = hk.transform_with_state(self.forward)
        self.optimizer = optax.adam(self.cfg.learning_rate)

        train_losses = []
        train_recon_errors = []
        train_perplexities = []
        train_vqvae_loss = []

        rng = jax.random.PRNGKey(42)
        train_dataset_iter = iter(train_dataset)
        params, state = self.forward.init(rng, jnp.asarray(next(train_dataset_iter)), is_training=True)
        opt_state = self.optimizer.init(params)

        for step in range(1, self.cfg.num_training_updates + 1):
            data = jnp.asarray(next(train_dataset_iter))
            params, state, opt_state, train_results = (
                self.train_step(params, state, opt_state, data))

            train_results = jax.device_get(train_results)
            train_losses.append(train_results['loss'])
            train_recon_errors.append(train_results['recon_error'])
            train_perplexities.append(train_results['vq_output']['perplexity'])
            train_vqvae_loss.append(train_results['vq_output']['loss'])

            if step % 100 == 0:
                print(f'[Step {step}/{self.cfg.num_training_updates}] ' +
                    ('train loss: %f ' % np.mean(train_losses[-100:])) +
                    ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                    ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
                    ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))
            if step % 500 == 0:
                # Put data through the model with is_training=False, so that in the case of
                # using EMA the codebook is not updated.
                train_reconstructions = self.forward.apply(params, state, rng, data, is_training=False)[0]['x_recon']
                valid_batch = jnp.asarray(next(iter(valid_dataset)))
                valid_reconstructions = self.forward.apply(params, state, rng, valid_batch,
                                                           is_training=False)[0]['x_recon']

                visualization.visualize_reconstructions(data,
                                                        train_reconstructions,
                                                        valid_batch,
                                                        valid_reconstructions,
                                                        filename=f'{run_dir}/reconstructions_{step}.png')
        jnp.save(self.cfg.save_path, params)

        train_batch = jnp.asarray(next(iter(train_dataset)))
        valid_batch = jnp.asarray(next(iter(valid_dataset)))

        # Put data through the model with is_training=False, so that in the case of
        # using EMA the codebook is not updated.
        train_reconstructions = self.forward.apply(params, state, rng, train_batch, is_training=False)[0]['x_recon']
        valid_reconstructions = self.forward.apply(params, state, rng, valid_batch, is_training=False)[0]['x_recon']

        visualization.visualize_reconstructions(train_batch,
                                                train_reconstructions,
                                                valid_batch,
                                                valid_reconstructions,
                                                filename=f'{run_dir}/reconstructions_final.png')


if __name__=='__main__':
    trainer = Trainer()
    trainer.train()