import pickle

import haiku as hk
import jax
import jax.numpy as jnp

from config import Config
from vqvae import VQVAEModel, Encoder, Decoder
import visualization

def load(path) -> hk.Params:
  with open(path, 'rb') as fp:
    return pickle.load(fp)

def generate_samples():
    seed = 63
    key = jax.random.PRNGKey(seed)

    cfg = Config()

    # load params
    params = load('/home/ben/vqvae_experiments/run_2021-11-24-10-57-00/weights.pkl_8000.pkl')

    # 4. Put parameters back onto the device.
    params = jax.device_put(params)

    # 5. Use them as you like.
    # f.apply(params, None, x)

    forward_fn = hk.transform_with_state(forward)
    key, subkey = jax.random.split(key)
    _, state = forward_fn.init(key, jax.random.normal(subkey, shape=(1,32,32,3)), is_training=False, cfg=cfg)
    # params, state = forward_fn.init(key, jax.random.normal(subkey, shape=(1,32,32,3)), is_training=False, cfg=cfg)
    train_reconstructions = forward_fn.apply(params, state, key, None, is_training=False, cfg=cfg)[0]
    visualization.visualize_samples(train_reconstructions)
    print('done')


# @jax.jit
# def sample(self, params, state, data):
#     model_output, state = self.forward.apply(params, state, None, data, is_training=False)
#     return state, model_output

def forward(data, is_training, cfg):
    encoder = Encoder(cfg.num_hiddens, cfg.num_residual_layers, cfg.num_residual_hiddens)
    decoder = Decoder(cfg.num_hiddens, cfg.num_residual_layers, cfg.num_residual_hiddens)
    pre_vq_conv1 = hk.Conv2D(output_channels=cfg.embedding_dim, kernel_shape=(1, 1), stride=(1, 1), name="to_vq")

    if cfg.vq_use_ema:
        vq_vae = hk.nets.VectorQuantizerEMA(
            embedding_dim=cfg.embedding_dim,
            num_embeddings=cfg.num_embeddings,
            commitment_cost=cfg.commitment_cost,
            decay=cfg.decay)
    else:
        vq_vae = hk.nets.VectorQuantizer(
            embedding_dim=cfg.embedding_dim,
            num_embeddings=cfg.num_embeddings,
            commitment_cost=cfg.commitment_cost)
    train_data_variance = 0.01 # TODO: change
    model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                    data_variance=train_data_variance)

    # return model(data['image'], is_training)
    return model.sample(key=hk.next_rng_key())

if __name__ == '__main__':
    generate_samples()