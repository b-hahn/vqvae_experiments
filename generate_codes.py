# for training data, generate codes. Just pickle them

import datetime
import itertools
from pathlib import Path
import pickle
from typing import Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tqdm

from config import Config
import dataloader
from vqvae import VQVAEModel, Encoder, Decoder

def cast_and_normalise_images(data_dict: Dict):
    """Convert images to floating point with the range [-0.5, 0.5]"""
    data_dict['image'] = (tf.cast(data_dict['image'], tf.float32) / 255.0) - 0.5
    return data_dict

def load(path: Path) -> hk.Params:
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def extract_codes():
    seed = 63
    key = jax.random.PRNGKey(seed)
    tf.random.set_seed(seed)
    cfg = Config()
    split = 'val'

    # load params
    weights_file = Path('/home/ben/vqvae_experiments/run_2021-12-02-15-22-23/weights.pkl_100000.pkl')
    weights_ts = weights_file.parent.name[4:]
    params = load(weights_file)

    # 4. Put parameters back onto the device.
    params = jax.device_put(params)

    forward_fn = hk.transform_with_state(forward)

    train_data_dict = dataloader.get_cifar_dataset(split=split)
    train_data_dict['idx'] = [i for i in range(train_data_dict['image'].shape[0])]
    train_dataset = tfds.as_numpy(
        tf.data.Dataset.from_tensor_slices(train_data_dict)
        .map(cast_and_normalise_images)
        .shuffle(10000)
        .repeat(1)
        .batch(cfg.batch_size, drop_remainder=False)
        .prefetch(-1))
    train_data_variance = np.var(train_data_dict['image'] / 255.0)
    # train_dataset_iter = iter(train_dataset)
    key, subkey = jax.random.split(key)

    #
    sample_train_batch, train_dataset_iter = itertools.tee(train_dataset, 2)
    train_batch = next(sample_train_batch)
    _, state = forward_fn.init(subkey, train_batch['image'], is_training=False, cfg=cfg)
    key, subkey = jax.random.split(key)
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    encodings_dir = Path(f"encodings_{weights_ts}")
    encodings_dir.mkdir(exist_ok=True)

    # save image+encoding
    encodings_numpy = []
    for step, train_batch in enumerate(train_dataset_iter):
        # train_batch = next(train_dataset_iter)
        model_output, state = forward_fn.apply(params, state, subkey, train_batch['image'], is_training=False, cfg=cfg)
        encodings = model_output['encodings'].reshape(train_batch['image'].shape[0], 64, 512)

        # save 1 encoding per image
        # for i in range(train_batch['image'].shape[0]):
        #     encodings_numpy.append()
            # np.save(encodings_dir / f"{train_batch['idx'][i]}", encodings[i, :, :])
        encodings_numpy.append(encodings)
        print(f"Processing {step}...", end="\r", flush=True)
    np.save(encodings_dir / f"encodings_{split}", np.concatenate(encodings_numpy))
    print(f'{np.concatenate(encodings_numpy).shape=}')
    # params, state = forward_fn.init(key, jax.random.normal(subkey, shape=(1,32,32,3)), is_training=False, cfg=cfg)
    # train_reconstructions = forward_fn.apply(params, state, key, None, is_training=False, cfg=cfg)[0]

def forward(batch, is_training, cfg):
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
    return model.get_code(batch)#, key=hk.next_rng_key())

if __name__ == '__main__':
    extract_codes()