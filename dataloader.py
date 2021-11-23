from pathlib import Path

import jax
import jax.numpy as jnp
import tensorflow as tf
from tensorflow._api.v2 import data
import tensorflow_datasets as tfds

def get_cifar_dataset(split: str) -> jnp.ndarray:
    cifar10 = tfds.as_numpy(tfds.load("cifar10", split="train+test", batch_size=-1))
    del cifar10["id"], cifar10["label"]
    # jax.tree_map(lambda x: f'{x.dtype.name}{list(x.shape)}', cifar10)
    if split == 'train':
        return jax.tree_map(lambda x: x[:40000], cifar10)
    if split == 'val':
            return jax.tree_map(lambda x: x[40000:50000], cifar10)
    if split == 'test':
            return jax.tree_map(lambda x: x[50000:], cifar10)


def get_swisstopo_dataset(split: str, img_w: int, img_h: int) -> jnp.ndarray:
    # dataset = tf.data.Dataset.list_files("tiles/swiss-map-raster25_2015_1328_krel_1.25_2056/*.png")
    # dataset_it = dataset.as_numpy_iterator()
    # jax.tree_map(lambda x: f'{x.dtype.name}{list(x.shape)}', dataset_it)
    # return jax.tree_map(lambda x: x[:100], dataset_it)
    data_dir = Path("/home/ben/vqvae_experiments/tiles/xs/")
    batch_size = 32

    if split == 'train':
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            labels=None,
            image_size=(img_w, img_h),
            batch_size=batch_size)
    elif split == 'val':
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            labels=None,
            image_size=(img_w, img_h),
            batch_size=batch_size)

    return dataset


if __name__ == '__main__':
    # ds_cifar = get_cifar_dataset(split='test')
    ds_swisstopo = get_swisstopo_dataset()

