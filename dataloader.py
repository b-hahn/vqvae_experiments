import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

def get_cifar_dataset(split: str) -> jnp.ndarray:
    cifar10 = tfds.as_numpy(tfds.load("cifar10", split="train+test", batch_size=-1))
    del cifar10["id"], cifar10["label"]
    jax.tree_map(lambda x: f'{x.dtype.name}{list(x.shape)}', cifar10)
    if split == 'train':
        return jax.tree_map(lambda x: x[:40000], cifar10)
    if split == 'val':
            return jax.tree_map(lambda x: x[40000:50000], cifar10)
    if split == 'test':
            return jax.tree_map(lambda x: x[50000:], cifar10)


def get_swisstopo_dataset() -> jnp.ndarray:
    dataset = tf.data.Dataset.list_files("tiles/swiss-map-raster25_2015_1328_krel_1.25_2056/*.png")
    dataset_it = dataset.as_numpy_iterator()
    jax.tree_map(lambda x: f'{x.dtype.name}{list(x.shape)}', dataset_ut)


if __name__ == '__main__':
    ds_cifar = get_cifar_dataset(split='test')
    ds_swisstopo = get_swisstopo_dataset()

