import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

def get_dataset(split: str) -> jnp.ndarray:
    cifar10 = tfds.as_numpy(tfds.load("cifar10", split="train+test", batch_size=-1))
    del cifar10["id"], cifar10["label"]
    jax.tree_map(lambda x: f'{x.dtype.name}{list(x.shape)}', cifar10)
    if split == 'train':
        return jax.tree_map(lambda x: x[:40000], cifar10)
    if split == 'val':
            return jax.tree_map(lambda x: x[40000:50000], cifar10)
    if split == 'test':
            return jax.tree_map(lambda x: x[50000:], cifar10)


if __name__ == '__main__':
    ds = get_dataset(split='test')

