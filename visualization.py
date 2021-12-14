import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np


def convert_batch_to_image_grid(image_batch, img_w: int, img_h: int):
    reshaped = (image_batch.reshape(4, 8, img_w, img_h, 3).reshape(4 * img_w, 8 * img_h, 3))
    reshaped = (image_batch.reshape(4, 8, img_w, img_h, 3).transpose([0, 2, 1, 3, 4]).reshape(4 * img_w, 8 * img_h, 3))
    return reshaped + 0.5

def visualize_reconstructions(train_batch, train_reconstructions, valid_batch, valid_reconstructions, filename: str) -> None:
    def add_grid():
        loc = plticker.MultipleLocator(base=32)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.grid(which='major', axis='both', linestyle='-')

    train_batch = train_batch['image']
    # img_w, img_h = train_batch['image'].shape[1], train_batch['image'].shape[2]
    img_w, img_h = train_batch.shape[1], train_batch.shape[2]
    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(2, 2, 1)
    ax.imshow(convert_batch_to_image_grid(train_batch, img_w, img_h), interpolation='nearest')
    add_grid()
    ax.set_title('training data originals')
    # plt.axis('off')

    ax = f.add_subplot(2, 2, 2)
    ax.imshow(convert_batch_to_image_grid(train_reconstructions, img_w, img_h), interpolation='nearest')
    add_grid()
    ax.set_title('training data reconstructions')
    # plt.axis('off')

    if not valid_batch is None and not valid_reconstructions is None:
        ax = f.add_subplot(2, 2, 3)
        ax.imshow(convert_batch_to_image_grid(valid_batch, img_w, img_h), interpolation='nearest')
        add_grid()
        ax.set_title('validation data originals')
        # plt.axis('off')

        ax = f.add_subplot(2, 2, 4)
        ax.imshow(convert_batch_to_image_grid(valid_reconstructions, img_w, img_h), interpolation='nearest')
        add_grid()
        ax.set_title('validation data reconstructions')
        # plt.axis('off')
    plt.savefig(filename)

    # train_reconstructions = forward.apply(params, state, rng, train_batch, is_training=False)[0]['x_recon']
    # from pathlib import Path
    # import pickle
    # save_path = Path('/content/drive/MyDrive/Colab Notebooks/vqvae/') / 'weights'
    # with open(save_path, 'w') as f:
    #   pickle.dump(params, f)


def visualize_samples(samples, filename='samples.png'):
    def add_grid():
        loc = plticker.MultipleLocator(base=32)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.grid(which='major', axis='both', linestyle='-')

    img_w, img_h = 128, 128
    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(convert_batch_to_image_grid(samples, img_w, img_h), interpolation='nearest')
    add_grid()
    ax.set_title('samples')
    plt.savefig(filename)
