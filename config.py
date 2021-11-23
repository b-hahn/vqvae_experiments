from dataclasses import dataclass

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

    save_path = "weights"

