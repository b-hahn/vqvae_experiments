from dataclasses import dataclass

import flax
from flax import linen as nn
from flax.core.frozen_dict import V
import jax
import jax.numpy as jnp

@dataclass
class Config:
    hidden_size: int = 128
    initializer_range: int = 0.5
    num_attn_heads: int = 3
    num_encoders: int = 1
    num_decoders: int = 1

class SelfAttention(nn.Module):
    def setup(self):
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, q, k, v, mask=None):
        # softmax of Q*K_T / sqrt(d_K) * V
        d_k = k.shape[0]  # TODO: which size to use here?
        batch_size = q.shape[0]
        tgt_len = q.shape[1]
        seq_len = k.shape[1]

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        scores = jax.lax.batch_matmul(q, k) / jnp.sqrt(d_k)
        # x = q * k.T / jnp.sqrt(d_k)
        # TODO: add masking here
        if mask is not None:
            expanded_mask = jnp.repeat(mask[:, None, :], repeats=(batch_size, tgt_len, seq_len))
            scores[expanded_mask == 0] = -float('inf')

        weights = nn.softmax(scores)
        outputs = jax.lax.batch_matmul(weights, v)
        return outputs


class MultiHeadAttention(nn.Module):
    def setup(self, d_model, num_attn_heads, dropout: float = 0.3) -> None:
        self.d_model = d_model
        self.num_heads = num_attn_heads
        self.dropout = dropout
        self.attn_output_size = self.d_model // self.num_heads

        self.attention_heads = [SelfAttention(d_model, self.attn_output_size) for _ in range(num_attn_heads)]
        self.output = nn.Dense(d_model)
        self.dtype = jnp.float32

    def __call__(self, q, k, v, mask):
        x = jnp.concatenate([attn(q, k, v, mask) for attn in self.attention_heads])
        x = self.output(x)
        return x

class EncoderLayer(nn.Module):
    def setup(self, d_model: int, num_heads: int, dropout: float = 0.3) -> None:
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        feedforward_size: int = 256
        self.feedforward = [
            nn.Dense(feedforward_size), nn.relu,
            nn.Dropout(rate=0.1),
            nn.Dense(feedforward_size),
            nn.Dropout(rate=0.1)
        ]

        self.layernorm_attn = nn.LayerNorm()
        self.layernorm_feedforward = nn.LayerNorm()

    def __call__(self, src, src_mask):
        x = src
        x = x + self.attn(q=x, k=x, v=x, mask=src_mask)
        x = self.layernorm_attn(x)
        x = x + self.feedforward(x)
        x = self.layernorm_feedforward(x)
        return x

class DecoderLayer(nn.Module):
    def setup(self, d_model: int, num_heads: int, dropout: float = 0.3) -> None:
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.masked_attn = MultiHeadAttention(d_model, num_heads, dropout)

        feedforward_size: int = 256
        self.feedforward = [
            nn.Dense(feedforward_size), nn.relu,
            nn.Dropout(rate=0.1),
            nn.Dense(feedforward_size),
            nn.Dropout(rate=0.1)
        ]

        self.layernorm_attn = nn.LayerNorm()
        self.layernorm_masked_attn = nn.LayerNorm()
        self.layernorm_feedforward = nn.LayerNorm()

    def __call__(self, tgt, enc, tgt_mask, enc_mask):
        x = tgt
        x = x + self.masked_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.layernorm_masked_attn(x)
        x = x + self.masked_attn(q=x, k=enc, v=enc, mask=enc_mask)
        x = self.layernorm_attn(x)
        x = x + self.feedforward(x)
        x = self.layernorm_feedforward(x)
        return x

class Encoder(nn.Module):
    def setup(self, d_model, num_heads, num_encoders: int) -> None:
        # create list of encoder heads
        self.encoder_layers = [EncoderLayer(d_model, num_heads) for _ in range(num_encoders)]

    def __call__(self, src, src_mask):
        output = src
        for layer in self.encoder_layers:
            output = layer(output, src_mask)
        return output

class Decoder(nn.Module):
    def setup(self, d_model: int, num_heads: int, num_decoders: int) -> None:
        # create list of encoder heads
        self.decoder_layers = [DecoderLayer(d_model, num_heads) for _ in range(num_decoders)]

    def __call__(self, tgt, enc, tgt_mask, enc_mask):
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, enc, tgt_mask, enc_mask)
        return output

class Transformer(nn.Module):
    def setup(self, config) -> None:
        self.config = config

        self.encoder = Encoder(config.num_attn_heads, config.num_encoders)
        self.decoder = Decoder(config.num_attn_heads, config.num_decoders)

    def __call__(self, src, target, src_mask, target_mask):
        # get embeddings for all inputs
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(target, enc_out, src_mask, target_mask)

        # self attention for N times in encoder

        # self + cross attention in decoder


if __name__ == '__main__':
    cfg = Config()
    tf = Transformer(cfg)


