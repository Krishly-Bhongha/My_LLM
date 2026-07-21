import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    EMBEDDING_DIM,
    NUM_HEADS,
    DROPOUT,
    MAX_CONTEXT,
)


class MultiHeadAttention(nn.Module):

    def __init__(self):

        super().__init__()

        assert EMBEDDING_DIM % NUM_HEADS == 0, \
            "EMBEDDING_DIM must be divisible by NUM_HEADS"

        self.embedding_dim = EMBEDDING_DIM
        self.num_heads = NUM_HEADS
        self.head_dim = EMBEDDING_DIM // NUM_HEADS

        # Precompute scaling factor
        self.scale = self.head_dim ** -0.5

        self.Wq = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
            bias=True
        )

        self.Wk = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
            bias=True
        )

        self.Wv = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
            bias=True
        )

        self.Wo = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
            bias=True
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        # Upper triangular causal mask
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    MAX_CONTEXT,
                    MAX_CONTEXT,
                    dtype=torch.bool
                ),
                diagonal=1
            )
        )

    def forward(self, x, attention_mask=None):

        batch_size, sequence_length, _ = x.shape

        # Project to Query, Key and Value

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Split into heads

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores.mul_(self.scale)

        scores.masked_fill_(
            self.mask[
                :sequence_length,
                :sequence_length
            ],
            float("-inf")
        )
        
        if attention_mask is not None:

            scores.masked_fill_(
                ~attention_mask[:, None, None, :],
                float("-inf")
            )

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(
            attention
        )

        output = torch.matmul(
            attention,
            V
        )

        # Merge heads

        output = output.transpose(
            1,
            2
        ).contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        return self.Wo(output)