import torch.nn as nn

from Transformer.config_template import (
    EMBEDDING_DIM,
    NUM_HEADS,
    DROPOUT,
)

class MultiHeadAttention(nn.Module):

    def __init__(self):

        super().__init__()

        assert EMBEDDING_DIM % NUM_HEADS == 0, \
            "EMBEDDING_DIM must be divisible by NUM_HEADS"

        self.embedding_dim = EMBEDDING_DIM
        self.num_heads = NUM_HEADS
        self.head_dim = EMBEDDING_DIM // NUM_HEADS

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

        self.dropout = nn.Dropout(DROPOUT)