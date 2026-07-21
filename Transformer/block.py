import torch.nn as nn

from config import (
    EMBEDDING_DIM,
    DROPOUT,
    FFN_MULTIPLIER,
)

from attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        # Multi-Head Self Attention
        self.attention = MultiHeadAttention()

        # Layer Normalization
        self.norm1 = nn.LayerNorm(
            EMBEDDING_DIM
        )

        self.norm2 = nn.LayerNorm(
            EMBEDDING_DIM
        )

        # Feed Forward Network
        self.feed_forward = nn.Sequential(

            nn.Linear(
                EMBEDDING_DIM,
                FFN_MULTIPLIER * EMBEDDING_DIM
            ),

            nn.GELU(),

            nn.Linear(
                FFN_MULTIPLIER * EMBEDDING_DIM,
                EMBEDDING_DIM
            ),

            nn.Dropout(
                DROPOUT
            )
        )


    def forward(self, x):
        # -------------------------------
        # Multi-Head Attention
        # -------------------------------

        x = x + self.attention(
            self.norm1(x)
        )
        # -------------------------------
        # Feed Forward Network
        # -------------------------------

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x