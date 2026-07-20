import math
import torch
import torch.nn as nn

from Transformer.config_template import (
    EMBEDDING_DIM,
    MAX_CONTEXT,
)


class PositionalEncoding(nn.Module):

    def __init__(self):

        super().__init__()

        position = torch.arange(
            MAX_CONTEXT,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(

            torch.arange(
                0,
                EMBEDDING_DIM,
                2,
                dtype=torch.float32
            )

            * (-math.log(10000.0) / EMBEDDING_DIM)

        )

        pe = torch.zeros(
            MAX_CONTEXT,
            EMBEDDING_DIM,
            dtype=torch.float32
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        """
        x shape:
        (batch_size, sequence_length, embedding_dim)
        """

        sequence_length = x.size(1)

        x = x + self.pe[:, :sequence_length]

        return x