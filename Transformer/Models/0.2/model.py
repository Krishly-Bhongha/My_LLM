import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_LAYERS,
)

from block import TransformerBlock
from pos_encode import PositionalEncoding


class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM
        )

        self.position = PositionalEncoding()

        self.blocks = nn.ModuleList([
            TransformerBlock()
            for _ in range(NUM_LAYERS)
        ])

        self.final_norm = nn.LayerNorm(
            EMBEDDING_DIM
        )

        self.output = nn.Linear(
            EMBEDDING_DIM,
            VOCAB_SIZE,
            bias=False
        )

    def forward(self, x):

        pass


def initialize_parameters():

    model = TransformerModel()

    torch.save(
        model.state_dict(),
        ROOT / "model.pt"
    )

    print("Model parameters initialized successfully.")


if __name__ == "__main__":

    initialize_parameters()
