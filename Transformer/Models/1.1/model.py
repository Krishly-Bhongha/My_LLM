import sys
from pathlib import Path

import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent
TRANSFORMER_DIR = MODEL_DIR.parent.parent

# Version-specific config.py
sys.path.insert(0, str(MODEL_DIR))

# Shared Transformer modules
sys.path.insert(1, str(TRANSFORMER_DIR))

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

    def forward(self, x, attention_mask=None):

        # --------------------------------------------
        # Token Embeddings
        # (B, S) -> (B, S, E)
        # --------------------------------------------

        x = self.embedding(x)

        # --------------------------------------------
        # Add Positional Encoding
        # --------------------------------------------

        x = self.position(x)

        # --------------------------------------------
        # Transformer Blocks
        # --------------------------------------------

        for block in self.blocks:

            x = block(
            x,
            attention_mask
        )

        # --------------------------------------------
        # Final Layer Normalization
        # --------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------
        # Vocabulary Projection
        # (B, S, E) -> (B, S, V)
        # --------------------------------------------

        logits = self.output(x)

        return logits


def initialize_parameters():

    model = TransformerModel()

    MODEL_FILE = MODEL_DIR / "model.pt"

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    print("Model parameters initialized successfully.")


if __name__ == "__main__":

    initialize_parameters()
