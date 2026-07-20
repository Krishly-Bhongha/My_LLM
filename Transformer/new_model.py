import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "Models"

CONFIG_FILE = ROOT / "config_template.py"

MODEL_TEMPLATE = """\
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

    def forward(self, x):

        pass


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
"""

def initialize_model():

    while True:

        version = input("Enter model version: ").strip()

        if version == "":
            print("Version cannot be empty.\n")
            continue

        model_dir = MODELS_DIR / version

        if model_dir.exists():
            print("Version already exists.\n")
            continue

        break

    model_dir.mkdir(parents=True)

    shutil.copy2(
        CONFIG_FILE,
        model_dir / "config.py"
    )

    with open(
        model_dir / "model.py",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(MODEL_TEMPLATE)

    print(f"\nCreated model version {version}")


if __name__ == "__main__":

    initialize_model()