from pathlib import Path
import torch
import torch.nn as nn

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "Models"


# ============================================================
# SELECT MODEL VERSION
# ============================================================

while True:

    VERSION = input(
        "Enter model version: "
    ).strip()

    MODEL_DIR = (
        MODELS_DIR
        / VERSION
    )

    if MODEL_DIR.is_dir():

        break

    print(
        f"Model version '{VERSION}' does not exist."
    )

    print(
        "Please enter a valid version.\n"
    )


import sys
import importlib
# ============================================================
# IMPORT VERSION-SPECIFIC MODULES
# ============================================================

# Highest priority:
# Models/<version>/config.py
# Models/<version>/model.py

sys.path.insert(
    0,
    str(MODEL_DIR)
)

# Second priority:
# Global Transformer files

sys.path.insert(
    1,
    str(ROOT)
)


# Prevent Python from using an old cached config/model
sys.modules.pop("config", None)
sys.modules.pop("model", None)


config = importlib.import_module(
    "config"
)

model_module = importlib.import_module(
    "model"
)

from model import TransformerModel

from loader import (
    create_dataloaders,
    PAD_ID,
)
# ============================================================
# CONFIG VARIABLES
# ============================================================

DEVICE = config.DEVICE

LEARNING_RATE = config.LEARNING_RATE

WEIGHT_DECAY = config.WEIGHT_DECAY

USE_AMP = config.USE_AMP

USE_COMPILE = config.USE_COMPILE

# ============================================================
# GPU SETTINGS
# ============================================================

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision(
        "high"
    )

# ============================================================
# MODEL
# ============================================================
MODEL_FILE = MODEL_DIR / "model.pt"

model = TransformerModel()

model.load_state_dict(

    torch.load(

        MODEL_FILE,

        map_location="cpu",

        weights_only=True
    )
)

model.to(DEVICE)

if USE_COMPILE:

    model = torch.compile(model)

# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    ignore_index=PAD_ID
)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)

# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(

    "cuda",

    enabled=(
        USE_AMP
        and DEVICE == "cuda"
    )
)