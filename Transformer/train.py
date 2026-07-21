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

# ============================================================
# DATALOADERS
# ============================================================

train_loader, val_loader = create_dataloaders()

print(f"Training batches   : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")

# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0

    for inputs, targets, attention_mask in train_loader:

        attention_mask = attention_mask.to(
            DEVICE,
            non_blocking=True
        )

        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE,
            enabled=USE_AMP
        ):

            logits = model(inputs, attention_mask=attention_mask)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        scaler.scale(loss).backward()

        if config.GRAD_CLIP > 0:

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.GRAD_CLIP
            )

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ============================================================
# VALIDATE ONE EPOCH
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for inputs, targets, attention_mask in val_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            attention_mask = attention_mask.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type=DEVICE,
                enabled=USE_AMP
            ):

                logits = model(
                    inputs,
                    attention_mask=attention_mask
                )

                loss = criterion(
                    logits.view(
                        -1,
                        logits.size(-1)
                    ),
                    targets.view(-1)
                )

            total_loss += loss.item()

    return total_loss / len(val_loader)


best_val_loss = float("inf")

for epoch in range(config.NUM_EPOCHS):

    train_loss = train_epoch()

    val_loss = validate()

    print(
        f"Epoch {epoch + 1}/{config.NUM_EPOCHS}"
        f" | Train Loss: {train_loss:.4f}"
        f" | Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )

        print(
            f"✓ Validation improved. Model saved."
            f" (Best Val Loss: {best_val_loss:.4f})"
        )