from pathlib import Path
import json

import torch

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = Path("Transformer") / "Dataset"

METADATA_FILE = DATASET_DIR / "metadata.json"

TRAIN_IDS_FILE = DATASET_DIR / "train_ids.pt"

# ============================================================
# DATASET METADATA
# ============================================================

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    METADATA = json.load(file)

VOCAB_SIZE = METADATA["vocab_size"]
NUM_TOKENS = METADATA["num_tokens"]
DTYPE = METADATA["dtype"]

PAD_ID = METADATA["pad_id"]
BOS_ID = METADATA["bos_id"]
EOS_ID = METADATA["eos_id"]

# ============================================================
# DATA CONFIGURATION
# ============================================================

# Number of sequences processed together
BATCH_SIZE = 32

# Minimum target context length.
#
# Once this many input tokens have been collected,
# the loader continues until the next <EOS>.
#
# Therefore, actual sequence length may be greater
# than BLOCK_SIZE.
BLOCK_SIZE = 128

# Fraction of sequences used for training
TRAIN_SPLIT = 0.9

# Shuffle training sequences every epoch
SHUFFLE = True

# Useful for reproducible train/validation split
RANDOM_SEED = 42

# Keep 0 on Windows initially.
# Can increase later if data loading becomes a bottleneck.
NUM_WORKERS = 0

# Whether to discard the final incomplete batch
DROP_LAST = False

# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_DIM = 256

NUM_HEADS = 4

NUM_LAYERS = 2

DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CONTEXT = 1024

FFN_MULTIPLIER = 4

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

NUM_EPOCHS = 500

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

GRAD_CLIP = 1.0

USE_AMP = True

USE_COMPILE = False

SAVE_EVERY = 1

LOG_EVERY = 100

PIN_MEMORY = True

PERSISTENT_WORKERS = True

PREFETCH_FACTOR = 4