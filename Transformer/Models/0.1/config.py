from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "Data"

TRANSFORMER_DIR = ROOT / "Transformer"

DATASET_DIR = TRANSFORMER_DIR / "Dataset"

VOCAB_FILE = DATA_DIR / "vocab.json"

TRAIN_IDS_FILE = DATASET_DIR / "train_ids.pt"


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
NUM_WORKERS = 2

# Whether to discard the final incomplete batch
DROP_LAST = False

# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_DIM = 128

NUM_HEADS = 1

NUM_LAYERS = 2

DROPOUT = 0.1

DEVICE = "cuda"

MAX_CONTEXT = 256

FFN_MULTIPLIER = 2