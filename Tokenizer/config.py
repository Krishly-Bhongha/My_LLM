from pathlib import Path
import os


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "Data"

RAW_DIR = DATA_DIR / "raw"

PREPROCESSED_DIR = DATA_DIR / "preprocessed"

TOKENIZER_DATA_DIR = DATA_DIR / "tokenizer"


# ============================================================
# DATASET
# ============================================================

PREPROCESSED_FILE = PREPROCESSED_DIR / "Tiny_Stories_clean.txt"


# Name without .txt
DATASET_NAME = PREPROCESSED_FILE.stem


# All tokenizer output for this dataset goes here.
DATASET_DIR = TOKENIZER_DATA_DIR / DATASET_NAME


# ============================================================
# TOKENIZER OUTPUT
# ============================================================

CORPUS_FILE = DATASET_DIR / "corpus.npy"

OFFSETS_FILE = DATASET_DIR / "offsets.npy"

VOCAB_FILE = DATASET_DIR / "vocab.json"

MERGES_FILE = DATASET_DIR / "merges.json"

METADATA_FILE = DATASET_DIR / "metadata.json"


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"

BOS_TOKEN = "<BOS>"

EOS_TOKEN = "<EOS>"

UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
]


# ============================================================
# INITIAL VOCABULARY
# ============================================================

MIN_CHAR_FREQ = 20

MAX_VOCAB_SIZE = 100000


# ============================================================
# BPE
# ============================================================

NUM_MERGES = 100000

MIN_PAIR_FREQ = 100

MIN_PAIR_FREQ_LOAD = 100


DO_NOT_MERGE = {
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    " ",
}


# ============================================================
# CHECKPOINTING
# ============================================================

CHECKPOINT_INTERVAL = 10000

HEAP_FILE = DATASET_DIR / "heap.npy"

MERGE_STATE_FILE = DATASET_DIR / "merge_state.json"

# ============================================================
# CPU
# ============================================================

CPU_WORKERS = max(
    1,
    (os.cpu_count() or 2) - 1
)


# ============================================================
# NUMPY
# ============================================================

TOKEN_DTYPE = "int32"

OFFSET_DTYPE = "int64"


# ============================================================
# PAIR ENCODING
# ============================================================

PAIR_BASE = 100000
