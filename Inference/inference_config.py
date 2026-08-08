from pathlib import Path

# -------------------------
# Paths
# -------------------------

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

MODELS_DIR = PROJECT_ROOT / "Transformer" / "Models"
DATA_DIR = PROJECT_ROOT / "Data"

VOCAB_FILE = DATA_DIR / "vocab.json"
MERGES_FILE = DATA_DIR / "merges.json"

# -------------------------
# Device
# -------------------------

DEVICE = "cuda"

# -------------------------
# Default Model
# -------------------------

DEFAULT_MODEL_VERSION = "2.0"

MODEL_FILENAME = "model.pt"
CONFIG_FILENAME = "config.py"

# -------------------------
# Generation
# -------------------------

MAX_NEW_TOKENS = 200

TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9

# -------------------------
# Conversation
# -------------------------

EXIT_COMMANDS = {
    "exit",
    "quit",
    "bye"
}

# -------------------------
# Special Tokens
# -------------------------

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"