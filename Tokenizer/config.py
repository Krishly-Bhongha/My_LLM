from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "Data"

CHAT_FILE = DATA_DIR / "cleaned_chat.txt"

VOCAB_FILE = DATA_DIR / "vocab.json"

TOKEN_FREQ_FILE = DATA_DIR / "token_freq.json"

MERGES_FILE = DATA_DIR / "merges.json"

MIN_CHAR_FREQ = 20

SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
]

MERGES_PER_RUN = 100

MIN_PAIR_FREQ = 10

MAX_VOCAB = 100000      # practically unlimited

DO_NOT_MERGE = SPECIAL_TOKENS + [" "]

TOKENIZED_FILE = DATA_DIR / "tokenized_data.json"