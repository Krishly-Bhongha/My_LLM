from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "Data"

CHAT_FILE = DATA_DIR / "cleaned_chat.txt"

VOCAB_FILE = DATA_DIR / "vocab.json"

TOKEN_FREQ_FILE = DATA_DIR / "token_freq.json"

MIN_CHAR_FREQ = 20

SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
]