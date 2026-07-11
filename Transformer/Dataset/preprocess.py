import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATA = ROOT / "Data"

DATASET = ROOT / "Transformer" / "Dataset"

VOCAB_FILE = DATA / "vocab.json"
TOKENIZED_FILE = DATA / "tokenized_data.json"

OUTPUT_FILE = DATASET / "train_ids.npy"
META_FILE = DATASET / "metadata.json"


def preprocess():

    with open(VOCAB_FILE, encoding="utf8") as f:
        vocab = json.load(f)

    with open(TOKENIZED_FILE, encoding="utf8") as f:
        corpus = json.load(f)

    ids = []

    for sentence in corpus:

        for token in sentence:

            ids.append(vocab[token])

    ids = np.array(ids, dtype=np.uint16)

    np.save(OUTPUT_FILE, ids)

    metadata = {
        "num_tokens": int(len(ids)),
        "vocab_size": len(vocab),
        "dtype": "uint16"
    }

    with open(META_FILE, "w", encoding="utf8") as f:
        json.dump(metadata, f, indent=4)

    print("Saved train_ids.npy")
    print(f"Total Tokens : {len(ids)}")
    print(f"Vocabulary   : {len(vocab)}")


if __name__ == "__main__":
    preprocess()