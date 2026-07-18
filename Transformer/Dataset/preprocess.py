import json
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATA = ROOT / "Data"
DATASET = ROOT / "Transformer" / "Dataset"

VOCAB_FILE = DATA / "vocab.json"
TOKENIZED_FILE = DATA / "tokenized_data.json"

OUTPUT_FILE = DATASET / "train_ids.pt"
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

    # Store directly as a PyTorch tensor
    ids = torch.tensor(ids, dtype=torch.long)
    
    torch.save(ids, OUTPUT_FILE)

    metadata = {
        "num_tokens": int(len(ids)),
        "vocab_size": len(vocab),
        "dtype": "long"
    }

    with open(META_FILE, "w", encoding="utf8") as f:
        json.dump(metadata, f, indent=4)

    print("Saved train_ids.pt")
    print(f"Total Tokens : {len(ids)}")
    print(f"Vocabulary   : {len(vocab)}")


if __name__ == "__main__":
    preprocess()