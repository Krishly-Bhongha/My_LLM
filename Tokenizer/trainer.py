import json

from config import *
from utils import save_json
from merge import count_pair_frequency, best_pair, merge_pair


def load_json(filepath):
    with open(filepath, "r", encoding="utf8") as f:
        return json.load(f)


def train_one_merge():

    corpus = load_json(TOKENIZED_FILE)
    vocab = load_json(VOCAB_FILE)

    pair_freq = count_pair_frequency(corpus)

    pair, freq = best_pair(pair_freq, MIN_PAIR_FREQ)

    if pair is None:
        print("No pair found with sufficient frequency.")
        return False

    print(f"Frequency: {freq} | Merging: {pair[0]} + {pair[1]}")

    corpus, new_token = merge_pair(corpus, pair)

    save_json(corpus, TOKENIZED_FILE)

    if new_token not in vocab:
        vocab[new_token] = len(vocab)
        save_json(vocab, VOCAB_FILE)

    return True
if __name__ == "__main__":
    train_one_merge()