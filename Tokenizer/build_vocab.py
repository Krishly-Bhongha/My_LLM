from collections import Counter

from preprocess import extract_message
from utils import save_json
from config import *


def build_vocab():

    char_freq = Counter()

    # First pass: count character frequencies
    with open(CHAT_FILE, encoding="utf8") as f:

        for line in f:

            message = extract_message(line)

            if not message:
                continue

            char_freq.update(message)

    vocab = {}

    next_id = 0

    # Add special tokens first
    for token in SPECIAL_TOKENS:
        vocab[token] = next_id
        next_id += 1

    # Add characters appearing >= MIN_CHAR_FREQ
    for ch in sorted(char_freq.keys()):

        if char_freq[ch] >= MIN_CHAR_FREQ:

            vocab[ch] = next_id
            next_id += 1

    # Save vocab and frequencies
    save_json(vocab, VOCAB_FILE)
    save_json(dict(char_freq), TOKEN_FREQ_FILE)

    # Second pass: create initial tokenized corpus
    corpus = []

    with open(CHAT_FILE, encoding="utf8") as f:

        for line in f:

            message = extract_message(line)

            if not message:
                continue

            sentence = ["<BOS>"]

            for ch in message:

                # Ignore characters not present in the vocabulary
                if ch in vocab:
                    sentence.append(ch)

            sentence.append("<EOS>")

            corpus.append(sentence)

    save_json(corpus, TOKENIZED_FILE)

    print(f"Unique characters found : {len(char_freq)}")
    print(f"Characters kept         : {len(vocab) - len(SPECIAL_TOKENS)}")
    print(f"Final vocabulary size   : {len(vocab)}")
    print(f"Sentences tokenized     : {len(corpus)}")


if __name__ == "__main__":
    build_vocab()