from collections import Counter
from multiprocessing import Pool

from config import (
    PREPROCESSED_FILE,
    VOCAB_FILE,
    SPECIAL_TOKENS,
    MIN_CHAR_FREQ,
    MAX_VOCAB_SIZE,
    CPU_WORKERS,
)

from utils import save_json


# ============================================================
# MULTIPROCESSING WORKER
# ============================================================

def count_characters(lines):
    """
    Count characters in a batch of stories.

    Each worker creates its own Counter.
    The counters are combined by the main process.
    """

    counter = Counter()

    for line in lines:
        counter.update(line)

    return counter


# ============================================================
# BATCH GENERATOR
# ============================================================

def generate_batches(file_path, batch_size):
    """
    Read the preprocessed file incrementally and yield batches
    of stories.

    The entire corpus is NOT loaded into RAM at once.
    """

    batch = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.rstrip("\n")

            if not line:
                continue

            batch.append(line)

            if len(batch) >= batch_size:

                yield batch

                batch = []

        # Remaining stories
        if batch:
            yield batch


# ============================================================
# BUILD VOCABULARY
# ============================================================

def build_vocab():

    """
    Build the initial character-level vocabulary.

    The initial vocabulary consists of:
        1. Special tokens
        2. Characters occurring at least MIN_CHAR_FREQ times

    Character counting uses multiprocessing.

    BPE-generated tokens are added later during training.
    """

    print("=" * 60)
    print("BUILDING INITIAL VOCABULARY")
    print("=" * 60)

    # --------------------------------------------------------
    # WORKER COUNT
    # --------------------------------------------------------

    workers = max(
        1,
        CPU_WORKERS,
    )

    print(
        f"\nCPU workers: {workers}"
    )

    # --------------------------------------------------------
    # COUNT CHARACTERS
    # --------------------------------------------------------

    print("\nReading:")
    print(
        f"  {PREPROCESSED_FILE}"
    )

    char_freq = Counter()

    # Number of stories sent to each worker at once.
    #
    # A larger batch reduces multiprocessing overhead.
    # Since each story is independent, there is no loss of data.
    batch_size = max(
        100,
        1000 // workers,
    )

    print(
        f"Batch size: {batch_size} stories"
    )

    # --------------------------------------------------------
    # MULTIPROCESSING
    # --------------------------------------------------------

    with Pool(
        processes=workers,
    ) as pool:

        batch_iterator = generate_batches(
            PREPROCESSED_FILE,
            batch_size,
        )

        for local_counter in pool.imap_unordered(
            count_characters,
            batch_iterator,
            chunksize=1,
        ):

            # Combine this worker's exact counts
            # into the global Counter.
            char_freq.update(
                local_counter
            )

    print(
        f"Unique characters found: "
        f"{len(char_freq):,}"
    )

    # --------------------------------------------------------
    # BUILD VOCABULARY
    # --------------------------------------------------------

    vocab = {}

    next_id = 0

    # Special tokens always come first.
    for token in SPECIAL_TOKENS:

        vocab[token] = next_id

        next_id += 1

    # --------------------------------------------------------
    # ADD CHARACTERS
    # --------------------------------------------------------

    # Sort characters for deterministic vocabulary IDs.
    #
    # This means running the vocabulary builder again on
    # exactly the same data produces exactly the same IDs.

    for char in sorted(
        char_freq.keys()
    ):

        frequency = char_freq[char]

        if frequency < MIN_CHAR_FREQ:
            continue

        # Don't exceed vocabulary limit.
        if next_id >= MAX_VOCAB_SIZE:
            break

        # Avoid accidentally colliding with a special token.
        if char in vocab:
            continue

        vocab[char] = next_id

        next_id += 1

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_json(
        vocab,
        VOCAB_FILE,
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    characters_kept = (
        len(vocab)
        - len(SPECIAL_TOKENS)
    )

    print()

    print(
        f"Characters kept       : "
        f"{characters_kept:,}"
    )

    print(
        f"Special tokens        : "
        f"{len(SPECIAL_TOKENS)}"
    )

    print(
        f"Initial vocabulary    : "
        f"{len(vocab):,}"
    )

    print(
        f"Minimum char frequency: "
        f"{MIN_CHAR_FREQ}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        f"  {VOCAB_FILE}"
    )

    print(
        "\nVocabulary complete."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_vocab()