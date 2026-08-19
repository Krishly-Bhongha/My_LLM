import json
from pathlib import Path

import numpy as np


# ============================================================
# JSON
# ============================================================

def save_json(data, filepath):
    """
    Save Python data as compact UTF-8 JSON.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            separators=(",", ":")
        )


def load_json(filepath):
    """
    Load JSON from a UTF-8 file.
    """

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# NUMPY
# ============================================================

def save_numpy(array, filepath):
    """
    Save a NumPy array.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        filepath,
        array
    )


def load_numpy(filepath):
    """
    Load a NumPy array without allowing Python objects.
    """

    return np.load(
        filepath,
        allow_pickle=False
    )


# ============================================================
# CORPUS
# ============================================================

def save_corpus(
    corpus,
    offsets,
    corpus_file,
    offsets_file
):
    """
    Save the flat integer corpus and its story offsets.
    """

    save_numpy(
        corpus,
        corpus_file
    )

    save_numpy(
        offsets,
        offsets_file
    )


def load_corpus(
    corpus_file,
    offsets_file
):
    """
    Load the flat integer corpus and story offsets.
    """

    corpus = load_numpy(
        corpus_file
    )

    offsets = load_numpy(
        offsets_file
    )

    return corpus, offsets


# ============================================================
# VALIDATION
# ============================================================

def validate_vocab(vocab):
    """
    Make sure vocabulary IDs are unique and contiguous.

    Example:

        {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "a": 3
        }

    is valid.

    Returns:
        True if valid.
    """

    ids = list(vocab.values())

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Vocabulary contains duplicate token IDs."
        )

    expected_ids = set(
        range(len(vocab))
    )

    actual_ids = set(
        int(token_id)
        for token_id in ids
    )

    if actual_ids != expected_ids:

        raise ValueError(
            "Vocabulary IDs must be contiguous "
            "starting from 0."
        )

    return True


def validate_corpus(
    corpus,
    offsets
):
    """
    Validate the flat corpus and story offsets.
    """

    if corpus.ndim != 1:

        raise ValueError(
            "Corpus must be a 1-dimensional array."
        )

    if offsets.ndim != 1:

        raise ValueError(
            "Offsets must be a 1-dimensional array."
        )

    if len(offsets) == 0:

        raise ValueError(
            "Offsets cannot be empty."
        )

    if offsets[0] != 0:

        raise ValueError(
            "The first offset must be 0."
        )

    if offsets[-1] != len(corpus):

        raise ValueError(
            "The final offset must equal "
            "the corpus length."
        )

    if np.any(
        offsets[1:] < offsets[:-1]
    ):

        raise ValueError(
            "Offsets must be monotonically increasing."
        )

    return True