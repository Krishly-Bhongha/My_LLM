import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from config import (
    PAD_ID,
    BOS_ID,
    EOS_ID,
    TRAIN_IDS_FILE,
    BATCH_SIZE,
    BLOCK_SIZE,
    TRAIN_SPLIT,
    SHUFFLE,
    RANDOM_SEED,
    NUM_WORKERS,
    DROP_LAST,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    PREFETCH_FACTOR,
)


# ============================================================
# DATASET
# ============================================================

class ChatDataset(Dataset):

    def __init__(self, token_ids):

        self.token_ids = token_ids

        # Stores:
        #
        # [
        #     (start_index, end_index),
        #     (start_index, end_index),
        #     ...
        # ]
        #
        # end_index is exclusive.

        self.sequences = []

        self._build_sequences()


    def _build_sequences(self):

        start = 0

        total_tokens = len(self.token_ids)


        while start < total_tokens - 1:

            # ------------------------------------------------
            # We need BLOCK_SIZE + 1 tokens.
            #
            # Example:
            #
            # sequence:
            #
            # [1, 5, 9, 3, 2]
            #
            # x:
            #
            # [1, 5, 9, 3]
            #
            # y:
            #
            # [5, 9, 3, 2]
            #
            # ------------------------------------------------

            end = min(
                start + BLOCK_SIZE + 1,
                total_tokens
            )


            # ------------------------------------------------
            # After reaching BLOCK_SIZE,
            # continue until the next EOS.
            # ------------------------------------------------

            while (
                end < total_tokens
                and self.token_ids[end - 1] != EOS_ID
            ):

                end += 1


            # Need at least 2 tokens for:
            #
            # x = sequence[:-1]
            # y = sequence[1:]

            if end - start >= 2:

                self.sequences.append(
                    (start, end)
                )


            # Next sequence starts after the previous one
            start = end


    def __len__(self):

        return len(self.sequences)


    def __getitem__(self, index):

        start, end = self.sequences[index]


        # Read the sequence from the tensor
        sequence = self.token_ids[start:end]

        # Input
        x = sequence[:-1]


        # Target is shifted by one token
        y = sequence[1:]


        return x, y


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_batch(batch):

    """
    Combines sequences of different lengths into one batch.

    Since sequences continue until <EOS>, their lengths
    may differ.

    Shorter sequences are padded using <PAD>.
    """

    inputs, targets = zip(*batch)


    # Longest sequence in this particular batch
    max_length = max(
        x.size(0)
        for x in inputs
    )


    current_batch_size = len(inputs)


    # --------------------------------------------------------
    # Create padded tensors
    # --------------------------------------------------------

    x_batch = torch.full(
        (
            current_batch_size,
            max_length
        ),
        PAD_ID,
        dtype=torch.long
    )


    y_batch = torch.full(
        (
            current_batch_size,
            max_length
        ),
        PAD_ID,
        dtype=torch.long
    )


    # --------------------------------------------------------
    # Attention mask
    #
    # True  = actual token
    # False = padding
    # --------------------------------------------------------

    attention_mask = torch.zeros(
        (
            current_batch_size,
            max_length
        ),
        dtype=torch.bool
    )


    # --------------------------------------------------------
    # Copy actual sequences into padded tensors
    # --------------------------------------------------------

    for i, (x, y) in enumerate(
        zip(inputs, targets)
    ):

        length = x.size(0)


        x_batch[i, :length] = x

        y_batch[i, :length] = y


        attention_mask[i, :length] = True


    return (
        x_batch,
        y_batch,
        attention_mask
    )


# ============================================================
# LOAD TOKEN IDS
# ============================================================

def load_token_ids():

    token_ids = torch.load(
        TRAIN_IDS_FILE,
        weights_only=True,
        mmap=True
    )

    return token_ids


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    token_ids = load_token_ids()


    full_dataset = ChatDataset(
        token_ids
    )


    total_sequences = len(
        full_dataset
    )


    train_size = int(
        total_sequences
        * TRAIN_SPLIT
    )


    val_size = (
        total_sequences
        - train_size
    )


    # Reproducible split

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )


    train_dataset, val_dataset = random_split(

        full_dataset,

        [
            train_size,
            val_size
        ],

        generator=generator
    )


    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders():

    train_dataset, val_dataset = (
        create_datasets()
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=SHUFFLE,

        collate_fn=collate_batch,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        persistent_workers=(
            PERSISTENT_WORKERS
            if NUM_WORKERS > 0
            else False
        ),

        prefetch_factor=(
            PREFETCH_FACTOR
            if NUM_WORKERS > 0
            else None
        ),

        drop_last=DROP_LAST
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        collate_fn=collate_batch,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        persistent_workers=(
            PERSISTENT_WORKERS
            if NUM_WORKERS > 0
            else False
        ),

        prefetch_factor=(
            PREFETCH_FACTOR
            if NUM_WORKERS > 0
            else None
        ),

        drop_last=False
    )


    return (
        train_loader,
        val_loader
    )


# ============================================================
# TEST LOADER
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader = (
        create_dataloaders()
    )


    print(
        f"Vocabulary size : {VOCAB_SIZE}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Block size      : {BLOCK_SIZE}"
    )

    print(
        f"Train batches   : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )


    # Get one batch

    x, y, attention_mask = next(
        iter(train_loader)
    )


    print("\nBatch shapes:")

    print(
        "Input          :",
        x.shape
    )

    print(
        "Target         :",
        y.shape
    )

    print(
        "Attention mask :",
        attention_mask.shape
    )


    print("\nFirst input sequence:")

    print(
        x[0]
    )


    print("\nFirst target sequence:")

    print(
        y[0]
    )


    print("\nFirst attention mask:")

    print(
        attention_mask[0]
    )