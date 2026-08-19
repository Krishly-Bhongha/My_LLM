from pathlib import Path
from multiprocessing import Pool

import numpy as np

from config import (
    PREPROCESSED_FILE,
    VOCAB_FILE,
    DATASET_DIR,
    TOKEN_DTYPE,
    OFFSET_DTYPE,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    CPU_WORKERS,
    MAX_VOCAB_SIZE,
    DO_NOT_MERGE,
    PAIR_BASE,
    MIN_PAIR_FREQ,
    MERGES_FILE,
    MERGE_STATE_FILE,
    MIN_PAIR_FREQ_LOAD,
)

from utils import (
    load_json,
    save_numpy,
    validate_vocab,
    save_json,
)


# ============================================================
# WORKER GLOBALS
# ============================================================

WORKER_VOCAB = None
WORKER_BOS_ID = None
WORKER_EOS_ID = None
WORKER_UNK_ID = None
WORKER_PAIR_BASE = PAIR_BASE
WORKER_DO_NOT_MERGE_IDS = None


# ============================================================
# WORKER INITIALIZATION
# ============================================================

def init_worker(
    vocab,
    bos_id,
    eos_id,
    unk_id,
    do_not_merge_ids,
):
    """
    Initialize read-only data inside each worker.
    """

    global WORKER_VOCAB
    global WORKER_BOS_ID
    global WORKER_EOS_ID
    global WORKER_UNK_ID
    global WORKER_PAIR_BASE
    global WORKER_DO_NOT_MERGE_IDS

    WORKER_VOCAB = vocab

    WORKER_BOS_ID = bos_id
    WORKER_EOS_ID = eos_id
    WORKER_UNK_ID = unk_id

    WORKER_PAIR_BASE = PAIR_BASE
    WORKER_DO_NOT_MERGE_IDS = do_not_merge_ids


# ============================================================
# WORKER
# ============================================================

def process_story(story):
    """
    Convert one story into:

        token IDs
        pair IDs for all mergeable adjacent pairs

    The pair ID is:

        left_id * PAIR_BASE + right_id

    Only pairs that are allowed to merge are included.
    """

    length = len(story)

    tokens = np.empty(
        length + 2,
        dtype=TOKEN_DTYPE,
    )

    tokens[0] = WORKER_BOS_ID

    unknown_count = 0

    for i, char in enumerate(story, start=1):

        token_id = WORKER_VOCAB.get(
            char,
            WORKER_UNK_ID,
        )

        if (
            token_id == WORKER_UNK_ID
            and char not in WORKER_VOCAB
        ):
            unknown_count += 1

        tokens[i] = token_id

    tokens[-1] = WORKER_EOS_ID

    # --------------------------------------------------------
    # BUILD PAIR IDS
    # --------------------------------------------------------

    if len(tokens) <= 1:

        pair_ids = np.empty(
            0,
            dtype=np.int64,
        )

    else:

        left = tokens[:-1].astype(
            np.int64,
            copy=False,
        )

        right = tokens[1:].astype(
            np.int64,
            copy=False,
        )

        # Pairs involving special/non-mergeable tokens are
        # excluded immediately.
        mask = (
            ~np.isin(
                left,
                WORKER_DO_NOT_MERGE_IDS,
            )
            &
            ~np.isin(
                right,
                WORKER_DO_NOT_MERGE_IDS,
            )
        )

        pair_ids = (
            left[mask] * WORKER_PAIR_BASE
            + right[mask]
        ).astype(
            np.int64,
            copy=False,
        )

    return tokens, pair_ids, unknown_count


# ============================================================
# READ STORIES
# ============================================================

def load_stories():
    """
    Read every complete preprocessed story.

    Each non-empty line represents one story.
    """

    stories = []

    with open(
        PREPROCESSED_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            story = line.rstrip("\n")

            if not story:
                continue

            stories.append(story)

    return stories


# ============================================================
# BUILD INITIAL CORPUS
# ============================================================

def build_corpus():

    print("=" * 60)
    print("BUILDING INITIAL BPE CORPUS")
    print("=" * 60)

    # ========================================================
    # LOAD VOCABULARY
    # ========================================================

    print("\nLoading vocabulary...")

    vocab = load_json(
        VOCAB_FILE
    )

    validate_vocab(
        vocab
    )

    print(
        f"Vocabulary size: {len(vocab):,}"
    )

    # --------------------------------------------------------
    # REQUIRED SPECIAL TOKENS
    # --------------------------------------------------------

    for token in (
        BOS_TOKEN,
        EOS_TOKEN,
        UNK_TOKEN,
    ):

        if token not in vocab:

            raise ValueError(
                f"{token!r} is missing from vocabulary."
            )

    bos_id = int(
        vocab[BOS_TOKEN]
    )

    eos_id = int(
        vocab[EOS_TOKEN]
    )

    unk_id = int(
        vocab[UNK_TOKEN]
    )

    # ========================================================
    # PAIR CONFIGURATION
    # ========================================================

    # Every token ID is smaller than MAX_VOCAB_SIZE.
    #
    # Therefore:
    #
    # pair_id = left * PAIR_BASE + right
    #
    # can uniquely represent every possible pair.

    # Convert non-mergeable token strings into IDs once.
    do_not_merge_ids = np.array(
        [
            vocab[token]
            for token in DO_NOT_MERGE
            if token in vocab
        ],
        dtype=np.int64,
    )

    print(
        f"Pair base       : {PAIR_BASE:,}"
    )

    print(
        f"Non-merge tokens: "
        f"{len(do_not_merge_ids)}"
    )

    # ========================================================
    # LOAD STORIES
    # ========================================================

    print("\nReading:")
    print(
        f"  {PREPROCESSED_FILE}"
    )

    stories = load_stories()

    if not stories:

        raise ValueError(
            "No stories were found."
        )

    print(
        f"Stories found: {len(stories):,}"
    )

    # ========================================================
    # MULTIPROCESSING
    # ========================================================

    workers = min(
        max(1, CPU_WORKERS),
        len(stories),
    )

    print(
        f"\nUsing {workers} worker processes."
    )

    print(
        "Converting stories and generating pair IDs..."
    )

    chunksize = max(
        1,
        len(stories) // (workers * 8),
    )

    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(
            vocab,
            bos_id,
            eos_id,
            unk_id,
            do_not_merge_ids,
        ),
    ) as pool:

        results = pool.map(
            process_story,
            stories,
            chunksize=chunksize,
        )

    # ========================================================
    # EXTRACT RESULTS
    # ========================================================

    story_arrays = [
        result[0]
        for result in results
    ]

    story_pair_arrays = [
        result[1]
        for result in results
    ]

    total_unknown = sum(
        result[2]
        for result in results
    )

    # ========================================================
    # STORY LENGTHS / OFFSETS
    # ========================================================

    story_lengths = np.fromiter(
        (
            len(story)
            for story in story_arrays
        ),
        dtype=OFFSET_DTYPE,
        count=len(story_arrays),
    )

    total_nodes = int(
        story_lengths.sum()
    )

    print(
        f"Total nodes: {total_nodes:,}"
    )

    offsets = np.empty(
        len(story_arrays) + 1,
        dtype=OFFSET_DTYPE,
    )

    offsets[0] = 0

    offsets[1:] = np.cumsum(
        story_lengths,
        dtype=OFFSET_DTYPE,
    )

    # ========================================================
    # ALLOCATE NODE ARRAYS
    # ========================================================

    tokens = np.empty(
        total_nodes,
        dtype=TOKEN_DTYPE,
    )

    next_node = np.full(
        total_nodes,
        -1,
        dtype=OFFSET_DTYPE,
    )

    prev_node = np.full(
        total_nodes,
        -1,
        dtype=OFFSET_DTYPE,
    )

    active = np.ones(
        total_nodes,
        dtype=np.bool_,
    )

    # ========================================================
    # PLACE STORIES INTO GLOBAL NODE ARRAYS
    # ========================================================

    current_position = 0

    for story_array in story_arrays:

        length = len(
            story_array
        )

        start = current_position
        end = start + length

        tokens[start:end] = story_array

        if length > 1:

            next_node[start:end - 1] = np.arange(
                start + 1,
                end,
                dtype=OFFSET_DTYPE,
            )

            prev_node[start + 1:end] = np.arange(
                start,
                end - 1,
                dtype=OFFSET_DTYPE,
            )

        current_position = end

    # ========================================================
    # VALIDATE STORY BOUNDARIES
    # ========================================================

    print(
        "\nValidating story boundaries..."
    )

    for i in range(
        len(offsets) - 1
    ):

        start = int(
            offsets[i]
        )

        end = int(
            offsets[i + 1]
        )

        if tokens[start] != bos_id:

            raise ValueError(
                f"Story {i} does not begin with <BOS>."
            )

        if tokens[end - 1] != eos_id:

            raise ValueError(
                f"Story {i} does not end with <EOS>."
            )

        if prev_node[start] != -1:

            raise ValueError(
                f"Invalid prev link at story {i}."
            )

        if next_node[end - 1] != -1:

            raise ValueError(
                f"Invalid next link at story {i}."
            )

    # ========================================================
    # BUILD PAIR OCCURRENCE INDEX
    # ========================================================

    print(
        "\nBuilding pair-frequency and occurrence index..."
    )

    # Each node can be the LEFT node of at most one pair:
    #
    #     node -> next[node]
    #
    # Therefore we can use node IDs directly as occurrence IDs.
    #
    # occ_next[node]
    #     next occurrence of the SAME pair
    #
    # occ_prev[node]
    #     previous occurrence of the SAME pair

    occ_next = np.full(
        total_nodes,
        -1,
        dtype=OFFSET_DTYPE,
    )

    occ_prev = np.full(
        total_nodes,
        -1,
        dtype=OFFSET_DTYPE,
    )

    # Python dictionaries are used only for the INITIAL
    # construction. merge.py will load compact arrays and
    # construct its lookup table from them.
    #
    # pair_id -> slot in pair arrays

    pair_to_slot = {}

    pair_counts_list = []
    pair_heads_list = []
    pair_tails_list = []

    current_position = 0

    total_pair_occurrences = 0

    for story_idx, pair_ids in enumerate(
        story_pair_arrays
    ):

        start = int(
            offsets[story_idx]
        )

        # pair_ids was generated only for mergeable pairs.
        #
        # We therefore need to recover the corresponding
        # global LEFT node IDs.
        #
        # We do this by examining the story's original
        # adjacent pairs.

        story_tokens = story_arrays[story_idx]

        if len(story_tokens) > 1:

            left_tokens = story_tokens[:-1]
            right_tokens = story_tokens[1:]

            mask = (
                ~np.isin(
                    left_tokens,
                    do_not_merge_ids,
                )
                &
                ~np.isin(
                    right_tokens,
                    do_not_merge_ids,
                )
            )

            local_left_nodes = np.flatnonzero(
                mask
            )

            global_left_nodes = (
                local_left_nodes
                + start
            )

            # The pair IDs and global node IDs correspond
            # one-to-one.

            if len(global_left_nodes) != len(pair_ids):

                raise RuntimeError(
                    "Pair/node occurrence count mismatch."
                )

            for pair_id, node_id in zip(
                pair_ids,
                global_left_nodes,
            ):

                pair_id = int(
                    pair_id
                )

                node_id = int(
                    node_id
                )

                # ------------------------------------------------
                # NEW PAIR
                # ------------------------------------------------

                slot = pair_to_slot.get(
                    pair_id
                )

                if slot is None:

                    slot = len(
                        pair_counts_list
                    )

                    pair_to_slot[
                        pair_id
                    ] = slot

                    pair_counts_list.append(
                        0
                    )

                    pair_heads_list.append(
                        node_id
                    )

                    pair_tails_list.append(
                        node_id
                    )

                else:

                    old_tail = pair_tails_list[
                        slot
                    ]

                    occ_next[
                        old_tail
                    ] = node_id

                    occ_prev[
                        node_id
                    ] = old_tail

                    pair_tails_list[
                        slot
                    ] = node_id

                pair_counts_list[
                    slot
                ] += 1

                total_pair_occurrences += 1

    # ========================================================
    # CONVERT PAIR STRUCTURES TO NUMPY
    # ========================================================

    pair_ids_array = np.fromiter(
        pair_to_slot.keys(),
        dtype=np.int64,
        count=len(pair_to_slot),
    )

    pair_counts = np.asarray(
        pair_counts_list,
        dtype=np.int64,
    )

    pair_heads = np.asarray(
        pair_heads_list,
        dtype=OFFSET_DTYPE,
    )

    pair_tails = np.asarray(
        pair_tails_list,
        dtype=OFFSET_DTYPE,
    )

    # ========================================================
    # VALIDATE PAIR INDEX
    # ========================================================

    if not (
        len(pair_ids_array)
        == len(pair_counts)
        == len(pair_heads)
        == len(pair_tails)
    ):

        raise RuntimeError(
            "Pair index arrays have inconsistent lengths."
        )

    # ========================================================
    # BUILD INITIAL MAX-FREQUENCY HEAP
    # ========================================================

    print(
        "\nBuilding initial pair-frequency heap..."
    )

    # Python heapq is a MIN heap.
    # Therefore frequencies are negated so that the
    # highest-frequency pair appears at the top.
    #
    # Each entry is:
    #
    #     [-frequency, pair_id]
    #
    # The heap is stored as a NumPy int64 array on disk.
    #
    # merge.py will load it and call heapq.heapify().

    valid_mask = (
        pair_counts >= MIN_PAIR_FREQ_LOAD
    )

    heap = np.column_stack(
        (
            -pair_counts[valid_mask],
            pair_ids_array[valid_mask],
        )
    ).astype(
        np.int64,
        copy=False,
    )

    print(
        f"Heap entries: {len(heap):,}"
    )
    # ========================================================
    # SAVE
    # ========================================================

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
    "tokens": DATASET_DIR / "tokens.npy",
    "next": DATASET_DIR / "next.npy",
    "prev": DATASET_DIR / "prev.npy",
    "active": DATASET_DIR / "active.npy",
    "offsets": DATASET_DIR / "offsets.npy",

    "pair_ids": DATASET_DIR / "pair_ids.npy",
    "pair_counts": DATASET_DIR / "pair_counts.npy",
    "pair_heads": DATASET_DIR / "pair_heads.npy",
    "pair_tails": DATASET_DIR / "pair_tails.npy",

    "occ_next": DATASET_DIR / "occ_next.npy",
    "occ_prev": DATASET_DIR / "occ_prev.npy",

    "heap": DATASET_DIR / "heap.npy",
}

    print(
        "\nSaving initial BPE structures..."
    )

    save_numpy(
        tokens,
        files["tokens"],
    )

    save_numpy(
        next_node,
        files["next"],
    )

    save_numpy(
        prev_node,
        files["prev"],
    )

    save_numpy(
        active,
        files["active"],
    )

    save_numpy(
        offsets,
        files["offsets"],
    )

    save_numpy(
        pair_ids_array,
        files["pair_ids"],
    )

    save_numpy(
        pair_counts,
        files["pair_counts"],
    )

    save_numpy(
        pair_heads,
        files["pair_heads"],
    )

    save_numpy(
        pair_tails,
        files["pair_tails"],
    )

    save_numpy(
        occ_next,
        files["occ_next"],
    )

    save_numpy(
        occ_prev,
        files["occ_prev"],
    )

    save_numpy(
        heap,
        files["heap"],
    )

    # ========================================================
    # INITIALIZE MERGE CHECKPOINT
    # ========================================================

    save_json(
        [],
        MERGES_FILE,
    )

    save_json(
        {
            "completed_merges": 0
        },
        MERGE_STATE_FILE,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("CORPUS BUILD COMPLETE")
    print("=" * 60)

    print(
        f"Stories              : {len(stories):,}"
    )

    print(
        f"Total nodes          : {total_nodes:,}"
    )

    print(
        f"Unique mergeable pairs: "
        f"{len(pair_ids_array):,}"
    )

    print(
        f"Pair occurrences     : "
        f"{total_pair_occurrences:,}"
    )

    print(
        f"Unknown characters   : "
        f"{total_unknown:,}"
    )

    print(
        f"\nToken dtype          : "
        f"{tokens.dtype}"
    )

    print(
        f"Node-index dtype     : "
        f"{next_node.dtype}"
    )

    print(
        f"Pair-ID dtype        : "
        f"{pair_ids_array.dtype}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        f"  {DATASET_DIR}"
    )

    print(
        "\nFiles:"
    )

    for file in files.values():

        print(
            f"  {file.name}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_corpus()