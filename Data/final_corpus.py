from pathlib import Path
import numpy as np
import sys
import os
from concurrent.futures import ProcessPoolExecutor


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

TOKENIZER_DIR = (
    ROOT / "tokenizer"
)


# ============================================================
# MULTIPROCESSING SETTINGS
# ============================================================

NUM_WORKERS = os.cpu_count() or 1

# Number of stories assigned to each worker task.
# Multiple tasks per worker allow better load balancing.
STORIES_PER_TASK = 1000


# ============================================================
# WORKER FUNCTION
# ============================================================

def reconstruct_story_chunk(args):
    """
    Reconstruct a range of stories directly into the final
    memory-mapped corpus.

    Workers independently open the input/output files.
    Each worker writes to a completely separate region of
    the output array, so no locking is required.
    """

    (
        dataset_dir,
        story_start,
        story_end,
        output_start,
    ) = args

    checkpoint_dir = (
        dataset_dir
        / "checkpoints"
        / "latest"
    )

    tokens_file = (
        checkpoint_dir / "tokens.npy"
    )

    next_file = (
        checkpoint_dir / "next.npy"
    )

    active_file = (
        checkpoint_dir / "active.npy"
    )

    offsets_file = (
        checkpoint_dir / "offsets.npy"
    )

    temp_corpus_file = (
        dataset_dir / "final_corpus.tmp.npy"
    )

    # --------------------------------------------------------
    # OPEN MEMORY-MAPPED INPUTS
    # --------------------------------------------------------

    tokens = np.load(
        tokens_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    next_node = np.load(
        next_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    active = np.load(
        active_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    offsets = np.load(
        offsets_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    # --------------------------------------------------------
    # OPEN MEMORY-MAPPED OUTPUT
    # --------------------------------------------------------

    final_corpus = np.load(
        temp_corpus_file,
        mmap_mode="r+",
        allow_pickle=False,
    )

    # --------------------------------------------------------
    # RECONSTRUCT STORIES
    # --------------------------------------------------------

    write_position = output_start

    for story_index in range(
        story_start,
        story_end,
    ):

        start_node = int(
            offsets[story_index]
        )

        end_node = int(
            offsets[story_index + 1]
        )

        node = start_node

        while node != -1:

            # ------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------

            if node < start_node or node >= end_node:

                raise RuntimeError(
                    "\nLinked list escaped story boundary.\n"
                    f"Story      : {story_index}\n"
                    f"Node       : {node}\n"
                    f"Story start: {start_node}\n"
                    f"Story end  : {end_node}"
                )

            if not active[node]:

                raise RuntimeError(
                    "\nEncountered inactive node while "
                    "following final linked list.\n"
                    f"Story : {story_index}\n"
                    f"Node  : {node}"
                )

            final_corpus[
                write_position
            ] = tokens[node]

            write_position += 1

            node = int(
                next_node[node]
            )

    return (
        story_start,
        story_end,
        write_position - output_start,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINAL CORPUS GENERATOR")
    print("=" * 60)

    # ========================================================
    # ASK FOR TOKENIZER FOLDER
    # ========================================================

    # final_corpus.py is inside:
    #
    #     My_LLM/Data/
    #
    # config.py is inside:
    #
    #     My_LLM/Tokenizer/
    #

    PROJECT_ROOT = ROOT.parent

    TOKENIZER_CONFIG_DIR = (
        PROJECT_ROOT / "Tokenizer"
    )

    sys.path.insert(
        0,
        str(TOKENIZER_CONFIG_DIR)
    )

    import config

    while True:

        user_input = input(
            "\nEnter tokenizer folder name "
            "(press Enter to use config.py): "
        ).strip()

        # ----------------------------------------------------
        # BLANK INPUT
        # ----------------------------------------------------

        if not user_input:

            preprocessed_file = (
                config.PREPROCESSED_FILE
            )

            dataset_name = (
                preprocessed_file.stem
            )

            print()
            print(
                "Using dataset from config.py:"
            )

            print(
                f"  {preprocessed_file}"
            )

        # ----------------------------------------------------
        # MANUAL INPUT
        # ----------------------------------------------------

        else:

            dataset_name = user_input

        # ----------------------------------------------------
        # CHECK FOLDER
        # ----------------------------------------------------

        dataset_dir = (
            TOKENIZER_DIR / dataset_name
        )

        if not dataset_dir.exists():

            print()
            print(
                "ERROR: Tokenizer folder does not exist:"
            )

            print(
                f"  {dataset_dir}"
            )

            continue

        break

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    checkpoint_dir = (
        dataset_dir
        / "checkpoints"
        / "latest"
    )

    if not checkpoint_dir.exists():

        raise RuntimeError(
            "\nFinal checkpoint does not exist:\n"
            f"{checkpoint_dir}\n\n"
            "Make sure BPE merging has completed "
            "and the final checkpoint was saved."
        )

    # ========================================================
    # FILES
    # ========================================================

    tokens_file = (
        checkpoint_dir / "tokens.npy"
    )

    next_file = (
        checkpoint_dir / "next.npy"
    )

    active_file = (
        checkpoint_dir / "active.npy"
    )

    offsets_file = (
        checkpoint_dir / "offsets.npy"
    )

    # ========================================================
    # VERIFY
    # ========================================================

    required_files = [
        tokens_file,
        next_file,
        active_file,
        offsets_file,
    ]

    for path in required_files:

        if not path.exists():

            raise RuntimeError(
                "\nRequired file missing:\n"
                f"{path}"
            )

    # ========================================================
    # LOAD
    # ========================================================

    print()
    print("Loading final BPE state...")

    tokens = np.load(
        tokens_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    next_node = np.load(
        next_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    active = np.load(
        active_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    offsets = np.load(
        offsets_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if len(tokens) != len(next_node):

        raise RuntimeError(
            "tokens.npy and next.npy have "
            "different lengths."
        )

    if len(tokens) != len(active):

        raise RuntimeError(
            "tokens.npy and active.npy have "
            "different lengths."
        )

    if len(offsets) < 2:

        raise RuntimeError(
            "offsets.npy does not contain any stories."
        )

    # ========================================================
    # COUNT FINAL TOKENS
    # ========================================================

    final_token_count = int(
        np.count_nonzero(active)
    )

    story_count = len(offsets) - 1

    print(
        f"Stories       : {story_count:,}"
    )

    print(
        f"Original nodes: {len(tokens):,}"
    )

    print(
        f"Final tokens  : {final_token_count:,}"
    )

    # ========================================================
    # COUNT TOKENS PER STORY
    # ========================================================
    #
    # This lets us know exactly where each worker should write
    # in the final corpus.
    #
    # No linked-list traversal is required here.
    #

    print()
    print("Calculating final story sizes...")

    story_lengths = np.empty(
        story_count,
        dtype=np.int64,
    )

    for story_index in range(
        story_count
    ):

        start = int(
            offsets[story_index]
        )

        end = int(
            offsets[story_index + 1]
        )

        story_lengths[
            story_index
        ] = np.count_nonzero(
            active[start:end]
        )

    calculated_total = int(
        story_lengths.sum()
    )

    if calculated_total != final_token_count:

        raise RuntimeError(
            "\nFinal token count mismatch.\n"
            f"Active count : {final_token_count:,}\n"
            f"Story counts : {calculated_total:,}"
        )

    # ========================================================
    # FINAL OFFSETS
    # ========================================================

    final_offsets = np.empty(
        story_count + 1,
        dtype=np.int64,
    )

    final_offsets[0] = 0

    final_offsets[1:] = np.cumsum(
        story_lengths,
        dtype=np.int64,
    )

    # ========================================================
    # TEMPORARY OUTPUT FILES
    # ========================================================

    final_corpus_file = (
        dataset_dir / "final_corpus.npy"
    )

    final_offsets_file = (
        dataset_dir / "final_offsets.npy"
    )

    temp_corpus_file = (
        dataset_dir / "final_corpus.tmp.npy"
    )

    temp_offsets_file = (
        dataset_dir / "final_offsets.tmp.npy"
    )

    # --------------------------------------------------------
    # REMOVE OLD TEMPORARY FILES
    # --------------------------------------------------------

    if temp_corpus_file.exists():
        temp_corpus_file.unlink()

    if temp_offsets_file.exists():
        temp_offsets_file.unlink()

    # ========================================================
    # CREATE MEMORY-MAPPED OUTPUT
    # ========================================================

    print()
    print(
        "Creating temporary final corpus..."
    )

    final_corpus = np.lib.format.open_memmap(
        temp_corpus_file,
        mode="w+",
        dtype=tokens.dtype,
        shape=(final_token_count,),
    )

    final_corpus.flush()

    del final_corpus

    # ========================================================
    # CREATE WORKER TASKS
    # ========================================================

    tasks = []

    for story_start in range(
        0,
        story_count,
        STORIES_PER_TASK,
    ):

        story_end = min(
            story_start + STORIES_PER_TASK,
            story_count,
        )

        output_start = int(
            final_offsets[story_start]
        )

        tasks.append(
            (
                dataset_dir,
                story_start,
                story_end,
                output_start,
            )
        )

    worker_count = min(
        NUM_WORKERS,
        len(tasks),
    )

    print()
    print(
        f"Workers        : {worker_count}"
    )

    print(
        f"Tasks          : {len(tasks):,}"
    )

    print(
        f"Stories/task   : {STORIES_PER_TASK:,}"
    )

    # ========================================================
    # MULTIPROCESSING
    # ========================================================

    print()
    print(
        "Reconstructing final corpus..."
    )

    completed_stories = 0

    with ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:

        futures = [
            executor.submit(
                reconstruct_story_chunk,
                task,
            )
            for task in tasks
        ]

        for future in futures:

            (
                story_start,
                story_end,
                written,
            ) = future.result()

            expected = int(
                final_offsets[story_end]
                - final_offsets[story_start]
            )

            if written != expected:

                raise RuntimeError(
                    "\nWorker wrote incorrect number "
                    "of tokens.\n"
                    f"Stories : "
                    f"{story_start}-{story_end}\n"
                    f"Expected: {expected:,}\n"
                    f"Written : {written:,}"
                )

            completed_stories += (
                story_end - story_start
            )

            print(
                f"  Processed "
                f"{completed_stories:,}/"
                f"{story_count:,} stories"
            )

    # ========================================================
    # FLUSH OUTPUT
    # ========================================================

    final_corpus = np.load(
        temp_corpus_file,
        mmap_mode="r+",
        allow_pickle=False,
    )

    final_corpus.flush()

    del final_corpus

    # ========================================================
    # SAVE OFFSETS
    # ========================================================

    np.save(
        temp_offsets_file,
        final_offsets,
        allow_pickle=False,
    )

    # ========================================================
    # VALIDATE FINAL OUTPUT
    # ========================================================

    print()
    print(
        "Validating generated corpus..."
    )

    generated_corpus = np.load(
        temp_corpus_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    generated_offsets = np.load(
        temp_offsets_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    if len(generated_corpus) != final_token_count:

        raise RuntimeError(
            "Generated corpus has incorrect size."
        )

    if len(generated_offsets) != story_count + 1:

        raise RuntimeError(
            "Generated offsets have incorrect size."
        )

    if int(
        generated_offsets[-1]
    ) != final_token_count:

        raise RuntimeError(
            "Final offsets do not match "
            "final corpus size."
        )

    del generated_corpus
    del generated_offsets

    # ========================================================
    # REPLACE OLD OUTPUT
    # ========================================================

    if final_corpus_file.exists():
        final_corpus_file.unlink()

    if final_offsets_file.exists():
        final_offsets_file.unlink()

    os.replace(
        temp_corpus_file,
        final_corpus_file,
    )

    os.replace(
        temp_offsets_file,
        final_offsets_file,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL CORPUS COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Tokenizer folder : {dataset_dir}"
    )

    print(
        f"Stories          : {story_count:,}"
    )

    print(
        f"Final tokens     : {final_token_count:,}"
    )

    print()
    print("Saved:")

    print(
        f"  {final_corpus_file}"
    )

    print(
        f"  {final_offsets_file}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()