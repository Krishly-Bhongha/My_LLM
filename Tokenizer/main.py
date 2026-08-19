from pathlib import Path
import subprocess
import sys
import importlib

import config


# ============================================================
# SCRIPT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PREPROCESS_SCRIPT = ROOT.parent / "Data" / "preprocess.py"
VOCAB_SCRIPT = ROOT / "vocab.py"
CORPUS_SCRIPT = ROOT / "corpus.py"
MERGE_SCRIPT = ROOT / "merge.py"


# ============================================================
# CORPUS FILES
# ============================================================

CORPUS_FILES = [
    config.DATASET_DIR / "tokens.npy",
    config.DATASET_DIR / "next.npy",
    config.DATASET_DIR / "prev.npy",
    config.DATASET_DIR / "active.npy",
    config.DATASET_DIR / "offsets.npy",

    config.DATASET_DIR / "pair_ids.npy",
    config.DATASET_DIR / "pair_counts.npy",
    config.DATASET_DIR / "pair_heads.npy",
    config.DATASET_DIR / "pair_tails.npy",

    config.DATASET_DIR / "occ_next.npy",
    config.DATASET_DIR / "occ_prev.npy",

    config.HEAP_FILE,
]


# ============================================================
# RUN SCRIPT
# ============================================================

def run_script(script):
    """
    Run another tokenizer script using the same Python
    interpreter running main.py.
    """

    print()
    print("=" * 60)
    print(f"RUNNING: {script.name}")
    print("=" * 60)
    print()

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ]
    )

    if result.returncode != 0:

        print()
        print("=" * 60)
        print(f"ERROR: {script.name} failed.")
        print("=" * 60)

        return False

    print()
    print(f"{script.name} completed successfully.")

    return True


# ============================================================
# FILE CHECK
# ============================================================

def file_exists(path):
    return path.exists()


# ============================================================
# CHECK CORPUS
# ============================================================

def corpus_exists():
    """
    The corpus stage is considered complete only if every
    structure required by merge.py exists.
    """

    return all(
        path.exists()
        for path in CORPUS_FILES
    )


# ============================================================
# CHECK BPE STATE
# ============================================================

def bpe_state_exists():
    """
    Determine whether merge.py has state it can load.

    corpus.py creates merge_state.json with completed_merges=0,
    so its existence is enough to establish an initialized
    BPE state.
    """

    return (
        config.MERGE_STATE_FILE.exists()
        and config.HEAP_FILE.exists()
        and config.MERGES_FILE.exists()
    )


# ============================================================
# PREPROCESS
# ============================================================

def ask_for_raw_file():
    """
    Ask for a raw filename.

    Blank input means:
        use the already-configured PREPROCESSED_FILE.

    If a filename is entered, keep asking until it exists.
    """

    print()
    print("=" * 60)
    print("DATASET SELECTION")
    print("=" * 60)

    print()
    print(
        "Configured preprocessed file:"
    )

    print(
        f"  {config.PREPROCESSED_FILE}"
    )

    print()
    print(
        "Raw data directory:"
    )

    print(
        f"  {config.RAW_DIR}"
    )

    print()
    print(
        "Enter a raw filename to preprocess it."
    )

    print(
        "Leave blank to use the configured "
        "preprocessed file."
    )

    while True:

        filename = input(
            "\nRaw filename: "
        ).strip()

        # ----------------------------------------------------
        # BLANK
        # ----------------------------------------------------

        if not filename:

            return None

        raw_file = (
            config.RAW_DIR
            / filename
        )

        # ----------------------------------------------------
        # EXISTS
        # ----------------------------------------------------

        if raw_file.is_file():

            return filename

        # ----------------------------------------------------
        # DOES NOT EXIST
        # ----------------------------------------------------

        print()
        print(
            "File not found:"
        )

        print(
            f"  {raw_file}"
        )

        print(
            "Please enter the filename again."
        )


def run_preprocessing(raw_filename):
    """
    Run preprocess.py on a raw dataset.

    Example:
        Tiny_Stories.txt
            ↓
        Tiny_Stories_clean.txt

    After successful preprocessing, update config.py so that
    PREPROCESSED_FILE points to the newly created file.
    """

    print()
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    raw_path = config.RAW_DIR / raw_filename

    # --------------------------------------------------------
    # OUTPUT FILENAME
    # --------------------------------------------------------
    #
    # preprocess.py converts:
    #
    #     something.txt
    #
    # into:
    #
    #     something_clean.txt
    #

    raw_path_obj = Path(raw_filename)

    output_filename = (
        raw_path_obj.stem
        + "_clean"
        + raw_path_obj.suffix
    )

    output_path = (
        config.PREPROCESSED_DIR
        / output_filename
    )

    print()
    print(
        f"Input : {raw_path}"
    )

    print(
        f"Output: {output_path}"
    )

    # --------------------------------------------------------
    # RUN PREPROCESSOR
    # --------------------------------------------------------

    result = subprocess.run(
        [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            raw_filename,
            "--output",
            output_filename,
        ]
    )

    if result.returncode != 0:

        print()
        print(
            "ERROR: preprocessing failed."
        )

        return False

    # --------------------------------------------------------
    # VERIFY OUTPUT
    # --------------------------------------------------------

    if not output_path.exists():

        print()
        print(
            "ERROR: preprocess.py finished but the "
            "expected output file was not created:"
        )

        print(
            f"  {output_path}"
        )

        return False

    # --------------------------------------------------------
    # UPDATE CONFIG.PY
    # --------------------------------------------------------

    update_preprocessed_filename(
        output_filename
    )

    refresh_config(output_filename)

    print()
    print(
        "Preprocessing completed successfully."
    )

    return True

def update_preprocessed_filename(filename):
    """
    Update only the filename portion of PREPROCESSED_FILE
    in config.py while preserving PREPROCESSED_DIR.
    """

    config_file = ROOT / "config.py"

    lines = config_file.read_text(
        encoding="utf-8"
    ).splitlines()

    new_lines = []

    for line in lines:

        stripped = line.strip()

        if stripped.startswith(
            "PREPROCESSED_FILE = PREPROCESSED_DIR /"
        ):

            new_lines.append(
                f'PREPROCESSED_FILE = PREPROCESSED_DIR / "{filename}"'
            )

        else:

            new_lines.append(line)

    config_file.write_text(
        "\n".join(new_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Updated config.py:"
    )

    print(
        f'  PREPROCESSED_FILE = PREPROCESSED_DIR / "{filename}"'
    )

def refresh_config(output_filename):
    global CORPUS_FILES

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    config.PREPROCESSED_FILE = (
        config.PREPROCESSED_DIR / output_filename
    )

    config.DATASET_NAME = (
        Path(output_filename).stem
    )

    config.DATASET_DIR = (
        config.TOKENIZER_DATA_DIR / config.DATASET_NAME
    )

    # --------------------------------------------------------
    # TOKENIZER FILES
    # --------------------------------------------------------

    config.CORPUS_FILE = (
        config.DATASET_DIR / "corpus.npy"
    )

    config.OFFSETS_FILE = (
        config.DATASET_DIR / "offsets.npy"
    )

    config.VOCAB_FILE = (
        config.DATASET_DIR / "vocab.json"
    )

    config.MERGES_FILE = (
        config.DATASET_DIR / "merges.json"
    )

    config.METADATA_FILE = (
        config.DATASET_DIR / "metadata.json"
    )

    config.HEAP_FILE = (
        config.DATASET_DIR / "heap.npy"
    )

    config.MERGE_STATE_FILE = (
        config.DATASET_DIR / "merge_state.json"
    )

    # --------------------------------------------------------
    # CORPUS FILES
    # --------------------------------------------------------

    CORPUS_FILES = [
        config.DATASET_DIR / "tokens.npy",
        config.DATASET_DIR / "next.npy",
        config.DATASET_DIR / "prev.npy",
        config.DATASET_DIR / "active.npy",
        config.DATASET_DIR / "offsets.npy",

        config.DATASET_DIR / "pair_ids.npy",
        config.DATASET_DIR / "pair_counts.npy",
        config.DATASET_DIR / "pair_heads.npy",
        config.DATASET_DIR / "pair_tails.npy",

        config.DATASET_DIR / "occ_next.npy",
        config.DATASET_DIR / "occ_prev.npy",

        config.HEAP_FILE,
    ]

    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    print()
    print("Configuration refreshed.")

    print(
        f"  Dataset      : {config.DATASET_NAME}"
    )

    print(
        f"  Preprocessed : {config.PREPROCESSED_FILE}"
    )

    print(
        f"  Dataset dir  : {config.DATASET_DIR}"
    )

    print(
        f"  Vocabulary   : {config.VOCAB_FILE}"
    )

# ============================================================
# VOCABULARY
# ============================================================

def get_current_vocab_file():
    """
    Get the vocabulary path directly from the currently
    configured preprocessed file.
    """

    dataset_name = (
        config.PREPROCESSED_FILE.stem
    )

    return (
        config.TOKENIZER_DATA_DIR
        / dataset_name
        / "vocab.json"
    )

def run_vocab():

    if not config.PREPROCESSED_FILE.exists():

        print()
        print(
            "ERROR: Preprocessed file does not exist:"
        )

        print(
            f"  {config.PREPROCESSED_FILE}"
        )

        return False

    return run_script(
        VOCAB_SCRIPT
    )


# ============================================================
# CORPUS
# ============================================================

def run_corpus():

    if not config.PREPROCESSED_FILE.exists():

        print()
        print(
            "ERROR: Preprocessed file does not exist:"
        )

        print(
            f"  {config.PREPROCESSED_FILE}"
        )

        return False

    vocab_file = get_current_vocab_file()

    print()
    print(
        f"Checking vocabulary:"
    )

    print(
        f"  {vocab_file}"
    )

    if not vocab_file.exists():

        print()
        print(
            "ERROR: Vocabulary file does not exist:"
        )

        print(
            f"  {vocab_file}"
        )

        return False

    return run_script(
        CORPUS_SCRIPT
    )

# ============================================================
# MERGE
# ============================================================

def run_merge():

    vocab_file = get_current_vocab_file()

    if not vocab_file.exists():

        print()
        print(
            "ERROR: Vocabulary file does not exist:"
        )

        print(
            f"  {vocab_file}"
        )

        return False

    if not corpus_exists():

        print()
        print(
            "ERROR: Corpus structures are incomplete."
        )

        print(
            "Run corpus.py first."
        )

        print()

        for path in CORPUS_FILES:

            if not path.exists():

                print(
                    f"Missing: {path.name}"
                )

        return False

    return run_script(
        MERGE_SCRIPT
    )


# ============================================================
# SMART FULL PIPELINE
# ============================================================

def run_full_pipeline():
    """
    Run the tokenizer pipeline intelligently.

    Existing completed stages are skipped.

    If the user enters a raw filename, preprocessing is run
    explicitly.

    If the user leaves the filename blank, the configured
    preprocessed file is used.
    """

    print()
    print("=" * 60)
    print("FULL TOKENIZER PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # ASK WHETHER TO PREPROCESS
    # --------------------------------------------------------

    raw_filename = ask_for_raw_file()

    # ========================================================
    # CASE 1:
    # USER ENTERED A RAW FILE
    # ========================================================

    if raw_filename is not None:

        if not run_preprocessing(
            raw_filename
        ):

            return

        # ----------------------------------------------------
        # Because we just generated a new preprocessed dataset,
        # the old vocabulary/corpus/BPE state may no longer
        # correspond to it.
        #
        # We therefore rebuild the downstream stages.
        # ----------------------------------------------------

        print()
        print(
            "New preprocessed data detected."
        )

        print(
            "Rebuilding vocabulary, corpus, "
            "and BPE state."
        )

        if not run_vocab():
            return

        if not run_corpus():
            return

        run_merge()

        return

    # ========================================================
    # CASE 2:
    # BLANK INPUT
    # ========================================================

    print()
    print(
        "Using configured preprocessed file:"
    )

    print(
        f"  {config.PREPROCESSED_FILE}"
    )

    if not config.PREPROCESSED_FILE.exists():

        print()
        print(
            "ERROR: Configured preprocessed file "
            "does not exist:"
        )

        print(
            f"  {config.PREPROCESSED_FILE}"
        )

        print()
        print(
            "Either create it with preprocess.py "
            "or enter a raw filename next time."
        )

        return

    # --------------------------------------------------------
    # VOCAB
    # --------------------------------------------------------

    if config.VOCAB_FILE.exists():

        print()
        print(
            "✓ Vocabulary already exists."
        )

    else:

        print()
        print(
            "Vocabulary not found."
        )

        print(
            "Building vocabulary..."
        )

        if not run_vocab():
            return

    # --------------------------------------------------------
    # CORPUS
    # --------------------------------------------------------

    if corpus_exists():

        print()
        print(
            "✓ Corpus structures already exist."
        )

    else:

        print()
        print(
            "Corpus structures not found."
        )

        print(
            "Building corpus + pair index + heap..."
        )

        if not run_corpus():
            return

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    if bpe_state_exists():

        print()
        print(
            "✓ BPE state found."
        )

        print(
            "Starting / resuming merge.py..."
        )

    else:

        print()
        print(
            "BPE state not found."
        )

        print(
            "Starting merge.py..."
        )

    run_merge()


# ============================================================
# MENU
# ============================================================

def print_menu():

    print()
    print("=" * 60)
    print("MY LLM TOKENIZER")
    print("=" * 60)

    print()
    print(
        f"Dataset:"
    )

    print(
        f"  {config.DATASET_NAME}"
    )

    print()
    print(
        f"Preprocessed file:"
    )

    print(
        f"  {config.PREPROCESSED_FILE}"
    )

    print()

    print(
        "1. Run full pipeline"
    )

    print(
        "2. Preprocess dataset"
    )

    print(
        "3. Build vocabulary"
    )

    print(
        "4. Build corpus + pair index + heap"
    )

    print(
        "5. Run / resume BPE merges"
    )

    print(
        "6. Exit"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    while True:

        print_menu()

        choice = input(
            "Choice: "
        ).strip()

        # ----------------------------------------------------
        # FULL PIPELINE
        # ----------------------------------------------------

        if choice == "1":

            run_full_pipeline()

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        elif choice == "2":

            raw_filename = ask_for_raw_file()

            if raw_filename is not None:

                run_preprocessing(
                    raw_filename
                )

            else:

                print()
                print(
                    "No raw filename entered."
                )

                print(
                    "Configured preprocessed file:"
                )

                print(
                    f"  {config.PREPROCESSED_FILE}"
                )

        # ----------------------------------------------------
        # VOCAB
        # ----------------------------------------------------

        elif choice == "3":

            run_vocab()

        # ----------------------------------------------------
        # CORPUS
        # ----------------------------------------------------

        elif choice == "4":

            run_corpus()

        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        elif choice == "5":

            run_merge()

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif choice == "6":

            print(
                "\nExiting tokenizer."
            )

            break

        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        else:

            print(
                "\nInvalid choice. "
                "Please enter 1-6."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()