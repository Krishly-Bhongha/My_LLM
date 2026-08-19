import sys
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

TOKENIZER_DIR = ROOT / "tokenizer"

PROJECT_ROOT = ROOT.parent

TOKENIZER_CONFIG_DIR = (
    PROJECT_ROOT / "Tokenizer"
)

sys.path.insert(
    0,
    str(TOKENIZER_CONFIG_DIR)
)

import config


# ============================================================
# GET DATASET NAME
# ============================================================

def get_dataset_name():

    dataset_name = input(
        "\nEnter tokenizer dataset name "
        "(press Enter to use config.py): "
    ).strip()

    if dataset_name:
        return dataset_name

    preprocessed_file = (
        config.PREPROCESSED_FILE
    )

    dataset_name = Path(
        preprocessed_file
    ).stem

    print()
    print("Using dataset from config.py:")
    print(f"  {preprocessed_file}")
    print(f"  Dataset name: {dataset_name}")

    return dataset_name


# ============================================================
# FINALIZE
# ============================================================

def finalize():

    print()
    print("=" * 60)
    print("FINALIZE TOKENIZER DATASET")
    print("=" * 60)

    dataset_name = get_dataset_name()

    dataset_dir = (
        TOKENIZER_DIR / dataset_name
    )

    checkpoints_dir = (
        dataset_dir / "checkpoints"
    )

    latest_dir = (
        checkpoints_dir / "latest"
    )

    # ========================================================
    # VERIFY DATASET
    # ========================================================

    if not dataset_dir.exists():

        print()
        print(
            "ERROR: Dataset folder does not exist:"
        )

        print(
            f"  {dataset_dir}"
        )

        return False

    if not latest_dir.exists():

        print()
        print(
            "ERROR: checkpoints/latest does not exist:"
        )

        print(
            f"  {latest_dir}"
        )

        return False

    # ========================================================
    # FILES TO KEEP
    # ========================================================

    KEEP_ROOT_FILES = {
        "final_corpus.npy",
        "final_offsets.npy",
    }

    KEEP_LATEST_FILES = {
        "vocab.json",
        "merges.json",
    }

    # ========================================================
    # FIND ITEMS TO DELETE
    # ========================================================

    items_to_delete = []

    # --------------------------------------------------------
    # 1. UNNECESSARY ITEMS DIRECTLY IN DATASET FOLDER
    # --------------------------------------------------------

    for item in dataset_dir.iterdir():

        # checkpoints handled separately
        if item == checkpoints_dir:
            continue

        if item.name in KEEP_ROOT_FILES:
            continue

        items_to_delete.append(item)

    # --------------------------------------------------------
    # 2. OLD CHECKPOINTS
    # --------------------------------------------------------

    for item in checkpoints_dir.iterdir():

        # latest is handled separately
        if item.name == "latest":
            continue

        items_to_delete.append(item)

    # --------------------------------------------------------
    # 3. UNNECESSARY FILES INSIDE latest
    # --------------------------------------------------------

    for item in latest_dir.iterdir():

        if item.name in KEEP_LATEST_FILES:
            continue

        items_to_delete.append(item)

    # ========================================================
    # SHOW WHAT WILL REMAIN
    # ========================================================

    print()
    print("=" * 60)
    print("FILES TO KEEP")
    print("=" * 60)

    print()
    print("Dataset root:")
    print("  final_corpus.npy")
    print("  final_offsets.npy")

    print()
    print("checkpoints/latest:")
    print("  vocab.json")
    print("  merges.json")

    # ========================================================
    # NOTHING TO DELETE
    # ========================================================

    if not items_to_delete:

        print()
        print("=" * 60)
        print("NOTHING NEEDS TO BE DELETED")
        print("=" * 60)

        return True

    # ========================================================
    # SHOW ITEMS TO DELETE
    # ========================================================

    print()
    print("=" * 60)
    print("ITEMS TO DELETE")
    print("=" * 60)

    for item in items_to_delete:

        try:
            relative_path = item.relative_to(
                dataset_dir
            )
        except ValueError:
            relative_path = item

        if item.is_dir():

            print(
                f"  [DIR]  {relative_path}"
            )

        else:

            print(
                f"  [FILE] {relative_path}"
            )

    # ========================================================
    # CONFIRM
    # ========================================================

    print()
    print(
        "Everything listed above will be permanently deleted."
    )

    print()
    print(
        "The final corpus files and the final "
        "vocab/merges files will remain."
    )

    confirmation = input(
        '\nType "DELETE" to continue: '
    ).strip()

    if confirmation != "DELETE":

        print()
        print(
            "Finalization cancelled."
        )

        return False

    # ========================================================
    # DELETE
    # ========================================================

    print()
    print(
        "Deleting unnecessary files..."
    )

    for item in items_to_delete:

        try:

            if item.is_dir():

                shutil.rmtree(item)

            else:

                item.unlink()

        except Exception as error:

            print()
            print(
                "ERROR while deleting:"
            )

            print(
                f"  {item}"
            )

            print(
                f"Reason: {error}"
            )

            return False

    # ========================================================
    # FINAL STRUCTURE
    # ========================================================

    print()
    print("=" * 60)
    print("FINALIZATION COMPLETE")
    print("=" * 60)

    print()
    print(
        "Final tokenizer structure:"
    )

    print()
    print(
        f"  {dataset_name}/"
    )

    print(
        "  ├── final_corpus.npy"
    )

    print(
        "  ├── final_offsets.npy"
    )

    print(
        "  └── checkpoints/"
    )

    print(
        "      └── latest/"
    )

    print(
        "          ├── vocab.json"
    )

    print(
        "          └── merges.json"
    )

    return True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    success = finalize()

    if not success:
        sys.exit(1)