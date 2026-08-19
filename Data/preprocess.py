from pathlib import Path
import argparse
import re

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(__file__).resolve().parent

RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"


# TinyStories uses this to separate stories.
END_OF_TEXT = "<|endoftext|>"


# ============================================================
# TINYSTORIES PREPROCESSING
# ============================================================

def preprocess_tinystories(text: str) -> list[str]:
    """
    Preprocess a TinyStories text file.

    Each <|endoftext|> marker represents the end of one story.

    Returns:
        A list containing one cleaned story per element.
    """

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Split into individual stories.
    raw_stories = text.split(END_OF_TEXT)

    # If the file was cut in the middle of a story, the final
    # segment does not have an END_OF_TEXT marker and is incomplete.
    # Ignore it.
    if not text.rstrip().endswith(END_OF_TEXT):
        raw_stories = raw_stories[:-1]

    stories = []

    for story in raw_stories:

        # Replace newlines inside a story with spaces.
        story = story.replace("\n", " ")

        # Collapse repeated whitespace.
        story = re.sub(r"\s+", " ", story)

        # Remove whitespace at the beginning/end.
        story = story.strip()

        # Ignore empty stories.
        if not story:
            continue

        stories.append(story)

    return stories


# ============================================================
# SAVE
# ============================================================

def save_stories(stories: list[str], output_file: Path):
    """
    Save one story per line.

    Each line represents one complete story.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline="\n"
    ) as f:

        for story in stories:

            f.write(story)
            f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Preprocess a dataset for the LLM tokenizer."
    )

    parser.add_argument(
        "filename",
        help="Name of the file inside Data/raw/"
    )

    parser.add_argument(
        "--type",
        choices=["tinystories"],
        default="tinystories",
        help="Type of dataset being processed."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output filename inside Data/preprocessed/"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    input_file = RAW_DIR / args.filename

    if not input_file.exists():

        print(
            f"ERROR: Input file not found:\n"
            f"  {input_file}"
        )

        return

    # --------------------------------------------------------
    # OUTPUT NAME
    # --------------------------------------------------------

    if args.output:

        output_filename = args.output

    else:

        input_name = input_file.stem

        output_filename = (
            f"{input_name}_clean.txt"
        )

    output_file = (
        PREPROCESSED_DIR
        / output_filename
    )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    print(
        f"\nInput : {input_file}"
    )

    print(
        f"Output: {output_file}"
    )

    print(
        f"Type  : {args.type}"
    )

    print("\nReading file...")

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    print(
        f"Raw characters: {len(text):,}"
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    if args.type == "tinystories":

        stories = preprocess_tinystories(
            text
        )

    else:

        raise ValueError(
            f"Unsupported dataset type: {args.type}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_stories(
        stories,
        output_file
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_characters = sum(
        len(story)
        for story in stories
    )

    print("\nPreprocessing complete.")

    print(
        f"Stories          : {len(stories):,}"
    )

    print(
        f"Characters       : {total_characters:,}"
    )

    if stories:

        print(
            f"Average length   : "
            f"{total_characters / len(stories):.1f}"
        )

    print(
        f"\nSaved to:\n"
        f"  {output_file}"
    )


if __name__ == "__main__":
    main()

