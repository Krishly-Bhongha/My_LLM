import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "Models"

CONFIG_FILE = ROOT / "config.py"

MODEL_TEMPLATE = """\
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from attention import MultiHeadAttention
from block import TransformerBlock
from pos_encode import PositionalEncoding


def initialize_parameters():

    pass


if __name__ == "__main__":

    initialize_parameters()
"""

def initialize_model():

    while True:

        version = input("Enter model version: ").strip()

        if version == "":
            print("Version cannot be empty.\n")
            continue

        model_dir = MODELS_DIR / version

        if model_dir.exists():
            print("Version already exists.\n")
            continue

        break

    model_dir.mkdir(parents=True)

    shutil.copy2(
        CONFIG_FILE,
        model_dir / "config.py"
    )

    with open(
        model_dir / "model.py",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(MODEL_TEMPLATE)

    print(f"\nCreated model version {version}")


if __name__ == "__main__":

    initialize_model()