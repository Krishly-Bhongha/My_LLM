import importlib.util
import sys
from pathlib import Path

import torch

from inference_config import (
    MODELS_DIR,
    DEFAULT_MODEL_VERSION,
    MODEL_FILENAME,
    DEVICE,
    EXIT_COMMANDS,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / "Tokenizer"))

from token_convert import TokenConverter
from generate import Generator


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


while True:

    version = input(
        f"Enter model version: "
    ).strip()

    if version == "":
        version = DEFAULT_MODEL_VERSION

    model_dir = MODELS_DIR / version

    if not model_dir.exists():
        print("Model version does not exist.\n")
        continue

    break


# Remove any cached config module
sys.modules.pop("config", None)

# Ensure the version directory is searched first
sys.path.insert(0, str(model_dir))

model_module = load_module(
    "model_definition",
    model_dir / "model.py"
)

# Remove it again afterwards
sys.path.pop(0)

model = model_module.TransformerModel()

state_dict = torch.load(
    model_dir / MODEL_FILENAME,
    map_location=DEVICE,
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

tokenizer = TokenConverter()

generator = Generator(
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
)

print("\nModel loaded successfully.")
print("Type 'exit' to quit.\n")

while True:

    prompt = input("You: ").strip()

    if prompt.lower() in EXIT_COMMANDS:
        break

    response = generator.generate(prompt)

    print(f"Bot: {response}\n")