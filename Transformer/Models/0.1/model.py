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
