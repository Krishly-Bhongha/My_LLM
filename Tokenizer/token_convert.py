import json

from config import VOCAB_FILE, MERGES_FILE, SPECIAL_TOKENS


class TokenConverter:

    def __init__(self):

        with open(VOCAB_FILE, "r", encoding="utf8") as f:
            self.token_to_id = json.load(f)

        with open(MERGES_FILE, "r", encoding="utf8") as f:
            self.merges = json.load(f)

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

        self.special_tokens = set(SPECIAL_TOKENS)

    def _apply_merges(self, tokens):
        """
        Replay every learned merge in order.
        """

        for left, right in self.merges:

            merged = left + right

            i = 0

            while i < len(tokens) - 1:

                if tokens[i] == left and tokens[i + 1] == right:

                    tokens[i] = merged
                    del tokens[i + 1]

                else:
                    i += 1

        return tokens

    def encode(self, text):
        """
        Convert text into token ids.
        """

        tokens = ["<BOS>"]

        for ch in text:

            if ch in self.token_to_id:
                tokens.append(ch)

        tokens.append("<EOS>")

        tokens = self._apply_merges(tokens)

        return [
            self.token_to_id[token]
            for token in tokens
        ]

    def decode(self, ids):
        """
        Convert token ids back into text.
        """

        pieces = []

        for idx in ids:

            token = self.id_to_token[int(idx)]

            if token in self.special_tokens:
                continue

            pieces.append(token)

        return "".join(pieces)


if __name__ == "__main__":

    tokenizer = TokenConverter()

    text = input("Prompt: ")

    ids = tokenizer.encode(text)

    print("\nEncoded IDs:")
    print(ids)

    print("\nDecoded Text:")
    print(tokenizer.decode(ids))
