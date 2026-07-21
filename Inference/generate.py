import torch

from inference_config import MAX_NEW_TOKENS
from sampling import sample


class Generator:

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.model.eval()

        self.eos_id = tokenizer.token_to_id["<EOS>"]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)

        x = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        generated = []

        for _ in range(max_new_tokens):

            # Forward pass
            logits = self.model(x)

            # Get logits for next token
            next_logits = logits[:, -1, :]

            # Sample next token
            next_token = sample(next_logits)

            # Stop at EOS
            if next_token == self.eos_id:
                break

            generated.append(next_token)

            # Append to context
            next_tensor = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=self.device
            )

            x = torch.cat((x, next_tensor), dim=1)

        return self.tokenizer.decode(generated)