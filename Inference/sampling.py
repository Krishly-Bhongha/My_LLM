import torch

from inference_config import TEMPERATURE, TOP_K, TOP_P


def sample(
    logits: torch.Tensor,
    temperature: float = TEMPERATURE,
    top_k: int | None = TOP_K,
    top_p: float | None = TOP_P,
) -> int:
    """
    Sample the next token from model logits.

    Args:
        logits: Tensor of shape (VOCAB_SIZE,) or (1, VOCAB_SIZE)
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        top_p: Top-p (nucleus) filtering.

    Returns:
        Token ID as an integer.
    """

    # Remove batch dimension if present
    if logits.dim() == 2:
        logits = logits.squeeze(0)

    # Apply temperature
    if temperature > 0:
        logits = logits / temperature

    # ---------- Top-k ----------
    if top_k is not None and 0 < top_k < logits.size(0):
        values, _ = torch.topk(logits, top_k)
        threshold = values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # ---------- Top-p ----------
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True
        )

        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        remove = cumulative_probs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False

        sorted_logits[remove] = float("-inf")

        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_indices, sorted_logits)

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Randomly sample according to the probability distribution
    token = torch.multinomial(probabilities, 1)

    return token.item()