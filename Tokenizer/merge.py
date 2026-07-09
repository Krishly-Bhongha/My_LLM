from collections import Counter
from config import DO_NOT_MERGE


def count_pair_frequency(corpus):
    """
    corpus:
        List[List[str]]

    Returns:
        Counter of adjacent pair frequencies.
    """

    pair_freq = Counter()

    for sentence in corpus:

        for i in range(len(sentence) - 1):

            left = sentence[i]
            right = sentence[i + 1]

            # Don't merge spaces or special tokens
            if left in DO_NOT_MERGE:
                continue

            if right in DO_NOT_MERGE:
                continue

            pair_freq[(left, right)] += 1

    return pair_freq


def best_pair(pair_freq, min_freq):
    """
    Returns the most frequent pair whose
    frequency >= min_freq.

    Returns:
        (pair, frequency)
        or
        (None, 0)
    """

    if len(pair_freq) == 0:
        return None, 0

    pair, freq = pair_freq.most_common(1)[0]

    if freq < min_freq:
        return None, 0

    return pair, freq


def merge_pair(corpus, pair):
    """
    Replace every occurrence of pair
    with one merged token.

    Example:

        ("H","e")

    becomes

        "He"
    """

    left, right = pair
    merged_token = left + right

    for sentence in corpus:

        i = 0

        while i < len(sentence) - 1:

            if sentence[i] == left and sentence[i + 1] == right:

                sentence[i] = merged_token
                del sentence[i + 1]

            else:
                i += 1

    return corpus, merged_token