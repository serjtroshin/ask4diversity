from typing import List

import numpy as np


def _distinct_ngram_diversity(
    responses: List[str], n=5, fail_token: str = None, ignore_fail_token: bool = True, tokenization_mode: str = "default"
) -> float:
    """
    Calculate the ratio of unique n-grams to total n-grams in a list of text responses.

    Steps included:
    1. Validate that n is a positive integer.
    2. Filter out invalid responses (non-str or empty).
    3. Early return 0.0 if fewer than 2 valid responses.
    4. If tokenization_mode="default": strip periods and newlines, split on spaces, ignore empty tokens.
       If tokenization_mode="math_steps": split on newlines, no cleaning (for math expressions).
       If tokenization_mode="math_expressions": do not split at all, no cleaning (for math expressions).
    5. Build sliding-window n-gram tuples.
    6. Flatten, then compute len(set)/len(list), returning 0.0 if no n-grams.

    :param responses: List[str] — the text responses to analyze.
    :param n:         int     — size of each n-gram (must be >0).
    :param fail_token: str    — token to ignore in diversity calculation.
    :param ignore_fail_token: bool — whether to ignore fail tokens.
    :param tokenization_mode: str — if "default", strip periods and newlines, split on spaces, ignore empty tokens.
                                 if "math_steps", split on newlines, no cleaning (for math expressions).
                                 if "math_expressions", do not split at all, no cleaning (for math expressions).
    :return:          float   — diversity score in [0.0, 1.0].
    """
    # 1. Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # 2. Filter invalid responses
    valid = [r for r in responses if isinstance(r, str) and r.strip()]
    valid = [r for r in valid if r != fail_token]  # remove fail tokens

    # 3. Early exit for small sets
    if len(valid) < 2:
        return 0.0

    # 4–5. Tokenize and build n-grams per response
    ngram_lists = []
    for text in valid:
        if tokenization_mode == "math_steps":
            # Split on newlines without cleaning (for math expressions)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
        elif tokenization_mode == "math_expressions":
            # Do not split at all, no cleaning (for math expressions)
            lines = [text]
        else:
            # Original behavior: strip periods & newlines, split on spaces
            cleaned = text.replace(".", "").replace("\n", "")
            lines = [w for w in cleaned.split(" ") if w]
        
        # sliding-window tuples
        ngrams = [tuple(lines[i : i + n]) for i in range(len(lines) - n + 1)]
        ngram_lists.append(ngrams)

    # flatten all n-grams
    all_ngrams = [gram for sub in ngram_lists for gram in sub]
    if not all_ngrams:
        return 0.0

    # 6. Ratio of unique to total n-grams
    return len(set(all_ngrams)) / len(all_ngrams)


def distinct_ngram_diversity(
    responses: List[str], n=5, fail_token: str = None, ignore_fail_token: bool = True, tokenization_mode: str = "default"
) -> float:
    """
    Calculate the diversity of a list of text responses based on unique n-grams.
    Average over n from 1 to 5.
    Strategy: Average pairwise diversity.
    By default, we allow any number of solutions > 1 given the prompt, so we set ignore_fail_token to True.
        ( currently only support ignore_fail_token=True)
    """
    if not ignore_fail_token:
        raise NotImplementedError(
            "Currently, ignore_fail_token must be True. This is a limitation of the current implementation."
        )
    diversity_scores = []
    for n_i in range(1, n + 1):
        _diversity = _distinct_ngram_diversity(
            responses, n=n_i, fail_token=fail_token, ignore_fail_token=ignore_fail_token, tokenization_mode=tokenization_mode
        )
        diversity_scores.append(_diversity)
    return np.mean(diversity_scores).item() if diversity_scores else 0.0


def effective_number_of_samples(
    responses: List[str], n=5, fail_token: str = None, tokenization_mode: str = "default"
) -> float:
    """
    Calculate the effective number of samples based on unique n-grams.
    This is a measure of how many unique samples are present in the responses.

    Sum_{i!=j} (distinct_ngram_diversity(responses[i], responses[j])) / (N - 1)

    :param responses: List[str] — the text responses to analyze.
    :param n:         int     — size of each n-gram (must be >0).
    :param fail_token: str    — token to ignore in diversity calculation.
    :param tokenization_mode: str — if "default", strip periods and newlines, split on spaces, ignore empty tokens.
                                 if "math_steps", split on newlines, no cleaning (for math expressions).
                                 if "math_expressions", do not split at all, no cleaning (for math expressions).
    :return:          float   — effective number of samples.
    """
    diversity_scores = [[None] * len(responses) for _ in range(len(responses))]
    for i in range(len(responses)):
        for j in range(0, len(responses)):
            _diversity = _distinct_ngram_diversity(
                [responses[i], responses[j]], n=n, fail_token=fail_token, tokenization_mode=tokenization_mode
            )  # normalize to [0, 1] range
            diversity_scores[i][j] = _diversity
            diversity_scores[j][i] = _diversity  # symmetric matrix
        diversity_scores[i][i] = 1.0  # self-comparison is always 0.0
    diversity_matric = np.array(diversity_scores)
    # print(f"Diversity matrix:\n{diversity_matric}")
    # estimate average diversity
    return (
        (diversity_matric.sum(-1) - 1).sum() / (len(responses) - 1)
        if len(responses) > 1
        else 0.0
    )  # we don't count self-comparison (1.0) in the average, so we subtract 1


def main():
    # Example usage
    responses = [
        "Another test response for diversity.",
        "Another test response for diversity.",
        "Another test response for diversity.",
        "Another test response for diversity.",
        "Another test response for diversity.",
        "Another test response for diversity.",
    ]
    diversity_score = _distinct_ngram_diversity(responses, n=2)
    print(f"Diversity score (n=2) _distinct_ngram_diversity: {diversity_score}")
    # These scores are from 1/N to 1.0. This example should return 1/N = 0.16666666666666666

    # This is expected diversity score normalized to [0, 1] range:
    print(
        "Average n-gram diversity: _distinct_ngram_diversity",
        distinct_ngram_diversity(responses, n=2),
    )  # should return 0.0
    print("Effective number of samples:", effective_number_of_samples(responses, n=2))
    print("--------------------")

    diverse_responses = [
        "This is a unique response",
        "That is another nice output",
        "Yet new generation",
    ]
    diverse_score = distinct_ngram_diversity(diverse_responses, n=2)
    print(f"Diversity score for diverse responses (n=2): {diverse_score}")

    print(
        "Effective number of samples:",
        effective_number_of_samples(diverse_responses, n=2),
        f"Out of {len(diverse_responses)} samples",
    )

    mixed_responses = [
        "This is a unique response",
        "That is another nice output",
        "Yet new generation",
        "This is another response",
        "This is a unique response",
    ]
    mixed_score = distinct_ngram_diversity(mixed_responses, n=2)
    print(f"Diversity score for mixed responses (n=2): {mixed_score}")
    print(
        "Effective number of samples:",
        effective_number_of_samples(mixed_responses, n=2),
        f"Out of {len(mixed_responses)} samples",
    )


if __name__ == "__main__":
    main()
