"""Teacher confidence in its category label, from top-k logprobs at the category step (FL-M4 T5).

Written by hand (the milestone's live-coding piece).
"""

from __future__ import annotations

import math


def category_confidence(
    top_logprobs: dict[int, float], first_tokens: dict[str, int]
) -> dict[str, float]:
    """Probability of each category, renormalized over the categories whose first token is in
    top_logprobs (others get 0.0). Raises ValueError if two categories share a first token."""
    seen: dict[int, str] = {}
    for category, token in first_tokens.items():
        if token in seen:
            raise ValueError(f"{seen[token]!r} and {category!r} share first token {token}")
        seen[token] = category

    present = {
        category: top_logprobs[token]
        for category, token in first_tokens.items()
        if token in top_logprobs and math.isfinite(top_logprobs[token])
    }

    if not present:
        return {category: 0.0 for category in first_tokens}

    m = max(present.values())
    weights = {category: math.exp(logprob - m) for category, logprob in present.items()}
    total = sum(weights.values())

    return {category: weights.get(category, 0.0) / total for category in first_tokens}
