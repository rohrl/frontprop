from typing import List

import numpy as np
import torch


class SimplePatterns:
    DIMS = [7, 7]
    ALL_PATTERNS = [t.float() for t in [
        torch.tensor([
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1]
        ]),
        torch.tensor([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ]),
        torch.tensor([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0]
        ]),
        torch.tensor([
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0]
        ]),
        torch.tensor([
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]),
        torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])]]

    def __init__(self):
        raise ValueError("This class should not be instantiated.")

    @staticmethod
    def get_simple_pattern(probs: List[float] = None, noise=0.0) -> torch.Tensor:
        """
        Randomly select a pattern given probabilities.
        """
        if probs is None:
            probs = [1.0] * len(SimplePatterns.ALL_PATTERNS)

        if len(probs) > len(SimplePatterns.ALL_PATTERNS):
            raise ValueError(f"Only {len(SimplePatterns.ALL_PATTERNS)} patterns available.")

        probs /= np.sum(probs)

        idx = np.random.choice(len(probs), p=probs)

        # add gaussian noise to the pattern
        if noise > 0:
            pattern = SimplePatterns.ALL_PATTERNS[idx].clone()
            pattern += torch.randn_like(pattern) * noise
            return pattern

        return SimplePatterns.ALL_PATTERNS[idx]

    @staticmethod
    def get_all_patterns(n: int = None) -> List[torch.Tensor]:
        n = n or len(SimplePatterns.ALL_PATTERNS)
        return SimplePatterns.ALL_PATTERNS[:n]

    @staticmethod
    def get_all_patterns_count():
        return len(SimplePatterns.ALL_PATTERNS)

    @staticmethod
    def get_pattern_dims():
        return SimplePatterns.DIMS
