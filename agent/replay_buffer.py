"""Experience replay buffer for policy-value MCTS training."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: list[TrainingExample] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.storage)

    def push(self, example: TrainingExample) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(example)
        else:
            self.storage[self.position] = example
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[TrainingExample]:
        indices = rng.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[index] for index in indices]
