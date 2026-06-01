"""Board symmetries for data augmentation."""

import numpy as np

from agent.replay_buffer import TrainingExample

_SYMMETRY_PERMS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),
    (6, 3, 0, 7, 4, 1, 8, 5, 2),
    (8, 7, 6, 5, 4, 3, 2, 1, 0),
    (2, 5, 8, 1, 4, 7, 0, 3, 6),
    (2, 1, 0, 5, 4, 3, 8, 7, 6),
    (6, 7, 8, 3, 4, 5, 0, 1, 2),
    (0, 3, 6, 1, 4, 7, 2, 5, 8),
    (8, 5, 2, 7, 4, 1, 6, 3, 0),
)


def apply_perm_vector(values: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    out = np.empty_like(values)
    for index, target in enumerate(perm):
        out[target] = values[index]
    return out


def augment_example(example: TrainingExample) -> list[TrainingExample]:
    augmented: list[TrainingExample] = []
    for perm in _SYMMETRY_PERMS:
        augmented.append(
            TrainingExample(
                state=apply_perm_vector(example.state, perm),
                policy=apply_perm_vector(example.policy, perm),
                value=example.value,
            )
        )
    return augmented
