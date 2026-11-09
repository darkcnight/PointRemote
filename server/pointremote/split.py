from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

Item = tuple[Path, int]


def stratified_split(
    items: list[Item], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Item], list[Item]]:
    rng = random.Random(seed)
    by_class: dict[int, list[Item]] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)
    train: list[Item] = []
    val: list[Item] = []
    for cls in sorted(by_class):
        group = sorted(by_class[cls], key=lambda it: str(it[0]))
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) >= 2 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val
