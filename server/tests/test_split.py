import random
from collections import Counter
from pathlib import Path

from pointremote.split import stratified_split


def _items(counts):
    items = []
    for cls, n in enumerate(counts):
        items += [(Path(f"c{cls}/{i:03}.jpg"), cls) for i in range(n)]
    return items


def test_split_is_stratified():
    train, val = stratified_split(_items([10, 20, 5]), 0.2, seed=1)
    assert Counter(c for _, c in val) == {0: 2, 1: 4, 2: 1}
    assert len(train) + len(val) == 35
    assert not set(train) & set(val)


def test_split_is_deterministic_and_order_independent():
    items = _items([10, 20, 5])
    shuffled = items[:]
    random.Random(99).shuffle(shuffled)
    assert stratified_split(items, 0.2, seed=3) == stratified_split(shuffled, 0.2, seed=3)


def test_different_seed_changes_split():
    items = _items([30])
    assert stratified_split(items, 0.2, seed=1)[1] != stratified_split(items, 0.2, seed=2)[1]


def test_single_item_class_goes_to_train():
    train, val = stratified_split(_items([1, 5]), 0.2, seed=0)
    assert [c for _, c in train].count(0) == 1
    assert all(c == 1 for _, c in val)
