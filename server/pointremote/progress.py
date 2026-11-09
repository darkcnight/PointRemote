from collections.abc import Callable

ProgressFn = Callable[[int, int, str], None]


def no_progress(epoch: int, total: int, message: str) -> None:
    pass
