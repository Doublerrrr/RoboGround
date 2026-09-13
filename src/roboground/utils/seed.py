"""随机种子与确定性设置。

注意：本项目大量使用 numpy 做几何运算，**不要**把 torch 的
`use_deterministic_algorithms(True)` 打开 —— 它会让部分 CUDA 算子报错，
且对几何精度没有实际收益。
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, *, deterministic: bool = False) -> int:
    """统一设置 python / numpy / torch 的随机种子。

    Parameters
    ----------
    seed
        种子值。
    deterministic
        是否让 cudnn 走确定性算法（会变慢，仅调试复现时开）。

    Returns
    -------
    int
        实际使用的种子。
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch 是可选的（几何/单测不需要）
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    return seed


class _Rng:
    """项目内统一的随机源，避免各处直接调用全局 np.random。"""

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.np = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.np = np.random.default_rng(self.seed)

    def choice(self, items, size=None, replace=True, p=None):
        return self.np.choice(items, size=size, replace=replace, p=p)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self.np.uniform(low, high, size=size)

    def integers(self, low, high=None, size=None):
        return self.np.integers(low, high, size=size)

    def permutation(self, n):
        return self.np.permutation(n)
