"""ChainForge — 概率编程 / 贝叶斯后验推断工具箱（纯 NumPy/SciPy 手写实现）。

旗舰算法 VarioNUTS：变分全秩预条件 NUTS + 诊断门控 + 诚实回退。
Author: 晨星
"""

__version__ = "0.1.0"
__author__ = "晨星"

from .core import Budget, Config, CountingModel, InferenceError, ReferencePosterior

__all__ = [
    "Budget",
    "Config",
    "CountingModel",
    "InferenceError",
    "ReferencePosterior",
]
