# -*- coding: utf-8 -*-
"""信号子系统入口。

暴露对外可用的函数：:func:compute_indicators、:func:build_candidates。
同时保留 gen_candidates 兼容旧接口，内部直接委托给新实现。
信号由插件系统加载，不在此模块内硬编码导入。
"""

from __future__ import annotations

from .indicators import compute_indicators
from .schemas import Candidate
from .builder import build_candidates


def gen_candidates(row, cfg, informative=None):
    """向后兼容的包装函数。"""

    return build_candidates(row, cfg, informative=informative)

# 导入内置信号，触发注册流程。
__all__ = [
    "Candidate",
    "compute_indicators",
    "build_candidates",
    "gen_candidates",
]
