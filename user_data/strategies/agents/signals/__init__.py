# -*- coding: utf-8 -*-
"""信号子系统入口。

暴露对外可用的函数：:func:compute_indicators、:func:build_candidates。
信号由插件系统加载，不在此模块内硬编码导入。
"""

from __future__ import annotations

from .indicators import compute_indicators
from .schemas import Candidate
from .builder import build_candidates

# 导入内置信号，触发注册流程。
__all__ = [
    "Candidate",
    "compute_indicators",
    "build_candidates",
]
