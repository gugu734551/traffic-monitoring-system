#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核心策略占位说明模块。

本仓库公开的是这套交通监控系统的**工程实现**：桌面端界面、检测与跟踪链路、
轨迹管理、表格统计、结果导出与用户系统。

课题的**核心策略**——即交通事件（闯红灯 / 逆行 / 疑似超速）的判定规则、
阈值整定与去重时机——属于论文核心内容，**不随本仓库开源**。

为了让读者能够完整地阅读并运行整条工程链路（视频读取 → YOLOv8 检测 →
DeepSORT 跟踪 → 事件判定 → 界面可视化 → CSV 导出），相关函数的位置、
签名与文档全部保留，只在函数体内调用本模块的 :func:`not_open_sourced`，
注入一个**安全默认值**并给出提示，而不是留下实现细节。
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

__all__ = ["CORE_STRATEGY_NOTICE", "not_open_sourced"]

CORE_STRATEGY_NOTICE = (
    "交通事件判定策略（闯红灯 / 逆行 / 疑似超速）属于本课题核心内容，未随本仓库开源；"
    "当前执行的是占位实现，返回安全默认值，不影响检测、跟踪与界面演示。"
)

# 同一处占位只在首次调用时告警，避免逐帧刷屏
_warned_features: set = set()


def not_open_sourced(feature: str, default: Optional[Any] = None) -> Any:
    """标记一次「核心策略未开源」的占位调用，并返回安全默认值。

    Args:
        feature: 被调用的策略名称，用于在告警信息中定位调用点。
        default: 安全默认值。例如布尔判定返回 ``False``、事件列表返回 ``[]``。

    Returns:
        传入的 ``default``。
    """
    if feature not in _warned_features:
        _warned_features.add(feature)
        warnings.warn(f"[{feature}] {CORE_STRATEGY_NOTICE}", RuntimeWarning, stacklevel=2)
    return default
