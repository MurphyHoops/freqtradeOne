# -*- coding: utf-8 -*-
"""全局状态持久化与恢复模块。

StateStore 将 GlobalState、EquityProvider 以及 ReservationAgent 的快照合并
写入磁盘，支持策略在崩溃或重启后恢复运行。
"""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    from freqtrade.exceptions import OperationalException  # type: ignore
except Exception:  # pragma: no cover - fallback when freqtrade is absent
    class OperationalException(RuntimeError):
        ...


class StateStore:
    """封装状态序列化与反序列化逻辑的持久化工具。"""

    def __init__(self, filepath: str, state, eq_provider, reservation_agent) -> None:
        """初始化 StateStore。

        Args:
            filepath: 状态文件的绝对路径。
            state: GlobalState 实例。
            eq_provider: EquityProvider 实例。
            reservation_agent: ReservationAgent 实例。
        """

        self.filepath = filepath
        self.state = state
        self.eq = eq_provider
        self.reservation = reservation_agent

    def save(self) -> None:
        """将当前状态写入磁盘，采用临时文件替换确保原子性。"""

        try:
            snapshot = {
                "global_state": self.state.to_snapshot(),
                "equity_provider": self.eq.to_snapshot(),
                "reservations": self.reservation.to_snapshot(),
            }
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, default=str)
            os.replace(tmp, self.filepath)
        except Exception as exc:
            print(f"[CRITICAL] State save failed: {exc}")

    def load_if_exists(self) -> None:
        """若状态文件存在则恢复内存状态，否则静默返回。"""

        if not os.path.isfile(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.state.restore_snapshot(snapshot.get("global_state", {}))
            self.eq.restore_snapshot(snapshot.get("equity_provider", {}))
            self.reservation.restore_snapshot(snapshot.get("reservations", {}))
            self.state.reset_cycle_after_restore()
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, IOError, PermissionError) as exc:
            raise OperationalException(
                "State file corrupted or unreadable. Manual intervention required to prevent debt reset."
            ) from exc
        except Exception as exc:
            raise OperationalException(
                "State file corrupted or unreadable. Manual intervention required to prevent debt reset."
            ) from exc
