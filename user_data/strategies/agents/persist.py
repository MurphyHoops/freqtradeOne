from __future__ import annotations

import json
import os
from typing import Optional


class StateStore:
    """StateStore 的职责说明。"""
    def __init__(self, filepath: str, state, eq_provider, reservation_agent) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.filepath = filepath
        self.state = state
        self.eq = eq_provider
        self.reservation = reservation_agent

    def save(self) -> None:
        """处理 save 的主要逻辑。"""
        try:
            snapshot = {
                "global_state": self.state.to_snapshot(),
                "equity_provider": self.eq.to_snapshot(),
                "reservations": self.reservation.to_snapshot(),
            }
            tmp = self.filepath + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(snapshot, handle)
            os.replace(tmp, self.filepath)
        except Exception as exc:
            print(f"[CRITICAL] State save failed: {exc}")

    def load_if_exists(self) -> None:
        """处理 load_if_exists 的主要逻辑。"""
        if not os.path.isfile(self.filepath):
            return
        try:
            with open(self.filepath, "r") as handle:
                snapshot = json.load(handle)
            self.state.restore_snapshot(snapshot.get("global_state", {}))
            self.eq.restore_snapshot(snapshot.get("equity_provider", {}))
            self.reservation.restore_snapshot(snapshot.get("reservations", {}))
            self.state.reset_cycle_after_restore()
        except Exception as exc:
            print(f"[WARN] State restore failed: {exc}")
