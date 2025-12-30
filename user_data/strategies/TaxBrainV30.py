# -*- coding: utf-8 -*-
"""TaxBrainV30: Tensor-Logic orchestrator."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import math
import sys
from pathlib import Path

import pandas as pd

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from user_data.strategies.TaxBrainV29 import TaxBrainV29
from user_data.strategies.agents.signals import schemas
from user_data.strategies.core.rejections import RejectReason


class TaxBrainV30(TaxBrainV29):
    """Tensor-Logic strategy that delegates logic to core modules."""

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        return self.matrix_engine.inject_features(df, pair)

    def _candidate_from_pool(
        self,
        pair: str,
        current_time: datetime,
        side: str,
        entry_tag: Optional[str],
        row_idx: Optional[int] = None,
    ) -> Optional[schemas.Candidate]:
        signal_id = None
        if entry_tag:
            try:
                signal_id = int(entry_tag)
            except Exception:
                signal_id = None
        if signal_id:
            raw = self.bridge.get_candidate_by_id(pair, current_time, signal_id, side, row_idx=row_idx)
        else:
            raw = self.bridge.get_best_candidate(pair, current_time, side, row_idx=row_idx)
        if not raw:
            return None
        meta = self.hub.meta_for_id(int(raw.get("signal_id", 0)))
        if not meta:
            return None
        raw_score = float(raw.get("raw_score", float("nan")))
        rr_ratio = float(raw.get("rr_ratio", float("nan")))
        expected_edge = float(raw.get("expected_edge", float("nan")))
        sl_pct = float(raw.get("sl_pct", float("nan")))
        tp_pct = float(raw.get("tp_pct", float("nan")))
        if not all(map(math.isfinite, (raw_score, rr_ratio, expected_edge, sl_pct, tp_pct))):
            return None
        return schemas.Candidate(
            direction=meta.direction,
            kind=meta.name,
            raw_score=raw_score,
            rr_ratio=rr_ratio,
            win_prob=expected_edge,
            expected_edge=expected_edge,
            squad=str(meta.squad),
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            exit_profile=meta.exit_profile,
            recipe=meta.recipe,
            timeframe=meta.timeframe,
            plan_timeframe=meta.plan_timeframe,
            plan_atr_pct=float(raw.get("plan_atr_pct"))
            if raw.get("plan_atr_pct") is not None
            else None,
        )

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        if df is None or df.empty:
            return df

        if "enter_long" not in df.columns:
            df["enter_long"] = 0
        if "enter_short" not in df.columns:
            df["enter_short"] = 0
        if "enter_tag" not in df.columns:
            df["enter_tag"] = None

        if "_signal_id_long" in df.columns:
            long_mask = pd.to_numeric(df["_signal_id_long"], errors="coerce").fillna(0) > 0
            df.loc[long_mask, "enter_long"] = 1
        if "_signal_id_short" in df.columns:
            short_mask = pd.to_numeric(df["_signal_id_short"], errors="coerce").fillna(0) > 0
            df.loc[short_mask, "enter_short"] = 1

        if "_signal_id" in df.columns:
            df["enter_tag"] = df["_signal_id"].apply(
                lambda x: str(int(x)) if pd.notna(x) and x > 0 else None
            )

        last_idx = df.index[-1]
        last_row = df.loc[last_idx]
        current_time = last_row.get("date", None)
        if current_time is None:
            current_time = last_idx if isinstance(last_idx, (pd.Timestamp, datetime)) else datetime.utcnow()

        signal_id = None
        if "_signal_id" in df.columns:
            try:
                signal_id = int(last_row.get("_signal_id", 0))
            except Exception:
                signal_id = None
        candidate = None
        if signal_id and current_time is not None:
            meta = self.hub.meta_for_id(signal_id)
            side = "buy" if meta and meta.direction == "long" else "sell"
            row_idx = self.bridge.get_row_index(pair, current_time)
            candidate = self._candidate_from_pool(pair, current_time, side, str(signal_id), row_idx=row_idx)
        self._update_last_signal(pair, candidate, last_row)
        try:
            if candidate and getattr(self, "global_backend", None):
                self.global_backend.record_signal_score(
                    pair, float(getattr(candidate, "expected_edge", 0.0))
                )
        except Exception:
            pass
        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        self.engine.sync_to_time(current_time)
        if not self.engine.is_permitted(pair, {"side": side, "time": current_time}):
            return False
        row_idx = self.bridge.get_row_index(pair, current_time)
        if row_idx is None:
            return False
        candidate = self._candidate_from_pool(pair, current_time, side, entry_tag, row_idx=row_idx)
        if not candidate:
            try:
                if getattr(self, "rejections", None):
                    context = {"side": side}
                    if entry_tag:
                        context["entry_tag"] = entry_tag
                    self.rejections.record(
                        RejectReason.NO_CANDIDATE,
                        pair=pair,
                        context=context,
                    )
            except Exception:
                pass
            return False
        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        self.engine.sync_to_time(current_time)
        row_idx = self.bridge.get_row_index(pair, current_time)
        if row_idx is None:
            return 0.0

        candidate = self._candidate_from_pool(pair, current_time, side, entry_tag, row_idx=row_idx)
        if not candidate:
            try:
                if getattr(self, "rejections", None):
                    context = {"side": side}
                    if entry_tag:
                        context["entry_tag"] = entry_tag
                    self.rejections.record(
                        RejectReason.NO_CANDIDATE,
                        pair=pair,
                        context=context,
                    )
            except Exception:
                pass
            return 0.0

        pst = self.state.get_pair_state(pair)
        pst.last_score = float(candidate.expected_edge)
        pst.last_dir = candidate.direction
        pst.last_sl_pct = float(candidate.sl_pct)
        pst.last_tp_pct = float(candidate.tp_pct)
        pst.last_kind = str(candidate.kind)
        pst.last_squad = str(candidate.squad)
        pst.last_exit_profile = candidate.exit_profile
        pst.last_recipe = candidate.recipe
        pst.last_atr_pct = float(candidate.plan_atr_pct) if candidate.plan_atr_pct is not None else 0.0

        if not self.engine.is_permitted(
            pair,
            {"side": side, "time": current_time, "score": float(candidate.expected_edge)},
        ):
            return 0.0

        stake, risk, bucket = self.sizer.compute(
            pair=pair,
            sl_pct=float(candidate.sl_pct),
            tp_pct=float(candidate.tp_pct),
            direction=str(candidate.direction),
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=leverage,
            plan_atr_pct=candidate.plan_atr_pct,
            exit_profile=candidate.exit_profile,
            bucket=None,
            current_rate=current_rate,
            score=float(candidate.expected_edge),
        )
        if stake <= 0 or risk <= 0:
            try:
                if getattr(self, "rejections", None):
                    self.rejections.record(
                        RejectReason.SIZER,
                        pair=pair,
                        context={"side": side},
                    )
            except Exception:
                pass
            return 0.0

        meta_payload = {
            "dir": candidate.direction,
            "kind": candidate.kind,
            "squad": candidate.squad,
            "sl_pct": candidate.sl_pct,
            "tp_pct": candidate.tp_pct,
            "exit_profile": candidate.exit_profile,
            "recipe": candidate.recipe,
            "plan_timeframe": candidate.plan_timeframe,
            "atr_pct": candidate.plan_atr_pct,
            "expected_edge": candidate.expected_edge,
            "raw_score": candidate.raw_score,
            "score": candidate.expected_edge,
        }
        reserved = self._reserve_risk_resources(
            pair=pair,
            stake=stake,
            risk=risk,
            bucket=bucket,
            sl=float(candidate.sl_pct),
            tp=float(candidate.tp_pct),
            direction=str(candidate.direction),
            current_rate=current_rate,
            meta=meta_payload,
        )
        if not reserved:
            try:
                if getattr(self, "rejections", None):
                    self.rejections.record(
                        RejectReason.RESERVATION,
                        pair=pair,
                        context={"side": side},
                    )
            except Exception:
                pass
            return 0.0
        return float(stake)
