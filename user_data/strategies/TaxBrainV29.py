from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import types
from typing import Any, Dict, Optional

import pandas as pd

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

MODULE_ALIAS = 'user_data.strategies.TaxBrainV29'
if MODULE_ALIAS not in sys.modules:
    sys.modules[MODULE_ALIAS] = sys.modules.get(__name__)

module_obj = sys.modules.get(__name__)
if module_obj is None:
    module_obj = types.ModuleType(__name__)
    sys.modules[__name__] = module_obj
module_obj.__dict__.update(globals())

from freqtrade.strategy import IStrategy

from user_data.strategies.agents.analytics import AnalyticsAgent
from user_data.strategies.agents.cycle import CycleAgent
from user_data.strategies.agents.execution import ExecutionAgent
from user_data.strategies.agents.exit import ExitPolicyV29
from user_data.strategies.agents.persist import StateStore
from user_data.strategies.agents.reservation import ReservationAgent
from user_data.strategies.agents.risk import RiskAgent
from user_data.strategies.agents.signal import Candidate, compute_indicators, gen_candidates
from user_data.strategies.agents.sizer import SizerAgent
from user_data.strategies.agents.tier import TierAgent, TierManager
from user_data.strategies.agents.treasury import TreasuryAgent
from user_data.strategies.config.v29_config import V29Config, apply_overrides


@dataclass
class ActiveTradeMeta:
    """ActiveTradeMeta 的职责说明。"""
    sl_pct: float
    tp_pct: float
    direction: str
    entry_bar_tick: int
    entry_price: float
    bucket: str
    icu_bars_left: Optional[int]


@dataclass
class PairState:
    """PairState 的职责说明。"""
    closs: int = 0
    local_loss: float = 0.0
    cooldown_bars_left: int = 0
    last_dir: Optional[str] = None
    last_score: float = 0.0
    last_squad: Optional[str] = None
    last_sl_pct: float = 0.0
    last_tp_pct: float = 0.0
    last_atr_pct: float = 0.0
    active_trades: Dict[str, ActiveTradeMeta] = field(default_factory=dict)


@dataclass
class TreasuryState:
    """TreasuryState 的职责说明。"""
    fast_alloc_risk: Dict[str, float] = field(default_factory=dict)
    slow_alloc_risk: Dict[str, float] = field(default_factory=dict)
    cycle_start_tick: int = 0
    cycle_start_equity: float = 0.0

    def to_snapshot(self) -> Dict[str, Any]:
        """处理 to_snapshot 的主要逻辑。"""
        return {
            "fast_alloc_risk": self.fast_alloc_risk,
            "slow_alloc_risk": self.slow_alloc_risk,
            "cycle_start_tick": self.cycle_start_tick,
            "cycle_start_equity": self.cycle_start_equity,
        }

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        """处理 restore_snapshot 的主要逻辑。"""
        self.fast_alloc_risk = {k: float(v) for k, v in payload.get("fast_alloc_risk", {}).items()}
        self.slow_alloc_risk = {k: float(v) for k, v in payload.get("slow_alloc_risk", {}).items()}
        self.cycle_start_tick = int(payload.get("cycle_start_tick", 0))
        self.cycle_start_equity = float(payload.get("cycle_start_equity", 0.0))


class GlobalState:
    """GlobalState 的职责说明。"""
    def __init__(self, cfg: V29Config) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.cfg = cfg
        self.per_pair: Dict[str, PairState] = {}
        self.debt_pool: float = 0.0
        self.trade_risk_ledger: Dict[str, float] = {}
        self.pair_risk_open: Dict[str, float] = {}
        self.treasury = TreasuryState()

        self.bar_tick: int = 0
        self.current_cycle_ts: Optional[float] = None
        self.last_finalized_bar_ts: Optional[float] = None
        self.reported_pairs_for_current_cycle: set[str] = set()
        self.last_finalize_walltime: float = time.time()

    def get_pair_state(self, pair: str) -> PairState:
        """处理 get_pair_state 的主要逻辑。"""
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]

    def get_total_open_risk(self) -> float:
        """处理 get_total_open_risk 的主要逻辑。"""
        return sum(self.pair_risk_open.values())

    def get_dynamic_portfolio_cap_pct(self, equity: float) -> float:
        """处理 get_dynamic_portfolio_cap_pct 的主要逻辑。"""
        base = self.cfg.portfolio_cap_pct_base
        if equity <= 0:
            return base * 0.5
        if (self.debt_pool / equity) > self.cfg.drawdown_threshold_pct:
            return base * 0.5
        return base

    def per_pair_cap_room(self, pair: str, equity: float, tier_pol, reserved: float) -> float:
        """处理 per_pair_cap_room 的主要逻辑。"""
        cap = tier_pol.per_pair_risk_cap_pct * equity
        used = self.pair_risk_open.get(pair, 0.0) + reserved
        return max(0.0, cap - used)

    def record_trade_open(
        self,
        pair: str,
        trade_id: str,
        real_risk: float,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        bucket: str,
        entry_price: float,
        tier_pol,
    ) -> None:
        """处理 record_trade_open 的主要逻辑。"""
        pst = self.get_pair_state(pair)
        self.trade_risk_ledger[trade_id] = float(real_risk)
        self.pair_risk_open[pair] = self.pair_risk_open.get(pair, 0.0) + float(real_risk)

        icu_left = tier_pol.icu_force_exit_bars if tier_pol.icu_force_exit_bars > 0 else None
        pst.active_trades[trade_id] = ActiveTradeMeta(
            sl_pct=float(sl_pct),
            tp_pct=float(tp_pct),
            direction=str(direction),
            entry_bar_tick=self.bar_tick,
            entry_price=float(entry_price),
            bucket=str(bucket),
            icu_bars_left=icu_left,
        )
        pst.cooldown_bars_left = max(pst.cooldown_bars_left, tier_pol.cooldown_bars)

    def record_trade_close(self, pair: str, trade_id: str, profit_abs: float, tier_mgr) -> None:
        """处理 record_trade_close 的主要逻辑。"""
        pst = self.get_pair_state(pair)
        was_risk = self.trade_risk_ledger.pop(trade_id, 0.0)
        self.pair_risk_open[pair] = max(0.0, self.pair_risk_open.get(pair, 0.0) - was_risk)
        if self.pair_risk_open.get(pair, 0.0) <= 1e-12:
            self.pair_risk_open[pair] = 0.0
        pst.active_trades.pop(trade_id, None)

        if profit_abs >= 0:
            tax = profit_abs * self.cfg.tax_rate_on_wins
            self.debt_pool = max(0.0, self.debt_pool - tax)
            pst.local_loss = max(0.0, pst.local_loss - profit_abs)
            pst.closs = max(0, pst.closs - 1)
            pol = tier_mgr.get(pst.closs)
            pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars_after_win)
        else:
            loss = abs(profit_abs)
            pst.local_loss += loss
            self.debt_pool += loss
            pst.closs += 1
            pol = tier_mgr.get(pst.closs)
            pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars)

    def to_snapshot(self) -> Dict[str, Any]:
        """处理 to_snapshot 的主要逻辑。"""
        per_pair_snap: Dict[str, Any] = {}
        for pair, pst in self.per_pair.items():
            per_pair_snap[pair] = {
                "closs": pst.closs,
                "local_loss": pst.local_loss,
                "cooldown_bars_left": pst.cooldown_bars_left,
                "last_dir": pst.last_dir,
                "last_score": pst.last_score,
                "last_squad": pst.last_squad,
                "last_sl_pct": pst.last_sl_pct,
                "last_tp_pct": pst.last_tp_pct,
                "last_atr_pct": pst.last_atr_pct,
                "active_trades": {
                    tid: {
                        "sl_pct": meta.sl_pct,
                        "tp_pct": meta.tp_pct,
                        "direction": meta.direction,
                        "entry_bar_tick": meta.entry_bar_tick,
                        "entry_price": meta.entry_price,
                        "bucket": meta.bucket,
                        "icu_bars_left": meta.icu_bars_left,
                    }
                    for tid, meta in pst.active_trades.items()
                },
            }
        return {
            "debt_pool": self.debt_pool,
            "per_pair": per_pair_snap,
            "trade_risk_ledger": self.trade_risk_ledger,
            "pair_risk_open": self.pair_risk_open,
            "treasury": self.treasury.to_snapshot(),
            "bar_tick": self.bar_tick,
            "current_cycle_ts": self.current_cycle_ts,
            "last_finalized_bar_ts": self.last_finalized_bar_ts,
            "last_finalize_walltime": self.last_finalize_walltime,
        }

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        """处理 restore_snapshot 的主要逻辑。"""
        self.debt_pool = float(payload.get("debt_pool", 0.0))
        self.trade_risk_ledger = {k: float(v) for k, v in payload.get("trade_risk_ledger", {}).items()}
        self.pair_risk_open = {k: float(v) for k, v in payload.get("pair_risk_open", {}).items()}
        self.bar_tick = int(payload.get("bar_tick", 0))
        self.current_cycle_ts = payload.get("current_cycle_ts")
        self.last_finalized_bar_ts = payload.get("last_finalized_bar_ts")
        self.last_finalize_walltime = float(payload.get("last_finalize_walltime", time.time()))

        self.per_pair = {}
        for pair, pst_payload in payload.get("per_pair", {}).items():
            pst = PairState(
                closs=int(pst_payload.get("closs", 0)),
                local_loss=float(pst_payload.get("local_loss", 0.0)),
                cooldown_bars_left=int(pst_payload.get("cooldown_bars_left", 0)),
                last_dir=pst_payload.get("last_dir"),
                last_score=float(pst_payload.get("last_score", 0.0)),
                last_squad=pst_payload.get("last_squad"),
                last_sl_pct=float(pst_payload.get("last_sl_pct", 0.0)),
                last_tp_pct=float(pst_payload.get("last_tp_pct", 0.0)),
                last_atr_pct=float(pst_payload.get("last_atr_pct", 0.0)),
            )
            for tid, meta_payload in pst_payload.get("active_trades", {}).items():
                pst.active_trades[tid] = ActiveTradeMeta(
                    sl_pct=float(meta_payload.get("sl_pct", 0.0)),
                    tp_pct=float(meta_payload.get("tp_pct", 0.0)),
                    direction=str(meta_payload.get("direction", "")),
                    entry_bar_tick=int(meta_payload.get("entry_bar_tick", 0)),
                    entry_price=float(meta_payload.get("entry_price", 0.0)),
                    bucket=str(meta_payload.get("bucket", "slow")),
                    icu_bars_left=(
                        int(meta_payload["icu_bars_left"])
                        if meta_payload.get("icu_bars_left") is not None
                        else None
                    ),
                )
            self.per_pair[pair] = pst

        self.treasury.restore_snapshot(payload.get("treasury", {}))

    def reset_cycle_after_restore(self) -> None:
        """处理 reset_cycle_after_restore 的主要逻辑。"""
        self.current_cycle_ts = None
        self.reported_pairs_for_current_cycle = set()
        self.last_finalize_walltime = time.time()


class EquityProvider:
    """EquityProvider 的职责说明。"""
    def __init__(self, initial_equity: float) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.equity_current = float(initial_equity)

    def to_snapshot(self) -> Dict[str, float]:
        """处理 to_snapshot 的主要逻辑。"""
        return {"equity_current": self.equity_current}

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        """处理 restore_snapshot 的主要逻辑。"""
        self.equity_current = float(payload.get("equity_current", self.equity_current))

    def get_equity(self) -> float:
        """处理 get_equity 的主要逻辑。"""
        return self.equity_current

    def on_trade_closed_update(self, profit_abs: float) -> None:
        """处理 on_trade_closed_update 的主要逻辑。"""
        self.equity_current += float(profit_abs)


class TaxBrainV29(IStrategy):
    """TaxBrainV29 的职责说明。"""
    timeframe = V29Config().timeframe
    can_short = True
    startup_candle_count = V29Config().startup_candle_count

    minimal_roi = {"0": 10_000.0}
    stoploss = -0.99
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_buy_signals = False
    ignore_sell_signals = True

    def __init__(self, config: Dict[str, Any]) -> None:
        """处理 __init__ 的主要逻辑。"""
        super().__init__(config)
        base_cfg = V29Config()
        self.cfg = apply_overrides(base_cfg, config.get("strategy_params", {}))

        self.timeframe = self.cfg.timeframe
        self.startup_candle_count = self.cfg.startup_candle_count
        try:
            self.__class__.timeframe = self.cfg.timeframe
            self.__class__.startup_candle_count = self.cfg.startup_candle_count
        except Exception:
            pass

        print(f"[TaxBrainV29] timeframe sync cfg={self.cfg.timeframe} instance={self.timeframe} class={getattr(self.__class__, 'timeframe', 'NA')}")
        print(f"[TaxBrainV29] ADX_{self.cfg.adx_len} active")

        self.tier_mgr = TierManager()
        self.tier_agent = TierAgent()
        user_data_dir = Path(config.get("user_data_dir", "."))
        initial_equity = float(config.get("dry_run_wallet", self.cfg.dry_run_wallet_fallback))
        self.eq_provider = EquityProvider(initial_equity)
        self.state = GlobalState(self.cfg)

        self.analytics = AnalyticsAgent(user_data_dir / "logs")
        self.reservation = ReservationAgent(self.cfg, analytics=self.analytics)
        self.treasury_agent = TreasuryAgent(self.cfg, self.tier_mgr)
        self.risk_agent = RiskAgent(self.cfg, self.reservation, self.tier_mgr)
        state_file = (user_data_dir / "taxbrain_v29_state.json").resolve()
        self.persist = StateStore(
            filepath=str(state_file),
            state=self.state,
            eq_provider=self.eq_provider,
            reservation_agent=self.reservation,
        )

        self.cycle_agent = CycleAgent(
            cfg=self.cfg,
            state=self.state,
            reservation=self.reservation,
            treasury=self.treasury_agent,
            risk=self.risk_agent,
            analytics=self.analytics,
            persist=self.persist,
            tier_mgr=self.tier_mgr,
        )
        self.sizer = SizerAgent(self.state, self.reservation, self.eq_provider, self.cfg, self.tier_mgr)
        self.exit_policy = ExitPolicyV29(self.state, self.eq_provider, self.cfg)
        self.execution = ExecutionAgent(self.state, self.reservation, self.eq_provider, self.cfg)

        self._last_signal: Dict[str, Optional[Candidate]] = {}
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}
        self._tf_sec = self._tf_to_sec(self.cfg.timeframe)

    @staticmethod
    def _tf_to_sec(tf: str) -> int:
        """处理 _tf_to_sec 的主要逻辑。"""
        if tf.endswith("m"):
            return int(tf[:-1]) * 60
        if tf.endswith("h"):
            return int(tf[:-1]) * 3600
        if tf.endswith("d"):
            return int(tf[:-1]) * 86400
        return 300

    def bot_start(self, **kwargs) -> None:
        """处理 bot_start 的主要逻辑。"""
        self.persist.load_if_exists()
        if self.state.treasury.cycle_start_tick == 0:
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = self.eq_provider.get_equity()

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """处理 populate_indicators 的主要逻辑。"""
        pair = metadata["pair"]
        pst = self.state.get_pair_state(pair)
        df = compute_indicators(df, self.cfg)

        if self.cfg.adx_len != 14 and "adx" in df.columns:
            self.analytics.log_debug(
                "adx_dynamic",
                f"Using ADX_{self.cfg.adx_len}",
                {"pair": pair},
            )

        if len(df) == 0:
            return df
        last = df.iloc[-1]
        last_ts = float(pd.to_datetime(last.get("date", pd.Timestamp.utcnow())).timestamp())

        candidates = gen_candidates(last)
        policy = self.tier_mgr.get(pst.closs)
        best = self.tier_agent.filter_best(policy, candidates)
        if best:
            pst.last_dir = best.direction
            pst.last_score = best.expected_edge
            pst.last_squad = best.kind
            pst.last_sl_pct = best.sl_pct
            pst.last_tp_pct = best.tp_pct
            pst.last_atr_pct = float(last.get("atr_pct", 0.0))
        else:
            pst.last_dir = None
            pst.last_squad = None
            pst.last_score = 0.0
            pst.last_sl_pct = 0.0
            pst.last_tp_pct = 0.0
        self._last_signal[pair] = best

        try:
            whitelist = self.dp.current_whitelist()
        except Exception:
            whitelist = [pair]

        self.cycle_agent.maybe_finalize(
            pair=pair,
            bar_ts=last_ts,
            whitelist=whitelist,
            timeframe_sec=self._tf_sec,
            eq_provider=self.eq_provider,
        )
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """处理 populate_entry_trend 的主要逻辑。"""
        df["enter_long"] = 0
        df["enter_short"] = 0
        pair = metadata["pair"]
        sig = self._last_signal.get(pair)
        if sig and len(df) > 0:
            if sig.direction == "long":
                df.loc[df.index[-1], "enter_long"] = 1
            else:
                df.loc[df.index[-1], "enter_short"] = 1
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """处理 populate_exit_trend 的主要逻辑。"""
        df["exit_long"] = 0
        df["exit_short"] = 0
        return df


    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """处理 confirm_trade_entry 的主要逻辑。"""
        pst = self.state.get_pair_state(pair)
        if pst.cooldown_bars_left > 0:
            return False
        if len(pst.active_trades) > 0:
            return False
        sig = self._last_signal.get(pair)
        if not sig or sig.direction != side.lower():
            return False
        self._pending_entry_meta[pair] = {
            "sl_pct": sig.sl_pct,
            "tp_pct": sig.tp_pct,
            "dir": sig.direction,
        }
        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """处理 custom_stake_amount 的主要逻辑。"""
        if abs(leverage - self.cfg.enforce_leverage) > 1e-6:
            return 0.0
        meta = self._pending_entry_meta.get(pair)
        if not meta:
            return 0.0
        if "stake_final" in meta:
            return float(meta["stake_final"])

        sl = float(meta["sl_pct"])
        tp = float(meta["tp_pct"])
        stake, risk, bucket = self.sizer.compute(
            pair=pair,
            sl_pct=sl,
            tp_pct=tp,
            direction=meta["dir"],
            min_stake=min_stake,
            max_stake=max_stake,
        )
        if stake <= 0 or risk <= 0:
            return 0.0

        rid = f"{pair}:{bucket}:{uuid.uuid4().hex}"
        self.reservation.reserve(pair, rid, risk, bucket)
        meta.update(
            {
                "stake_final": stake,
                "risk_final": risk,
                "reservation_id": rid,
                "bucket": bucket,
                "entry_price": current_rate,
            }
        )
        return float(stake)

    def leverage(self, *args, **kwargs) -> float:
        """处理 leverage 的主要逻辑。"""
        return self.cfg.enforce_leverage

    def order_filled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """处理 order_filled 的主要逻辑。"""
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        is_open = bool(getattr(trade, "is_open", True))
        if is_open:
            meta = self._pending_entry_meta.get(pair)
            if not meta:
                return
            opened = self.execution.on_open_filled(pair, trade, order, meta, self.tier_mgr)
            if opened:
                self._pending_entry_meta.pop(pair, None)
                self.persist.save()
        else:
            closed = self.execution.on_close_filled(pair, trade, order, self.tier_mgr)
            if closed:
                self.analytics.log_exit(pair, trade_id, "close_filled")
                self.persist.save()

    def order_cancelled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """处理 order_cancelled 的主要逻辑。"""
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", meta.get("reservation_id") if meta else "NA"))) if trade else meta.get("reservation_id", "NA") if meta else "NA"
            self.analytics.log_exit(pair, trade_id, "order_cancelled")

    def order_rejected(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """处理 order_rejected 的主要逻辑。"""
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", meta.get("reservation_id") if meta else "NA"))) if trade else meta.get("reservation_id", "NA") if meta else "NA"
            self.analytics.log_exit(pair, trade_id, "order_rejected")

    def _handle_cancel_or_reject(self, pair: str) -> tuple[bool, Optional[dict[str, Any]]]:
        """处理 _handle_cancel_or_reject 的主要逻辑。"""
        meta = self._pending_entry_meta.get(pair)
        meta_snapshot = dict(meta) if meta else None
        rid = meta.get("reservation_id") if meta else None
        released = self.execution.on_cancel_or_reject(pair, rid)
        if released:
            self._pending_entry_meta.pop(pair, None)
            self.persist.save()
        return released, meta_snapshot

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """处理 custom_stoploss 的主要逻辑。"""
        pst = self.state.get_pair_state(pair)
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        meta = pst.active_trades.get(tid)

        sl_pct = None
        tp_pct = None
        atr_pct_hint = 0.0

        try:
            sl_pct = getattr(trade, "get_custom_data", lambda *_: None)("sl_pct")
            tp_pct = getattr(trade, "get_custom_data", lambda *_: None)("tp_pct")
        except Exception:
            pass
        try:
            if (sl_pct is None or sl_pct <= 0) and hasattr(trade, "user_data"):
                sl_pct = float(trade.user_data.get("sl_pct", 0.0))
            if (tp_pct is None or tp_pct <= 0) and hasattr(trade, "user_data"):
                tp_pct = float(trade.user_data.get("tp_pct", 0.0))
        except Exception:
            pass
        if (sl_pct is None or sl_pct <= 0) and meta:
            sl_pct = meta.sl_pct
        if (tp_pct is None or tp_pct <= 0) and meta:
            tp_pct = meta.tp_pct
        if meta:
            atr_pct_hint = pst.last_atr_pct

        if sl_pct is None or sl_pct <= 0:
            return -0.03

        if tp_pct and current_profit and current_profit > 0:
            try:
                if current_profit >= float(tp_pct) * float(self.cfg.breakeven_lock_frac_of_tp):
                    try:
                        analyzed = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                        pairdf = analyzed[0] if isinstance(analyzed, (list, tuple)) else analyzed
                        if len(pairdf) > 0 and "atr" in pairdf.columns:
                            atr_pct_hint = float(pairdf["atr"].iloc[-1] / max(pairdf["close"].iloc[-1], 1e-12))
                    except Exception:
                        pass
                    dist = self.exit_policy.early_lock_distance(trade, current_rate, current_profit, atr_pct_hint)
                    if dist is not None:
                        self.analytics.log_exit(pair, tid, "breakeven_lock")
                        return max(-abs(float(sl_pct)), dist)
            except Exception:
                pass

        return -abs(float(sl_pct))

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """处理 custom_exit 的主要逻辑。"""
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        reason = self.exit_policy.decide(
            pair, tid, float(current_profit) if current_profit is not None else None
        )
        if reason:
            self.analytics.log_exit(pair, tid, reason)
        return reason






