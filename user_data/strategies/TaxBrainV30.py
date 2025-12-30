# -*- coding: utf-8 -*-
"""TaxBrainV30: Tensor-Logic orchestrator (thin strategist)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import sys
import math

import pandas as pd

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

try:
    from freqtrade.enums import RunMode
except Exception:  # pragma: no cover
    RunMode = None

try:
    from freqtrade.strategy import IStrategy
except Exception:  # pragma: no cover
    class IStrategy:  # type: ignore
        def __init__(self, *args, **kwargs):
            if not hasattr(self, "dp"):
                self.dp = type("DP", (), {})()

from user_data.strategies.agents.exits import rules_threshold  # noqa: F401
from user_data.strategies.config.v30_config import V30Config, apply_overrides
from user_data.strategies.agents.portfolio.treasury import TreasuryAgent
from user_data.strategies.agents.portfolio.tier import TierAgent, TierManager
from user_data.strategies.agents.portfolio.sizer import SizerAgent
from user_data.strategies.agents.portfolio.risk import RiskAgent
from user_data.strategies.agents.portfolio.reservation import ReservationAgent
from user_data.strategies.agents.portfolio.persist import StateStore
from user_data.strategies.agents.portfolio.execution import ExecutionAgent
from user_data.strategies.agents.portfolio.cycle import CycleAgent
from user_data.strategies.agents.portfolio.analytics import AnalyticsAgent
from user_data.strategies.agents.portfolio.global_backend import (
    GlobalRiskBackend,
    LocalGlobalBackend,
    RedisGlobalBackend,
)
from user_data.strategies.agents.market.sensor import MarketSensor
from user_data.strategies.agents.exits.facade import ExitFacade
try:
    from user_data.strategies.agents.exits.router import EXIT_ROUTER, SLContext, TPContext
except Exception:  # pragma: no cover
    EXIT_ROUTER = None
    SLContext = None  # type: ignore
    TPContext = None  # type: ignore
from user_data.strategies.agents.exits.exit import ExitPolicyV30, ExitTags
from user_data.strategies.agents.signals import schemas
from user_data.strategies.core.engine import Engine, GlobalState
from user_data.strategies.core.bridge import ZeroCopyBridge
from user_data.strategies.core.hub import SignalHub
from user_data.strategies.core.rejections import RejectTracker, RejectReason
from user_data.strategies.vectorized.matrix_engine import MatrixEngine
from user_data.strategies.core import strategy_helpers as helpers


class _NoopStateStore:
    def save(self) -> None:  # pragma: no cover - trivial
        return

    def load_if_exists(self) -> None:  # pragma: no cover - trivial
        return


class EquityProvider:
    def __init__(self, initial_equity: float) -> None:
        self.equity_current = float(initial_equity)

    def to_snapshot(self) -> Dict[str, float]:
        return {"equity_current": self.equity_current}

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        self.equity_current = float(payload.get("equity_current", self.equity_current))

    def get_equity(self) -> float:
        return self.equity_current

    def on_trade_closed_update(self, profit_abs: float) -> None:
        self.equity_current += float(profit_abs)


class TaxBrainV30(IStrategy):
    timeframe = V30Config().system.timeframe
    can_short = True
    startup_candle_count = V30Config().system.startup_candle_count
    use_custom_roi = False
    use_custom_stoploss = False
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_buy_signals = False
    ignore_sell_signals = True
    _indicator_requirements_map: Dict[Optional[str], set[str]] = {}

    def __init__(self, config: Dict[str, Any]) -> None:
        base_cfg = V30Config()
        self.cfg = apply_overrides(base_cfg, config.get("strategy_params", {}))
        self.timeframe = self.cfg.system.timeframe
        self.startup_candle_count = self.cfg.system.startup_candle_count
        stoploss_val = getattr(self.cfg, "stoploss", None)
        try:
            stoploss_val = float(stoploss_val)
        except (TypeError, ValueError):
            stoploss_val = float(self.cfg.trading.sizing.enforce_leverage) * -0.2
        self.stoploss = stoploss_val
        minimal_roi = getattr(self.cfg, "minimal_roi", None)
        if isinstance(minimal_roi, dict):
            self.minimal_roi = dict(minimal_roi)
        else:
            self.minimal_roi = {"0": 0.50 * float(self.cfg.trading.sizing.enforce_leverage)}
        try:
            self.__class__.timeframe = self.cfg.system.timeframe
            self.__class__.startup_candle_count = self.cfg.system.startup_candle_count
        except Exception:
            pass
        self.hub = SignalHub(self.cfg)
        self.hub.discover()
        self._factor_requirements = self.hub.factor_requirements
        self._indicator_requirements = self.hub.indicator_requirements
        self.__class__._indicator_requirements_map = self._indicator_requirements
        config_timeframes = tuple(getattr(self.cfg, "informative_timeframes", ()))
        inferred_timeframes = tuple(tf for tf in self._factor_requirements.keys() if tf)
        self._informative_timeframes = helpers.derive_informative_timeframes(
            config_timeframes, inferred_timeframes, self.timeframe
        )
        helpers.register_informative_methods(self)
        self._enabled_signal_specs = self.hub.enabled_specs
        super().__init__(config)
        self.tier_mgr = TierManager(self.cfg)
        self.tier_agent = TierAgent()
        user_data_dir = Path(config.get("user_data_dir", "."))
        initial_equity = float(config.get("dry_run_wallet", self.cfg.system.dry_run_wallet_fallback))
        self.eq_provider = EquityProvider(initial_equity)

        runmode_cfg = config.get("runmode") if isinstance(config, dict) else None
        runmode_token = str(getattr(runmode_cfg, "value", runmode_cfg or "")).lower()
        force_local_backend = False
        if RunMode is not None and runmode_cfg in (getattr(RunMode, "HYPEROPT", None), getattr(RunMode, "BACKTEST", None)):
            force_local_backend = True
        elif runmode_token:
            force_local_backend = any(key in runmode_token for key in ("hyperopt", "backtest"))

        backend_mode = str(
            getattr(getattr(self.cfg, "system", None), "global_backend_mode", getattr(self.cfg, "global_backend_mode", "local"))
        ).lower()
        if force_local_backend:
            backend_mode = "local"
        if backend_mode == "redis":
            self.global_backend = RedisGlobalBackend(
                host=self.cfg.system.redis_host,
                port=self.cfg.system.redis_port,
                db=self.cfg.system.redis_db,
                password=None,
                namespace=self.cfg.system.redis_namespace,
            )
        elif backend_mode == "local":
            self.global_backend = LocalGlobalBackend()
        else:
            raise ValueError(f"Unknown global_backend_mode: {self.cfg.system.global_backend_mode}")

        sensor_cfg = getattr(self.cfg, "sensor", None)
        sensor_weights = dict(getattr(sensor_cfg, "weights", {}) or {})
        sensor_entropy = float(
            getattr(sensor_cfg, "entropy_factor", getattr(getattr(self.cfg, "risk", None), "entropy_factor", 0.4))
        )
        system_cfg = getattr(self.cfg, "system", None)
        sensor_strict = bool(getattr(system_cfg, "market_sensor_strict", False)) if system_cfg else False
        self.market_sensor = MarketSensor(
            self.global_backend,
            weights=sensor_weights,
            entropy_factor=sensor_entropy,
            strict=sensor_strict,
        )

        self.state = GlobalState(self.cfg, backend=self.global_backend)
        self.analytics = AnalyticsAgent(user_data_dir / "logs")
        self.reservation = ReservationAgent(self.cfg, analytics=self.analytics, backend=self.global_backend)
        self.treasury_agent = TreasuryAgent(self.cfg, self.tier_mgr, backend=self.global_backend)
        self.risk_agent = RiskAgent(self.cfg, self.reservation, self.tier_mgr, backend=self.global_backend)
        system_cfg = getattr(self.cfg, "system", None)
        self.rejections = RejectTracker(
            log_enabled=bool(getattr(system_cfg, "rejection_log_enabled", False)) if system_cfg else False,
            stats_enabled=bool(getattr(system_cfg, "rejection_stats_enabled", True)) if system_cfg else True,
        )
        self.exit_facade = ExitFacade(self.cfg, self.tier_mgr) if ExitFacade else None
        if self.exit_facade:
            self.exit_facade.attach_strategy(self)
            self.exit_facade.set_dataprovider(getattr(self, "dp", None))
        self.exit_policy = ExitPolicyV30(self.state, self.eq_provider, self.cfg, dp=getattr(self, "dp", None))
        self.exit_policy.set_strategy(self)
        state_file = (user_data_dir / "taxbrain_v30_state.json").resolve()
        self._runmode_token: str = self._compute_runmode_token()
        self._persist_enabled: bool = not self._is_backtest_like_runmode()
        self.persist = StateStore(
            filepath=str(state_file),
            state=self.state,
            eq_provider=self.eq_provider,
            reservation_agent=self.reservation,
        )
        if not self._persist_enabled:
            self.persist = _NoopStateStore()
        self.cycle_agent = CycleAgent(
            cfg=self.cfg,
            state=self.state,
            reservation=self.reservation,
            treasury=self.treasury_agent,
            risk=self.risk_agent,
            analytics=self.analytics,
            persist=self.persist,
            tier_mgr=self.tier_mgr,
            backend=self.global_backend,
        )
        self.sizer = SizerAgent(
            self.state, self.reservation, self.eq_provider, self.cfg, self.tier_mgr, backend=self.global_backend
        )
        self.execution = ExecutionAgent(self.state, self.reservation, self.eq_provider, self.cfg)
        self._last_signal: Dict[str, Optional[schemas.Candidate]] = {}
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}
        self._tf_sec = helpers.tf_to_sec(self.cfg.system.timeframe)
        self._candidate_pool_limit = int(max(1, getattr(self.cfg, "candidate_pool_max_per_side", 4)))
        self.bridge = ZeroCopyBridge(self)
        self.matrix_engine = MatrixEngine(self, self.hub, self.bridge)
        self.engine = Engine(
            cfg=self.cfg,
            state=self.state,
            eq_provider=self.eq_provider,
            treasury_agent=self.treasury_agent,
            reservation=self.reservation,
            risk_agent=self.risk_agent,
            analytics=self.analytics,
            persist=self.persist,
            tier_mgr=self.tier_mgr,
            tf_sec=self._tf_sec,
            is_backtest_like=self._is_backtest_like_runmode,
            rejections=self.rejections,
        )
        self.cycle_agent.engine = self.engine

    def set_dataprovider(self, dp):
        try:
            super().set_dataprovider(dp)
        except AttributeError:
            pass
        self.dp = dp
        if hasattr(self, "exit_policy") and self.exit_policy is not None:
            self.exit_policy.dp = dp
        if hasattr(self, "sizer"):
            self.sizer.set_dataprovider(dp)
        if self.exit_facade:
            self.exit_facade.set_dataprovider(dp)

    def _compute_runmode_token(self) -> str:
        cfg_mode = self.config.get("runmode") if hasattr(self, "config") else None
        if cfg_mode is not None:
            return str(getattr(cfg_mode, "value", cfg_mode)).lower()
        strategy_mode = getattr(self, "runmode", None)
        if strategy_mode is not None:
            return str(getattr(strategy_mode, "value", strategy_mode)).lower()
        dp_mode = getattr(getattr(self, "dp", None), "runmode", None)
        if dp_mode is not None:
            return str(getattr(dp_mode, "value", dp_mode)).lower()
        return ""

    def _ensure_runmode_token(self) -> str:
        if not self._runmode_token:
            self._runmode_token = self._compute_runmode_token()
        return self._runmode_token

    def _is_backtest_like_runmode(self) -> bool:
        token = self._ensure_runmode_token() or ""
        return any(key in token for key in ("backtest", "hyperopt"))

    def _prepare_informative_frame(self, frame: Optional[pd.DataFrame], timeframe: str) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty:
            return frame
        return frame

    @staticmethod
    def _derived_factor_columns_missing(df: pd.DataFrame, timeframes: Iterable[Optional[str]]) -> bool:
        return helpers.derived_factor_columns_missing(df, timeframes)

    def informative_pairs(self):
        try:
            whitelist = self.dp.current_whitelist()
        except Exception:
            whitelist = self.config.get("exchange", {}).get("pair_whitelist", [])
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        if whitelist and self._informative_timeframes:
            for pair in whitelist:
                for tf in self._informative_timeframes:
                    key = (pair, tf)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
        system_cfg = getattr(self.cfg, "system", None)
        sensor_enabled = bool(getattr(system_cfg, "market_sensor_enabled", True)) if system_cfg else True
        sensor_backtest = bool(getattr(system_cfg, "market_sensor_in_backtest", False)) if system_cfg else False
        if sensor_enabled and (not self._is_backtest_like_runmode() or sensor_backtest):
            sensor_pairs = getattr(system_cfg, "sensor_pairs", ("BTC/USDT", "ETH/USDT")) if system_cfg else ()
            for pair in sensor_pairs:
                sp = (pair, self.timeframe)
                if sp not in seen:
                    seen.add(sp)
                    pairs.append(sp)
        return pairs

    def get_informative_row(self, pair: str, timeframe: str) -> Optional[pd.Series]:
        return self.bridge.get_informative_row(pair, timeframe)

    def get_informative_value(
        self, pair: str, timeframe: str, column: str, default: Optional[float] = None
    ) -> Optional[float]:
        row = self.get_informative_row(pair, timeframe)
        if row is None:
            return default
        try:
            if column in row:
                return row[column]
        except Exception:
            pass
        suffix = helpers.timeframe_suffix_token(timeframe)
        if suffix:
            alt_col = f"{column}_{suffix}"
            try:
                if alt_col in row:
                    return row[alt_col]
            except Exception:
                pass
        try:
            return row.get(column, default)
        except Exception:
            return default

    def _update_last_signal(self, pair: str, candidate: Optional[schemas.Candidate], row: pd.Series) -> None:
        helpers.update_last_signal(self, pair, candidate, row)

    def _reserve_backend_risk(self, pair: str, risk: float) -> bool:
        return helpers.reserve_backend_risk(self, pair, risk)

    def _reserve_risk_resources(
        self,
        pair: str,
        stake: float,
        risk: float,
        bucket: str,
        sl: float,
        tp: float,
        direction: str,
        current_rate: float,
        meta: Dict[str, Any],
    ) -> bool:
        return helpers.reserve_risk_resources(
            self,
            pair=pair,
            stake=stake,
            risk=risk,
            bucket=bucket,
            sl=sl,
            tp=tp,
            direction=direction,
            current_rate=current_rate,
            meta=meta,
        )

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        return self.matrix_engine.inject_features(df, pair)

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
            sig = pd.to_numeric(df["_signal_id"], errors="coerce")
            mask = sig.notna() & (sig > 0)
            df["enter_tag"] = None
            df.loc[mask, "enter_tag"] = sig.loc[mask].astype(int).astype(str)

        last_idx = df.index[-1]
        last_row = df.iloc[-1]
        current_time = last_row.get("date", None)
        if current_time is None:
            current_time = last_idx if isinstance(last_idx, (pd.Timestamp, datetime)) else datetime.utcnow()

        meta = self.bridge.get_row_meta(pair, current_time)
        candidate = helpers.candidate_from_meta(self, pair, meta) if meta else None
        self._update_last_signal(pair, candidate, last_row)
        try:
            if candidate and getattr(self, "global_backend", None):
                self.global_backend.record_signal_score(pair, float(getattr(candidate, "expected_edge", 0.0)))
        except Exception:
            pass
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if "exit_long" not in df.columns:
            df["exit_long"] = 0
        if "exit_short" not in df.columns:
            df["exit_short"] = 0
        if "exit_tag" not in df.columns:
            df["exit_tag"] = None
        meta = {
            "pair": metadata.get("pair"),
            "timeframe": getattr(self, "timeframe", None) or self.timeframe,
        }
        try:
            if self.exit_facade:
                df = self.exit_facade.apply_vector(
                    df=df,
                    metadata=meta,
                    state=self.state,
                    timeframe_col=None,
                )
            elif EXIT_ROUTER is not None:
                df = EXIT_ROUTER.apply_vector_exits(
                    df=df,
                    metadata=meta,
                    dp=self.dp,
                    cfg=self.cfg,
                    state=self.state,
                    timeframe_col=None,
                )
        except Exception:
            pass
        return df

    def confirm_trade_exit(
        self,
        pair: str,
        trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime | None = None,
        **kwargs,
    ) -> bool:
        return True

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
        result = self.bridge.get_side_meta(pair, current_time, side, entry_tag)
        if not result:
            helpers.update_rejection(self, RejectReason.NO_CANDIDATE, pair, {"side": side, "entry_tag": entry_tag})
            return False
        meta, meta_info = result

        context: Dict[str, Any] = {
            "side": side,
            "time": current_time,
            "score": float(meta.get("expected_edge", 0.0) or 0.0),
            "kind": meta_info.name,
            "recipe": meta_info.recipe,
            "squad": meta_info.squad,
        }
        return self.engine.is_permitted(pair, context)

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
        result = self.bridge.get_side_meta(pair, current_time, side, entry_tag)
        if not result:
            helpers.update_rejection(self, RejectReason.NO_CANDIDATE, pair, {"side": side, "entry_tag": entry_tag})
            return 0.0
        meta, meta_info = result

        if self._is_backtest_like_runmode():
            pst = self.state.get_pair_state(pair)
            pst.last_score = float(meta.get("expected_edge", 0.0) or 0.0)
            pst.last_dir = meta_info.direction
            pst.last_sl_pct = float(meta.get("sl_pct", 0.0) or 0.0)
            pst.last_tp_pct = float(meta.get("tp_pct", 0.0) or 0.0)
            pst.last_kind = str(meta_info.name)
            pst.last_squad = str(meta_info.squad)
            pst.last_exit_profile = meta_info.exit_profile
            pst.last_recipe = meta_info.recipe
            pst.last_atr_pct = float(meta.get("plan_atr_pct", 0.0) or 0.0)

        self.engine.sync_to_time(current_time)
        context = {
            "side": side,
            "time": current_time,
            "score": float(meta.get("expected_edge", 0.0) or 0.0),
            "kind": meta_info.name,
            "recipe": meta_info.recipe,
            "squad": meta_info.squad,
        }
        if not self.engine.is_permitted(pair, context):
            return 0.0

        stake, risk, bucket = self.sizer.compute(
            pair=pair,
            sl_pct=float(meta.get("sl_pct", 0.0) or 0.0),
            tp_pct=float(meta.get("tp_pct", 0.0) or 0.0),
            direction=str(meta_info.direction),
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=leverage,
            plan_atr_pct=meta.get("plan_atr_pct"),
            exit_profile=meta_info.exit_profile,
            bucket=None,
            current_rate=current_rate,
            score=float(meta.get("expected_edge", 0.0) or 0.0),
        )
        if stake <= 0 or risk <= 0:
            helpers.update_rejection(self, RejectReason.SIZER, pair, {"side": side})
            return 0.0

        meta_payload = {
            "dir": meta_info.direction,
            "kind": meta_info.name,
            "squad": meta_info.squad,
            "sl_pct": meta.get("sl_pct"),
            "tp_pct": meta.get("tp_pct"),
            "exit_profile": meta_info.exit_profile,
            "recipe": meta_info.recipe,
            "plan_timeframe": meta_info.plan_timeframe,
            "atr_pct": meta.get("plan_atr_pct"),
            "expected_edge": meta.get("expected_edge"),
            "raw_score": meta.get("raw_score"),
            "score": meta.get("expected_edge"),
        }
        reserved = self._reserve_risk_resources(
            pair=pair,
            stake=stake,
            risk=risk,
            bucket=bucket,
            sl=float(meta.get("sl_pct", 0.0) or 0.0),
            tp=float(meta.get("tp_pct", 0.0) or 0.0),
            direction=str(meta_info.direction),
            current_rate=current_rate,
            meta=meta_payload,
        )
        if not reserved:
            helpers.update_rejection(self, RejectReason.RESERVATION, pair, {"side": side})
            return 0.0
        return float(stake)

    def leverage(self, *args, **kwargs) -> float:
        """Return configured leverage multiplier."""
        return self.cfg.trading.sizing.enforce_leverage

    def bot_start(self, **kwargs) -> None:
        self._runmode_token = self._compute_runmode_token()
        backtest_like = self._is_backtest_like_runmode()
        self._persist_enabled = not backtest_like
        try:
            self.logger.info(
                f"[tier] runmode={self._runmode_token or 'unknown'} backtest_like={backtest_like} persist={'on' if self._persist_enabled else 'off'}"
            )
        except Exception:
            pass
        if self._persist_enabled:
            self.persist.load_if_exists()
        else:
            self.persist = _NoopStateStore()
            self.cycle_agent.persist = self.persist
        if self.state.treasury.cycle_start_tick == 0:
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = self.eq_provider.get_equity()

    def order_filled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """Handle entry/exit fills, update risk ledgers and persistence."""
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        side = getattr(order, "ft_order_side", None)

        is_entry = side == getattr(trade, "entry_side", None)
        is_exit = side == getattr(trade, "exit_side", None)

        if is_entry:
            meta = self._pending_entry_meta.pop(pair, None)
            opened = self.execution.on_open_filled(
                pair=pair,
                trade=trade,
                order=order,
                pending_meta=meta,
                tier_mgr=self.tier_mgr,
            )
            if opened:
                try:
                    self.state.get_pair_state(pair)._cooldown_last_ts = float(current_time.timestamp())
                except Exception:
                    pass
            if opened and self._persist_enabled:
                self.persist.save()
        elif is_exit:
            processed = getattr(self, "_bt_closed_trades", None)
            if processed is None:
                processed = set()
                self._bt_closed_trades = processed

            if self._is_backtest_like_runmode() and trade_id in processed:
                return

            if self._is_backtest_like_runmode() and current_time:
                self.engine.sync_to_time(current_time)

            profit_abs = getattr(trade, "close_profit_abs", 0.0)
            if profit_abs is None or profit_abs == 0.0:
                fill_price = getattr(order, "average", getattr(order, "price", getattr(trade, "close_rate", 0.0)))
                profit_abs = trade.calc_profit(fill_price)

            if profit_abs != 0:
                try:
                    trade.close_profit_abs = profit_abs
                    trade.profit_abs = profit_abs
                except Exception:
                    pass

            closed = self.execution.on_close_filled(
                pair=pair,
                trade=trade,
                order=order,
                tier_mgr=self.tier_mgr,
            )

            if closed:
                try:
                    self.state.get_pair_state(pair)._cooldown_last_ts = float(current_time.timestamp())
                except Exception:
                    pass

                reason = getattr(trade, "exit_reason", "unknown_fill")
                fill_price = getattr(order, "average", getattr(order, "price", getattr(trade, "close_rate", 0.0)))

                self._log_full_exit_state(
                    pair=pair,
                    trade=trade,
                    reason=reason,
                    pnl_abs=profit_abs,
                    exit_rate=fill_price,
                    exit_time=current_time,
                )

            if self._is_backtest_like_runmode():
                processed.add(trade_id)

            if closed and self._persist_enabled:
                self.persist.save()
        else:
            return

    def order_cancelled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = (
                str(getattr(trade, "trade_id", getattr(trade, "id", meta.get("reservation_id") if meta else "NA")))
                if trade
                else meta.get("reservation_id", "NA") if meta else "NA"
            )
            tag = ExitTags.ORDER_CANCELLED if ExitTags else "order_cancelled"
            self.analytics.log_exit(pair, trade_id, tag)

    def order_rejected(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = (
                str(getattr(trade, "trade_id", getattr(trade, "id", meta.get("reservation_id") if meta else "NA")))
                if trade
                else meta.get("reservation_id", "NA") if meta else "NA"
            )
            tag = ExitTags.ORDER_REJECTED if ExitTags else "order_rejected"
            self.analytics.log_exit(pair, trade_id, tag)

    def _handle_cancel_or_reject(self, pair: str) -> tuple[bool, Optional[dict[str, Any]]]:
        meta = self._pending_entry_meta.get(pair)
        meta_snapshot = dict(meta) if meta else None
        rid = meta.get("reservation_id") if meta else None
        released = self.execution.on_cancel_or_reject(pair, rid)
        if released:
            self._pending_entry_meta.pop(pair, None)
            if self._persist_enabled:
                self.persist.save()
        return released, meta_snapshot

    @staticmethod
    def _apply_leverage_pct(pct: Optional[float], trade) -> Optional[float]:
        if pct is None:
            return None
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            return None
        if pct <= 0:
            return None

        lev = getattr(trade, "leverage", 1.0) or 1.0
        try:
            lev = float(lev)
        except (TypeError, ValueError):
            lev = 1.0
        if lev <= 0:
            lev = 1.0

        return pct * lev

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
        if EXIT_ROUTER is None or SLContext is None:
            return self.stoploss
        try:
            ctx = SLContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=float(current_profit or 0.0),
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            sl_pct = EXIT_ROUTER.sl_best(ctx, base_sl_pct=None)
        except Exception:
            sl_pct = None

        sl_pct = self._apply_leverage_pct(sl_pct, trade)
        if sl_pct is None or sl_pct <= 0:
            return self.stoploss
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
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        reason = self.exit_policy.decide(
            pair,
            tid,
            float(current_profit) if current_profit is not None else None,
            trade=trade,
        )
        if not reason:
            reason = self._atr_entry_exit_reason(
                pair,
                trade,
                current_time,
                current_rate,
                current_profit,
            )
        if not reason:
            return None
        try:
            if hasattr(trade, "exit_reason"):
                setattr(trade, "exit_reason", reason)
        except Exception:
            pass
        try:
            if hasattr(trade, "set_custom_data"):
                trade.set_custom_data("exit_tag", reason)
        except Exception:
            pass
        try:
            self.analytics.log_exit(pair, tid, reason)
        except Exception:
            pass
        return reason

    def custom_roi(
        self,
        pair: str,
        trade,
        current_time: datetime,
        trade_duration: int,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> Optional[float]:
        if EXIT_ROUTER is None or TPContext is None:
            return None
        try:
            ctx = TPContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=0.0,
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            tp_pct = EXIT_ROUTER.tp_best(ctx, base_tp_pct=None)
        except Exception:
            tp_pct = None

        tp_pct = self._apply_leverage_pct(tp_pct, trade)
        if tp_pct is None or tp_pct <= 0:
            return None
        return float(tp_pct)

    def _router_sl_tp_pct(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_profit: float,
    ) -> tuple[Optional[float], Optional[float]]:
        if EXIT_ROUTER is None or SLContext is None or TPContext is None:
            return None, None
        try:
            sl_ctx = SLContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=float(current_profit or 0.0),
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            sl_pct = EXIT_ROUTER.sl_best(sl_ctx, base_sl_pct=None)
        except Exception:
            sl_pct = None
        try:
            tp_ctx = TPContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=float(current_profit or 0.0),
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            tp_pct = EXIT_ROUTER.tp_best(tp_ctx, base_tp_pct=None)
        except Exception:
            tp_pct = None
        return (
            float(sl_pct) if sl_pct and sl_pct > 0 else None,
            float(tp_pct) if tp_pct and tp_pct > 0 else None,
        )

    def _get_current_atr_abs(
        self,
        pair: str,
        current_time: datetime,
        profile: Optional[Any] = None,
    ) -> Optional[float]:
        dp = getattr(self, "dp", None)
        if dp is None:
            return None
        atr_tf = None
        if profile is not None:
            atr_tf = getattr(profile, "atr_timeframe", None)
        if not atr_tf:
            atr_tf = getattr(self.cfg, "timeframe", None) or self.timeframe
        try:
            analyzed = dp.get_analyzed_dataframe(pair, atr_tf)
        except Exception:
            return None
        df = analyzed[0] if isinstance(analyzed, (list, tuple)) else analyzed
        if df is None or df.empty:
            return None
        try:
            upto = df.loc[:current_time] if current_time is not None else df
        except Exception:
            try:
                ct = current_time.replace(tzinfo=None) if getattr(current_time, "tzinfo", None) else current_time
                upto = df.loc[:ct]
            except Exception:
                upto = df
        if upto is None or upto.empty:
            return None
        row = upto.iloc[-1]
        try:
            if "atr" in row.index:
                atr_val = float(row["atr"])
                if math.isfinite(atr_val) and atr_val > 0:
                    return atr_val
        except Exception:
            pass
        try:
            atr_pct = float(row.get("atr_pct", 0.0))
            close = float(row.get("close", 0.0))
            if atr_pct > 0 and close > 0:
                atr_val = atr_pct * close
                if math.isfinite(atr_val) and atr_val > 0:
                    return atr_val
        except Exception:
            return None
        return None

    def _atr_entry_exit_reason(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> Optional[str]:
        if self.exit_facade is None:
            return None
        try:
            _, profile_def, _ = self.exit_facade.resolve_trade_plan(pair, trade, current_time)
        except Exception:
            profile_def = None
        if not profile_def:
            return None
        k_sl = float(getattr(profile_def, "atr_mul_sl", 0.0) or 0.0)
        k_tp = getattr(profile_def, "atr_mul_tp", None)
        if k_tp is None or k_tp <= 0:
            k_tp = max(k_sl * 2.0, 0.0)
        k_tp = float(k_tp)
        atr_abs = self._get_current_atr_abs(pair, current_time, profile_def)
        if atr_abs is None or atr_abs <= 0:
            return None
        entry = float(trade.open_rate)
        price = float(current_rate)
        is_short = bool(getattr(trade, "is_short", False))
        sl_abs = k_sl * atr_abs if k_sl > 0 else None
        tp_abs = k_tp * atr_abs if k_tp > 0 else None
        if not is_short:
            if sl_abs is not None:
                sl_price = entry - sl_abs
                if price <= sl_price:
                    return "atr_entry_sl"
            if tp_abs is not None:
                tp_price = entry + tp_abs
                if price >= tp_price:
                    return "atr_entry_tp"
        else:
            if sl_abs is not None:
                sl_price = entry + sl_abs
                if price >= sl_price:
                    return "atr_entry_sl"
            if tp_abs is not None:
                tp_price = entry - tp_abs
                if price <= tp_price:
                    return "atr_entry_tp"
        return None

    def _log_full_exit_state(
        self,
        pair: str,
        trade,
        reason: str,
        pnl_abs: float,
        exit_rate: float | None = None,
        exit_time: datetime | None = None,
    ):
        try:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
            pst = self.state.get_pair_state(pair)
            tier_pol = self.tier_mgr.get(pst.closs)
            tier_name = getattr(tier_pol, "name", "unknown")

            entry_price = getattr(trade, "open_rate", 0.0)
            amount = getattr(trade, "amount", 0.0)
            stake_amount = getattr(trade, "stake_amount", 0.0)
            open_date = getattr(trade, "open_date", None)

            final_exit_price = exit_rate if exit_rate is not None else getattr(trade, "close_rate", 0.0)
            final_exit_time = exit_time if exit_time is not None else getattr(trade, "close_date", None)

            is_short = getattr(trade, "is_short", False)
            k_val = self.state.treasury.k_short if is_short else self.state.treasury.k_long

            backend_debt = 0.0
            if self.state.backend:
                try:
                    backend_debt = float(self.state.backend.get_snapshot().debt_pool)
                except Exception:
                    pass

            details = {
                "pnl_abs": pnl_abs,
                "tier": tier_name,
                "closs": pst.closs,
                "local_loss": pst.local_loss,
                "central_debt_local": self.state.debt_pool,
                "central_debt_backend": backend_debt,
                "score": pst.last_score,
                "k_val": k_val,
                "cooldown": pst.cooldown_bars_left,
                "entry_price": entry_price,
                "exit_price": final_exit_price,
                "amount": amount,
                "stake": stake_amount,
                "open_at": str(open_date) if open_date else "",
                "close_at": str(final_exit_time) if final_exit_time else "",
                "duration": str(final_exit_time - open_date) if (final_exit_time and open_date) else "",
            }
            self.analytics.log_exit(pair, trade_id, reason, **details)
        except Exception as exc:
            print(f"Log exit state failed: {exc}")
