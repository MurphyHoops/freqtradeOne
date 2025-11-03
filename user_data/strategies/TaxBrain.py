# ICUBrainV26.py
#
# V26: 多币 ICU 风控大脑策略
#
# 关键特性：
# - Tier 医院分级: closs(伤情) -> TierPolicy 决定允许信号/冷却/ICU时限/ sizing 模式
# - 疼痛池 global_loss_bucket: 全局血压，用于 risk_off 和 组合降档
# - allocator 分诊台: 谁又疼又值得救 -> 分配急救风险预算
# - PositionSizerV26: 基于 Tier + allocator + VaR cap + 回血需求来决定仓位
# - finalize_bar_cycle(): 每根真实K线推进一次“时间”，递减冷却/ICU计时，自愈疼痛池，重算 allocator
#   并立刻持久化 -> 断电重启不丢 ICU 倒计时 / 冷却剩余时间
# - 风险预约 (reservation): 在 custom_stake_amount 先占用 VaR 配额，避免并发下单时双花组合风险
# - order_filled(): 实际成交后把预约的风险记入在市风险，并在平仓时把盈利/亏损回流到 closs / 疼痛池
# - custom_stoploss(): 多层兜底，防止重启后老仓裸奔 -99%
#
# 重要假设：
#   * 我们要求 leverage == 1.0
#     因为 exit_policy / stoploss / TP 逻辑全部默认 “价格变化 ~= 当前盈利比率”。
#     如果你要 >1x 杠杆，必须把这些逻辑同步放大/缩小。
#
# freqtrade回调对齐 (基于 2025-05-18 coveralls 上 freqtrade/strategy/interface.py):
# - confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs) -> bool
#   side 是 "long"/"short"，不是 "buy"/"sell"。:contentReference[oaicite:13]{index=13}
# - custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
#       min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float
#   我们会计算仓位，做风险预约，返回最终下单金额。:contentReference[oaicite:14]{index=14}
# - order_filled(self, pair, trade, order, current_time, **kwargs) -> None
#   每次订单完全成交都会调用（进场/出场/止损/强平/翻向），
#   我们在这里登记开仓、注销平仓，更新全局风险/疼痛池/ICU状态，再持久化。:contentReference[oaicite:15]{index=15}
# - custom_stoploss(self, pair, trade, current_time, current_rate, current_profit,
#       after_fill, **kwargs) -> float | None
#   current_profit 是 ratio (0.05=+5%)，
#   我们返回一个负数距离，例如 -0.02=止损在现价下方2%。
#   多层兜底，防止重启后老仓走全局-0.99。:contentReference[oaicite:16]{index=16}
#
# - custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> str | None
#   我们用 ExitPolicy.decide() 来给理由："tp_hit","early_lock","icu_timeout","flip_long","risk_off",...
#
# 注意：本策略使用 pandas_ta 计算技术指标（ema/rsi/atr/adx）。请确保环境已安装 pandas_ta。


from freqtrade.strategy import IStrategy
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import math
import os
import json
import time
import uuid
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Tier / Candidate 建模
# ---------------------------------------------------------------------------

@dataclass
class TierPolicy:
    name: str
    allowed_kinds: set[str]
    min_raw_score: float
    min_rr_ratio: float
    min_edge: float
    sizing_algo: str  # "BASELINE" or "TARGET_RECOVERY"
    k_mult_base_pct: float        # baseline 风险预算系数 (equity * k_mult_base_pct)
    recovery_factor: float        # 回血时想要追回 local_loss 的倍率
    local_recovery_tries: int     # 还能打几针回血单
    cooldown_bars: int
    cooldown_bars_after_win: int
    per_pair_risk_cap_pct: float  # 单币 VaR 上限 (相对 equity)
    max_stake_notional_pct: float # 单笔单子的最大名义金额 (相对 equity)
    fast_multiplier_cap: float    # allocator 给你的“加速”最多放大到 baseline 的多少倍
    icu_force_exit_bars: int      # 这笔仓位的 ICU 倒计时，<=0时强平；None/0 代表不用 ICU 倒计时硬平


class TierManager:
    """
    管理 closs -> TierPolicy 的映射。
    closs 低 = 健康；closs 中等 = 受伤准备回血；closs 高 = ICU，极度保守，仓位小 + 倒计时强平。
    """

    def __init__(self):
        # 你可以根据策略口味微调这些参数
        # Tier0: 健康。允许探索性的 setup，快速试错。
        self._tier0 = TierPolicy(
            name="T0_healthy",
            allowed_kinds={"mean_rev_long", "pullback_long", "trend_short"},
            min_raw_score=0.10,
            min_rr_ratio=1.2,
            min_edge=0.002,  # ~0.2% 期望edge
            sizing_algo="BASELINE",
            k_mult_base_pct=0.005,     # baseline: 愿意冒 0.5% equity 的 VaR
            recovery_factor=1.0,
            local_recovery_tries=1,    # V26: 给健康币 1 次试错，不是 0
            cooldown_bars=5,
            cooldown_bars_after_win=2,
            per_pair_risk_cap_pct=0.03,     # 单币最多 3% VaR
            max_stake_notional_pct=0.15,    # 单笔最多动用 15% equity 名义
            fast_multiplier_cap=2.0,
            icu_force_exit_bars=0,     # 健康仓不强制 ICU 倒计时
        )

        # Tier1-2: 受伤，回血阶段。允许的 setup 收窄，侧重高置信度回血刀法。
        self._tier12 = TierPolicy(
            name="T12_recovery",
            allowed_kinds={"pullback_long", "trend_short"},  # 砍掉最激进的 mean_rev_long
            min_raw_score=0.15,
            min_rr_ratio=1.4,
            min_edge=0.003,   # 0.3% edge
            sizing_algo="TARGET_RECOVERY",
            k_mult_base_pct=0.004,     # baseline稍微小点
            recovery_factor=1.5,       # 想追回 local_loss 的 1.5 倍
            local_recovery_tries=2,
            cooldown_bars=10,
            cooldown_bars_after_win=4,
            per_pair_risk_cap_pct=0.02,   # 单币 VaR 上限 2%
            max_stake_notional_pct=0.12,
            fast_multiplier_cap=1.5,
            icu_force_exit_bars=30,    # 30 bars 内必须解决掉
        )

        # Tier3+: ICU。极度保守，小剂量回血 + 硬性倒计时。
        self._tier3p = TierPolicy(
            name="T3p_ICU",
            allowed_kinds={"trend_short", "pullback_long"},  # 甚至更收敛
            min_raw_score=0.2,
            min_rr_ratio=1.6,
            min_edge=0.004,    # 0.4%
            sizing_algo="TARGET_RECOVERY",
            k_mult_base_pct=0.003,     # baseline 更小
            recovery_factor=2.0,
            local_recovery_tries=3,
            cooldown_bars=20,
            cooldown_bars_after_win=6,
            per_pair_risk_cap_pct=0.01,
            max_stake_notional_pct=0.10,
            fast_multiplier_cap=1.0,   # ICU 不允许加速膨胀
            icu_force_exit_bars=20,    # ICU仓 20 bars 必须退出
        )

    def get_tier_policy(self, closs: int) -> TierPolicy:
        if closs <= 0:
            return self._tier0
        elif 1 <= closs <= 2:
            return self._tier12
        else:
            return self._tier3p


@dataclass
class Candidate:
    direction: str    # "long" or "short"
    kind: str         # 哪种 setup: "mean_rev_long" / "pullback_long" / "trend_short"
    sl_pct: float     # 止损距离(正值), 例如0.01=1%
    tp_pct: float     # 目标收益(正值)
    raw_score: float  # 信号质量原始分
    rr_ratio: float   # tp_pct/sl_pct
    win_prob: float   # 粗略胜率估计
    expected_edge: float  # 期望edge: win_prob*tp - (1-win_prob)*sl


# ---------------------------------------------------------------------------
# 2. 全局状态 / PairState / 风险账本 / 持久化
# ---------------------------------------------------------------------------

@dataclass
class ActiveTradeMeta:
    sl_pct: float
    tp_pct: float
    direction: str            # "long" / "short"
    icu_bars_left: Optional[int]
    entry_bar_tick: int       # bar_tick when opened


@dataclass
class PairState:
    # 伤情状态
    closs: int = 0
    local_loss: float = 0.0
    local_tries_left: int = 0
    cooldown_bars_left: int = 0

    # 最近一根K线看到的最佳交易方向/评分 (allocator用)
    last_dir: Optional[str] = None
    last_score: float = 0.0

    # 当前活跃仓位
    active_trades: Dict[str, ActiveTradeMeta] = field(default_factory=dict)


class EquityProvider:
    """
    我们自己的权益跟踪器。我们把真实 realized PnL 累加进来，
    并持久化在 state 文件里，确保重启后 sizing 还知道账户大概多少。
    """
    def __init__(self, initial_equity: float = 1000.0):
        self.equity_current = float(initial_equity)

    def to_snapshot(self) -> Dict[str, Any]:
        return {"equity_current": self.equity_current}

    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        self.equity_current = float(snap.get("equity_current", self.equity_current))

    def get_equity(self) -> float:
        return self.equity_current

    def on_trade_closed_update(self, profit_abs: float) -> None:
        self.equity_current += float(profit_abs)


class GlobalState:
    """
    这是整个 ICU 大脑的内存 + 生命体征账本。
    - per_pair[pair] : PairState
    - global_loss_bucket : 全局疼痛池(血压)
    - pair_risk_open[pair] : 该币当前挂着的真实 VaR 敞口
    - trade_risk_ledger[trade_id] : 每个活跃仓位的真实 VaR
    - reservations : 下单前的风险预约，避免并发下单超配
    - bar_tick : 我们自己的bar计数器(每 finalize 一次+1)
    - current_cycle_ts / last_finalized_bar_ts / reported_pairs_for_current_cycle :
        用来确保一根K线只 finalize 一次
    """

    def __init__(self):
        self.per_pair: Dict[str, PairState] = {}
        self.global_loss_bucket: float = 0.0

        # 风险暴露账本(真实VaR)
        self.trade_risk_ledger: Dict[str, float] = {}
        self.pair_risk_open: Dict[str, float] = {}

        # 预约（未真正成交，但已“占坑”的风险）
        self.reservations: Dict[str, Tuple[str, float]] = {}
        self.reserved_pair_risk: Dict[str, float] = {}
        self.reserved_portfolio_risk: float = 0.0

        # allocator 分配结果: pair -> 这根bar它的急救风险预算(美元VaR)
        self.allocations: Dict[str, float] = {}

        # bar时钟
        self.bar_tick: int = 0
        self.current_cycle_ts: Optional[float] = None
        self.last_finalized_bar_ts: Optional[float] = None
        self.reported_pairs_for_current_cycle: set[str] = set()
        self.last_finalize_walltime: float = time.time()

        # 参数
        self.pain_decay_factor_per_bar: float = 0.999
        self.win_tax_pct: float = 0.2  # 赢单抽税回血全局疼痛池

    # ---- PairState helpers ----
    def get_pair_state(self, pair: str, tier_mgr: TierManager) -> PairState:
        if pair not in self.per_pair:
            pol0 = tier_mgr.get_tier_policy(0)
            self.per_pair[pair] = PairState(
                closs=0,
                local_loss=0.0,
                local_tries_left=pol0.local_recovery_tries,
                cooldown_bars_left=0,
                last_dir=None,
                last_score=0.0,
                active_trades={},
            )
        return self.per_pair[pair]

    # ---- Snapshot / persistence ----
    def to_snapshot(self) -> Dict[str, Any]:
        # active_trades 里有 dataclass，要手动序列化
        per_pair_snap: Dict[str, Any] = {}
        for pair, pst in self.per_pair.items():
            per_pair_snap[pair] = {
                "closs": pst.closs,
                "local_loss": pst.local_loss,
                "local_tries_left": pst.local_tries_left,
                "cooldown_bars_left": pst.cooldown_bars_left,
                "last_dir": pst.last_dir,
                "last_score": pst.last_score,
                "active_trades": {
                    tid: {
                        "sl_pct": tmeta.sl_pct,
                        "tp_pct": tmeta.tp_pct,
                        "direction": tmeta.direction,
                        "icu_bars_left": tmeta.icu_bars_left,
                        "entry_bar_tick": tmeta.entry_bar_tick,
                    }
                    for tid, tmeta in pst.active_trades.items()
                },
            }

        snap = {
            "per_pair": per_pair_snap,
            "global_loss_bucket": self.global_loss_bucket,
            "trade_risk_ledger": self.trade_risk_ledger,
            "pair_risk_open": self.pair_risk_open,
            "reservations": self.reservations,
            "reserved_pair_risk": self.reserved_pair_risk,
            "reserved_portfolio_risk": self.reserved_portfolio_risk,
            "allocations": self.allocations,
            "bar_tick": self.bar_tick,
            "current_cycle_ts": self.current_cycle_ts,
            "last_finalized_bar_ts": self.last_finalized_bar_ts,
            "reported_pairs_for_current_cycle": list(self.reported_pairs_for_current_cycle),
            "last_finalize_walltime": self.last_finalize_walltime,
            "pain_decay_factor_per_bar": self.pain_decay_factor_per_bar,
            "win_tax_pct": self.win_tax_pct,
        }
        return snap

    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        self.per_pair = {}
        per_pair_snap = snap.get("per_pair", {})
        for pair, pst in per_pair_snap.items():
            atmap = {}
            for tid, tmeta in pst.get("active_trades", {}).items():
                atmap[tid] = ActiveTradeMeta(
                    sl_pct=float(tmeta["sl_pct"]),
                    tp_pct=float(tmeta["tp_pct"]),
                    direction=str(tmeta["direction"]),
                    icu_bars_left=tmeta["icu_bars_left"],
                    entry_bar_tick=int(tmeta["entry_bar_tick"]),
                )
            self.per_pair[pair] = PairState(
                closs=int(pst["closs"]),
                local_loss=float(pst["local_loss"]),
                local_tries_left=int(pst["local_tries_left"]),
                cooldown_bars_left=int(pst["cooldown_bars_left"]),
                last_dir=pst["last_dir"],
                last_score=float(pst["last_score"]),
                active_trades=atmap,
            )

        self.global_loss_bucket = float(snap.get("global_loss_bucket", 0.0))
        self.trade_risk_ledger = {
            k: float(v) for k, v in snap.get("trade_risk_ledger", {}).items()
        }
        self.pair_risk_open = {
            k: float(v) for k, v in snap.get("pair_risk_open", {}).items()
        }
        self.reservations = {
            k: (v[0], float(v[1]))
            for k, v in snap.get("reservations", {}).items()
        }
        self.reserved_pair_risk = {
            k: float(v) for k, v in snap.get("reserved_pair_risk", {}).items()
        }
        self.reserved_portfolio_risk = float(
            snap.get("reserved_portfolio_risk", 0.0)
        )
        self.allocations = {
            k: float(v) for k, v in snap.get("allocations", {}).items()
        }
        self.bar_tick = int(snap.get("bar_tick", 0))
        self.current_cycle_ts = snap.get("current_cycle_ts", None)
        self.last_finalized_bar_ts = snap.get("last_finalized_bar_ts", None)
        self.reported_pairs_for_current_cycle = set(
            snap.get("reported_pairs_for_current_cycle", [])
        )
        self.last_finalize_walltime = float(
            snap.get("last_finalize_walltime", time.time())
        )
        self.pain_decay_factor_per_bar = float(
            snap.get("pain_decay_factor_per_bar", self.pain_decay_factor_per_bar)
        )
        self.win_tax_pct = float(
            snap.get("win_tax_pct", self.win_tax_pct)
        )

    def reset_cycle_after_restore(self) -> None:
        """
        重启后我们不想立刻把上一根K线“再 finalize 一遍”。
        我们保留 last_finalized_bar_ts，但把本轮的上报集清空，
        并把 last_finalize_walltime 设成当前时间，避免超时触发重复 finalize。
        """
        self.reported_pairs_for_current_cycle = set()
        self.current_cycle_ts = None
        self.last_finalize_walltime = time.time()

    # ---- 风险预约 (reservation) ----
    def reserve_risk(self, pair: str, reservation_id: str, risk_amount: float) -> None:
        """
        在 custom_stake_amount() 决定仓位后、下单前，
        我们先“预留”这笔真实VaR(风险敞口)，避免同一轮里别的币重复分配这段风险。
        """
        if reservation_id in self.reservations:
            return
        self.reservations[reservation_id] = (pair, float(risk_amount))
        self.reserved_portfolio_risk += float(risk_amount)
        self.reserved_pair_risk[pair] = self.reserved_pair_risk.get(pair, 0.0) + float(risk_amount)

    def release_reservation(self, reservation_id: str) -> Tuple[str, float]:
        """
        把这笔预留风险释放出来。
        我们在 order_filled() 里 - 如果订单真的成交，就会把这笔风险转正到 real exposure；
        无论如何，reservation 也要从预留账本里移除，避免重复计数。
        """
        if reservation_id not in self.reservations:
            return ("", 0.0)
        pair, risk_amt = self.reservations.pop(reservation_id)
        self.reserved_portfolio_risk -= float(risk_amt)
        self.reserved_pair_risk[pair] = max(
            0.0, self.reserved_pair_risk.get(pair, 0.0) - float(risk_amt)
        )
        if self.reserved_pair_risk[pair] <= 1e-12:
            self.reserved_pair_risk[pair] = 0.0
        return (pair, float(risk_amt))

    def get_total_risk(self, include_reserved: bool = True) -> float:
        """
        当前全组合风险暴露(真实 VaR)，可选把预留风险也算进去，
        用于 sizer 的 VaR cap（防止并发下单双花）。
        """
        base = sum(self.pair_risk_open.values())
        if include_reserved:
            base += self.reserved_portfolio_risk
        return base

    def get_pair_risk(self, pair: str, include_reserved: bool = True) -> float:
        base = self.pair_risk_open.get(pair, 0.0)
        if include_reserved:
            base += self.reserved_pair_risk.get(pair, 0.0)
        return base

    # ---- allocator / finalize / 风控心跳 ----

    def allocator_update_allocations(
        self,
        equity: float,
        fast_pool_pct: float,
        injury_norm: float,
        allocator_pain_cap: float,
    ) -> Dict[str, float]:
        """
        分诊台：
        - last_score 高 = 当前setup质量好
        - local_loss 高 = 这币在流血
        组合成权重 -> 分配“急救池 VaR”（fast_pool_pct * equity）
        """
        weights: Dict[str, float] = {}
        for pair, pst in self.per_pair.items():
            if pst.last_score > 0.0 and pst.last_dir:
                pain_term = 1.0
                if injury_norm > 0:
                    pain_term += (pst.local_loss / (equity * injury_norm))
                pain_term = min(pain_term, allocator_pain_cap)
                w = pst.last_score * pain_term
                if w > 0:
                    weights[pair] = w

        total_w = sum(weights.values())
        alloc: Dict[str, float] = {}
        if total_w > 0:
            total_budget = fast_pool_pct * equity
            for pair, w in weights.items():
                alloc[pair] = total_budget * (w / total_w)
        else:
            alloc = {}

        self.allocations = alloc
        return alloc

    def decrement_timers_and_decay_pain(self) -> None:
        """
        finalize_bar_cycle() 时调用：
        - 冷却计时 -1
        - 各仓位 ICU 倒计时 -1
        - 疼痛池指数自愈
        """
        # 冷却 / ICU 倒计时
        for pst in self.per_pair.values():
            if pst.cooldown_bars_left > 0:
                pst.cooldown_bars_left -= 1
            for tmeta in pst.active_trades.values():
                if tmeta.icu_bars_left is not None and tmeta.icu_bars_left > 0:
                    tmeta.icu_bars_left -= 1

        # 疼痛池自愈
        self.global_loss_bucket *= self.pain_decay_factor_per_bar

    def get_dynamic_portfolio_cap_pct(
        self,
        drawdown_threshold_pct: float,
        portfolio_cap_pct_base: float,
        equity: float,
    ) -> float:
        """
        如果疼痛池/权益超阈值(说明血压过高)，全局 VaR 上限砍半。
        """
        if equity <= 0:
            return portfolio_cap_pct_base * 0.5
        if (self.global_loss_bucket / equity) > drawdown_threshold_pct:
            return portfolio_cap_pct_base * 0.5
        return portfolio_cap_pct_base

    def record_trade_open(
        self,
        pair: str,
        trade_id: str,
        real_risk: float,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        tier_policy: TierPolicy,
        tier_mgr: TierManager,
    ) -> None:
        """
        把刚刚真正成交的仓位登记进系统大脑：
        - pair_risk_open / trade_risk_ledger
        - active_trades[trade_id]
        - 给这个 pair 上 cooldown
        """
        pst = self.get_pair_state(pair, tier_mgr)

        # 更新风险账本
        self.trade_risk_ledger[trade_id] = float(real_risk)
        self.pair_risk_open[pair] = self.pair_risk_open.get(pair, 0.0) + float(real_risk)

        # ICU 倒计时：无论当前 closs 是否 ICU，只要 tier_policy 给了 icu_force_exit_bars>0
        icu_bars = tier_policy.icu_force_exit_bars if tier_policy.icu_force_exit_bars > 0 else None
        pst.active_trades[trade_id] = ActiveTradeMeta(
            sl_pct=float(sl_pct),
            tp_pct=float(tp_pct),
            direction=str(direction),
            icu_bars_left=icu_bars,
            entry_bar_tick=self.bar_tick,
        )

        # 冷却期(至少这么长)
        pst.cooldown_bars_left = max(
            pst.cooldown_bars_left,
            tier_policy.cooldown_bars,
        )

    def record_trade_close(
        self,
        pair: str,
        trade_id: str,
        profit_abs: float,
        tier_mgr: TierManager,
    ) -> None:
        """
        平仓后清算这笔仓位带来的后果：
        - VaR暴露释放
        - closs / local_loss / local_tries_left / cooldown / 疼痛池 / 疗效
        """
        pst = self.get_pair_state(pair, tier_mgr)

        # 释放风险账本
        was_risk = self.trade_risk_ledger.pop(trade_id, 0.0)
        self.pair_risk_open[pair] = max(
            0.0,
            self.pair_risk_open.get(pair, 0.0) - was_risk,
        )
        if self.pair_risk_open[pair] <= 1e-12:
            self.pair_risk_open[pair] = 0.0

        # 移除 active_trades
        if trade_id in pst.active_trades:
            pst.active_trades.pop(trade_id, None)

        # 根据盈亏 更新伤情&疼痛池
        pol_now = tier_mgr.get_tier_policy(pst.closs)

        if profit_abs >= 0:
            # 赢单 -> 回血+降级
            tax = profit_abs * self.win_tax_pct
            self.global_loss_bucket = max(0.0, self.global_loss_bucket - tax)

            if pst.closs > 0:
                pst.closs -= 1
            pst.local_loss = 0.0

            # 新tier的 tries_left
            pol_after = tier_mgr.get_tier_policy(pst.closs)
            pst.local_tries_left = pol_after.local_recovery_tries

            # 赢后冷却(短一点)
            pst.cooldown_bars_left = max(
                pst.cooldown_bars_left,
                pol_after.cooldown_bars_after_win,
            )
        else:
            # 亏单 -> 累计损失，扣 tries
            loss_abs = abs(profit_abs)
            pst.local_loss += loss_abs
            pst.local_tries_left -= 1

            if pst.local_tries_left <= 0:
                # 彻底失败 -> 升级病房
                self.global_loss_bucket += pst.local_loss
                pst.local_loss = 0.0
                pst.closs += 1

                pol_after = tier_mgr.get_tier_policy(pst.closs)
                pst.local_tries_left = pol_after.local_recovery_tries

                # 严格冷却
                pst.cooldown_bars_left = max(
                    pst.cooldown_bars_left,
                    pol_after.cooldown_bars,
                )

    # ---- bar finalize ----
    def finalize_bar_cycle(
        self,
        equity_provider: EquityProvider,
        fast_pool_pct: float,
        injury_norm: float,
        allocator_pain_cap: float,
        drawdown_threshold_pct: float,
        portfolio_cap_pct_base: float,
    ) -> None:
        """
        在一根K线真正结束时调用：
        - bar_tick += 1
        - 冷却/ICU倒计时-1
        - 疼痛池指数衰减
        - allocator 分配新的急救预算
        """
        self.bar_tick += 1
        self.decrement_timers_and_decay_pain()

        # 重新计算 allocator 分诊
        eq = equity_provider.get_equity()
        self.allocator_update_allocations(
            equity=eq,
            fast_pool_pct=fast_pool_pct,
            injury_norm=injury_norm,
            allocator_pain_cap=allocator_pain_cap,
        )

        # 动态组合 VaR cap 会在 sizer 里读取，用 get_dynamic_portfolio_cap_pct()
        # 在这里我们不需要主动改别的状态。

    def maybe_finalize_bar_cycle(
        self,
        pair: str,
        bar_ts: float,
        whitelist: list[str],
        timeframe_sec: int,
        force_finalize_mult: float,
        equity_provider: EquityProvider,
        fast_pool_pct: float,
        injury_norm: float,
        allocator_pain_cap: float,
        drawdown_threshold_pct: float,
        portfolio_cap_pct_base: float,
        persist_callback,
    ) -> None:
        """
        我们在每次 populate_indicators() 的最后调用。
        逻辑：
        1. 如果检测到 candle timestamp 变了 -> 认为进入新bar，清空这一bar的上报集
        2. 标记当前pair已上报
        3. 如果(所有白名单pair都上报) or (超时)，
           且这是一个新的bar_ts(比 last_finalized_bar_ts 新)，
           则 finalize_bar_cycle() + save()，并记下 last_finalized_bar_ts。
        """

        now = time.time()

        # 如果是新bar ts，说明开始了一个新的K线
        if self.current_cycle_ts is None or bar_ts > float(self.current_cycle_ts):
            self.current_cycle_ts = float(bar_ts)
            self.reported_pairs_for_current_cycle = set()

        # 标记这个pair已在本bar上报
        self.reported_pairs_for_current_cycle.add(pair)

        # 判定：是否所有白名单都上报？
        # 为了鲁棒性：只检查仍在 whitelist 的pair
        all_reported = all(
            p in self.reported_pairs_for_current_cycle
            for p in whitelist
        )

        # 判定：是否该bar_ts比上次 finalize 的bar_ts更新？
        newer_than_last = (
            self.last_finalized_bar_ts is None
            or float(self.current_cycle_ts) > float(self.last_finalized_bar_ts)
        )

        # 判定：是否超时？(某些pair没来了，不能永远卡住)
        timeout_sec = force_finalize_mult * timeframe_sec
        force_due_timeout = (now - self.last_finalize_walltime) >= timeout_sec

        if newer_than_last and (all_reported or force_due_timeout):
            # 真正 finalize
            self.finalize_bar_cycle(
                equity_provider=equity_provider,
                fast_pool_pct=fast_pool_pct,
                injury_norm=injury_norm,
                allocator_pain_cap=allocator_pain_cap,
                drawdown_threshold_pct=drawdown_threshold_pct,
                portfolio_cap_pct_base=portfolio_cap_pct_base,
            )
            self.last_finalize_walltime = now
            self.last_finalized_bar_ts = float(self.current_cycle_ts)

            # finalize 完成后我们立刻落盘，确保 ICU / cooldown / 疼痛池 / allocator 进度持久化
            persist_callback()

            # 重置当前bar的上报集，准备下一根bar
            # 注意：不要把 current_cycle_ts 清掉，
            # 因为这一根bar已经 finalize 结束，下一次新的K线出现时会重新覆盖它。
            self.reported_pairs_for_current_cycle = set()


class StateStore:
    """
    负责把 GlobalState + EquityProvider 持久化到磁盘。
    我们在 finalize_bar_cycle() 和 每次 order_filled() 后都会 save()。
    """
    def __init__(self, filepath: str, gstate: GlobalState, eqp: EquityProvider):
        self.filepath = filepath
        self.state = gstate
        self.eqp = eqp

    def save(self) -> None:
        snap = {
            "global_state": self.state.to_snapshot(),
            "equity_provider": self.eqp.to_snapshot(),
        }
        tmpfile = self.filepath + ".tmp"
        with open(tmpfile, "w") as f:
            json.dump(snap, f)
        os.replace(tmpfile, self.filepath)

    def load_if_exists(self) -> None:
        if not os.path.isfile(self.filepath):
            return
        with open(self.filepath, "r") as f:
            snap = json.load(f)
        gss = snap.get("global_state", {})
        eqs = snap.get("equity_provider", {})

        self.state.restore_snapshot(gss)
        self.eqp.restore_snapshot(eqs)
        # 避免“开机瞬间又把上一根bar finalize 一遍”
        self.state.reset_cycle_after_restore()


# ---------------------------------------------------------------------------
# 3. PositionSizerV26
#    - 把 TierPolicy + allocator + VaR cap 等等 -> stake amount
#    - 生成真实风险 real_risk, 用于预留/登记
# ---------------------------------------------------------------------------

class PositionSizerV26:
    def __init__(
        self,
        gstate: GlobalState,
        eqp: EquityProvider,
        tier_mgr: TierManager,
        drawdown_threshold_pct: float,
        portfolio_cap_pct_base: float,
    ):
        self.state = gstate
        self.eqp = eqp
        self.tier_mgr = tier_mgr
        self.drawdown_threshold_pct = drawdown_threshold_pct
        self.portfolio_cap_pct_base = portfolio_cap_pct_base

    def compute_position(
        self,
        pair: str,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        tier_policy: TierPolicy,
        min_stake_exchange: Optional[float],
        max_stake_exchange: float,
    ) -> Tuple[float, float]:
        """
        返回 (stake_nominal, real_risk)
        stake_nominal: 实际下单的名义金额
        real_risk:     stake_nominal * sl_pct  -> 真实VaR，用来入账
        """

        equity = self.eqp.get_equity()
        pst = self.state.get_pair_state(pair, self.tier_mgr)

        # 1) baseline 风险预算
        base_budget = tier_policy.k_mult_base_pct * equity

        # 2) 回血需求 (TARGET_RECOVERY)
        recovery_budget = base_budget
        if tier_policy.sizing_algo == "TARGET_RECOVERY":
            # 想赚回 local_loss * recovery_factor
            desired_profit = pst.local_loss * tier_policy.recovery_factor
            # 为了拿到这么多利润, 你需要的名义资金 ~ desired_profit / tp_pct
            if tp_pct > 0:
                stake_from_tp = desired_profit / tp_pct
                # 这对应的最大亏损大约是 stake_from_tp * sl_pct
                risk_from_tp = stake_from_tp * sl_pct
                # 我们把 risk_from_tp 当作“回血优先级”预算需求
                recovery_budget = max(base_budget, risk_from_tp)

        # 3) allocator 的紧急分诊预算
        alloc_extra = self.state.allocations.get(pair, 0.0)
        # 理论候选风险需求
        candidate_risk_need = recovery_budget + alloc_extra
        # fast_multiplier_cap 限制别无限放大
        candidate_risk_need = min(
            candidate_risk_need,
            recovery_budget * tier_policy.fast_multiplier_cap,
        )

        # 4) 单币 VaR cap / 组合 VaR cap (含预约)
        pair_cap = tier_policy.per_pair_risk_cap_pct * equity
        portfolio_cap_pct_dyn = self.state.get_dynamic_portfolio_cap_pct(
            drawdown_threshold_pct=self.drawdown_threshold_pct,
            portfolio_cap_pct_base=self.portfolio_cap_pct_base,
            equity=equity,
        )
        portfolio_cap = portfolio_cap_pct_dyn * equity

        pair_used = self.state.get_pair_risk(pair, include_reserved=True)
        portfolio_used = self.state.get_total_risk(include_reserved=True)

        pair_room = max(0.0, pair_cap - pair_used)
        port_room = max(0.0, portfolio_cap - portfolio_used)

        risk_room = min(pair_room, port_room)
        risk_final = min(candidate_risk_need, risk_room)

        if risk_final <= 0 or sl_pct <= 0:
            return (0.0, 0.0)

        # 5) 把最终风险预算 risk_final -> 名义仓位
        stake_nominal = risk_final / sl_pct

        # 6) 名义金额硬限制：
        #    - 单笔最大名义金额(相对equity)
        #    - 交易所最大下单金额
        stake_cap_notional = tier_policy.max_stake_notional_pct * equity
        stake_nominal = min(stake_nominal, stake_cap_notional, max_stake_exchange)

        # 7) 交易所最小下单额
        if min_stake_exchange is not None:
            stake_nominal = max(stake_nominal, min_stake_exchange)

        # 边界
        if stake_nominal <= 0:
            return (0.0, 0.0)

        # 8) 重新以实际 stake_nominal 计算真实VaR
        real_risk = stake_nominal * sl_pct
        return (stake_nominal, real_risk)


# ---------------------------------------------------------------------------
# 4. ExitPolicy
#    - ICU 倒计时
#    - TP / half-TP early lock
#    - flip (反手)
#    - global risk_off
# ---------------------------------------------------------------------------

class ExitPolicy:
    def __init__(self, gstate: GlobalState, eqp: EquityProvider,
                 drawdown_threshold_pct: float):
        self.state = gstate
        self.eqp = eqp
        self.drawdown_threshold_pct = drawdown_threshold_pct

    def decide(
        self,
        pair: str,
        pst: PairState,
        tmeta: ActiveTradeMeta,
        current_profit_ratio: float,
    ) -> Optional[str]:
        """
        返回 exit_reason 字符串:
          "tp_hit", "early_lock", "icu_timeout", "flip_long", "flip_short", "risk_off"
        或 None 表示继续持有
        """
        # 1) 直接达到 TP
        if current_profit_ratio >= tmeta.tp_pct:
            return "tp_hit"

        # 2) 提前锁盈：至少到达 TP 一半，且 >0
        if current_profit_ratio > 0 and current_profit_ratio >= tmeta.tp_pct * 0.5:
            return "early_lock"

        # 3) ICU 倒计时到零就强平
        if (tmeta.icu_bars_left is not None) and (tmeta.icu_bars_left <= 0):
            return "icu_timeout"

        # 4) flip: 如果该币当前最有利方向已经和仓位反了，
        #    且评分足够高（last_score>0.01）
        if pst.last_dir and pst.last_dir != tmeta.direction and pst.last_score >= 0.01:
            # 我们不在这一步直接下反向单，而是先平旧仓
            if pst.last_dir == "long":
                return "flip_long"
            else:
                return "flip_short"

        # 5) 全局高血压 risk_off:
        eq = self.eqp.get_equity()
        if eq > 0:
            stress_ratio = self.state.global_loss_bucket / eq
        else:
            stress_ratio = 999.0

        if stress_ratio > self.drawdown_threshold_pct and current_profit_ratio < 0:
            # 全局很痛，又是在亏的单 -> 尽快砍掉减压
            return "risk_off"

        return None


# ---------------------------------------------------------------------------
# 5. ICUBrainV26 Strategy 主体
# ---------------------------------------------------------------------------

class ICUBrainV26(IStrategy):
    """
    多币 ICU 风控大脑。
    """

    timeframe = "5m"
    can_short = True

    # 我们完全用 custom_exit / custom_stoploss，所以把官方 ROI 等机制尽量钝化:
    minimal_roi = {"0": 10_000.0}  # essentially never hit by built-in roi logic
    stoploss = -0.99               # hard safety net；我们用 custom_stoploss 来收紧
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_buy_signals = False
    ignore_sell_signals = True

    startup_candle_count = 210  # for ema200 etc.

    # 这些参数是风控/调度的关键常量
    drawdown_threshold_pct = 0.15       # 全局疼痛>15% equity -> risk_off & VaR降档
    portfolio_cap_pct_base = 0.20       # 正常时组合最多愿意暴露20% VaR
    fast_pool_pct = 0.05                # allocator急救池总预算=5% equity
    allocator_injury_norm = 0.02        # local_loss 相对于 equity 的归一化尺度
    allocator_pain_cap = 3.0            # allocator里 pain_term 的封顶
    force_finalize_mult = 2.0           # 超时系数，超时也会 finalize，一般=2倍timeframe长度
    # 注：pain_decay_factor_per_bar 在 GlobalState 里默认0.999
    #     (= 每根bar疼痛池自愈约0.1%)，可以按品味调

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # 构建核心大脑组件
        self.tier_mgr = TierManager()
        self.state = GlobalState()
        self.eqp = EquityProvider(initial_equity=float(config.get("dry_run_wallet", 1000.0)))
        # 持久化文件
        statedir = config.get("user_data_dir", ".")
        self.state_store = StateStore(
            filepath=os.path.join(statedir, "icubrain_state.json"),
            gstate=self.state,
            eqp=self.eqp,
        )

        # sizer / exit_policy
        self.sizer = PositionSizerV26(
            gstate=self.state,
            eqp=self.eqp,
            tier_mgr=self.tier_mgr,
            drawdown_threshold_pct=self.drawdown_threshold_pct,
            portfolio_cap_pct_base=self.portfolio_cap_pct_base,
        )
        self.exit_policy = ExitPolicy(
            gstate=self.state,
            eqp=self.eqp,
            drawdown_threshold_pct=self.drawdown_threshold_pct,
        )

        # runtime caches
        # 最后一根bar给这个pair挑出来的最佳 candidate
        self._last_signal: Dict[str, Optional[Candidate]] = {}
        # 在 confirm_trade_entry / custom_stake_amount 之间流转的结构化信息
        #  { pair: {
        #       "sl_pct":.., "tp_pct":.., "direction":..,
        #       "tier_policy": TierPolicy,
        #       "expected_edge":..,
        #       "stake_final":..,
        #       "risk_final":..,
        #       "reservation_id":..
        #    } }
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}

        # 缓存 timeframe(秒)
        self._timeframe_sec = self._timeframe_to_sec(self.timeframe)

    # ---- freqtrade lifecycle hooks ----

    def bot_start(self, **kwargs) -> None:
        """
        机器人启动时加载持久化状态，恢复 ICU、冷却、疼痛池、equity 等。
        """
        self.state_store.load_if_exists()
        # 非常关键：我们依赖“盈亏比≈价格波动”来判断 TP/SL/early_lock
        # 所以不允许机器人偷偷上杠杆>1
        # 注意：freqtrade会在 custom_stake_amount() 里告诉我们 leverage，
        # 我们也会再次assert。
        return

    # ---- Helpers: timeframe -> seconds ---
    @staticmethod
    def _timeframe_to_sec(tf: str) -> int:
        # minimal util for common tfs like "5m","1h"
        if tf.endswith("m"):
            return int(tf[:-1]) * 60
        if tf.endswith("h"):
            return int(tf[:-1]) * 3600
        if tf.endswith("d"):
            return int(tf[:-1]) * 86400
        # fallback
        return 300

    # ---- Indicator calculation and Candidate generation ----

    def _gen_engine_candidates_for_row(self, row: pd.Series) -> list[Candidate]:
        """
        根据最后一根K的指标，产出多种候选信号。
        这里我们用非常朴素的启发式信号：
        - mean_rev_long : RSI<25, price < ema50 - 1.5*ATR_pct
        - pullback_long : uptrend(ema50>ema200, ADX>20), 回踩到 ema50附近
        - trend_short   : downtrend(ema50<ema200, ADX>25), 价格反弹到 ema50上方
        每个候选会给出 sl_pct / tp_pct / raw_score 等。
        """

        out: list[Candidate] = []
        close = float(row["close"])
        ema50 = float(row["ema50"])
        ema200 = float(row["ema200"])
        rsi_v = float(row["rsi"])
        atr_pct = float(row["atr_pct"])
        adx_v = float(row["adx"])

        # mean_rev_long
        # oversold mean reversion: 价格严重击穿ema50，rsi极低
        if rsi_v < 25 and close < ema50 * 0.985:
            sl_pct = atr_pct * 1.2     # e.g. 1.2 * ATR%
            tp_pct = atr_pct * 2.4     # RR ~2
            raw_score = max(0.0, (25 - rsi_v) / 25.0)  # rsi越低越好
            rr_ratio = tp_pct / max(sl_pct, 1e-9)
            # 胜率估计：0.52 + 0.4*raw_score, clamp [0.5,0.9]
            win_prob = min(0.9, max(0.5, 0.52 + 0.4 * raw_score))
            expected_edge = win_prob * tp_pct - (1.0 - win_prob) * sl_pct
            out.append(
                Candidate(
                    direction="long",
                    kind="mean_rev_long",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    raw_score=raw_score,
                    rr_ratio=rr_ratio,
                    win_prob=win_prob,
                    expected_edge=expected_edge,
                )
            )

        # pullback_long
        # 上升趋势里(ema50>ema200, ADX>20)的回踩
        if ema50 > ema200 and adx_v > 20 and close < ema50 * 0.99:
            sl_pct = atr_pct * 1.0
            tp_pct = atr_pct * 2.0
            raw_score = 0.5 * max(0.0, (ema50 / max(ema200, 1e-9) - 1.0)) \
                        + 0.5 * max(0.0, (adx_v - 20.0) / 20.0)
            rr_ratio = tp_pct / max(sl_pct, 1e-9)
            # 胜率估计：0.55 + 0.4*raw_score (趋势内回踩，略更高)
            win_prob = min(0.95, max(0.5, 0.55 + 0.4 * raw_score))
            expected_edge = win_prob * tp_pct - (1.0 - win_prob) * sl_pct
            out.append(
                Candidate(
                    direction="long",
                    kind="pullback_long",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    raw_score=raw_score,
                    rr_ratio=rr_ratio,
                    win_prob=win_prob,
                    expected_edge=expected_edge,
                )
            )

        # trend_short
        # 下行趋势(ema50<ema200, ADX>25)里，反弹到ema50上方的做空
        if ema50 < ema200 and adx_v > 25 and close > ema50 * 1.01:
            sl_pct = atr_pct * 1.2
            tp_pct = atr_pct * 2.4
            raw_score = 0.5 * max(0.0, (adx_v - 25.0) / 25.0) \
                        + 0.5 * max(0.0, (1.0 - ema50 / max(ema200,1e-9)))
            rr_ratio = tp_pct / max(sl_pct, 1e-9)
            # 胜率估计：0.50 + 0.4*raw_score (下跌顺势做空)
            win_prob = min(0.95, max(0.5, 0.50 + 0.4 * raw_score))
            expected_edge = win_prob * tp_pct - (1.0 - win_prob) * sl_pct
            out.append(
                Candidate(
                    direction="short",
                    kind="trend_short",
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    raw_score=raw_score,
                    rr_ratio=rr_ratio,
                    win_prob=win_prob,
                    expected_edge=expected_edge,
                )
            )

        return out

    def _filter_and_rank_candidates(
        self,
        tier_pol: TierPolicy,
        cands: list[Candidate],
    ) -> Optional[Candidate]:
        """
        Tier-aware 过滤：
        - kind 允许吗？
        - raw_score >= min_raw_score?
        - rr_ratio >= min_rr_ratio?
        - expected_edge >= min_edge?
        然后按 (expected_edge, raw_score) 排序，取最优。
        """
        valid: list[Candidate] = []
        for c in cands:
            if c.kind not in tier_pol.allowed_kinds:
                continue
            if c.raw_score < tier_pol.min_raw_score:
                continue
            if c.rr_ratio < tier_pol.min_rr_ratio:
                continue
            if c.expected_edge < tier_pol.min_edge:
                continue
            valid.append(c)
        if not valid:
            return None
        # 排序：优先 expected_edge，再 raw_score
        valid.sort(key=lambda x: (x.expected_edge, x.raw_score), reverse=True)
        return valid[0]

    # ---- freqtrade mandatory pandas steps ----

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        计算该pair的技术指标 -> 生成 candidate -> 更新 last_dir/last_score ->
        报告给 GlobalState -> maybe_finalize_bar_cycle() -> 可能触发 finalize + save().
        """

        pair = metadata["pair"]
        pst = self.state.get_pair_state(pair, self.tier_mgr)

        # 计算指标
        dataframe["ema50"] = ta.ema(dataframe["close"], length=50)
        dataframe["ema200"] = ta.ema(dataframe["close"], length=200)
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)

        # ATR 和 ATR%
        atr_df = ta.atr(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            length=14,
        )
        dataframe["atr"] = atr_df
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        # ADX
        adx_df = ta.adx(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            length=14,
        )
        # pandas_ta.adx() 返回多列 { "ADX_14", "DMP_14", "DMN_14" }
        if isinstance(adx_df, pd.DataFrame) and "ADX_14" in adx_df.columns:
            dataframe["adx"] = adx_df["ADX_14"]
        else:
            # fallback
            dataframe["adx"] = adx_df if isinstance(adx_df, pd.Series) else 20.0

        if len(dataframe) == 0:
            return dataframe

        last_row = dataframe.iloc[-1]

        # 生成候选
        raw_cands = self._gen_engine_candidates_for_row(last_row)
        tier_pol = self.tier_mgr.get_tier_policy(pst.closs)
        best_cand = self._filter_and_rank_candidates(tier_pol, raw_cands)

        # 更新 pair 的 last_dir / last_score
        if best_cand is not None:
            pst.last_dir = best_cand.direction
            pst.last_score = best_cand.expected_edge
        else:
            pst.last_dir = None
            pst.last_score = 0.0

        # 缓存给 confirm_trade_entry 用
        self._last_signal[pair] = best_cand

        # ---- bar心跳+持久化 ----
        # freqtrade 的 dataframe 通常有 'date' (tz-naive datetime).
        # 用最后这根K线的 timestamp 作为 bar_ts。
        bar_ts = float(pd.to_datetime(last_row["date"]).timestamp())

        # whitelist: 当前在跑的交易对
        try:
            whitelist = self.dp.current_whitelist()
        except Exception:
            whitelist = [pair]

        self.state.maybe_finalize_bar_cycle(
            pair=pair,
            bar_ts=bar_ts,
            whitelist=whitelist,
            timeframe_sec=self._timeframe_sec,
            force_finalize_mult=self.force_finalize_mult,
            equity_provider=self.eqp,
            fast_pool_pct=self.fast_pool_pct,
            injury_norm=self.allocator_injury_norm,
            allocator_pain_cap=self.allocator_pain_cap,
            drawdown_threshold_pct=self.drawdown_threshold_pct,
            portfolio_cap_pct_base=self.portfolio_cap_pct_base,
            persist_callback=lambda: self.state_store.save(),
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        freqtrade 只真正关心最后一行的 enter_long / enter_short 触发开仓信号。
        回测里这会不完美，因为我们没有把历史 closs 演化回放到整张表，
        但实盘决策没问题。
        """
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        pair = metadata["pair"]
        best_cand = self._last_signal.get(pair)

        if best_cand is not None and len(dataframe) > 0:
            if best_cand.direction == "long":
                dataframe.loc[dataframe.index[-1], "enter_long"] = 1
            elif best_cand.direction == "short":
                dataframe.loc[dataframe.index[-1], "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        我们不使用 freqtrade 的 exit_signals 做平仓，
        平仓靠 custom_exit() / custom_stoploss() / order_filled()全套 ICU 规则。
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # -------------------------------------------------------------------
    # 6. 确认入场 / sizing / 预约风险
    # -------------------------------------------------------------------

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
        """
        在订单真正下到交易所"前一瞬间"调用。
        我们在这里检查：
        - 冷却期
        - 是否已持有仓位（不允许同币并发多单）
        - 方向一致
        - sl/tp 合法
        并把 tier_policy / sl_pct / tp_pct 缓存进 _pending_entry_meta，供 custom_stake_amount 使用。

        freqtrade 官方明确说明：这个回调执行时机非常敏感，避免重计算和外部IO。:contentReference[oaicite:17]{index=17}
        """

        pst = self.state.get_pair_state(pair, self.tier_mgr)

        # 冷却期内禁止进新仓
        if pst.cooldown_bars_left > 0:
            return False

        # 单币不允许有多仓同时开着
        if len(pst.active_trades) > 0:
            return False

        best_cand = self._last_signal.get(pair)
        if not best_cand:
            return False

        # side 是 "long"/"short"
        desired_dir = side.lower()
        if best_cand.direction != desired_dir:
            return False

        if best_cand.sl_pct is None or best_cand.tp_pct is None:
            return False
        if best_cand.sl_pct <= 0 or best_cand.tp_pct <= 0:
            return False

        tier_pol = self.tier_mgr.get_tier_policy(pst.closs)
        self._pending_entry_meta[pair] = {
            "sl_pct": best_cand.sl_pct,
            "tp_pct": best_cand.tp_pct,
            "direction": best_cand.direction,
            "expected_edge": best_cand.expected_edge,
            "tier_policy": tier_pol,
            # stake_final / risk_final / reservation_id
            # 会在 custom_stake_amount() 里补
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
        """
        决定最终下单金额 (名义仓位)，并且在这个时刻进行“风险预约”。
        freqtrade 会在 confirm_trade_entry() 之后马上调用它。:contentReference[oaicite:18]{index=18}

        我们会：
        1. 用 PositionSizerV26 计算 stake_nominal, real_risk
        2. assert 杠杆 == 1.0 (我们整个 ICU 模型假设的是 1x )
        3. 把 real_risk 以 reservation 的形式占坑
        4. 缓存 stake_final / risk_final / reservation_id 给 order_filled() 用
        """

        # 我们的ICU风控逻辑假设利润比率≈价格波动，必须是1x杠杆
        if abs(leverage - 1.0) > 1e-6:
            # 直接拒单
            return 0.0

        meta = self._pending_entry_meta.get(pair)
        if not meta:
            return 0.0

        # 如果已经算过stake_final(比如freqtrade多次询问)，直接复用
        if "stake_final" in meta and "risk_final" in meta and "reservation_id" in meta:
            return float(meta["stake_final"])

        sl_pct = meta["sl_pct"]
        tp_pct = meta["tp_pct"]
        direction = meta["direction"]
        tier_pol = meta["tier_policy"]

        stake_nominal, real_risk = self.sizer.compute_position(
            pair=pair,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            direction=direction,
            tier_policy=tier_pol,
            min_stake_exchange=min_stake,
            max_stake_exchange=max_stake,
        )

        if stake_nominal <= 0:
            return 0.0

        # 预约风险：生成 reservation_id
        reservation_id = f"{pair}:{side}:{uuid.uuid4().hex}"
        self.state.reserve_risk(pair, reservation_id, real_risk)

        # 把这些记回 meta, 让 order_filled() 能读取并转正
        meta["stake_final"] = stake_nominal
        meta["risk_final"] = real_risk
        meta["reservation_id"] = reservation_id

        return float(stake_nominal)

    # -------------------------------------------------------------------
    # 7. 止损逻辑 / 主动平仓逻辑
    # -------------------------------------------------------------------

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
        """
        返回 stoploss 距离 (负数)，例如 -0.02 => 止损在现价下方2%。
        我们优先从 trade 的 custom_data / user_data 中取 sl_pct；
        如果丢失(比如重启后对象不带了)，就从 GlobalState 里的 active_trades 持久化记录兜底；
        如果还是拿不到，就用一个保守保护止损(比如 -0.03)。
        绝不返回 None 让它掉回全局 -0.99，避免老仓裸奔。:contentReference[oaicite:19]{index=19}
        """
        # 尝试从 trade.get_custom_data("sl_pct")
        sl_from_trade = None
        try:
            sl_from_trade = trade.get_custom_data("sl_pct")
        except Exception:
            sl_from_trade = None

        if sl_from_trade is None:
            # 尝试从 trade.user_data
            try:
                sl_from_trade = float(trade.user_data.get("sl_pct", None))
            except Exception:
                sl_from_trade = None

        if sl_from_trade is None:
            # 最后兜底：从 GlobalState.active_trades
            pair_state = self.state.get_pair_state(pair, self.tier_mgr)
            tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
            if tid in pair_state.active_trades:
                sl_from_trade = pair_state.active_trades[tid].sl_pct

        if sl_from_trade is None:
            # 最绝望的fallback，强行给个-3%
            return -0.03

        sl_pct = abs(float(sl_from_trade))
        if sl_pct <= 0:
            return -0.03

        # freqtrade 期望的是一个负数ratio
        return -sl_pct

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        ICU 护士台：
        - tp_hit / early_lock
        - icu_timeout
        - flip_xxx
        - risk_off
        返回 exit_reason 字符串 -> freqtrade 将立刻发出平仓单。
        """

        pst = self.state.get_pair_state(pair, self.tier_mgr)
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))

        tmeta = pst.active_trades.get(tid)
        if tmeta is None:
            return None

        # current_profit 是 ratio(0.05=+5%)，正好跟我们的 tp_pct/sl_pct 同量纲
        reason = self.exit_policy.decide(
            pair=pair,
            pst=pst,
            tmeta=tmeta,
            current_profit_ratio=float(current_profit),
        )
        if reason:
            return reason
        return None

    # -------------------------------------------------------------------
    # 8. order_filled(): 真实成交后 -> 风险账本入/出院 + 持久化
    # -------------------------------------------------------------------

    def order_filled(
        self,
        pair: str,
        trade,
        order,
        current_time: datetime,
        **kwargs,
    ) -> None:
        """
        freqtrade 会在任何订单完全成交后调用这个回调，包括建仓/加仓/平仓/止损/强平。:contentReference[oaicite:20]{index=20}

        我们用它来：
        - 如果是首次把仓位打进市场(从无到有)，就把这笔仓登记进 GlobalState：
            pair_risk_open, trade_risk_ledger, active_trades, cooldown, ICU倒计时等
          并释放那笔 reservation。
        - 如果是把仓位完全关掉(从有到无)，就计算 profit_abs，
          调用 record_trade_close() 更新 closs / 疼痛池 / cooldown / equity_current，
          然后落盘。

        注意：我们只允许单币单仓(不加码/分批)，所以逻辑较简单：
        如果 trade 现在 is_open == True 而 active_trades 里还没有，就视为开仓；
        如果 trade 现在 is_open == False 而 active_trades 里还在，就视为平仓。
        """

        pst = self.state.get_pair_state(pair, self.tier_mgr)
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        is_open_now = bool(getattr(trade, "is_open", True))

        # 读取 profit_abs:
        # close_profit_abs 在 trade 关闭后通常存在；
        # 否则 fallback 到 trade.profit_abs。
        profit_abs = 0.0
        if hasattr(trade, "close_profit_abs") and trade.close_profit_abs is not None:
            profit_abs = float(trade.close_profit_abs)
        elif hasattr(trade, "profit_abs") and trade.profit_abs is not None:
            profit_abs = float(trade.profit_abs)

        # --- case A: 新开仓 ---
        if is_open_now and tid not in pst.active_trades:
            meta = self._pending_entry_meta.get(pair, None)
            if meta is None:
                # 可能是某种我们没认的加仓 / 复活，
                # 这里就不做登记，避免重复。
                return

            # 从 pending_meta 拿我们之前计算的 risk_final / sl_pct / tp_pct / direction / reservation_id
            sl_pct = meta["sl_pct"]
            tp_pct = meta["tp_pct"]
            direction = meta["direction"]
            tier_pol = meta["tier_policy"]
            real_risk = float(meta.get("risk_final", 0.0))
            reservation_id = meta.get("reservation_id")

            # 把reservation转正到在市风险账本(pair_risk_open / trade_risk_ledger)
            self.state.record_trade_open(
                pair=pair,
                trade_id=tid,
                real_risk=real_risk,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                direction=direction,
                tier_policy=tier_pol,
                tier_mgr=self.tier_mgr,
            )

            # 释放reservation占用
            if reservation_id:
                self.state.release_reservation(reservation_id)

            # 把 sl/tp 等信息写回 trade.user_data，
            # 这样 custom_stoploss / custom_exit 可以用第一优先级直接读它，
            # (注意：不同freqtrade版本对 user_data / custom_data 持久性可能有差异，
            #  我们依然保留 GlobalState 兜底。)
            try:
                trade.set_custom_data("sl_pct", sl_pct)
                trade.set_custom_data("tp_pct", tp_pct)
            except Exception:
                pass
            try:
                if hasattr(trade, "user_data") and isinstance(trade.user_data, dict):
                    trade.user_data["sl_pct"] = sl_pct
                    trade.user_data["tp_pct"] = tp_pct
            except Exception:
                pass

            # 清掉 pending_meta
            self._pending_entry_meta.pop(pair, None)

            # 持久化状态(开仓后立即落盘，避免断电丢 ICU 倒计时初始值)
            self.state_store.save()
            return

        # --- case B: 平仓 ---
        if (not is_open_now) and (tid in pst.active_trades):
            # 更新大脑的创伤状态、疼痛池、closs、cooldown等
            self.state.record_trade_close(
                pair=pair,
                trade_id=tid,
                profit_abs=profit_abs,
                tier_mgr=self.tier_mgr,
            )
            # 更新 equity
            self.eqp.on_trade_closed_update(profit_abs)

            # 持久化(平仓后落盘，避免重启后closs/冷却/疼痛池回档)
            self.state_store.save()
            return

        # 其他情况(比如部分加仓/部分减仓、止损单但仓还在等)
        # 我们目前不支持多段加码，所以忽略。
        return

    # -------------------------------------------------------------------
    # 9. leverage() 回调
    # -------------------------------------------------------------------

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        我们的整个止损/TP/early_lock/ICU逻辑假设 "收益率 ~= 价格变动"，
        即杠杆=1x。
        我们会直接返回 1.0。
        """
        return 1.0
