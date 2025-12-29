# V29 invariants

This document captures the non-negotiable behavioral contracts for TaxBrain V29.

## Debt model
- Central debt (GlobalState.debt_pool) increases only when a trade hits max tier loss rollover; it decays by pain_decay_per_bar each bar.
- Local debt (PairState.local_loss) accumulates on losses below max tier; profits first offset local_loss, then affect central debt.
- On profit >= 0: local_loss is reduced by profit; excess profit repays central debt (cannot go below zero).
- On max tier loss rollover: local_loss and the loss amount are socialized into central debt; closs resets to 0.

## State machine
- bar_tick advances once per bar (live finalize) or by floor(delta / tf_sec) in backtest sync.
- cooldown_bars_left and icu_bars_left decrement per bar; never below zero.
- cycle: if bar_tick - cycle_start_tick >= cycle_len_bars and cycle PnL >= 0, then debt_pool, local_loss, and closs reset to 0 (if enabled).

## Signal gating
- TierPolicy filters candidates by min_raw_score, min_rr_ratio, and min_edge.
- Gatekeeping applies a single min_score threshold (when enabled).
- Entry is rejected if cooldown is active, single_position_only is violated, or guard bus rejects.

## Sizing and allocation
- Central (fluid) allocation uses score-shaping with beta pressure; local allocation uses local_loss.
- Sizer uses sl_pct or ATR-based stop (per exit profile) to compute risk and nominal size.
- Reservation happens before order placement; TTL decay runs per bar.

## Statistical consistency
- Candidate scoring uses the same win_prob/regime shaping in vectorized and scalar paths.
- enter_tag is strictly the string form of signal_id; no JSON metadata.
