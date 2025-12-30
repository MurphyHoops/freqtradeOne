# Plugin package for entry signals.

SIGNAL_PLUGIN_MAP = {
    "mean_rev_long": "user_data.strategies.plugins.signals.mean_reversion",
    "pullback_long": "user_data.strategies.plugins.signals.pullback",
    "trend_short": "user_data.strategies.plugins.signals.trend_short",
    "newbars_breakout_long_5m": "user_data.strategies.plugins.signals.newbars",
    "newbars_breakout_long_30m": "user_data.strategies.plugins.signals.newbars",
    "newbars_breakdown_short_5m": "user_data.strategies.plugins.signals.newbars",
    "newbars_breakdown_short_30m": "user_data.strategies.plugins.signals.newbars",
}

__all__ = ["SIGNAL_PLUGIN_MAP"]
