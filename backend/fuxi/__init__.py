"""伏羲動態滾動量化系統 (Fuxi-Adaptive Rolling Quant) v4.0.

Production implementation of `docs/fuxi-adaptive-rolling-quant/fuxi-adaptive-rolling-quant.md`.

Module map (§ numbers refer to the spec):

* ``features``   §2   Point-in-time multi-timeframe features + collinearity prune
* ``labels``     §3   Primary direction + directional triple barrier + sample uniqueness
* ``ratchet``    §5.3 Asymmetric compounding ratchet + cold reserve vault
* ``risk``       §5.1/5.2/5.4 Net EV, liquidation guard, fractional Kelly, monthly smoothing
* ``rolling``    §4   Hybrid anti-amnesia dataset, champion/challenger, orphan isolation
* ``validation`` §7   DSR / Monte Carlo / friction / monthly consistency gates
* ``loop3``      §6.2+§4.1 Parameter-plateau search over (N, M, x, y) + feature reselect
* ``data``       --   Binance USDT-M 5m/15m/30m feed (5m base axis)
* ``backtest``   --   Bar-by-bar closed-loop runner wiring the above together
"""

from __future__ import annotations

__all__ = [
    "features",
    "labels",
    "ratchet",
    "risk",
    "rolling",
    "validation",
    "loop3",
    "data",
    "backtest",
]
