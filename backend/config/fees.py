"""Single source of truth for trading fee + slippage assumptions.

Both the paper-trading path (``services/paper_broker.py``) and the backtest
path (``backtest/execution_simulator.py``) must price the same trade the same
way. Before this module existed the two paths hardcoded different numbers
(paper: 1.5bp/4.5bp/2bp vs backtest: 0.035%/0.02%/0.05%), which made paper
verification of backtested strategies meaningless.

Conventions
-----------
* All rates are stored in **basis points** (1 bp = 0.01% = 0.0001).
* Unit conversion (``percent = bps / 100``, ``fraction = bps / 10000``)
  happens **once at the boundary** (constructor / ``from_percent``), never
  scattered across call sites.
* Defaults come from env so ops can align them without a code change:
  ``MAKER_FEE_BPS`` (1.5), ``TAKER_FEE_BPS`` (4.5), ``SLIPPAGE_BPS`` (2.0).
  Explicit constructor args win unless the corresponding env var is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_from_env(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class FeeSchedule:
    """Maker/taker fee + slippage schedule, all in basis points."""

    maker_fee_bps: float = field(default_factory=lambda: _default_from_env("MAKER_FEE_BPS", 1.5))
    taker_fee_bps: float = field(default_factory=lambda: _default_from_env("TAKER_FEE_BPS", 4.5))
    slippage_bps: float = field(default_factory=lambda: _default_from_env("SLIPPAGE_BPS", 2.0))

    @classmethod
    def from_percent(
        cls,
        *,
        maker_fee_pct: float,
        taker_fee_pct: float,
        slippage_pct: float,
    ) -> "FeeSchedule":
        """Build from percent-unit values (``percent = bps / 100``).

        This is the single boundary conversion point for callers that still
        speak in percent (e.g. legacy ``BacktestConfig`` fields).
        """
        return cls(
            maker_fee_bps=maker_fee_pct * 100.0,
            taker_fee_bps=taker_fee_pct * 100.0,
            slippage_bps=slippage_pct * 100.0,
        )

    def fee(self, notional: float, is_maker: bool = False) -> float:
        """Fee in quote currency for a notional."""
        bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        return abs(notional) * bps / 10000.0

    def entry_price(self, mid: float, is_buy: bool) -> float:
        """Fill price for opening (with slippage). 買入滑高，賣出滑低。"""
        adj = self.slippage_bps / 10000.0
        return mid * (1 + adj) if is_buy else mid * (1 - adj)

    def exit_price(self, mid: float, is_buy: bool) -> float:
        """Fill price for closing (with slippage). 賣出滑低，買回滑高。"""
        adj = self.slippage_bps / 10000.0
        return mid * (1 - adj) if is_buy else mid * (1 + adj)


DEFAULT_FEES = FeeSchedule()
