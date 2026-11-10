"""§5.3 非對稱階梯式棘輪複利與安全金庫.

``active_pool`` trades; ``reserve_vault`` never re-enters risk. At each flat
settlement that prints a new high-water mark, ``(1 - α)`` of the incremental
profit is skimmed into the vault and the Kelly base only steps up by ``α``.
Intraday drawdown shrinks the base immediately (asymmetric defence).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RatchetStatus:
    total_equity: float
    active_pool: float
    reserve_vault: float
    ratchet_hwm: float
    kelly_base_capital: float

    def as_dict(self) -> dict:
        return {
            "total_equity": self.total_equity,
            "active_pool": self.active_pool,
            "reserve_vault": self.reserve_vault,
            "ratchet_hwm": self.ratchet_hwm,
            "kelly_base_capital": self.kelly_base_capital,
        }


class AsymmetricCompoundingRatchet:
    """Active pool + cold vault with stepwise ratcheted Kelly base."""

    def __init__(self, initial_capital: float = 1000.0, reinvest_ratio: float = 0.60) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0.0 <= reinvest_ratio <= 1.0:
            raise ValueError("reinvest_ratio α must be in [0, 1]")
        self.w0 = float(initial_capital)
        self.alpha = float(reinvest_ratio)
        self.active_pool = float(initial_capital)
        self.reserve_vault = 0.0
        self.ratchet_hwm = float(initial_capital)
        self.period_start_active = float(initial_capital)

    # ------------------------------------------------------------------ #
    def update_intraday_pnl(self, net_pnl_u: float) -> None:
        self.active_pool = max(0.0, self.active_pool + float(net_pnl_u))

    def settle_rolling_period(self, is_flat: bool = True) -> RatchetStatus:
        """Skim ``(1-α)·ΔΠ`` into the vault on a new HWM, then lock the base."""
        if is_flat:
            total = self.active_pool + self.reserve_vault
            if total > self.ratchet_hwm:
                new_profit = total - self.ratchet_hwm
                skimmed = (1.0 - self.alpha) * new_profit
                self.reserve_vault += skimmed
                self.active_pool -= skimmed
                self.ratchet_hwm = total
            self.period_start_active = self.active_pool
        return self.get_status()

    def get_ratcheted_base_capital(self) -> float:
        intraday_dd = max(0.0, self.period_start_active - self.active_pool)
        locked_growth = self.alpha * max(0.0, self.ratchet_hwm - self.w0)
        w_base = self.w0 + locked_growth - intraday_dd
        return float(max(0.0, min(w_base, self.active_pool)))

    def get_status(self) -> RatchetStatus:
        return RatchetStatus(
            total_equity=self.active_pool + self.reserve_vault,
            active_pool=self.active_pool,
            reserve_vault=self.reserve_vault,
            ratchet_hwm=self.ratchet_hwm,
            kelly_base_capital=self.get_ratcheted_base_capital(),
        )
