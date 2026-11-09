"""§5.1/5.2/5.4 永續淨期望值、強平防禦、凱利注碼與月度平滑.

Sizing pipeline per signal bar::

    net EV  →  liquidation-safe leverage cap  →  fractional Kelly × η_vol × η_month
    × challenger-throttle  →  discrete bucket margin  →  guardrail cap

Fee inputs arrive in **basis points** via ``config.fees.FeeSchedule`` (the repo
single source of truth). ``expected_loss_pct`` (``x/y``) and ``win_pct``
(``x``) are fractions. Conversions happen here at the boundary, once.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:  # keep `fuxi` importable standalone
    sys.path.insert(0, str(_BACKEND))

from config.fees import FeeSchedule  # noqa: E402
from config.guardrails import MAX_SINGLE_COIN  # noqa: E402

EPS = 1e-8

# §5.4 defaults
DEFAULT_MONTHLY_PROFIT_TARGET = 0.10
DEFAULT_MONTHLY_SOFT_DD = 0.03
DEFAULT_MONTHLY_HARD_DD = 0.06
DEFAULT_MONTHLY_DD_TOLERANCE = 0.08

MIN_EV_THRESHOLD = 0.0015  # δ_min: net-EV gate per trade


@dataclass
class SizingDecision:
    margin: float
    leverage: float
    exposure: float
    ev: float
    eta_vol: float = 1.0
    eta_month: float = 1.0
    month_mode: str = "MONTHLY_NORMAL"
    approved: bool = False

    def as_dict(self) -> dict:
        return {
            "margin": self.margin,
            "leverage": self.leverage,
            "exposure": self.exposure,
            "ev": self.ev,
            "eta_vol": self.eta_vol,
            "eta_month": self.eta_month,
            "month_mode": self.month_mode,
            "approved": self.approved,
        }


class PerpetualRiskAndSmoothingEngine:
    """Net-EV gate + liquidation cap + fractional Kelly + monthly throttle."""

    def __init__(
        self,
        kelly_c: float = 0.25,
        max_cap: float = 0.20,
        mmr: float = 0.004,
        liq_buffer: float = 0.005,
        target_atr_norm: float = 0.004,
        monthly_profit_target: float = DEFAULT_MONTHLY_PROFIT_TARGET,
        monthly_soft_dd: float = DEFAULT_MONTHLY_SOFT_DD,
        monthly_hard_dd: float = DEFAULT_MONTHLY_HARD_DD,
        discrete_buckets: Optional[List[float]] = None,
        fees: Optional[FeeSchedule] = None,
    ) -> None:
        self.c = float(kelly_c)
        self.max_cap = float(max_cap)
        self.mmr = float(mmr)
        self.liq_buffer = float(liq_buffer)
        self.target_atr_norm = float(target_atr_norm)
        self.monthly_profit_target = float(monthly_profit_target)
        self.monthly_soft_dd = float(monthly_soft_dd)
        self.monthly_hard_dd = float(monthly_hard_dd)
        self.buckets = sorted(discrete_buckets) if discrete_buckets else None
        self.fees = fees or FeeSchedule()

    # ------------------------------------------------------------------ #
    # §5.4 monthly + vol scalars
    # ------------------------------------------------------------------ #
    def compute_monthly_and_vol_scalars(
        self,
        current_equity: float,
        month_start_equity: float,
        month_hwm_equity: float,
        current_norm_atr: float,
    ) -> tuple[float, float, str]:
        eta_vol = float(np.clip(self.target_atr_norm / (current_norm_atr + EPS), 0.5, 1.5))
        mtd_return = (current_equity - month_start_equity) / (month_start_equity + EPS)
        mtd_dd = (month_hwm_equity - current_equity) / (month_hwm_equity + EPS)

        if mtd_return <= -self.monthly_hard_dd or mtd_dd >= DEFAULT_MONTHLY_DD_TOLERANCE:
            return eta_vol, 0.0, "MONTHLY_HALT_CIRCUIT_BREAKER"
        if mtd_return <= -self.monthly_soft_dd:
            return eta_vol, 0.5, "MONTHLY_DEFENSIVE_THROTTLE"
        if mtd_return >= self.monthly_profit_target:
            return eta_vol, 0.5, "MONTHLY_PROFIT_LOCK_SKIMMING"
        return eta_vol, 1.0, "MONTHLY_NORMAL"

    # ------------------------------------------------------------------ #
    # §5.1 net expected value
    # ------------------------------------------------------------------ #
    def net_expected_value(
        self,
        prob_cal: float,
        x: float,
        y: float,
        direction: int,
        funding_rate: float = 0.0,
        expected_funding_counts: int = 1,
    ) -> tuple[float, float, float]:
        """Returns ``(ev, cost_win, cost_loss)`` as fractions of notional."""
        maker = self.fees.maker_fee_bps / 10000.0
        taker = self.fees.taker_fee_bps / 10000.0
        slip = self.fees.slippage_bps / 10000.0
        funding_cost = direction * funding_rate * expected_funding_counts

        cost_win = maker + maker + funding_cost
        cost_loss = maker + taker + slip + funding_cost
        net_win = x - cost_win
        net_loss = x / y + cost_loss
        if net_win <= 0 or net_loss <= 0:
            return -1.0, cost_win, cost_loss
        ev = prob_cal * net_win - (1.0 - prob_cal) * net_loss
        return ev, cost_win, cost_loss

    # ------------------------------------------------------------------ #
    # §5.2 liquidation-safe leverage cap
    # ------------------------------------------------------------------ #
    def max_safe_leverage(self, expected_loss_pct: float, funding_cost: float = 0.0) -> float:
        """``z_safe = floor(1 / (x/y + Cost_loss + MMR + buffer))``, min 1."""
        maker = self.fees.maker_fee_bps / 10000.0
        taker = self.fees.taker_fee_bps / 10000.0
        slip = self.fees.slippage_bps / 10000.0
        cost_loss = maker + taker + slip + funding_cost
        denom = expected_loss_pct + cost_loss + self.mmr + self.liq_buffer
        if denom <= 0:
            return 1.0
        return float(max(1.0, math.floor(1.0 / denom)))

    # ------------------------------------------------------------------ #
    # §5.1 + §5.4 composite sizing
    # ------------------------------------------------------------------ #
    def compute_safe_sizing(
        self,
        prob_cal: float,
        direction: int,
        x: float,
        y: float,
        target_leverage: float,
        ratcheted_base_capital: float,
        total_equity: float,
        month_start_equity: float,
        month_hwm_equity: float,
        rolling_penalty_scalar: float = 1.0,
        current_norm_atr: float = 0.004,
        funding_rate: float = 0.0001,
        expected_funding_counts: int = 1,
    ) -> SizingDecision:
        sl_pct = x / y
        funding_cost = direction * funding_rate * expected_funding_counts

        ev, cost_win, cost_loss = self.net_expected_value(
            prob_cal, x, y, direction, funding_rate, expected_funding_counts
        )
        eta_vol, eta_month, month_mode = self.compute_monthly_and_vol_scalars(
            total_equity, month_start_equity, month_hwm_equity, current_norm_atr
        )
        if ev <= MIN_EV_THRESHOLD or eta_month == 0.0:
            return SizingDecision(0.0, 0.0, 0.0, ev, eta_vol, eta_month, month_mode, False)

        actual_leverage = float(min(target_leverage, self.max_safe_leverage(sl_pct, funding_cost)))

        net_win = x - cost_win
        net_loss = sl_pct + cost_loss
        b_net = net_win / (net_loss + EPS)
        f_star = prob_cal - (1.0 - prob_cal) / (b_net + EPS)
        if f_star <= 0.0:
            return SizingDecision(0.0, actual_leverage, 0.0, ev, eta_vol, eta_month, month_mode, False)

        f_final = min(f_star * self.c * eta_vol * eta_month * rolling_penalty_scalar, self.max_cap)
        raw_margin = ratcheted_base_capital * f_final

        if self.buckets:
            valid = [b for b in self.buckets if b <= raw_margin]
            final_margin = float(max(valid)) if valid else 0.0
        else:
            final_margin = float(raw_margin)

        # Deploy guardrail: single-coin notional may not exceed MAX_SINGLE_COIN.
        if total_equity > 0:
            cap_margin = MAX_SINGLE_COIN * total_equity / max(actual_leverage, EPS)
            if final_margin * 1.0 > cap_margin and self.buckets:
                valid = [b for b in self.buckets if b <= min(raw_margin, cap_margin)]
                final_margin = float(max(valid)) if valid else 0.0
            else:
                final_margin = float(min(final_margin, cap_margin))

        return SizingDecision(
            margin=final_margin,
            leverage=actual_leverage,
            exposure=final_margin * actual_leverage,
            ev=ev,
            eta_vol=eta_vol,
            eta_month=eta_month,
            month_mode=month_mode,
            approved=final_margin > 0,
        )
