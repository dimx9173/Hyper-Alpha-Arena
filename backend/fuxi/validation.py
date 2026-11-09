"""§7 機構級收益驗證四大關卡.

Gate 1 — Deflated Sharpe Ratio (Bailey & López de Prado): the annualized
Sharpe estimate must clear the expected maximum under the null of
``num_trials`` tries, corrected for skew/kurtosis. ``dsr >= 0.95`` to promote.

Gate 2 — Monte Carlo trade reshuffling (B=10_000): 95th-percentile max
drawdown ≤ 20% and ruin (MDD ≥ 30%) probability < 0.5%.

Gate 3 — Friction stress: at 1.5× two-sided cost, annualized SR stays > 1.0
and net-profit retention ≥ 50%.

Gate 4 — Monthly consistency: winning-month share ≥ 65% (single-sided — the
old 85% ceiling rejected systems that win too often), max time underwater
≤ 30 days, net retention after removing top-5 trades ≥ 50%. A sample spanning
fewer than three months is annotated with a reason instead of being forced to
a verdict.

``PromotionGate.evaluate`` ties all four to the §4.3 handoff rule. Every gate
prices cost through ``config.fees.FeeSchedule`` — the repo single source of
truth — never through a hardcoded per-trade number.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:  # keep `fuxi` importable standalone
    sys.path.insert(0, str(_BACKEND))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kurtosis, norm, skew  # noqa: E402

from config.fees import FeeSchedule  # noqa: E402

from .risk import round_trip_cost_fractions  # noqa: E402

MC_SIMULATIONS = 10_000
MC_MDD_LIMIT = 0.20
MC_RUIN_THRESHOLD = 0.30
MC_RUIN_LIMIT = 0.005

FRICTION_MULTIPLIER = 1.5
FRICTION_MIN_SR = 1.0
FRICTION_MIN_RETENTION = 0.50

MONTHLY_WIN_SHARE_LO = 0.65
MONTHLY_MIN_MONTHS = 3
MAX_TUW_DAYS = 30
TOP5_MIN_RETENTION = 0.50

TRADING_DAYS_PER_YEAR = 365.0


def infer_periods_per_year(timestamps) -> float:
    """Annualization period count implied by a timestamp span.

    ``timestamps`` is either a ``pd.DatetimeIndex`` (5m bars, daily bars, …) or
    an integer array of epoch milliseconds. Returns observations per year; the
    caller combines it with the number of trades to get trades/year.

    Raises ``ValueError`` when the span cannot support an annualization, so no
    caller silently inherits a fabricated unit scale.
    """
    if isinstance(timestamps, pd.DatetimeIndex):
        ts = np.asarray(timestamps.asi8, dtype=np.int64) // 10**6
    else:
        ts = np.asarray(timestamps, dtype=np.int64)
    if ts.size < 2:
        raise ValueError("cannot infer periods_per_year from fewer than 2 timestamps")
    span_days = float(np.max(ts) - np.min(ts)) / 86_400_000.0
    if not np.isfinite(span_days) or span_days <= 0.0:
        raise ValueError(
            "cannot infer periods_per_year: timestamps span 0 days (all exits at one instant)"
        )
    return float(ts.size) * TRADING_DAYS_PER_YEAR / span_days


class ProfitValidationSuite:
    """The four institutional gates as pure functions of trade returns."""

    # ------------------------------------------------------------------ #
    # Gate 1: DSR
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_deflated_sharpe_ratio(
        returns: np.ndarray,
        *,
        periods_per_year: float,
        num_trials: int = 100,
        trials_sharpe_std: float = 0.5,
    ) -> dict:
        """Deflated Sharpe ratio on an **annualized** Sharpe scale.

        ``returns`` are per-period (per-trade) returns and ``periods_per_year``
        is how many of them make a year (365 for daily bars, trades/year for
        trade returns). Per review F1 the unit scale is explicit and mandatory:

        * ``sr_ann = sr_per_unit * sqrt(periods_per_year)``
        * ``T`` is the sample length in **years** (``n / periods_per_year``)
        * ``SR0`` keeps the original expected-maximum formula
        * the skew/kurtosis denominator is evaluated on ``sr_ann``

        There is no ``trades_per_year=None → n`` fallback: a caller that cannot
        state its annualization must not receive a number.
        """
        ppy = float(periods_per_year)
        if not np.isfinite(ppy) or ppy <= 0.0:
            raise ValueError(
                f"periods_per_year must be positive and finite, got {periods_per_year!r}"
            )
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        T = len(r)
        if T < 10 or float(np.std(r, ddof=1)) == 0.0:
            return {
                "sr_est": 0.0,
                "sr_ann": 0.0,
                "sr_null": 0.0,
                "dsr": 0.0,
                "n": T,
                "years": 0.0,
                "periods_per_year": ppy,
            }

        sr_unit = float(np.mean(r) / np.std(r, ddof=1))
        sr_ann = sr_unit * float(np.sqrt(ppy))
        sk = float(skew(r))
        kr = float(kurtosis(r, fisher=False))
        gamma = 0.5772156649
        K = max(2, int(num_trials))
        z1 = float(norm.ppf(1.0 - 1.0 / K))
        z2 = float(norm.ppf(1.0 - 1.0 / (K * np.e)))
        sr_null = float(trials_sharpe_std * ((1.0 - gamma) * z1 + gamma * z2))

        denom_var = 1.0 - sk * sr_ann + ((kr - 1.0) / 4.0) * (sr_ann**2)
        denom = float(np.sqrt(max(denom_var, 1e-8)))
        years = T / ppy
        if years < 1.0:
            # A sub-one-year sample cannot support an "annual" confidence level:
            # return the point estimates with an explicit short-sample flag
            # instead of clamping to 1e-8 and inventing a verdict.
            return {
                "sr_est": sr_ann,
                "sr_ann": sr_ann,
                "sr_unit": sr_unit,
                "sr_null": sr_null,
                "dsr": 0.0,
                "n": T,
                "years": years,
                "periods_per_year": ppy,
                "short_sample": True,
            }
        z = ((sr_ann - sr_null) * np.sqrt(years - 1.0)) / denom
        return {
            # ``sr_est`` is the annualized Sharpe actually tested (kept under
            # its historical key so existing report consumers stay valid).
            "sr_est": sr_ann,
            "sr_ann": sr_ann,
            "sr_unit": sr_unit,
            "sr_null": sr_null,
            "dsr": float(norm.cdf(z)),
            "n": T,
            "years": years,
            "periods_per_year": ppy,
        }

    # ------------------------------------------------------------------ #
    # Gate 2: Monte Carlo reshuffling
    # ------------------------------------------------------------------ #
    @staticmethod
    def run_monte_carlo_reshuffling(
        trade_returns: np.ndarray,
        n_simulations: int = MC_SIMULATIONS,
        ruin_threshold: float = MC_RUIN_THRESHOLD,
        seed: int = 42,
    ) -> dict:
        tr = np.asarray(trade_returns, dtype=float)
        tr = tr[np.isfinite(tr)]
        n = len(tr)
        if n == 0:
            return {"mdd_median": 0.0, "mdd_95th": 0.0, "risk_of_ruin": 0.0, "n": 0}

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_simulations, n))
        sims = tr[idx]
        curves = np.cumprod(1.0 + sims, axis=1)
        peaks = np.maximum.accumulate(curves, axis=1)
        dds = (peaks - curves) / np.maximum(peaks, 1e-8)
        max_dds = np.max(dds, axis=1)
        return {
            "mdd_median": float(np.median(max_dds)),
            "mdd_95th": float(np.percentile(max_dds, 95)),
            "risk_of_ruin": float(np.mean(max_dds >= ruin_threshold)),
            "n": n,
        }

    # ------------------------------------------------------------------ #
    # Gate 3: friction stress
    # ------------------------------------------------------------------ #
    @staticmethod
    def friction_sensitivity(
        trade_returns: np.ndarray,
        *,
        cost_loss: float,
        periods_per_year: float,
        multiplier: float = FRICTION_MULTIPLIER,
        funding_cost: float = 0.0,
    ) -> dict:
        """Re-price every trade at ``multiplier``x friction; check SR and retention.

        The trades here are a *mixed* set with no win/loss labels, so the
        conservative round-trip leg (``cost_loss``: taker+slippage on both legs)
        plus ``funding_cost`` is charged - review F3. SR is annualized with the
        caller-supplied ``periods_per_year`` (trades/year); there is no implicit
        ``n`` fallback.
        """
        ppy = float(periods_per_year)
        if not np.isfinite(ppy) or ppy <= 0.0:
            raise ValueError(
                f"periods_per_year must be positive and finite, got {periods_per_year!r}"
            )
        tr = np.asarray(trade_returns, dtype=float)
        tr = tr[np.isfinite(tr)]
        base_total = float(np.sum(tr))
        if len(tr) == 0 or base_total <= 0:
            return {
                "passed": False,
                "sr": 0.0,
                "retention": 0.0,
                "stressed_total": 0.0,
                "periods_per_year": ppy,
            }

        extra_cost = (multiplier - 1.0) * (cost_loss + funding_cost)
        stressed = tr - extra_cost
        stressed_total = float(np.sum(stressed))
        retention = stressed_total / base_total if base_total != 0 else 0.0

        per_trade = float(np.mean(stressed) / (np.std(stressed, ddof=1) + 1e-12))
        sr = per_trade * float(np.sqrt(ppy))
        passed = bool(sr > FRICTION_MIN_SR and retention >= FRICTION_MIN_RETENTION)
        return {
            "passed": passed,
            "sr": float(sr),
            "retention": float(retention),
            "stressed_total": stressed_total,
            "periods_per_year": ppy,
        }

    # ------------------------------------------------------------------ #
    # Gate 4: monthly consistency
    # ------------------------------------------------------------------ #
    @staticmethod
    def monthly_consistency(
        trade_returns: np.ndarray,
        trade_timestamps_ms: np.ndarray,
        max_tuw_days: int = MAX_TUW_DAYS,
    ) -> dict:
        tr = np.asarray(trade_returns, dtype=float)
        ts = np.asarray(trade_timestamps_ms, dtype=np.int64)
        if len(tr) == 0:
            return {"passed": False, "reason": "no trades"}

        order = np.argsort(ts)
        tr, ts = tr[order], ts[order]
        months = pd.to_datetime(ts, unit="ms", utc=True).tz_convert(None).to_period("M")
        monthly = pd.Series(tr).groupby(months).sum()
        n_months = len(monthly)
        win_share = float((monthly > 0).mean()) if n_months else 0.0

        equity = np.cumprod(1.0 + tr)
        peaks = np.maximum.accumulate(equity)
        underwater = equity < peaks
        # longest underwater stretch measured in calendar days
        tuw_days = 0.0
        if underwater.any():
            days = (ts - ts[0]) / 86_400_000.0
            best, start = 0.0, None
            for i, uw in enumerate(underwater):
                if uw and start is None:
                    start = days[i]
                if not uw and start is not None:
                    best = max(best, days[i] - start)
                    start = None
            if start is not None:
                best = max(best, days[-1] - start)
            tuw_days = float(best)

        total = float(np.sum(tr))
        top5 = float(np.sum(np.sort(tr)[-5:])) if len(tr) >= 5 else float(np.sum(tr[tr > 0]))
        retention = (total - top5) / total if total > 0 else 0.0

        checks = {
            # Single-sided: ≥65% winning months. The old [65%, 85%] band
            # rejected a system precisely for winning more often than 85%,
            # which is self-refuting (review F2).
            "win_share_ok": bool(win_share >= MONTHLY_WIN_SHARE_LO),
            "tuw_ok": bool(tuw_days <= max_tuw_days),
            "top5_retention_ok": bool(retention >= TOP5_MIN_RETENTION),
        }
        out = {
            "passed": bool(all(checks.values())),
            "win_share": win_share,
            "n_months": int(n_months),
            "max_tuw_days": tuw_days,
            "top5_retention": float(retention),
            **checks,
        }
        if n_months < MONTHLY_MIN_MONTHS:
            # Too few months to judge consistency: report why the number is
            # thin instead of inventing a verdict from one month of data.
            out["reason"] = (
                f"n_months {n_months} < {MONTHLY_MIN_MONTHS}: monthly share is not "
                "statistically meaningful over so short a sample"
            )
        return out


class PromotionGate:
    """§9 governance: all four gates must pass before a challenger promotes."""

    def __init__(
        self,
        dsr_min: float = 0.95,
        mdd95_max: float = MC_MDD_LIMIT,
        ruin_max: float = MC_RUIN_LIMIT,
        fees: Optional[FeeSchedule] = None,
        funding_cost: float = 0.0,
        dsr_trials: int = 100,
        dsr_trials_std: float = 0.5,
    ) -> None:
        self.dsr_min = dsr_min
        self.mdd95_max = mdd95_max
        self.ruin_max = ruin_max
        self.fees = fees or FeeSchedule()
        self.funding_cost = float(funding_cost)
        self.dsr_trials = dsr_trials
        self.dsr_trials_std = dsr_trials_std
        # Cost legs come from FeeSchedule (§5.1), never a hardcoded 8bp/10bp.
        self.cost_win, self.cost_loss = round_trip_cost_fractions(self.fees, self.funding_cost)

    def evaluate(
        self, trade_returns: np.ndarray, trade_timestamps_ms: np.ndarray
    ) -> dict:
        # These are per-trade returns, so the annualization unit is trades/year,
        # inferred from the exit-timestamp span.
        periods_per_year = infer_periods_per_year(trade_timestamps_ms)
        dsr = ProfitValidationSuite.compute_deflated_sharpe_ratio(
            trade_returns,
            periods_per_year=periods_per_year,
            num_trials=self.dsr_trials,
            trials_sharpe_std=self.dsr_trials_std,
        )
        mc = ProfitValidationSuite.run_monte_carlo_reshuffling(trade_returns)
        fr = ProfitValidationSuite.friction_sensitivity(
            trade_returns,
            cost_loss=self.cost_loss,
            periods_per_year=periods_per_year,
            funding_cost=self.funding_cost,
        )
        mo = ProfitValidationSuite.monthly_consistency(trade_returns, trade_timestamps_ms)
        verdict = bool(
            dsr["dsr"] >= self.dsr_min
            and mc["mdd_95th"] <= self.mdd95_max
            and mc["risk_of_ruin"] < self.ruin_max
            and fr["passed"]
            and mo["passed"]
        )
        return {
            "passed": verdict,
            "dsr": dsr,
            "monte_carlo": mc,
            "friction": fr,
            "monthly": mo,
        }
