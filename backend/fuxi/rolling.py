"""§4 三層異步滾動循環、抗遺忘視窗與無縫交接.

Loop 1 (every 5m close): inference only — lives in ``backtest.py``.
Loop 2 (every 7/14d): freeze ``(features, N/M/x/y)``, retrain LGBM weights +
Isotonic on the hybrid 70/30 dataset, produce a challenger, duel it against
the champion on the OOS window.
Loop 3 (every 30d or PSI > 0.25): feature-subset reselect + ``(N,M,x,y)``
plateau search via ``J(θ) = μ_SR − λ·σ_SR``; ``PASS_REGIME`` sleep when every
plateau median falls below 0.5 (§4.2 revenge-overfitting guard).

Orphan isolation (§4.3): every open position carries its birth
``PositionContract`` (model id, TP/SL, expiry). A champion swap never rewrites
an old contract; the new model only prices new entries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .validation import ProfitValidationSuite

EPS = 1e-8

# Promotion needs a material OOS edge over the champion (§4.3).
CHALLENGER_EPSILON_MARGIN = 0.0005
# Losing duel → defensive derate of the Kelly scalar (§4.3).
CHALLENGER_LOSS_THROTTLE = 0.75
# §4.2 revenge-overfitting guard floor. Canonical here because loop3 imports
# this module (the reverse import would be circular).
PASS_REGIME_MEDIAN_SR = 0.5

MODEL_ID_PREFIX = "FUXI"


@dataclass
class PositionContract:
    """Exit contract sealed at entry; owned by the model that opened it."""

    position_id: str
    model_id: str
    direction: int
    entry_price: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    open_bar_idx: int
    max_hold_bars: int
    x: float = 0.0
    y: float = 0.0


@dataclass
class TrainedModel:
    model_id: str
    booster: lgb.LGBMClassifier
    calibrator: IsotonicRegression
    params: Dict[str, float]
    feature_cols: List[str]

    def calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict_proba(X[self.feature_cols])[:, 1]
        return np.asarray(self.calibrator.transform(np.asarray(raw, dtype=float)))


class RollingClosedLoopController:
    """Owns the champion, challenger duels, and the orphan-contract registry."""

    def __init__(
        self,
        stress_ratio: float = 0.30,
        dsr_threshold: float = 0.95,
        dsr_trials: int = 30,
        dsr_trials_sharpe_std: float = 0.20,
    ) -> None:
        if not 0.0 <= stress_ratio < 1.0:
            raise ValueError("stress_ratio ρ must be in [0, 1)")
        self.stress_ratio = float(stress_ratio)
        self.dsr_threshold = float(dsr_threshold)
        self.dsr_trials = int(dsr_trials)
        self.dsr_trials_sharpe_std = float(dsr_trials_sharpe_std)

        self.champion: Optional[TrainedModel] = None
        self.kelly_penalty_scalar: float = 1.0
        self.open_contracts: Dict[str, PositionContract] = {}
        self.pass_regime: bool = False
        self._model_seq: int = 0

    # ------------------------------------------------------------------ #
    # champion helpers
    # ------------------------------------------------------------------ #
    @property
    def champion_id(self) -> str:
        return self.champion.model_id if self.champion else "INIT_CHAMPION_v0"

    @property
    def champion_params(self) -> Dict[str, float]:
        if self.champion:
            return dict(self.champion.params)
        return {"x": 0.025, "y": 2.0, "M": 15}

    def _next_model_id(self) -> str:
        self._model_seq += 1
        return f"{MODEL_ID_PREFIX}_C{self._model_seq:04d}_{uuid.uuid4().hex[:6]}"

    # ------------------------------------------------------------------ #
    # §4.2 hybrid anti-amnesia dataset
    # ------------------------------------------------------------------ #
    def build_hybrid_anti_amnesia_dataset(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        w_full: np.ndarray,
        recent_window_size: int = 2500,
        atr_col: str = "tf5m_norm_atr",
        stress_quantile: float = 0.95,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        n_total = len(X_full)
        if n_total <= recent_window_size:
            return X_full, y_full, np.asarray(w_full, dtype=float)

        X_recent = X_full.iloc[-recent_window_size:]
        y_recent = y_full.iloc[-recent_window_size:]
        w_recent = np.asarray(w_full, dtype=float)[-recent_window_size:]

        X_hist = X_full.iloc[:-recent_window_size]
        y_hist = y_full.iloc[:-recent_window_size]
        w_hist = np.asarray(w_full, dtype=float)[:-recent_window_size]

        if atr_col in X_hist.columns and len(X_hist) > 50:
            cutoff = float(np.percentile(X_hist[atr_col].to_numpy(dtype=float), stress_quantile * 100))
            stress_idx = np.where(X_hist[atr_col].to_numpy(dtype=float) >= cutoff)[0]
        else:
            stress_idx = np.arange(len(X_hist))

        if len(stress_idx) == 0:
            return X_recent, y_recent, w_recent

        n_target = int(recent_window_size * (self.stress_ratio / max(1.0 - self.stress_ratio, EPS)))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(stress_idx, size=min(n_target, len(stress_idx)), replace=False)

        X_hybrid = pd.concat([X_hist.iloc[chosen], X_recent], axis=0)
        y_hybrid = pd.concat([y_hist.iloc[chosen], y_recent], axis=0)
        w_hybrid = np.concatenate([w_hist[chosen], w_recent])
        return X_hybrid, y_hybrid, w_hybrid

    # ------------------------------------------------------------------ #
    # Loop 2: train challenger + duel
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_lgbm(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        w_train: np.ndarray,
        feature_cols: List[str],
        seed: int = 42,
        n_estimators: int = 80,
        max_depth: int = 3,
        learning_rate: float = 0.05,
    ) -> tuple[lgb.LGBMClassifier, IsotonicRegression]:
        n = len(X_train)
        split = max(1, int(n * 0.85))
        booster = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            verbose=-1,
        )
        booster.fit(
            X_train.iloc[:split][feature_cols],
            y_train.iloc[:split],
            sample_weight=np.asarray(w_train, dtype=float)[:split],
        )
        cal_raw = booster.predict_proba(X_train.iloc[split:][feature_cols])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.asarray(cal_raw, dtype=float), y_train.iloc[split:].to_numpy(dtype=float))
        return booster, calibrator

    def train_and_evaluate_challenger(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        w_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        val_realized_rets: np.ndarray,
        candidate_params: Dict[str, float],
        feature_cols: Optional[List[str]] = None,
        entry_threshold: float = 0.55,
        fee_per_trade: float = 0.0010,
        seed: int = 42,
    ) -> dict:
        """Train a challenger and duel it against the champion on OOS data.

        Promotion (§4.3): ``DSR(chal) ≥ threshold`` AND
        ``E_chal > E_champ + ε``. First model deploys directly.

        ``primary_dir`` is a label-side column (known only ex-post at bar t);
        it is excluded from model features unless explicitly requested.
        """
        cols = list(feature_cols) if feature_cols else [c for c in X_train.columns if c != "primary_dir"]
        cols = [c for c in cols if c in X_train.columns and c in X_val.columns]
        if not cols:
            raise ValueError("no usable feature columns")
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                f"empty train ({len(X_train)}) or validation ({len(X_val)}) frame — "
                "check the split before duelling"
            )
        booster, calibrator = self._fit_lgbm(X_train, y_train, w_train, cols, seed=seed)
        challenger = TrainedModel(
            self._next_model_id(), booster, calibrator, dict(candidate_params), cols
        )

        chal_probs = challenger.calibrated_proba(X_val)
        chal_mask = chal_probs > entry_threshold
        chal_rets = np.asarray(val_realized_rets, dtype=float)[chal_mask] - fee_per_trade
        chal_ev = float(np.mean(chal_rets)) if len(chal_rets) > 5 else -0.01
        # DSR must score the trades the challenger would actually take, not the
        # whole validation window (which includes bars it correctly skips).
        dsr_info = ProfitValidationSuite.compute_deflated_sharpe_ratio(
            chal_rets,
            num_trials=self.dsr_trials,
            trials_sharpe_std=self.dsr_trials_sharpe_std,
        )
        chal_dsr = float(dsr_info["dsr"])

        if self.champion is None:
            self.champion = challenger
            self.kelly_penalty_scalar = 1.0
            self.pass_regime = False
            return {
                "status": "INITIAL_CHAMPION_DEPLOYED",
                "active_id": self.champion.model_id,
                "chal_ev": chal_ev,
                "chal_dsr": chal_dsr,
            }

        champ_probs = self.champion.calibrated_proba(X_val)
        champ_mask = champ_probs > entry_threshold
        champ_rets = np.asarray(val_realized_rets, dtype=float)[champ_mask] - fee_per_trade
        champ_ev = float(np.mean(champ_rets)) if len(champ_rets) > 5 else -0.01

        if (
            chal_dsr >= self.dsr_threshold
            and chal_ev > 0.0
            and chal_ev > champ_ev + CHALLENGER_EPSILON_MARGIN
        ):
            prev_id = self.champion.model_id
            self.champion = challenger
            self.kelly_penalty_scalar = 1.0
            return {
                "status": "CHALLENGER_PROMOTED",
                "prev_id": prev_id,
                "active_id": self.champion.model_id,
                "chal_ev": chal_ev,
                "champ_ev": champ_ev,
                "chal_dsr": chal_dsr,
                "orphan_positions_locked": len(self.open_contracts),
            }

        self.kelly_penalty_scalar = CHALLENGER_LOSS_THROTTLE
        return {
            "status": "CHAMPION_RETAINED_WITH_THROTTLE",
            "active_id": self.champion.model_id,
            "chal_ev": chal_ev,
            "champ_ev": champ_ev,
            "chal_dsr": chal_dsr,
            "kelly_penalty_scalar": self.kelly_penalty_scalar,
        }

    # ------------------------------------------------------------------ #
    # §4.3 orphan isolation: positions carry their birth contract
    # ------------------------------------------------------------------ #
    def register_new_position(
        self,
        pos_id: str,
        direction: int,
        entry_price: float,
        margin: float,
        leverage: float,
        bar_idx: int,
    ) -> PositionContract:
        p = self.champion_params
        x, y, M = float(p["x"]), float(p["y"]), int(p["M"])
        sl_pct = x / y
        contract = PositionContract(
            position_id=pos_id,
            model_id=self.champion_id,
            direction=int(direction),
            entry_price=float(entry_price),
            margin=float(margin),
            leverage=float(leverage),
            tp_price=float(entry_price) * (1.0 + direction * x),
            sl_price=float(entry_price) * (1.0 - direction * sl_pct),
            open_bar_idx=int(bar_idx),
            max_hold_bars=M,
            x=x,
            y=y,
        )
        self.open_contracts[pos_id] = contract
        return contract

    def close_position(self, pos_id: str) -> Optional[PositionContract]:
        return self.open_contracts.pop(pos_id, None)
