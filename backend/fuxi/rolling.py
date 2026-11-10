"""§4 三層異步滾動循環、抗遺忘視窗與無縫交接.

Loop 1 (every 5m close): inference only — lives in ``backtest.py``.
Loop 2 (every 7/14d): freeze ``(features, N/M/x/y)``, retrain LGBM weights +
Isotonic on the hybrid 70/30 dataset, produce a challenger, duel it against
the champion on the OOS window.
Loop 3 (every 30d): feature-subset reselect + ``(N,M,x,y)``
plateau search via ``J(θ) = μ_SR − λ·σ_SR``; ``PASS_REGIME`` sleep when every
plateau's max mean **annualized** Sharpe falls below 0.5 (§4.2
revenge-overfitting guard, review F4).

Orphan isolation (§4.3): every open position carries its birth
``PositionContract`` (model id, TP/SL, expiry). A champion swap never rewrites
an old contract; the new model only prices new entries.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config.fees import FeeSchedule  # noqa: E402

from .risk import round_trip_cost_fractions  # noqa: E402
from .validation import ProfitValidationSuite, infer_periods_per_year  # noqa: E402

EPS = 1e-8

# Promotion needs a material OOS edge over the champion (§4.3).
CHALLENGER_EPSILON_MARGIN = 0.0005
# Losing duel → defensive derate of the Kelly scalar (§4.3).
CHALLENGER_LOSS_THROTTLE = 0.75
# §4.2 revenge-overfitting guard floor, in ANNUALIZED Sharpe units (same scale
# as the §7 friction gate's ``sr``; review F4). Canonical here because loop3
# imports this module (the reverse import would be circular).
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

    def save(self, directory: Path) -> Path:
        """把冠軍模型落盤，讓重啟後的進程不用重訓即可推理。

        目錄結構：``booster.txt`` + ``isotonic.npz`` + ``meta.json``。

        為何不用 pickle：pickle 把 sklearn/LGBM 物件的類別路徑與內部
        記憶體佈局一起序列化，跨版本（sklearn/LGBM 升級、欄位增刪）即
        反序列化失敗或靜默錯位；且 pickle 可執行任意程式碼，不適合當
        模型存檔格式。這裡只存「重建所需的最小資料」。

        為何存 LGBM 原生格式：``booster_.save_model`` 寫出的是純文字
        樹結構（split/gain/leaf value），不依賴 Python 物件佈局，
        跨版本可讀，且 ``lgb.Booster(model_file=...)`` 載入後的預測值
        與原模型逐位一致。
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        booster_path = directory / "booster.txt"
        iso_path = directory / "isotonic.npz"
        meta_path = directory / "meta.json"
        # LGBM 原生文字格式：只存樹，不存 Python 包裝層。
        self.booster.booster_.save_model(str(booster_path))
        # Isotonic 只存擬合陣列：重建時調 _build_f 重做插值函式即可，
        # 不 pickle 內含 scipy interp1d 閉包的 sklearn 物件。
        cal = self.calibrator
        np.savez(
            str(iso_path),
            X_thresholds_=np.asarray(cal.X_thresholds_),
            y_thresholds_=np.asarray(cal.y_thresholds_),
            X_min_=np.asarray(cal.X_min_),
            X_max_=np.asarray(cal.X_max_),
        )
        _objective = getattr(self.booster, "_objective", None)
        meta = {
            "model_id": self.model_id,
            "params": dict(self.params),
            "feature_cols": list(self.feature_cols),
            "lgbm_params": self.booster.get_params(),
            "classes": np.asarray(self.booster.classes_).tolist(),
            "classes_dtype": str(np.asarray(self.booster.classes_).dtype),
            "n_classes": int(self.booster.n_classes_),
            "n_features": int(getattr(self.booster, "n_features_in_", self.booster._Booster.num_feature())),
            "fitted_with_feature_names": bool(getattr(self.booster, "_fitted_with_feature_names", False)),
            "objective": _objective if isinstance(_objective, str) else None,
            "calibrator": {
                "y_min": cal.y_min,
                "y_max": cal.y_max,
                "increasing": cal.increasing,
                "increasing_": bool(getattr(cal, "increasing_", cal.increasing is True)),
                "out_of_bounds": cal.out_of_bounds,
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "TrainedModel":
        """``save`` 的逆操作：重建 booster 與 calibrator，推理逐位一致。"""
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        # LGBM 包裝層用建構參數重建，樹本體從原生文字檔載入。
        booster = lgb.LGBMClassifier(**meta["lgbm_params"])
        booster._Booster = lgb.Booster(model_file=str(directory / "booster.txt"))
        booster._classes = np.asarray(meta["classes"], dtype=np.dtype(meta["classes_dtype"]))
        booster._n_classes = int(meta["n_classes"])
        booster._objective = meta["objective"]
        # predict() 走 _le.inverse_transform；predict_proba 不用它，但一併
        # 重建才能讓載入後的模型保有完整 sklearn 介面。_LGBMLabelEncoder
        # 是 lightgbm.sklearn 的私有類，版本升級可能消失：延遲匯入並容忍
        # 失敗——推理路徑（calibrated_proba → predict_proba）不依賴它。
        try:
            from lightgbm.sklearn import _LGBMLabelEncoder
        except ImportError:  # pragma: no cover - 新版 lightgbm 可能改名
            _LGBMLabelEncoder = None
        if _LGBMLabelEncoder is not None:
            booster._le = _LGBMLabelEncoder()
            booster._le.classes_ = booster._classes
        booster.fitted_ = True
        # 補回 sklearn 側的擬合屬性，讓 n_features_in_ / feature_name_ 等
        # 內省介面在重啟後仍可用（predict 路徑本身不讀它們）。
        booster._n_features = booster._Booster.num_feature()
        booster._n_features_in = int(meta.get("n_features", booster._n_features))
        booster._fitted_with_feature_names = bool(meta.get("fitted_with_feature_names", False))
        # Isotonic 用存檔陣列重建，再調 _build_f 重做插值函式 f_。
        iso = np.load(str(directory / "isotonic.npz"))
        cal_kwargs = dict(meta["calibrator"])
        increasing_fitted = cal_kwargs.pop("increasing_", None)
        calibrator = IsotonicRegression(**cal_kwargs)
        calibrator.X_thresholds_ = np.asarray(iso["X_thresholds_"])
        calibrator.y_thresholds_ = np.asarray(iso["y_thresholds_"])
        calibrator.X_min_ = float(np.asarray(iso["X_min_"]))
        calibrator.X_max_ = float(np.asarray(iso["X_max_"]))
        calibrator.increasing_ = increasing_fitted if increasing_fitted is not None else cal_kwargs["increasing"]
        calibrator._build_f(calibrator.X_thresholds_, calibrator.y_thresholds_)
        return cls(
            model_id=meta["model_id"],
            booster=booster,
            calibrator=calibrator,
            params=dict(meta["params"]),
            feature_cols=list(meta["feature_cols"]),
        )


class RollingClosedLoopController:
    """Owns the champion, challenger duels, and the orphan-contract registry."""

    def __init__(
        self,
        stress_ratio: float = 0.30,
        dsr_threshold: float = 0.95,
        dsr_trials: int = 100,
        dsr_trials_sharpe_std: float = 0.5,
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

    def set_pass_regime(self, value: bool) -> None:
        """§4.2 latch: Loop 3 (or its caller) decides whether the regime sleeps.

        Review F4: ``pass_regime`` used to be written only inside the duel,
        which meant a caller could compute a PASS verdict and have the very
        next champion deploy silently re-arm trading. The controller exposes
        this setter so the decision outlives ordinary Loop-2 churn.
        """
        self.pass_regime = bool(value)

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

    @staticmethod
    def _net_trade_rets(
        realized_rets: np.ndarray,
        mask: np.ndarray,
        fees: Optional[FeeSchedule],
        index: pd.Index,
        fallback_ppy: float,
        expired: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """Net returns and annualization unit for the trades a model takes.

        Review F3: a take-profit exit rests at the maker limit (``cost_win``);
        a stop-loss exit and a time exit both cross the book (``cost_loss``).
        Review F-A: the sign of ``realized_ret`` cannot pick the leg, because
        the labeller marks every time exit as ``meta_label=0`` while its
        realized return can still be positive (``labels.py:84-109``). Passing
        ``expired`` takes the conservative leg for those rows instead of
        charging a maker fee the live path never pays.

        The annualization unit is periods/year implied by the trade timestamp
        span. Only when that span cannot support one (fewer than two exits, or
        every exit on the same instant) does the caller's declared
        ``periods_per_year`` apply.
        """
        selected = np.asarray(realized_rets, dtype=float)[mask]
        cost_win, cost_loss = round_trip_cost_fractions(fees or FeeSchedule())
        use_win = selected > 0.0
        if expired is not None:
            use_win = use_win & ~np.asarray(expired, dtype=bool)[mask]
        net = selected - np.where(use_win, cost_win, cost_loss)
        try:
            ppy = infer_periods_per_year(index[mask])
        except (ValueError, IndexError, TypeError):
            ppy = float(fallback_ppy)
        return net, ppy

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
        fees=None,
        seed: int = 42,
        periods_per_year: float = 365.0,
        val_expired: Optional[np.ndarray] = None,
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
        chal_rets, chal_ppy = self._net_trade_rets(
            val_realized_rets, chal_mask, fees, X_val.index, periods_per_year,
            expired=val_expired,
        )
        chal_ev = float(np.mean(chal_rets)) if len(chal_rets) > 5 else -0.01
        # DSR must score the trades the challenger would actually take, not the
        # whole validation window (which includes bars it correctly skips).
        dsr_info = ProfitValidationSuite.compute_deflated_sharpe_ratio(
            chal_rets,
            periods_per_year=chal_ppy,
            num_trials=self.dsr_trials,
            trials_sharpe_std=self.dsr_trials_sharpe_std,
        )
        # Review F-H: a sub-year validation window cannot support an annual
        # DSR (years = span/365 < 1 forces dsr 0). Report it as an explicit
        # insufficient-window condition instead of letting 0.0 masquerade as
        # "the challenger failed the bar".
        chal_dsr = float(dsr_info["dsr"])
        chal_short = bool(dsr_info.get("short_sample", False))

        if self.champion is None:
            self.champion = challenger
            self.kelly_penalty_scalar = 1.0
            # Review F4: a champion deployment resets the throttle but must
            # NOT clear a PASS_REGIME verdict — that latch is Loop 3's decision
            # and outlives Loop-2 churn.
            return {
                "status": "INITIAL_CHAMPION_DEPLOYED",
                "active_id": self.champion.model_id,
                "chal_ev": chal_ev,
                "chal_dsr": chal_dsr,
            }

        champ_probs = self.champion.calibrated_proba(X_val)
        champ_mask = champ_probs > entry_threshold
        champ_rets, _ = self._net_trade_rets(
            val_realized_rets, champ_mask, fees, X_val.index, periods_per_year,
            expired=val_expired,
        )
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
                "short_sample": chal_short,
                "orphan_positions_locked": len(self.open_contracts),
            }
        # A losing duel derates Kelly to 0.75 (§4.3) and leaves the PASS_REGIME
        # latch exactly as Loop 3 set it (review F4). A short validation
        # window (review F-H) cannot clear an annual DSR by construction, so
        # say so instead of reporting a 0.0 that reads as "failed the bar".
        self.kelly_penalty_scalar = CHALLENGER_LOSS_THROTTLE
        return {
            "status": "CHAMPION_RETAINED_WITH_THROTTLE",
            "active_id": self.champion.model_id,
            "chal_ev": chal_ev,
            "champ_ev": champ_ev,
            "chal_dsr": chal_dsr,
            "short_sample": chal_short,
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
