"""§3 兩階段元標籤：把 Fuxi 的特徵與標籤管線接成一顆可訓練的 LGBM 腦。

這一層只做四件事，其餘全部複用既有模組，不重寫也複製一份：

1. 從 ``crypto_klines`` 讀 5m K 線成 DataFrame（PIT 切片，不含未來）。
2. 呼叫 ``fuxi.features.MultiTimeframeFeaturePipeline`` 建 5m/15m/30m 特徵。
3. 呼叫 ``fuxi.labels.MetaLabelingEngine`` 產生三重障礙標籤與唯一性權重。
4. 訓練 LGBM + Isotonic 校準，對外只輸出 ``p_cal``。

為什麼不改 GA 骨架
------------------
本模組不 import ``ga_annual``、不改 ``Genome``、不改 ``refit_cycle``。它是一個
純函數式的訓練器，輸入 K 線，輸出校準後勝率。要把這顆腦接進 GA 或 StrategySpec
是後續任務，接點由呼叫端決定，所以現有 19 基因路徑的行為完全不變。

v15 約束：M 下限 12 小時等價 bars
--------------------------------
短持倉在 5m 軸上沒有可學習的邊際。實測合成序列 x=3%、y=2 時，M=12 根（1h）的
TP 命中率是 0.000，M=144 根（12h）才到 0.280。以時間而非根數表達這個下限，
換軸時才不會悄悄失效：``min_hold_bars(bar_minutes)`` 是唯一的換算點。

前視防線
--------
* 高階週期（15m/30m）只以「已收盤」的 bar 併入，由 ``fuxi.features`` 的
  ``merge_asof(direction="backward")`` 保證，索引是 close time。
* 訓練與校準切片之間留 ``M`` 根的 embargo。標籤的結局要等 M 根之後才知道，
  沒有這段間隔時，校準段的標籤會由訓練段尾部的未來資料決定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

try:  # 執行時 backend/ 已在 sys.path 上（與 strategy_spec 同一慣例）
    from fuxi.features import WARM_START_BARS, MultiTimeframeFeaturePipeline
    from fuxi.labels import MetaLabelingEngine, build_model_frame
except ImportError:  # pragma: no cover - 以套件形式匯入時的退路
    from backend.fuxi.features import (  # type: ignore
        WARM_START_BARS,
        MultiTimeframeFeaturePipeline,
    )
    from backend.fuxi.labels import (  # type: ignore
        MetaLabelingEngine,
        build_model_frame,
    )

EPS = 1e-8

# v15：持倉時間下限。以分鐘為單位陳述，bars 由 min_hold_bars 換算。
MIN_HOLD_MINUTES = 720.0          # 12h
BASE_BAR_MINUTES = 5              # 主軸
KLINES_PERIOD = "5m"


def min_hold_bars(
    bar_minutes: int = BASE_BAR_MINUTES,
    min_hold_minutes: float = MIN_HOLD_MINUTES,
) -> int:
    """v15 的 M 下限換算：12 小時在給定 K 線週期下等於幾根。

    換軸只需要改 ``bar_minutes``；把下限寫成常數根數會在換軸時失效。
    """
    if bar_minutes <= 0:
        raise ValueError(f"bar_minutes 必須為正，got {bar_minutes!r}")
    if min_hold_minutes <= 0:
        raise ValueError(f"min_hold_minutes 必須為正，got {min_hold_minutes!r}")
    return int(math.ceil(float(min_hold_minutes) / float(bar_minutes)))


@dataclass
class MetaLabelConfig:
    """三重障礙與 LGBM 的超參數。``M_bars`` 受 v15 下限約束。"""

    x: float = 0.03
    y: float = 2.0
    M_bars: int = 144            # = min_hold_bars(5)，12h
    lag_n: int = 5
    entry_threshold: float = 0.55
    n_estimators: int = 80
    max_depth: int = 3
    learning_rate: float = 0.05
    calibrate_fraction: float = 0.15
    embargo_bars: Optional[int] = None   # None -> 取 M_bars
    seed: int = 42
    bar_minutes: int = BASE_BAR_MINUTES

    def __post_init__(self) -> None:
        if self.x <= 0:
            raise ValueError(f"x 必須為正，got {self.x!r}")
        if self.y <= 0:
            raise ValueError(f"y 必須為正，got {self.y!r}")
        if self.lag_n < 1:
            raise ValueError(f"lag_n 必須 >= 1，got {self.lag_n!r}")
        if not 0.0 < self.calibrate_fraction < 0.5:
            raise ValueError(
                f"calibrate_fraction 必須在 (0, 0.5)，got {self.calibrate_fraction!r}"
            )
        floor = min_hold_bars(self.bar_minutes)
        if int(self.M_bars) < floor:
            raise ValueError(
                f"v15 約束：M_bars={self.M_bars} 低於下限 {floor} 根"
                f"（{MIN_HOLD_MINUTES / 60:g}h / {self.bar_minutes}m 軸），短持倉不做"
            )
        if self.embargo_bars is not None and int(self.embargo_bars) < 0:
            raise ValueError(
                f"embargo_bars 不可為負，got {self.embargo_bars!r}"
            )

    @property
    def effective_embargo_bars(self) -> int:
        """標籤結局需 M 根才揭曉，預設 embargo 就取 M。"""
        if self.embargo_bars is None:
            return int(self.M_bars)
        return int(self.embargo_bars)


def load_bars_5m(
    db,
    symbol: str,
    exchange: str = "hyperliquid",
    *,
    start_sec: int,
    end_sec: int,
    period: str = KLINES_PERIOD,
) -> pd.DataFrame:
    """從 ``crypto_klines`` 讀指定區間，索引為 bar 的 open time（UTC）。

    ``crypto_klines.timestamp`` 是 period 的起始秒，與
    ``HistoricalDataProvider._query_klines_from_db`` 同一語義，所以引擎看到的
    bar 邊界與訓練時完全一致。

    回傳空 DataFrame 而不是偽造資料：資料不足時由呼叫端決定是拒訓或換標的。
    """
    from sqlalchemy import text as _sql_text

    rows = db.execute(
        _sql_text(
            """
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM crypto_klines
            WHERE symbol = :symbol AND period = :period AND exchange = :exchange
              AND timestamp >= :start_ts AND timestamp <= :end_ts
            ORDER BY timestamp ASC
            """
        ),
        {
            "symbol": symbol,
            "period": period,
            "exchange": exchange,
            "start_ts": int(start_sec),
            "end_ts": int(end_sec),
        },
    ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    idx = pd.to_datetime([int(r[0]) for r in rows], unit="s", utc=True)
    frame = pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5] or 0.0) for r in rows],
        },
        index=idx,
    )
    frame.index.name = "open_time"
    return frame


def build_meta_frame(
    bars: pd.DataFrame,
    config: MetaLabelConfig,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """K 線 -> (X, y, w, realized_ret, expired)，欄位同名同義於 Fuxi。

    暖啟動不足時回空框，讓呼叫端明確拒訓，而不是用不完整特徵硬訓。
    """
    if bars is None or len(bars) == 0:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype=float), np.empty(0), np.empty(0), np.empty(0)

    features = MultiTimeframeFeaturePipeline(lag_n=config.lag_n).build_multi_tf_features(bars)
    engine = MetaLabelingEngine(x=config.x, y=config.y, M=config.M_bars)
    primary = engine.generate_primary_signals(bars)
    meta = engine.apply_triple_barrier_and_weights(bars, primary)
    X, y, w, realized, expired = build_model_frame(features, meta)
    if len(X) <= WARM_START_BARS:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype=float), np.empty(0), np.empty(0), np.empty(0)
    return X, y, w, realized, expired


@dataclass
class MetaLabelModel:
    """訓練完成的腦。對外只承諾 ``p_cal``。"""

    booster: lgb.LGBMClassifier
    calibrator: IsotonicRegression
    feature_cols: List[str]
    config: MetaLabelConfig
    n_train: int = 0
    n_calibrate: int = 0
    train_span: tuple = field(default_factory=tuple)

    def p_cal(self, X: pd.DataFrame) -> np.ndarray:
        """校準後勝率 P(m=1 | X, d)。缺欄或非有限值一律回 NaN，不補 0。"""
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"p_cal 缺少特徵欄：{missing}")
        if len(X) == 0:
            return np.empty(0, dtype=float)
        raw = np.asarray(
            self.booster.predict_proba(X[self.feature_cols])[:, 1], dtype=float
        )
        return np.asarray(self.calibrator.transform(raw), dtype=float)


def fit_meta_label_model(
    bars: pd.DataFrame,
    config: Optional[MetaLabelConfig] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> MetaLabelModel:
    """用歷史 K 線訓練 LGBM + Isotonic，回傳可算 ``p_cal`` 的模型。

    ``feature_cols`` 為 None 時取全部數值特徵（含 ``primary_dir``）。指定時只留
    存在的欄，避免 Loop 3 重選出的子集在換窗後消失時靜默擴張。

    時間切分為前段訓練、後段校準，中間夾 ``config.effective_embargo_bars`` 根。
    """
    cfg = config or MetaLabelConfig()
    X, y, w, _realized, _expired = build_meta_frame(bars, cfg)
    if len(X) == 0:
        raise ValueError(
            f"特徵或標籤不足：需要 > {WARM_START_BARS} 列可用樣本，"
            f"got {len(bars)} 根原始 K 線"
        )

    cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if feature_cols is not None:
        wanted = list(feature_cols)
        cols = [c for c in wanted if c in cols]
        if not cols:
            raise ValueError(f"feature_cols 全部不存在：{wanted}")

    n = len(X)
    n_cal = max(1, int(round(n * float(cfg.calibrate_fraction))))
    gap = int(cfg.effective_embargo_bars)
    n_train = n - n_cal - gap
    if n_train < 20:
        raise ValueError(
            f"訓練列不足：n={n}、calibrate={n_cal}、embargo={gap} 只留 {n_train} 列"
        )

    X_train = X.iloc[:n_train][cols]
    y_train = y.iloc[:n_train]
    w_train = np.asarray(w, dtype=float)[:n_train]
    cal_start = n_train + gap
    X_cal = X.iloc[cal_start:][cols]
    y_cal = y.iloc[cal_start:]
    if len(X_cal) == 0:
        raise ValueError(f"embargo 吃光校準段：n_train={n_train}、gap={gap}、n={n}")

    booster = lgb.LGBMClassifier(
        n_estimators=int(cfg.n_estimators),
        max_depth=int(cfg.max_depth),
        learning_rate=float(cfg.learning_rate),
        random_state=int(cfg.seed),
        verbose=-1,
    )
    booster.fit(X_train, y_train, sample_weight=w_train)

    raw_cal = np.asarray(booster.predict_proba(X_cal)[:, 1], dtype=float)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, y_cal.to_numpy(dtype=float))

    return MetaLabelModel(
        booster=booster,
        calibrator=calibrator,
        feature_cols=list(cols),
        config=cfg,
        n_train=int(n_train),
        n_calibrate=int(len(X_cal)),
        train_span=(X.index[0], X.index[-1]),
    )


def calibrated_series(
    bars: pd.DataFrame,
    config: Optional[MetaLabelConfig] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.Series:
    """便利函數：對同一段 K 線回傳逐列的 ``p_cal``（索引 = 特徵列時間）。

    只給整段推論用。訓練與推論要分離時請用 ``fit_meta_label_model``。
    """
    cfg = config or MetaLabelConfig()
    model = fit_meta_label_model(bars, cfg, feature_cols=feature_cols)
    X, _y, _w, _r, _e = build_meta_frame(bars, cfg)
    return pd.Series(model.p_cal(X), index=X.index, name="p_cal")
