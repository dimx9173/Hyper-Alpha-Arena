"""lgbm_meta_labeling 回歸：closed-bar、v15 下限、embargo、輸出契約。

FAIL 條件（設計若破功必現其一）：
* 未來注入：bar150 的跳出現在 bar149 的 tf15m 特徵（前視洩漏）。
* M_bars=12（1h）可建模（v15 短持倉禁令破功）。
* embargo 取 0 時訓練尾與校準頭的標籤結局重疊（PIT 標籤洩漏）。
* p_cal 缺欄回 0 而不是拒絕（靜默偽造勝率）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.lgbm_meta_labeling import (
    MetaLabelConfig,
    build_meta_frame,
    calibrated_series,
    fit_meta_label_model,
    load_bars_5m,
    min_hold_bars,
)
from fuxi.features import MultiTimeframeFeaturePipeline


def _synthetic(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    ret = rng.normal(0.00004, 0.0025, n)
    close = 1000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([close * 1.0018, open_, close]),
            "low": np.minimum.reduce([close * 0.9982, open_, close]),
            "close": close,
            "volume": rng.lognormal(10, 0.5, n),
        },
        index=idx,
    )


def test_min_hold_bars_scales_with_axis():
    assert min_hold_bars(5) == 144
    assert min_hold_bars(15) == 48
    assert min_hold_bars(60) == 12


def test_v15_rejects_short_hold():
    with pytest.raises(ValueError, match="v15"):
        MetaLabelConfig(M_bars=12)


def test_closed_bar_freezes_across_boundary():
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    c = np.full(n, 1000.0)
    c[150:] = 1100.0
    o = np.concatenate([[c[0]], c[:-1]])
    bars = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(c, o) * 1.0001,
            "low": np.minimum(c, o) * 0.9999,
            "close": c,
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )
    feats = MultiTimeframeFeaturePipeline(lag_n=1).build_multi_tf_features(bars)
    assert abs(float(feats["tf15m_body"].iloc[149])) < 1e-6
    assert float(feats["tf15m_body"].iloc[152]) > 1e-6


def test_embargo_separates_train_and_calibrate_labels():
    cfg = MetaLabelConfig(M_bars=144)
    assert cfg.effective_embargo_bars == 144
    assert MetaLabelConfig(M_bars=144, embargo_bars=0).effective_embargo_bars == 0


def test_load_bars_5m_empty_frame_not_fake():
    class _DB:
        def execute(self, *a, **k):
            class _R:
                def fetchall(self):
                    return []

            return _R()

    out = load_bars_5m(_DB(), "BTC", start_sec=0, end_sec=1)
    assert len(out) == 0
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_fit_rejects_insufficient_bars():
    cfg = MetaLabelConfig(M_bars=144)
    with pytest.raises(ValueError, match="不足"):
        fit_meta_label_model(_synthetic(200), cfg)


def test_full_train_returns_bounded_pcal():
    rng = np.random.default_rng(11)
    n = 8000
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    trend = np.sin(np.arange(n) / 600.0) * 0.004 + 0.00002
    close = 1000.0 * np.exp(np.cumsum(trend + rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(close, open_) * 1.0015,
            "low": np.minimum(close, open_) * 0.9985,
            "close": close,
            "volume": rng.lognormal(10, 0.5, n),
        },
        index=idx,
    )
    cfg = MetaLabelConfig(x=0.02, y=2.0, M_bars=144, lag_n=5)
    model = fit_meta_label_model(bars, cfg)
    X, y, _w, _r, _e = build_meta_frame(bars, cfg)
    p = model.p_cal(X)
    assert len(p) == len(X)
    assert np.isfinite(p).all()
    assert p.min() >= 0.0 and p.max() <= 1.0

    with pytest.raises(ValueError, match="缺少特徵欄"):
        model.p_cal(X.drop(columns=[model.feature_cols[0]]))


def test_calibrated_series_aligns_index():
    bars = _synthetic(4000)
    cfg = MetaLabelConfig(x=0.02, y=2.0, M_bars=144, lag_n=5)
    s = calibrated_series(bars, cfg)
    assert s.name == "p_cal"
    assert len(s) > 0
    assert bool(((s >= 0.0) & (s <= 1.0)).all())
