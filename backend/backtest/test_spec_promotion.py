"""spec 上線判準的回歸測試：promotion verdict 與 promote 路徑的 fail-closed。

FAIL 條件（設計若破功必現其一）：
* 樣本不足或缺時間戳時回 passed=True（假造「已測量」）。
* 同一刻出場（無法年化）時拋例外或回 passed=True。
* 虧損 regime 拿到 passed=True（門檻形同虛設）。
* 強邊際長樣本拿不到 passed=True（門檻永遠擋人，等於無法上線）。
* `_run_spec_refit` 在 promote_ok=False 時仍寫入 promote 事件。
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import backtest.strategy_spec as spec_mod


def _verdict(returns, stamps):
    return spec_mod._promotion_verdict(
        {"trade_returns": list(returns), "trade_exit_ms": list(stamps)}
    )


def _timeline(n: int, span_days: float) -> np.ndarray:
    """n 筆交易均勻分布在 span_days 天內的出場時間戳（毫秒）。"""
    return (np.arange(n, dtype=np.int64) * int(span_days * 86_400_000) // n)


def test_insufficient_trades_never_passes():
    v = _verdict([0.01], [1])
    assert v["available"] is False
    assert v["reason"] == "insufficientTrades"
    assert v["passed"] is False
    assert v["trades"] == 1


def test_mismatched_timestamps_never_passes():
    v = _verdict([0.01, -0.02, 0.03], [1, 2])
    assert v["passed"] is False
    assert v["reason"] == "insufficientTrades"


def test_same_instant_exits_are_not_annualizable():
    v = _verdict([0.01, -0.02, 0.015], [5, 5, 5])
    assert v["available"] is False
    assert v["passed"] is False
    assert v["reason"].startswith("cannotAnnualize")


def test_losing_regime_is_rejected():
    rng = np.random.default_rng(3)
    n = 1600
    v = _verdict(rng.normal(-0.004, 0.02, n), _timeline(n, 400))
    assert v["available"] is True
    assert v["passed"] is False


def test_strong_edge_over_long_sample_passes():
    rng = np.random.default_rng(21)
    n = 4000
    v = _verdict(rng.normal(0.008, 0.006, n), _timeline(n, 900))
    assert v["available"] is True
    assert v["passed"] is True, v
    # 四關都必須真的被評估，不能只靠其中一關。
    assert set(v) >= {"dsr", "monte_carlo", "friction", "monthly"}
    assert v["dsr"]["dsr"] >= 0.95
    assert v["friction"]["passed"] is True
    assert v["monthly"]["passed"] is True


def test_refit_refuses_promote_when_verdict_fails(tmp_path, monkeypatch):
    """promote_ok=False 時不得在 ledger 寫入 promote 事件。"""
    refit = importlib.import_module("backtest.refit_cycle")
    ledger = tmp_path / "ledger.jsonl"
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({"BTC": []}))

    def _fake_eval(*args, **kwargs):
        return {
            "spec": {}, "gate_annual": True, "pnl_365d": 99.0,
            "maxdd_365d": 1.0, "sharpe_365d": 3.0, "pf": 9.0, "pf_ok": True,
            "trades_365d": 500, "win_rate": 60.0, "fitness": 9.0,
            "spec_id": "deadbeef",
            "promotion": {"available": False, "reason": "insufficientTrades",
                          "passed": False, "trades": 1},
            "promote_ok": False,
            "window_consistency": 0.0, "windows_passed": 0, "windows_total": 1,
            "window_detail": [],
        }

    monkeypatch.setattr(spec_mod, "evaluate_lgbm_spec", _fake_eval)
    args = type("A", (), {
        "bars_5m_json": str(bars_path), "ledger": str(ledger), "symbols": ["BTC"],
        "window_days": 30, "eval_days": 7, "max_triggers": 8000,
        "spec_json": None, "note": "",
    })()

    rc = refit._run_spec_refit(args)
    assert rc == 1
    assert not ledger.exists() or ledger.read_text().strip() == ""


def test_refit_promotes_when_verdict_passes(tmp_path, monkeypatch):
    refit = importlib.import_module("backtest.refit_cycle")
    ledger = tmp_path / "ledger.jsonl"
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({"BTC": []}))

    def _fake_eval(*args, **kwargs):
        return {
            "spec": {}, "gate_annual": True, "pnl_365d": 99.0,
            "maxdd_365d": 1.0, "sharpe_365d": 3.0, "pf": 9.0, "pf_ok": True,
            "trades_365d": 500, "win_rate": 60.0, "fitness": 9.0,
            "spec_id": "deadbeef",
            "promotion": {"available": True, "passed": True,
                          "dsr": {"dsr": 0.99}},
            "promote_ok": True,
            "window_consistency": 1.0, "windows_passed": 1, "windows_total": 1,
            "window_detail": [],
        }

    monkeypatch.setattr(spec_mod, "evaluate_lgbm_spec", _fake_eval)
    args = type("A", (), {
        "bars_5m_json": str(bars_path), "ledger": str(ledger), "symbols": ["BTC"],
        "window_days": 30, "eval_days": 7, "max_triggers": 8000,
        "spec_json": None, "note": "",
    })()

    rc = refit._run_spec_refit(args)
    assert rc == 0
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["event"] == "promote"
    assert rows[0]["kind"] == "spec"
    assert rows[0]["promotion"]["passed"] is True


def test_window_slice_is_respected():
    """不同 (end, days) 必須切出不同資料段，否則 consistency 結構性恆定。"""
    import pandas as pd

    n = 70000
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    bars = pd.DataFrame({"close": np.arange(n, dtype=float)}, index=idx)
    end_ms = int(idx.max().timestamp() * 1000)
    early_end = end_ms - 180 * 86_400_000

    a = spec_mod._slice_window(bars, early_end, 180)
    b = spec_mod._slice_window(bars, end_ms, 180)
    assert len(a) > 0 and len(b) > 0
    assert not a.index.equals(b.index)
    assert a.index.max() <= b.index.min()
    assert spec_mod._slice_window(bars, end_ms, 0).empty
