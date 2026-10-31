"""前視回歸：HistoricalDataProvider.get_indicator 不得偷看未來。

FAIL 條件（修前必現）：t=25 先算一次 EMA20 把全窗快取寫死，
推進到 t=60 重算仍回舊值 15.5（正確 50.5），與全新 provider 在 t=60
直接算的結果不一致。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.historical_data_provider import HistoricalDataProvider


def _provider():
    p = HistoricalDataProvider.__new__(HistoricalDataProvider)
    p.db = MagicMock()
    p.symbols = ["BTC"]
    p.exchange = "hyperliquid"
    p.current_time_ms = 0
    p._kline_cache = {}
    p._price_cache = {}
    p._indicator_cache = {}
    p._market_metrics_empty = True
    p._query_log = []
    p._kline_cache["BTC_1h"] = [
        {"timestamp": t, "open": t, "high": t, "low": t,
         "close": float(t), "volume": 1.0}
        for t in range(1, 61)
    ]
    return p


def test_indicator_does_not_see_future():
    p = _provider()
    p.current_time_ms = 25 * 1000
    r25 = p.get_indicator("BTC", "EMA20", "1h")["value"]
    assert r25 is not None and abs(r25 - 15.5) < 1e-9, r25

    # 時間推進後重算，必須等於全新 provider 在同一時刻直接算的值
    p.current_time_ms = 60 * 1000
    r60 = p.get_indicator("BTC", "EMA20", "1h")["value"]

    q = _provider()
    q.current_time_ms = 60 * 1000
    fresh60 = q.get_indicator("BTC", "EMA20", "1h")["value"]

    assert abs(r60 - fresh60) < 1e-9, f"前視：推進後 {r60} != 新鮮 {fresh60}"
    assert abs(r60 - 50.5) < 1e-9, f"前視：t=60 EMA20 應為 50.5，得 {r60}"


if __name__ == "__main__":
    test_indicator_does_not_see_future()
    print("PASS: indicator 不偷看未來")
