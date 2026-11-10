"""pytest 收集設定。

以下三個檔案是 **script 風格**的檢查腳本，不是 pytest 測試：它們在模組層
直接執行檢查、把結果印到 stdout，最後用 ``sys.exit(0/1)`` 表達結論。

pytest 一旦 import 到它們，``sys.exit`` 會以 INTERNALERROR 中斷整個
session，連無關的測試都收不到。所以這裡排除收集，改用直接執行：

    cd backend
    .venv/bin/python backtest/test_ga_scoring_regression.py
    .venv/bin/python backtest/test_max_triggers_units.py
    .venv/bin/python backtest/test_engine_accounting.py

已知 ``test_ga_scoring_regression.py`` 有 4 項斷言與現行 ``ga_annual.py``
不符（fitness 公式、探索期懲罰、交易數無關性、worker 參數傳遞），且在本次
改動前的乾淨樹上同樣失敗。那是斷言過時，不是回歸；修它要改的是斷言，
不是被測程式，屬於獨立工作。
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent

collect_ignore = [
    str(_HERE / "backtest" / "test_ga_scoring_regression.py"),
    str(_HERE / "backtest" / "test_max_triggers_units.py"),
    str(_HERE / "backtest" / "test_engine_accounting.py"),
]
