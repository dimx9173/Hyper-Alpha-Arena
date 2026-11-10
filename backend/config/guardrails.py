"""Single source of truth for live-deploy guardrails.

實盤護欄（`services/program_execution_service.py` 的 portfolio guard，
約 L1109-1145）與 paper 驗證（`services/paper_broker.py` 的 `_check_guards`）
原本各自硬編碼 2x / 0.4 / 0.6；GA fitness（`backtest/ga_annual.py` 的
`gate_violation`）則完全不知道這些護欄，導致搜出 `portion 0.45 x leverage 3
= 1.35x` 這種回測高分、實盤 100% 被 `single_coin_cap` 擋下的不可部署解。

本模組是三方的共用常數來源。改護欄只改這裡。

Conventions
-----------
* 比率單位：equity 的倍數（0.4 = 單幣名目上限為權益的 40%）。
* `single_coin_notional(portion, leverage)` 把基因的下單參數換算成
  實盤護欄看到的名目倍數：`_new_notional = available_balance * portion *
  leverage`（見 `program_execution_service.py:1131-1132`），stationary
  假設下以 total_equity 為分母，即 `portion * leverage`。
"""

from __future__ import annotations

# 總暴露上限：(total_notional + new_notional) <= equity * MAX_TOTAL_LEVERAGE
# 來源：program_execution_service.py:1133 / paper_broker.py:219
MAX_TOTAL_LEVERAGE: float = 2.0

# 單幣上限：new_notional <= equity * MAX_SINGLE_COIN
# 來源：program_execution_service.py:1137 / paper_broker.py:213
MAX_SINGLE_COIN: float = 0.4

# 同簇上限：(cluster_notional + new_notional) <= equity * MAX_CLUSTER
# 來源：program_execution_service.py:1142 / paper_broker.py:226
MAX_CLUSTER: float = 0.6

# 同簇幣種集合。來源：program_execution_service.py:1117 / paper_broker.py:134
CLUSTER: frozenset = frozenset({"BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "NEAR"})


def single_coin_notional(portion: float, leverage: int) -> float:
    """基因下單參數對應的單幣名目（equity 倍數）。

    實盤路徑：`_new_notional = available_balance * portion * leverage`。
    GA 評分時無帳戶狀態，取 stationary 假設
   （available_balance ≈ total_equity，即空倉起手第一筆的視角）。
    """
    return float(portion) * float(leverage or 1)


def single_coin_violation(portion: float, leverage: int) -> float:
    """相對超標量：0 = 可部署；1.0 = 超標 100%（名目達上限兩倍）。

    與 `gate_violation` 現有 PnL/DD 項同量綱（相對違規量），直接進 `max()`。

    fail-closed：portion/leverage 非有限或名目為負（髒基因）一律回 1.0，
    與 max_trades 硬否決同級，不讓格式越爛懲罰越小。
    """
    import math as _math
    try:
        notional = single_coin_notional(float(portion), int(leverage))
    except (TypeError, ValueError):
        return 1.0
    if not _math.isfinite(notional) or notional < 0:
        return 1.0
    if MAX_SINGLE_COIN <= 0:
        return 0.0
    return max(0.0, (notional - MAX_SINGLE_COIN) / MAX_SINGLE_COIN)
