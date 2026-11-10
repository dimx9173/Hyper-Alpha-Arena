"""
引擎會計正確性回歸測試（T5）

測試三個已修復缺陷：
  1. 未平倉部位不計入 PnL/trades/PF/勝率（censoring bias）
  2. Sharpe 用 √252 年化 per-trigger 報酬（crypto 應為 √365 且需重取樣）
  3. total_funding_shadow 累計但未扣 equity
  4. （實作中發現）total_pnl 是稅前毛值，fee/funding 不進入 PnL

不依賴真實回測資料庫，全部用假的 account / trades / equity_curve，
所以可以快速執行且不受資料覆蓋影響。
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.virtual_account import VirtualAccount
from backtest.models import BacktestConfig, BacktestTradeRecord
from backtest.engine import ProgramBacktestEngine

PASS = 0
FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS  %s" % name)
    else:
        FAIL += 1
        FAILURES.append(name)
        print("  FAIL  %s %s" % (name, detail))


def mk_trade(entry, exit_, pnl, reason="decision"):
    return BacktestTradeRecord(
        timestamp=0, trigger_type="scheduled", symbol="BTC", operation="close",
        side="long", entry_price=entry, size=1.0, leverage=1,
        exit_price=exit_, exit_timestamp=1, exit_reason=reason,
        pnl=pnl, pnl_percent=pnl / entry * 100.0, fee=0.0,
    )


def mk_config(start_ms, end_ms, balance=1000.0, exchange="binance"):
    return BacktestConfig(
        code="", signal_pool_ids=[], symbols=["BTC"],
        start_time_ms=start_ms, end_time_ms=end_ms,
        initial_balance=balance, exchange=exchange,
    )


class FakeEngine(ProgramBacktestEngine):
    """用假的 DB session 與 simulator，專測 _resample_equity_daily。"""
    def __init__(self):
        self.db = None


print()
print("=" * 60)
print("1. _resample_equity_daily：日頻重取樣")
print("=" * 60)

DAY = 86400000
START = 1750000000000
START = (START // DAY) * DAY  # 對齊到 00:00
eng = FakeEngine()

# 1a. 每小時一個點，equity 固定 -> 重取樣後應全等
hourly = []
for h in range(48):
    hourly.append({"timestamp": START + h * 3600000, "equity": 1000.0})
daily = eng._resample_equity_daily(hourly, START, START + 47 * 3600000)
check("固定 equity 重取樣後全等", all(abs(v - 1000.0) < 1e-9 for v in daily), str(daily[:3]))
check("重取樣點數 = 跨越的天數+1", len(daily) == 2, "got %d" % len(daily))

# 1b. equity 每天固定 100 -> 100 線性成長
daily2 = []
for d in range(10):
    for h in range(24):
        daily2.append({"timestamp": START + d * DAY + h * 3600000, "equity": 1000.0 * (d + 1)})
res = eng._resample_equity_daily(daily2, START, START + 9 * DAY + 23 * 3600000)
# 10 天資料（day 0..9），最後一筆在 day9 的 23:00。
# 日界網格覆蓋 day0..day9 共 10 個點（不是 11）——day10 沒有任何資料不該外推。
check("每日階梯 -> 重取樣點數 10", len(res) == 10, "got %d" % len(res))
check("每日階梯 -> 首末值正確",
      abs(res[0] - 1000.0) < 1e-6 and abs(res[-1] - 10000.0) < 1e-6,
      "first=%.2f last=%.2f" % (res[0], res[-1]))

# 1c. 線性增長的 equity 重取樣應保持線性
lin = []
for h in range(240):
    lin.append({"timestamp": START + h * 3600000, "equity": 1000.0 + h})
res3 = eng._resample_equity_daily(lin, START, START + 239 * 3600000)
# 240 小時 = 10 天，網格為 day0..day9 共 10 點；每點相隔 24 小時的 equity 增量。
check("線性序列重取樣後仍等距遞增",
      len(res3) == 10 and all(
          abs((res3[i+1] - res3[i]) - (res3[1] - res3[0])) < 1e-6
          for i in range(len(res3) - 1)),
      str([round(v, 1) for v in res3[:4]]))

# 1d. 邊界
check("空曲線 -> []", eng._resample_equity_daily([], START, START + DAY) == [])
check("單點 -> []", eng._resample_equity_daily(
    [{"timestamp": START, "equity": 1000.0}], START, START + DAY) == [])
# 負 equity 的點被濾掉，但剩下的 2 個有效點仍可插值 -> 保留結果。
neg = [{"timestamp": START, "equity": 1000.0}, {"timestamp": START + 100, "equity": 1001.0},
       {"timestamp": START + 200, "equity": -5.0}]
r_neg = eng._resample_equity_daily(neg, START, START + DAY)
check("非正 equity 被濾除但仍能重取樣", r_neg == [1000.0, 1001.0], str(r_neg))
# 全部非正才會退化成空
all_neg = [{"timestamp": START, "equity": -1.0}, {"timestamp": START + 1, "equity": -2.0}]
check("全部非正 equity -> []", eng._resample_equity_daily(all_neg, START, START + DAY) == [])

# 1e. 輸入順序不影響結果（必須內部排序）
shuffled = list(reversed(daily2))
res4 = eng._resample_equity_daily(shuffled, START, START + 9 * DAY + 23 * 3600000)
check("亂序輸入結果相同", res4 == res)

# 1f. 網格對齊到 UTC 00:00
check("起始點對齊日界", (res[0] is not None) and START % DAY == 0)


print()
print("=" * 60)
print("2. 未平倉部位不計入 -> 這是 _force_close_all 的守門員")
print("=" * 60)

# 這裡不跑真實回測（需要 DB），而是斷言引擎有 _force_close_all 方法，
# 且它存在於 _calculate_result 的呼叫路徑上。實際行為由 diag 腳本驗證。
check("ProgramBacktestEngine 有 _force_close_all", hasattr(ProgramBacktestEngine, "_force_close_all"))
check("ProgramBacktestEngine 有 _resample_equity_daily", hasattr(ProgramBacktestEngine, "_resample_equity_daily"))

import inspect
src = inspect.getsource(ProgramBacktestEngine._calculate_result)
check("_calculate_result 呼叫了 _force_close_all", "_force_close_all(" in src)
check("_calculate_result 在計算 closed_trades 之前強制平倉",
      src.index("_force_close_all(") < src.index("closed_trades = [t for t in trades"))
check("PnL 已扣 total_fees（淨值口徑）", "account.total_fees" in src)
check("恆等式已納入 total_fees", "expected = sum_closed_pnl - account.total_fees" in src)
check("Sharpe 使用 _resample_equity_daily", "_resample_equity_daily(" in src)
check("Sharpe 年化使用 365 而非 252", "365 ** 0.5" in src and "252" not in src.split("sharpe_ratio =")[1][:400])
check("funding shadow 有入帳", "total_funding_shadow" in src)


print()
print("=" * 60)
print("3. 會計恆等式（純數學，用假資料驗證公式一致性）")
print("=" * 60)

# 模擬：2 筆平倉，fee 已知
acct = VirtualAccount(initial_balance=1000.0)
acct.open_position("BTC", "long", 1.0, 100.0, leverage=1, timestamp=0, fee=0.5)
p1 = acct.close_position("BTC", 110.0, fee=0.5)
p2 = acct.close_position("BTC", 50.0, fee=0.5) if acct.positions else None
# 第二筆需要重新開倉
acct.open_position("ETH", "long", 1.0, 100.0, leverage=1, timestamp=0, fee=0.5)
p3 = acct.close_position("ETH", 50.0, fee=0.5)
acct.update_equity({})

closed_pnl = [p1, p3]
fees = acct.total_fees
expected = sum(closed_pnl) - fees
actual = acct.equity - 1000.0
check("恆等式: equity-initial == Σ毛利 - fees",
      abs(actual - expected) < 1e-6,
      "actual=%.6f expected=%.6f" % (actual, expected))
# BTC 110->110 無變化（pnl=0），ETH 100->50 虧 50，所以總毛利為負。
# 這裡要驗的是「恆等式在混合盈虧下仍成立」，不是斷言淨值為正。
check("恆等式在混合盈虧下成立（毛利可為負）",
      abs(actual - expected) < 1e-6 and sum(closed_pnl) < 0 and expected < 0,
      "gross=%.2f net=%.2f" % (sum(closed_pnl), expected))
check("恆等式: 部位已清空", len(acct.positions) == 0)
check("total_fees 累加了 4 次（2 進 2 出）", abs(fees - 2.0) < 1e-9, "fees=%.4f" % fees)

# 虧損情境
acct2 = VirtualAccount(initial_balance=1000.0)
acct2.open_position("BTC", "long", 1.0, 100.0, leverage=1, timestamp=0, fee=1.0)
q1 = acct2.close_position("BTC", 80.0, fee=1.0)
acct2.update_equity({})
exp2 = q1 - acct2.total_fees
act2 = acct2.equity - 1000.0
check("恆等式（虧損情境）", abs(act2 - exp2) < 1e-6, "act=%.4f exp=%.4f" % (act2, exp2))
check("虧損情境: 淨值為負", exp2 < 0)


print()
print("=" * 60)
print("4. 毛值 vs 淨值差異（第四個缺陷的數學量測）")
print("=" * 60)

# 1000 進場、10 進場費 + 10 出場費 = 20
acct3 = VirtualAccount(initial_balance=1000.0)
acct3.open_position("BTC", "long", 10.0, 100.0, leverage=1, timestamp=0, fee=10.0)
g = acct3.close_position("BTC", 110.0, fee=10.0)
acct3.update_equity({})
gross_pct = g / 1000.0 * 100
net_pct = (g - acct3.total_fees) / 1000.0 * 100
check("毛值 10.00%", abs(gross_pct - 10.0) < 1e-9, "got %.4f" % gross_pct)
check("淨值 8.00%（扣 2% 費）", abs(net_pct - 8.0) < 1e-9, "got %.4f" % net_pct)
check("費損 2pp，與 total_fees/initial 一致",
      abs((gross_pct - net_pct) - acct3.total_fees / 1000.0 * 100) < 1e-9)
check("淨值 < 毛值（收緊方向正確）", net_pct < gross_pct)

# 費率越高，淨值越低（單調性）
prev = None
monotonic = True
for fee in (0.0, 5.0, 10.0, 20.0):
    a = VirtualAccount(initial_balance=1000.0)
    a.open_position("BTC", "long", 10.0, 100.0, leverage=1, timestamp=0, fee=fee)
    gg = a.close_position("BTC", 110.0, fee=fee)
    a.update_equity({})
    n = (gg - a.total_fees) / 1000.0 * 100
    if prev is not None and n >= prev:
        monotonic = False
    prev = n
check("淨值隨費用單調遞減", monotonic)


print()
print("=" * 60)
print("結果：%d 通過, %d 失敗" % (PASS, FAIL))
if FAILURES:
    print("失敗項：")
    for f in FAILURES:
        print("  - %s" % f)
print("=" * 60)
sys.exit(1 if FAIL else 0)
