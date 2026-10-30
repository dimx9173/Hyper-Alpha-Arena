"""
第五個缺陷的回歸測試：max_triggers 單位錯誤（毫秒當秒用）

缺陷（backend/backtest/ab_validators.py, annual_validator 原始碼）：
    trigger = 900 if max_triggers is None else max(300, 86400000 // max(1, max_triggers // 365))

86400000 是「毫秒 / 天」，但 scheduled_interval_sec 的單位是「秒」，間隔大了 1000 倍。

實測影響：ga_annual.py 傳 max_triggers=50000、backtest_days=1095，
  缺陷版 trigger = 635294 秒 = 7.35 天
  → 三年視窗只有 148 個 trigger（預期 50000，缺口 338 倍）
  → 逐年視窗只有 49 個 trigger（缺口 1020 倍）

惡化因素：_run_one 內部另有正確計算 total_days * 86400 / max_triggers，
但 max(635294, 1892) 讓錯誤值勝出，正確值被完全架空，缺陷因此長期隱形。

修法：抽出單一實算來源 _compute_trigger_interval_sec（單位：秒），
      annual_validator 與 _run_one 都呼叫它，消除兩份不一致的推導。

本測試不依賴回測資料庫，直接驗證共用函式的數值行為與原始碼結構。
"""
import re
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "ab_validators.py"
src = SRC_PATH.read_text()

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


print()
print("=" * 68)
print("1. 原始碼結構：單一實算來源，無毫秒常數")
print("=" * 68)

check("共用函式 _compute_trigger_interval_sec 已存在",
      "def _compute_trigger_interval_sec" in src)

fn = re.search(
    r"def _compute_trigger_interval_sec\(max_triggers,\s*window_days,\s*floor=(\d+)\)"
    r"(.*?)\n\ndef ",
    src, re.S)
check("能定位共用函式本體", fn is not None)


def strip_docstrings(text):
    """移除 docstring，避免把記錄缺陷的註解誤判成可執行程式碼。"""
    out = []
    i = 0
    q = chr(34) * 3
    while i < len(text):
        if text.startswith(q, i):
            end = text.find(q, i + 3)
            if end < 0:
                break
            i = end + 3
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


fn_raw = fn.group(2) if fn else ""
fn_body = strip_docstrings(fn_raw)

check("共用函式使用 86400（秒/天）", "86400" in fn_body, "body=%r" % fn_body[:120])
check("共用函式不含 86400000（毫秒/天）", "86400000" not in fn_body)

# 兩處呼叫點都必須走共用函式
av_call = re.search(r"trigger=900 if max_triggers is None else ([^\n]+)", src)
check("annual_validator 走共用函式",
      av_call is not None and "_compute_trigger_interval_sec(" in av_call.group(1),
      "got=%s" % (av_call.group(1) if av_call else None))
ro_call = re.search(r"optimal_trigger = ([^\n]+)", src)
check("_run_one 也走共用函式",
      ro_call is not None and "_compute_trigger_interval_sec(" in ro_call.group(1),
      "got=%s" % (ro_call.group(1) if ro_call else None))

# 可執行程式碼中不可出現 86400000（註解可提，用來記錄缺陷）
code_only = strip_docstrings(src)
code_only = "\n".join(
    l for l in code_only.splitlines()
    if not l.strip().startswith("#"))
check("可執行程式碼中沒有 86400000（僅註解可提）",
      "86400000" not in code_only)

print()
print("=" * 68)
print("2. 共用函式數值行為（真實 import，非重寫）")
print("=" * 68)

sys.path.insert(0, str(SRC_PATH.parent.parent))
try:
    from backtest.ab_validators import _compute_trigger_interval_sec as C
    IMPORTED = True
except Exception as exc:  # pragma: no cover
    C = None
    IMPORTED = False
    print("  (import 失敗：%s)" % exc)

check("能 import 共用函式", IMPORTED)

if IMPORTED:
    for mt, days in ((50000, 1095), (50000, 365), (30000, 365), (3000, 60)):
        got = C(mt, days)
        actual = days * 86400 // got if got else 0
        ratio = mt / actual if actual else float("inf")
        check("max_triggers=%-6d %4d 天 → 間隔 %-5d 秒 → %6d 個 trigger（比率 %.2f）"
              % (mt, days, got, actual, ratio),
              0.5 <= ratio <= 2.0, "比率 %.2f" % ratio)

    # 單位正確性：間隔必須是「秒」，且遠小於一天
    for mt, days in ((50000, 1095), (50000, 365), (3000, 60)):
        got = C(mt, days)
        check("max_triggers=%-6d %4d 天 → 間隔 %.4f 天（小於 1 天，單位正確）"
              % (mt, days, got / 86400.0),
              0 < got < 86400, "got=%d 秒" % got)

    # 邊界
    check("max_triggers=0 → 回傳 floor", C(0, 365) == 300, "got=%s" % C(0, 365))
    check("max_triggers=None → 回傳 floor", C(None, 365) == 300)
    check("極小視窗 → 不低於 floor", C(50000, 1) == 300, "got=%s" % C(50000, 1))

    # 單調性：視窗越長，間隔越長（否則三年 run 會比一年 run 密）
    seq = [C(50000, d) for d in (90, 180, 365, 730, 1095)]
    check("間隔隨視窗長度單調遞增",
          all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)), str(seq))

print()
print("=" * 68)
print("3. 缺陷版必須被這些測試抓到（證明測試有鑑別力）")
print("=" * 68)

BAD = "max(300, 86400000 // max(1, max_triggers // 365))"


def bad_interval(max_triggers):
    ns = {"max": max, "max_triggers": max_triggers}
    return eval(BAD, {"__builtins__": {}}, ns)


bad = bad_interval(50000)
check("缺陷版在 max_triggers=50000 產生 %.1f 天的間隔（離譜）"
      % (bad / 86400.0),
      bad / 86400.0 > 7, "got=%d 秒" % bad)
check("缺陷版三年視窗只有 %d 個 trigger（預期 50000）"
      % (1095 * 86400 // bad),
      1095 * 86400 // bad < 500)

# 只改單位（86400000 -> 86400，維持原本「365 天基準」的算法）：
#   635294 / 635 = 1000 倍
unit_only_bad = eval("max(300, 86400000 // max(1, 50000 // 365))",
                     {"__builtins__": {}}, {"max": max})
unit_only_good = eval("max(300, 86400 // max(1, 50000 // 365))",
                      {"__builtins__": {}}, {"max": max})
check("只改單位：缺陷版 / 修好版 = %.0f 倍（預期 1000）"
      % (unit_only_bad / unit_only_good),
      950 < unit_only_bad / unit_only_good < 1050,
      "%.1f" % (unit_only_bad / unit_only_good))

if IMPORTED:
    good = C(50000, 1095)
    # 本次修復同時改了「計算基準」（365 天基準 -> 視窗長度基準），
    # 所以總倍率不是 1000，而是 336；但關鍵是實際 trigger 數要對上 max_triggers。
    check("本次修復總倍率 %.0fx，且三年 trigger 數 50004 ≈ 50000（正確）"
          % (bad / good),
          50000 <= 1095 * 86400 // good <= 50100,
          "got %d" % (1095 * 86400 // good))

print()
print("=" * 68)
print("結果：%d 通過, %d 失敗" % (PASS, FAIL))
if FAILURES:
    print("失敗項：")
    for f in FAILURES:
        print("  - %s" % f)
print("=" * 68)
sys.exit(1 if FAIL else 0)
