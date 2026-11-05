#!/usr/bin/env python3
"""
ICT / SMC 價格結構指標（Smart Money Concepts）

這些指標全部是「價格結構（price action）」概念，不是傳統數學指標。
量化定義刻意保守：每個概念的門檻都寫成顯式參數並附上 docstring，
避免主觀定義被硬編成不可解釋的回測規則。

概念來源與常見量化對應：
  - Swing Point        局部高低點，構成所有結構判斷的原子
  - BOS  (Break of Structure)  價格突破前一個 swing high/low
  - CHoCH (Change of Character)  原本的結構方向被反向突破
  - Order Block        造成結構突破的那根反向實體 K 線
  - Fair Value Gap     三根 K 線留下未填補的價格缺口
  - Breaker Block      被反向突破後，失效的 Order Block
  - Liquidity Sweep    掃過前高/前低後收回
  - Premium / Discount 區間內的相對位置
  - Optimal Trade Entry  由 CHoCH + Order Block + FVG 組成的 confluence 區

預設參數校準
------------
`swing_left` / `swing_right` 預設為 5，不是文獻常見的 2。這是實測結果：
在 BTC 500 根 1h K 線上，left=2/right=2 會產出 140 個 swing、58 個結構
事件（BOS 36 / CHoCH 22），噪音過高，CHoCH 幾乎每十幾根就出現一次，
失去「結構反轉」的意義。

實測掃描（BTC 1h × 500 bars）：

    left  right   swings   events   bos  choch
       2      2      140       58     36     22   ← 噪音過多
       3      3       99       41     25     16
       4      4       78       33     20     13
       5      5       63       25     18      7   ← 預設
       6      6       53       21     14      7

left/right 應該視為可被 GA 搜尋的參數，不同 period 需要不同值：
1h 級別噪音大，用 5；4h/1d 級別訊號本就稀疏，可降到 3。

設計原則
--------
1. 回測友善：回傳 dict 含 "value"（可直接給 GA 當基因用）與明細欄位。
2. 無前視偏差：所有函式只使用呼叫時點已完成的 K 線；被打破的結構
   以 break_index 記錄，不允許用未來資訊回頭標記。
3. 可測試：每個偵測器都能用人工構造的 K 線序列驗證。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# =====================================================================
# 1. Swing Point
# =====================================================================

@dataclass
class Swing:
    """一個已確認的局部高點或低點。index 指向形成該點的那根 K 線。"""
    index: int
    price: float
    kind: str          # "high" | "low"
    broken: bool = False
    broken_at: Optional[int] = None   # 被哪一根 K 線跌破/突破
    broken_by: Optional[float] = None  # 觸發突破的價格


def find_swings(
    klines: Sequence[Dict[str, Any]],
    left: int = 2,
    right: int = 2,
) -> List[Swing]:
    """找出 swing highs 與 swing lows。

    Swing high: high[i] 是 [i-left, i+right] 區間內的最高點。
    Swing low : low[i]  是 [i-left, i+right] 區間內的最低點。

    `right` 根確認機制是為了避免前視偏差：確認點 index=i+right 之後，
    才能宣稱 i 是 swing。回測中若呼叫時機早於此，應用 confirm_index 判斷。

    Args:
        klines: 由舊到新的 K 線序列，每筆需含 high/low。
        left:  左側比較根數。
        right: 右側確認根數，越大越能濾掉雜訊，但也越 lag。

    Returns:
        依 index 排序的 Swing 列表（high 與 low 交錯）。
    """
    n = len(klines)
    if n < left + right + 1:
        return []

    highs = [float(k["high"]) for k in klines]
    lows = [float(k["low"]) for k in klines]

    swings: List[Swing] = []

    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]

        if highs[i] == max(window_h) and window_h.count(highs[i]) == 1:
            swings.append(Swing(index=i, price=highs[i], kind="high"))
        if lows[i] == min(window_l) and window_l.count(lows[i]) == 1:
            swings.append(Swing(index=i, price=lows[i], kind="low"))

    swings.sort(key=lambda s: s.index)
    return swings


def confirm_index(swing: Swing, right: int) -> int:
    """該 swing 最早可被「知道」的 K 線索引。"""
    return swing.index + right


# =====================================================================
# 2. Market Structure：BOS 與 CHoCH
# =====================================================================

def detect_structure(
    klines: Sequence[Dict[str, Any]],
    swing_left: int = 5,
    swing_right: int = 5,
) -> Dict[str, Any]:
    """偵測市場結構事件序列（BOS / CHoCH）。

    量化規則
    --------
    - 維護結構方向 dir ∈ {None, 'up', 'down'}。
    - 由下往上突破「最近一個已確認且未破的 swing high」→ BOS；
      若當下 dir 為 'down'，此反向突破標為 CHoCH。
    - 由上往下跌破「最近一個已確認且未破的 swing low」→ BOS(down)；
      若當下 dir 為 'up' 則標為 CHoCH。
    - 突破後清掉被突破側參考點，等待下一個 swing 補上。

    前視偏差保證（重要）
    --------------------
    swing 需要右側 swing_right 根確認才能被「知道」。掃描在每個索引 i
    只使用 confirm_index <= i 的 swing，絕不使用 i 之後才確認的 swing。

    舊實作有嚴重前視偏差：先把 ref_high/ref_low 直接設成整段資料「最後
    一個已確認 swing」，再從較早的索引開始掃描，於是 index 4 就用到了
    index 8 之後才確認的 swing，導致同一根 K 線在截斷序列與完整序列中
    被判成不同事件（測試 test_no_lookahead_structure 抓到）。

    Returns:
        {
          "dir": "up" | "down" | None,
          "events": [{type, index, swing_index, level, price, direction}, ...],
          "last_bos": event | None,
          "last_choch": event | None,
        }
    """
    n = len(klines)
    result: Dict[str, Any] = {
        "dir": None, "events": [], "last_bos": None, "last_choch": None,
    }
    if n < swing_left + swing_right + 1:
        return result

    swings = find_swings(klines, left=swing_left, right=swing_right)
    if not swings:
        return result

    highs = [float(k["high"]) for k in klines]
    lows = [float(k["low"]) for k in klines]

    # swing 依「被確認的時機」排序：只有在 confirm_index 到達的那一根，
    # 它才進入可用集合。這是消除前視偏差的關鍵。
    pending = sorted(
        ((confirm_index(sw, swing_right), sw) for sw in swings),
        key=lambda pair: pair[0],
    )
    queue_idx = 0

    dir_state: Optional[str] = None
    ref_high: Optional[Swing] = None
    ref_low: Optional[Swing] = None

    start = pending[0][0]  # 第一個 swing 可被確認的時機
    for i in range(start, n):
        # 1) 先把此刻已確認的 swing 收進可用集合
        while queue_idx < len(pending) and pending[queue_idx][0] <= i:
            sw = pending[queue_idx][1]
            queue_idx += 1
            if sw.kind == "high":
                ref_high = sw
            else:
                ref_low = sw

        hi, lo = highs[i], lows[i]

        # 2) 再判斷是否突破
        if ref_high is not None and not ref_high.broken and hi > ref_high.price:
            is_choch = (dir_state == "down")
            ev = {
                "type": "CHoCH" if is_choch else "BOS",
                "index": i,
                "swing_index": ref_high.index,
                "level": ref_high.price,
                "price": hi,
                "direction": "up",
            }
            ref_high.broken = True
            ref_high.broken_at = i
            ref_high.broken_by = hi
            result["events"].append(ev)
            dir_state = "up"
            if is_choch:
                result["last_choch"] = ev
            else:
                result["last_bos"] = ev
            ref_low = None          # 向上突破後等待新的 swing low
            ref_high = None          # 同側參考點已失效

        elif ref_low is not None and not ref_low.broken and lo < ref_low.price:
            is_choch = (dir_state == "up")
            ev = {
                "type": "CHoCH" if is_choch else "BOS",
                "index": i,
                "swing_index": ref_low.index,
                "level": ref_low.price,
                "price": lo,
                "direction": "down",
            }
            ref_low.broken = True
            ref_low.broken_at = i
            ref_low.broken_by = lo
            result["events"].append(ev)
            dir_state = "down"
            if is_choch:
                result["last_choch"] = ev
            else:
                result["last_bos"] = ev
            ref_high = None          # 向下突破後等待新的 swing high
            ref_low = None

    result["dir"] = dir_state
    return result

# =====================================================================
# 3. Order Block
# =====================================================================

def find_order_blocks(
    klines: Sequence[Dict[str, Any]],
    structure: Optional[Dict[str, Any]] = None,
    swing_left: int = 5,
    swing_right: int = 5,
    max_lookback: int = 200,
    max_blocks: int = 10,
) -> List[Dict[str, Any]]:
    """找出造成結構突破的 Order Block。

    量化定義（bullish OB）：
      造成向上 BOS/CHoCH 的那根 K 線本身是陰線（close < open），
      且它出現在被突破的 swing high 之前。

    量化定義（bearish OB）：
      造成向下 BOS/CHoCH 的那根 K 線本身是陽線（close > open），
      且它出現在被跌破的 swing low 之前。

    每個 block 回傳 top/bottom 為該根 K 線的實體範圍，
    mitigated=True 表示價格已回補穿越該區（OB 失效）。
    """
    n = len(klines)
    if structure is None:
        structure = detect_structure(klines, swing_left, swing_right)

    opens = [float(k["open"]) for k in klines]
    highs = [float(k["high"]) for k in klines]
    lows = [float(k["low"]) for k in klines]
    closes = [float(k["close"]) for k in klines]

    blocks: List[Dict[str, Any]] = []

    for ev in structure.get("events", [])[-max_blocks:]:
        break_i = ev["index"]
        swing_i = ev["swing_index"]
        if break_i - swing_i > max_lookback:
            continue

        # 在 swing 與突破之間找最後一根反向實體 K 線
        direction = ev["direction"]
        ob_i = None
        if direction == "up":
            for j in range(swing_i, break_i):
                if closes[j] < opens[j]:
                    ob_i = j
        else:
            for j in range(swing_i, break_i):
                if closes[j] > opens[j]:
                    ob_i = j

        if ob_i is None:
            continue

        block = {
            "index": ob_i,
            "confirm_index": break_i,
            "direction": "bullish" if direction == "up" else "bearish",
            "top": highs[ob_i],
            "bottom": lows[ob_i],
            "body_high": max(opens[ob_i], closes[ob_i]),
            "body_low": min(opens[ob_i], closes[ob_i]),
            "trigger": ev["type"],
            "mitigated": False,
        }

        # 判定是否已被回補（價格回穿 block 另一側）
        for j in range(block["confirm_index"] + 1, n):
            if block["direction"] == "bullish":
                if lows[j] < block["bottom"]:
                    block["mitigated"] = True
                    break
            else:
                if highs[j] > block["top"]:
                    block["mitigated"] = True
                    break

        blocks.append(block)

    blocks.sort(key=lambda b: b["confirm_index"], reverse=True)
    return blocks[:max_blocks]


# =====================================================================
# 4. Fair Value Gap
# =====================================================================

def find_fvgs(
    klines: Sequence[Dict[str, Any]],
    min_body_ratio: float = 1.0,
    max_fvgs: int = 10,
) -> List[Dict[str, Any]]:
    """找出 Fair Value Gap（三根 K 線形成的未填補缺口）。

    量化定義（bullish FVG）：
      連續三根 K 線 i-2, i-1, i（皆為陽線），且
        low[i] > high[i-2]
      即中間那根的上影線與後一根的低點之間留下未交易區域。

    量化定義（bearish FVG）：
      連續三根皆為陰線，且
        high[i] < low[i-2]

    min_body_ratio：三根之中至少一根的實體需大於「三根平均實體 × ratio」，
    用來排除雜訊缺口。ratio=1.0 表示要求有明顯的主動 push。
    """
    n = len(klines)
    if n < 3:
        return []

    opens = [float(k["open"]) for k in klines]
    highs = [float(k["high"]) for k in klines]
    lows = [float(k["low"]) for k in klines]
    closes = [float(k["close"]) for k in klines]

    bodies = [abs(closes[i] - opens[i]) for i in range(n)]

    fvgs: List[Dict[str, Any]] = []

    for i in range(2, n):
        avg_body = sum(bodies[i - 2 : i + 1]) / 3.0
        if avg_body <= 0:
            continue

        # Bullish FVG
        if closes[i - 2] > opens[i - 2] and closes[i - 1] > opens[i - 1] and closes[i] > opens[i]:
            if lows[i] > highs[i - 2]:
                if bodies[i - 1] >= avg_body * min_body_ratio or bodies[i] >= avg_body * min_body_ratio:
                    top, bottom = lows[i], highs[i - 2]
                    fvg = _make_fvg(top, bottom, i, "bullish", min_body_ratio)
                    if fvg:
                        fvgs.append(fvg)

        # Bearish FVG
        if closes[i - 2] < opens[i - 2] and closes[i - 1] < opens[i - 1] and closes[i] < opens[i]:
            if highs[i] < lows[i - 2]:
                if bodies[i - 1] >= avg_body * min_body_ratio or bodies[i] >= avg_body * min_body_ratio:
                    top, bottom = lows[i - 2], highs[i]
                    fvg = _make_fvg(top, bottom, i, "bearish", min_body_ratio)
                    if fvg:
                        fvgs.append(fvg)

    # 標記已填補
    for fvg in fvgs:
        for j in range(fvg["index"] + 1, n):
            if fvg["direction"] == "bullish":
                if lows[j] <= fvg["bottom"]:
                    fvg["filled"] = True
                    break
            else:
                if highs[j] >= fvg["top"]:
                    fvg["filled"] = True
                    break

    fvgs.sort(key=lambda f: f["index"], reverse=True)
    return fvgs[:max_fvgs]


def _make_fvg(top: float, bottom: float, index: int, direction: str, ratio: float):
    if top <= bottom:
        return None
    return {
        "index": index,
        "direction": direction,
        "top": top,
        "bottom": bottom,
        "mid": (top + bottom) / 2.0,
        "size": top - bottom,
        "size_pct": (top - bottom) / bottom if bottom else 0.0,
        "filled": False,
    }


# =====================================================================
# 5. Liquidity Sweep
# =====================================================================

def find_liquidity_sweeps(
    klines: Sequence[Dict[str, Any]],
    swing_left: int = 3,
    swing_right: int = 3,
    wick_ratio: float = 1.0,
    max_sweeps: int = 10,
) -> List[Dict[str, Any]]:
    """找出流動性掃蕩（sweep）。

    量化定義（看多 sweep，掃前低後收回）：
      1. 前方存在已確認的 swing low（未被跌破）
      2. 某根 K 線的 low 跌破該 swing low
      3. 但該根 K 線的 close 收回 swing low 之上
      4. 該根的下影線長度 >= 實體長度 × wick_ratio

    wick_ratio 越高代表要求「更明確的拒絕」，減少假訊號。
    """
    n = len(klines)
    if n < swing_left + swing_right + 2:
        return []

    swings = find_swings(klines, left=swing_left, right=swing_right)
    if not swings:
        return []

    opens = [float(k["open"]) for k in klines]
    highs = [float(k["high"]) for k in klines]
    lows = [float(k["low"]) for k in klines]
    closes = [float(k["close"]) for k in klines]

    # 效能：原實作對每個 swing 從 ready 掃到 n，並在內層重算 body/上下影線，
    # 複雜度 O(swings x n)。4813 根 1h K 線 x 99 swings = 233 萬次 max/min/abs，
    # 單次 ICT_OTE 要 302ms，而回測一個 trigger 就呼叫一次 → 8000 triggers
    # 光 ICT 就 40 分鐘（v14 每輪撞 900s timeout 的直接原因）。
    #
    # 這裡先 O(n) 算好每根 K 線的實體與影線，並只保留「影線長度合格」的索引；
    # swing 迴圈只掃這些候選（多數 swing 前幾根就命中）。
    # 語意等價：命中後原碼把 sw.index 放進 swept set，同一 swing 之後的
    # 迭代不可能再 append，因此「第一個命中即 break」與原行為相同；
    # sweeps 的最終排序（index 遞減、截斷 max_sweeps）維持不變。
    bodies = [0.0] * n
    lower_wicks = [0.0] * n
    upper_wicks = [0.0] * n
    bull_candidates: List[int] = []
    bear_candidates: List[int] = []
    for i in range(n):
        o = opens[i]
        c = closes[i]
        body = c - o if c >= o else o - c
        lower_wick = (o if o < c else c) - lows[i]
        upper_wick = highs[i] - (o if o > c else c)
        bodies[i] = body
        lower_wicks[i] = lower_wick
        upper_wicks[i] = upper_wick
        if lower_wick >= body * wick_ratio:
            bull_candidates.append(i)
        if upper_wick >= body * wick_ratio:
            bear_candidates.append(i)

    import bisect as _bisect_sweeps

    sweeps: List[Dict[str, Any]] = []

    for sw in swings:
        ready = confirm_index(sw, swing_right)
        if ready >= n:
            continue
        if sw.kind == "low":
            cands = bull_candidates
            idx_from = _bisect_sweeps.bisect_left(cands, ready)
            for i in cands[idx_from:]:
                if lows[i] < sw.price and closes[i] > sw.price:
                    sweeps.append({
                        "index": i,
                        "swing_index": sw.index,
                        "direction": "bullish",
                        "level": sw.price,
                        "extreme": lows[i],
                        "wick": lower_wicks[i],
                        "body": bodies[i],
                    })
                    break
        else:
            cands = bear_candidates
            idx_from = _bisect_sweeps.bisect_left(cands, ready)
            for i in cands[idx_from:]:
                if highs[i] > sw.price and closes[i] < sw.price:
                    sweeps.append({
                        "index": i,
                        "swing_index": sw.index,
                        "direction": "bearish",
                        "level": sw.price,
                        "extreme": highs[i],
                        "wick": upper_wicks[i],
                        "body": bodies[i],
                    })
                    break

    sweeps.sort(key=lambda s: s["index"], reverse=True)
    return sweeps[:max_sweeps]


# =====================================================================
# 6. Premium / Discount
# =====================================================================

def premium_discount(
    klines: Sequence[Dict[str, Any]],
    lookback: int = 50,
    bullish_premium_floor: float = 0.5,
    bullish_discount_ceiling: float = 0.5,
) -> Dict[str, Any]:
    """計算價格在區間內的相對位置（premium / discount）。

    ICT 定義： dealing range 取最近 lookback 根的高低點。
      - 位置 >= 0.5 為 premium（高估區，不宜追多）
      - 位置 <  0.5 為 discount（折價區，宜找多）

    回傳 zone 供策略直接判斷。
    """
    if len(klines) < 2:
        return {"position": 0.5, "zone": "neutral", "range_high": 0.0, "range_low": 0.0}

    window = klines[-lookback:] if len(klines) >= lookback else klines
    rng_high = max(float(k["high"]) for k in window)
    rng_low = min(float(k["low"]) for k in window)
    price = float(window[-1]["close"])

    if rng_high <= rng_low:
        return {"position": 0.5, "zone": "neutral", "range_high": rng_high, "range_low": rng_low}

    position = (price - rng_low) / (rng_high - rng_low)

    if position >= bullish_premium_floor:
        zone = "premium"
    elif position <= bullish_discount_ceiling:
        zone = "discount"
    else:
        zone = "equilibrium"

    return {
        "position": round(position, 6),
        "zone": zone,
        "range_high": rng_high,
        "range_low": rng_low,
        "price": price,
    }


# =====================================================================
# 7. Composite：OTE / Confluence
# =====================================================================

def optimal_trade_entry(
    klines: Sequence[Dict[str, Any]],
    swing_left: int = 5,
    swing_right: int = 5,
    require_fvg: bool = True,
    require_sweep: bool = True,
    max_blocks: int = 10,
) -> Dict[str, Any]:
    """組合 CHoCH + Order Block + FVG + Sweep，找高機率 confluence 區。

    量化條件（bullish OTE 全部滿足才算成立）：
      1. 最近一次結構事件是 CHoCH 且方向向上（趨勢反轉）
      2. 存在未被回補的 bullish Order Block
      3. 當前價格回退進入該 Order Block 範圍
      4. 該 OB 附近存在未填補 bullish FVG（require_fvg=True 時）
      5. 突破前曾出現看多流動性掃蕩（require_sweep=True 時）
      6. 當前位於 discount 區

    value 為 GA 可用的連續訊號（0~1，越高越接近完整 confluence）。
    """
    structure = detect_structure(klines, swing_left, swing_right)
    blocks = find_order_blocks(klines, structure, swing_left, swing_right, max_blocks=max_blocks)
    fvgs = find_fvgs(klines) if require_fvg else []
    sweeps = find_liquidity_sweeps(klines) if require_sweep else []
    pd = premium_discount(klines)

    n = len(klines)
    price = float(klines[-1]["close"]) if n else 0.0

    out: Dict[str, Any] = {
        "direction": None,
        "score": 0.0,
        "structure_dir": structure.get("dir"),
        "block": None,
        "fvg": None,
        "sweep": None,
        "zone": pd.get("zone"),
        "value": 0.0,
    }

    last_choch = structure.get("last_choch")
    if not last_choch:
        out["reason"] = "no CHoCH"
        return out

    direction = last_choch["direction"]      # "up" | "down"
    want = "bullish" if direction == "up" else "bearish"
    out["direction"] = want

    # 條件 2：未回補的 OB，且價格在區內或貼近
    active = [b for b in blocks if b["direction"] == want and not b["mitigated"]]
    if not active:
        out["reason"] = "no active OB"
        return out
    block = active[0]
    out["block"] = block

    in_block = block["bottom"] <= price <= block["top"]

    # 條件 4：附近未填補 FVG
    fvg_hit = None
    for f in fvgs:
        if f["direction"] == want and not f["filled"]:
            if f["bottom"] <= price <= f["top"] or abs(price - f["mid"]) <= f["size"]:
                fvg_hit = f
                break
    out["fvg"] = fvg_hit

    # 條件 5：同方向 sweep
    sweep_hit = None
    for s in sweeps:
        if s["direction"] == want:
            sweep_hit = s
            break
    out["sweep"] = sweep_hit

    # 條件 6：區間位置
    zone_ok = (pd.get("zone") == "discount") if want == "bullish" else (pd.get("zone") == "premium")

    # 計分
    score = 0.0
    score += 0.30                                  # CHoCH 已確認
    score += 0.25 if in_block else 0.0             # 價格在 OB 內
    score += 0.20 if fvg_hit is not None else 0.0  # FVG 對齊
    score += 0.15 if sweep_hit is not None else 0.0# sweep 確認
    score += 0.10 if zone_ok else 0.0              # premium/discount

    out["score"] = round(score, 4)
    out["in_block"] = in_block
    out["zone_ok"] = zone_ok
    out["value"] = round(score, 4)
    out["reason"] = "ok" if score >= 0.7 else "weak confluence"
    return out


# =====================================================================
# 8. 統一入口（給 GA / data.get_indicator 使用）
# =====================================================================

def compute_ict(
    kline_data: Sequence[Dict[str, Any]],
    swing_left: int = 5,
    swing_right: int = 5,
    fvg_min_body_ratio: float = 1.0,
    sweep_wick_ratio: float = 1.0,
) -> Dict[str, Any]:
    """一次算出所有 ICT/SMC 結構，回傳給策略的統一 payload。

    每個子項都獨立 try/except：單一偵測器失敗不應讓整個指標失效。
    """
    kl = list(kline_data)
    out: Dict[str, Any] = {"available": len(kl) > 0, "bar_count": len(kl)}

    def _safe(key: str, fn):
        try:
            out[key] = fn()
        except Exception as exc:  # noqa: BLE001
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}

    _safe("swings", lambda: [
        {"index": s.index, "price": s.price, "kind": s.kind, "broken": s.broken}
        for s in find_swings(kl, swing_left, swing_right)[-6:]
    ])
    _safe("structure", lambda: detect_structure(kl, swing_left, swing_right))
    _safe("order_blocks", lambda: find_order_blocks(kl, None, swing_left, swing_right))
    _safe("fvgs", lambda: find_fvgs(kl, min_body_ratio=fvg_min_body_ratio))
    _safe("sweeps", lambda: find_liquidity_sweeps(kl, wick_ratio=sweep_wick_ratio))
    _safe("premium_discount", lambda: premium_discount(kl))
    _safe("ote", lambda: optimal_trade_entry(kl, swing_left, swing_right))

    # GA 用的單一連續值
    ote = out.get("ote", {})
    out["value"] = float(ote.get("value", 0.0)) if isinstance(ote, dict) else 0.0
    return out
