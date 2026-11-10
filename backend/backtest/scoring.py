"""GA/Fuxi 共用計分層（scoring layer）。

從 ``backtest.ga_annual`` 抽出：純計分與門檻，不含搜尋演算法。
``ga_annual`` 與 ``strategy_spec`` 皆從此 import；本模組不回頭 import 兩者
（``ab_validators`` 僅在 ``_gate_thresholds`` 內延遲 import）。

搬移原則：原樣搬移，不改公式、不改數值、不改註解。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from config.guardrails import single_coin_violation

logger = logging.getLogger(__name__)


# ========== 交易頻率約束（2026-10-07 新增）==========
#
# 為什麼需要這個
# ----------------
# iter10 Round 7 通過嚴格 Annual Gate 六項全數（PnL 22.53%、Sharpe 2.135），
# 但鄰域穩健性測試顯示它是針峰：每個基因擾動 ±5% 後，鄰居中位 PnL
# 掉到 −17.55%（保留率 −77.9%）。
#
# 逐基因診斷找出成因：崩潰的基因（ema_fast/ema_slow/ema_tol_buy/donchian_thr）
# 全部直接決定「有多少交易」。交易數從 35 跳到 124 或掉到 23，PnL 就翻轉。
# 穩健的基因（portion/sl_pct/tp_pct）只影響「每筆賺多少」，擾動後仍在 11~25%。
#
# 所以問題不是「指標算錯」，是「績效估計的樣本量不足」。
# 35 筆交易、20 個標的 = 每標的年均不到 2 筆。單一標的的行為
# （一次軋空、一根假突破）就能翻轉總 PnL。
#
# 解決方式：把交易數當成硬約束寫進 fitness，讓 GA 自己去找到
# 「在可估計樣本量下仍有 edge」的參數組合，而不是靠少數交易堆出高 PnL。
#
# 區間設定依據
# ------------
# 下界 60：年化 Sharpe 的標準誤約 sqrt((1+SR²/2)/N)，N=60 時 SR=2 的
#          95% CI 約 [0.2, 3.8]，勉強可與 0 區分。N=35 時是 [−1.4, 5.7]。
# 上界 150：低於 730 的硬上限，但太高會讓手續費侵蝕 edge。
#          37 筆是過少，730 筆是過度交易，中間區間才有意義。
TRADE_COUNT_MIN = 60
TRADE_COUNT_MAX = 150


def trade_count_penalty(trades_365d: int) -> float:
    """交易頻率懲罰（加權懲罰項，可正可負）。

    設計為「高於上界才罰，低於下界也罰」：
      - 低於 TRADE_COUNT_MIN：樣本不足，績效估計不可信 → 線性懲罰
      - 落在區間內：回傳 0（不獎勵也不懲罰，避免 GA 為了刷分而湊交易）
      - 高於 TRADE_COUNT_MAX：成本侵蝕 → 平方懲罰（更嚴）

    回傳正值代表需要從 fitness 扣除的量。
    """
    t = float(trades_365d or 0)

    if t < TRADE_COUNT_MIN:
        # 線性懲罰，最多 -0.6（TRADE_COUNT_MIN=60 時 t=0 為 0.6）
        ratio = 1.0 - (t / float(TRADE_COUNT_MIN))
        return TRADE_COUNT_PENALTY_W * min(ratio, 1.0)

    if t > TRADE_COUNT_MAX:
        # 平方懲罰：150→200 只罰一點，150→730 罰很重
        excess = (t - TRADE_COUNT_MAX) / float(TRADE_COUNT_MAX)
        return TRADE_COUNT_PENALTY_W * min(excess ** 2 * 3.0, 1.5)

    return 0.0


# 權重說明：
#   0.30 足以讓 35 筆的個體（penalty ≈ 0.125）明顯落後於 60 筆的個體，
#   但不至於讓頻率壓過 PnL/DD 這兩個主要目標。
#   與 GATE_PENALTY_W（嚴格期）同量級，確保兩個懲罰項不會互相蓋過。
TRADE_COUNT_PENALTY_W = 0.30


# ========== PnL 正規化（10% ~ 720%） ==========
# 下限配合 2026-09-27 放寬後的 PnL_365D_PCT = 20.0：
# 20% 的 gate 通過者必須拿到正分，否則評分與 gate 互相矛盾。

PNL_SCORE_FLOOR_PCT = 10.0    # 起始門檻：低於此不給 PnL 分
PNL_SCORE_CEIL_PCT = 720.0    # 策略 PnL 上限：此值對應滿分
PNL_SCORE_MAX = 2.0           # 滿分
PNL_SCORE_NEG_FLOOR = -0.25   # 虧損端下限，避免極端虧損數值失控
PNL_SCORE_SPAN = (PNL_SCORE_CEIL_PCT - PNL_SCORE_FLOOR_PCT) / PNL_SCORE_MAX  # = 355.0


def _pnl_score(pnl_pct: float) -> float:
    """PnL 正規化分數：10% -> 0.0，720% -> 2.0，線性且單調。

    720% 封頂（策略 PnL 上限）。虧損端以相同斜率延伸，下限 -0.25。
    """
    try:
        p = float(pnl_pct)
    except (TypeError, ValueError):
        return PNL_SCORE_NEG_FLOOR
    if p != p:            # NaN
        return PNL_SCORE_NEG_FLOOR
    # 先換算成 score 單位再套下限；若先對百分比套 -0.25 會讓虧損區全部被壓平
    return max((min(p, PNL_SCORE_CEIL_PCT) - PNL_SCORE_FLOOR_PCT) / PNL_SCORE_SPAN,
               PNL_SCORE_NEG_FLOOR)


# ========== MaxDD 懲罰（C1 連續單調） ==========
# 2026-09-27 修正：舊曲線 1-ratio^2 / -0.25(ratio-1)^2 在 dd=DD_LIMIT_PCT 處導數斷裂。
# 實測後果：dd 4.99% 的 PnL/DD 匯率是 103:1，dd 5.01% 卻掉到 0.1:1，
# 懲罰強度驟降 1996 倍，5%~8% 形成「免費回撤帶」，GA 會直接跨過去。
# 新曲線在門檻處左右導數相同，且移除 DD_SCORE_FLOOR 飽和段
# （舊 floor 在 dd>=19.14% 之後把所有個體壓成同一分，排序訊息歸零）。

DD_LIMIT_PCT = 8.0          # 對齊放寬後的 MAXDD_365D_PCT
DD_DERIV_AT_LIMIT = 1.75    # 門檻處的目標導數（對 x=dd/DD_LIMIT_PCT 的一階導數）
DD_OVER_LIN = DD_DERIV_AT_LIMIT / DD_LIMIT_PCT   # 0.21875，與門檻內側導數一致
DD_OVER_QUAD = 0.005        # 超標後的二次項，讓高回撤持續拉開差距
DD_SCORE_FLOOR = -1.0e4     # 僅用於 NaN/負值等無效輸入，不再飽和有效區間


def dd_penalty_score(maxdd_pct: float) -> float:
    """MaxDD 懲罰分數（C1 連續、嚴格單調、無飽和段）。

    以嚴格門檻 DD_LIMIT_PCT = 8% 為目標：
      - 0%  -> 1.000（滿分）
      - 4%  -> 0.688
      - 8%  -> 0.000（觸線）
      - 12% -> -0.955
      - 20% -> -3.345
    門檻內外一階導數皆為 -1.75/DD_LIMIT_PCT，跨門檻時懲罰不會瞬間消失。
    """
    try:
        dd = float(maxdd_pct)
    except (TypeError, ValueError):
        return DD_SCORE_FLOOR
    if dd != dd or dd < 0:      # NaN 或負值
        return DD_SCORE_FLOOR
    x = dd / DD_LIMIT_PCT
    if x <= 1.0:
        score = 1.0 - 0.25 * x - 0.75 * x * x
    else:
        over = dd - DD_LIMIT_PCT
        score = -DD_OVER_LIN * over - DD_OVER_QUAD * over * over
    return max(score, DD_SCORE_FLOOR)


# ========== PF 正規化（探索/嚴格共用，消除兩期語義相反） ==========
# 2026-09-27 修正三個問題：
#   1. 嚴格版 pf is None 給滿分 2.0，探索版給 0 分 -> 兩期方向相反。
#      實測排序反轉：完全過 gate 的個體 0.3843，實測 gen943（PF=None）0.4662，
#      不合格者反而高 21.3%。PF=None 是資料/樣本不足訊號，不是優勢，一律給 0。
#   2. 探索版 `if pf > 1.0` 是硬 if，PF=0.9999 與 1.0001 差 0.05 分（無謂階梯）。
#   3. 兩期 cap 不同（嚴格 4.0 / 探索 2.0）。
# 統一為 0.8 -> 0.0、2.0 -> 1.0 的線性映射，無 if、無 None 特例、兩期一致。

PF_SCORE_LO = 0.8
PF_SCORE_HI = 2.0
PF_SCORE_SPAN = PF_SCORE_HI - PF_SCORE_LO   # = 1.2


def pf_score(pf: Optional[float]) -> float:
    """PF 正規化分數：0.8 -> 0.0，2.0 -> 1.0，之上封頂。PF=None 一律 0 分。"""
    if pf is None:
        return 0.0
    try:
        v = float(pf)
    except (TypeError, ValueError):
        return 0.0
    if v != v:                 # NaN
        return 0.0
    return min(max((v - PF_SCORE_LO) / PF_SCORE_SPAN, 0.0), 1.0)


# ========== 連續 gate 懲罰（取代 -1000 硬懲罰） ==========
# 2026-09-27 修正：舊版 `if not result.gate_annual: return -1000.0` 造成平坦地景。
# 實測嚴格期 0/500 通過 -> 全族群 fitness 恆等 -1000，tournament selection 退化成
# 隨機抽籤，後半 500 代等於空轉。改為連續懲罰後，接近 gate 的個體有可爬的坡。
# 懲罰取「最差單項」的相對違規量（而非加總），避免一項超標就被多重扣分。

# GATE_PENALTY_W 在 2026-09-27 由 200.0 調降到 2.0。
# 量測該輪 30 代 smoke 的 600 個個體：base 分數（不含懲罰）的實用動態範圍
# 只有 p25->p75 = 0.756，標準差 0.810；而 W=200 時平均懲罰高達 240，是 base
# 動態範圍的 317 倍。結果是品質訊號被完全淹沒，GA 只剩「最小化違規量」一個方向：
# 30 代下來 best 永遠為負、多樣性從 25.7 崩到 1~6、無任何改進趨勢。
# 懲罰必須與 base 同量綱才能同時保留兩個方向。W=2.0 時平均懲罰 2.4，
# 約為 base 動態範圍的 3.2 倍，維持邊界約束優先但仍看得到品質差異。
GATE_PENALTY_W = 2.0
# 2026-09-27：多視窗一致性懲罰。
# 單一三年視窗的 fitness 會獎勵「只在強行情賺錢」的策略（實測 5 個候選
# 三年平均 22~83%，但逐年分解為 71.7/30.2/-2.6 等，最近一年全數失敗）。
# 這個係數乘上「未通過嚴格 gate 的視窗比例」。2 視窗時：
#   全數通過 -> 0；錯過 1 個 -> -1.0；錯過 2 個 -> -2.0。
WINDOW_CONSISTENCY_PENALTY = 1.0
# 探索期使用較弱的同一組懲罰（2026-09-27）。
# 實測 30 代 smoke 第 1-3 代：fitness 前几名全是 1~4 筆交易的個體，且全部無法
# 通過任何 gate（最低需 20 筆）。原因是探索期若完全不施加 gate 梯度，
# 少交易個體天然擁有低 DD 與高 PF，評分会主動偏好它們，族群在探索期就往錯方向漂。
# 加上探索期懲罰後：
#   - 探索期就有朝最終 gate 拉扯的梯度，不必先漂歪再回頭
#   - 消除第 15->16 代（exploration->strict）的目標函數跳變
# 權重取嚴格期的一半，保留探索性，不會退化成純 gate 優化。
EXPLORATION_GATE_PENALTY_W = 1.0
# 違規量上界：只防數值失控，不在 1.0 截斷。
# 若在 1.0 截斷，所有「違規 >= 100%」的個體會並排在同一個 -200，
# 等於把舊版 -1000 的平坦地景換個數字重現。
GATE_VIOLATION_CAP = 20.0


def _gate_thresholds(phase: str = "strict") -> Dict[str, float]:
    """從 ab_validators 讀取 gate 門檻，避免兩處常數漂移。

    phase="strict"       -> annual_validator 的嚴格期門檻
    phase="exploration"  -> exploration_validator 的放寬期門檻

    fitness 必須用「當前實際在判生死的那組」門檻，否則兩者最佳化不同目標：
    探索期用嚴格期門檻算違規量時，trades 門檻是 20 而非 60，
    19 筆交易的個體只被罰 0.05 分，卻被 exploration gate 直接否決。
    """
    try:
        from backtest.ab_validators import (
            PNL_365D_PCT, MAXDD_365D_PCT, MIN_PF, MIN_SHARPE,
            MIN_WIN_PCT, MIN_TRADES_365D, MAX_TRADES_PER_YEAR, MIN_PF_STRONG,
            EXPLORATION_PNL_365D_PCT, EXPLORATION_MAXDD_365D_PCT,
            EXPLORATION_PF, EXPLORATION_SHARPE, EXPLORATION_WIN_PCT,
            EXPLORATION_PF_STRONG, EXPLORATION_TRADES,
        )
        if phase == "exploration":
            return {
                "pnl": float(EXPLORATION_PNL_365D_PCT),
                "dd": float(EXPLORATION_MAXDD_365D_PCT),
                "pf": float(EXPLORATION_PF),
                "sharpe": float(EXPLORATION_SHARPE),
                "win": float(EXPLORATION_WIN_PCT),
                "trades": float(EXPLORATION_TRADES),
                "max_trades": float(MAX_TRADES_PER_YEAR),
                "pf_strong": float(EXPLORATION_PF_STRONG),
            }
        return {
            "pnl": float(PNL_365D_PCT), "dd": float(MAXDD_365D_PCT),
            "pf": float(MIN_PF), "sharpe": float(MIN_SHARPE),
            "win": float(MIN_WIN_PCT), "trades": float(MIN_TRADES_365D),
            "max_trades": float(MAX_TRADES_PER_YEAR),
            "pf_strong": float(MIN_PF_STRONG),
        }
    except Exception:
        if phase == "exploration":
            return {"pnl": 10.0, "dd": 10.0, "pf": 1.2, "sharpe": 0.3,
                    "win": 35.0, "trades": 60.0, "max_trades": 730.0,
                    "pf_strong": 1.6}
        return {"pnl": 20.0, "dd": 8.0, "pf": 1.5, "sharpe": 0.5,
                "win": 35.0, "trades": 20.0, "max_trades": 730.0,
                "pf_strong": 2.0}


def gate_violation(result, phase: str = "strict") -> float:
    """相對違規量：0 = 完全通過該階段 gate；1 = 至少一項差距達門檻的 100%。

    必須與 ab_validators 對應 validator 的 cond 判斷保持一致，
    否則 fitness 與 gate 會各自最佳化不同目標。
    phase 決定用哪一組門檻（strict / exploration）。
    """
    t = _gate_thresholds(phase)
    dd = float(result.maxdd_365d if result.maxdd_365d is not None else 100.0)
    # 2026-09-27 修正：pf_ok 原本是 0/1 二元項。
    # 實測 30 代 smoke：族群 600 個個體的 violation 中位數正好 1.000，
    # 代表幾乎全數被 pf_ok 的二元 1.0 釘死。PF 從 0.0 爬到 1.49（距門檻只差 0.01）
    # 懲罰完全不變，GA 在最需要爬升的區段沒有任何梯度。
    # 這與「-1000 斷崖」是同一類問題，只是換成「1.0 平坦高原」。
    # 改為連續邊際：pf 0.0 -> 1.0、1.2 -> 0.2、1.49 -> 0.007、1.5 -> 0。
    if result.pf is None:
        pf_violation = 1.0
    else:
        pf_violation = max(0.0, (t["pf"] - float(result.pf)) / t["pf"])
    # 勝率改為替代條件（與 validator 的 cond["win"] 相同）：
    # 勝率達門檻，或 PF 達到較強水準。勝率不是門檻重點，因此不作為獨立違規項。
    # 上限壓到 0.5，避免像原本的 1.0 一樣遮蔽其他所有連續項的梯度。
    pf_strong_ok = result.pf is not None and float(result.pf) >= t.get("pf_strong", 2.0)
    win_ok = (float(result.win_rate) > t["win"]) or pf_strong_ok
    # 實盤可部署性（與 gate 同權重，同進 max）：
    # portion x leverage = 單幣名目（equity 倍數），超過實盤 single_coin_cap
    #（config/guardrails.py MAX_SINGLE_COIN = 0.4，對應
    # program_execution_service.py:1137 / paper_broker.py:213）即違規。
    # 用相對超標量（0 = 可部署，1.0 = 超標 100%），與 PnL/DD 項同量綱。
    # genome 缺失時（舊 seed log 的 _SeedResult 無此屬性）回退為 0，不誤殺。
    _genome = getattr(result, "genome", None)
    _portion = getattr(_genome, "portion", None)
    _leverage = getattr(_genome, "leverage", None)
    if _portion is None or _leverage is None:
        # 舊 seed log 的 _SeedResult 無 genome：回退為 0，不誤殺（會在 log 留痕請見下）。
        deploy_violation = 0.0
        logger.warning("gate_violation: result 無 genome portion/leverage，deploy 項回退為 0")
    else:
        # fail-closed 住在 single_coin_violation 內部（nan/inf/負值/轉型失敗一律 1.0）；
        # 這裡只接結果，不重複守衛。
        deploy_violation = single_coin_violation(_portion, _leverage)
    v = max(
        (t["pnl"] - float(result.pnl_365d)) / t["pnl"],
        (dd - t["dd"]) / t["dd"],
        (t["sharpe"] - float(result.sharpe_365d)) / t["sharpe"],
        (t["trades"] - float(result.trades_365d)) / t["trades"],
        pf_violation,
        0.0 if win_ok else 0.5,
        # 730 筆/年 是硬性否決（過度交易/過擬合），與 validator 的
        # `max_trades: not overtraded` 語義一致，因此用二元而非相對違規。
        # 若用相對違規，731 筆只會得到 0.00137 違規（懲罰 0.27 分），
        # 等於讓 730 筆上限在 fitness 層形同虛設。
        0.0 if float(result.trades_365d) <= t["max_trades"] else 1.0,
        deploy_violation,
    )
    return max(v, 0.0)


def calculate_fitness(result) -> float:
    """多目標適應度計算 - 嚴格閘門版（連續懲罰，無 -1000 斷崖）"""
    # 正規化各指標
    pnl_score = _pnl_score(result.pnl_365d)                # 10% = 0.0, 720% = 2.0
    dd_score = dd_penalty_score(result.maxdd_365d)         # C1 連續：0%=1.0, 8%=0, >8% 負值
    sharpe_score = min(result.sharpe_365d / 1.0, 2.0)      # 1.0 = 1.0, 2.0 = 2.0
    pf_s = pf_score(result.pf)                             # 0.8 = 0.0, 2.0 = 1.0, None = 0

    # 加權總分
    fitness = (
        0.40 * pnl_score +
        0.30 * dd_score +
        0.20 * sharpe_score +
        0.10 * pf_s
    )
    # 交易頻率懲罰（2026-10-07 新增，見 TRADE_COUNT_MIN/MAX 註解）。
    # Round 7 的 PnL 22.53% 建立在恰好 35 筆交易上，擾動後即崩潰。
    # 這個懲罰項讓 GA 傾向「樣本量足以支撐績效估計」的解。
    fitness -= trade_count_penalty(result.trades_365d)

    # 連續 gate 懲罰：取代舊的 -1000 硬懲罰，未過 gate 也保有排序梯度。
    # 不在 1.0 截斷：截斷會讓「嚴重違規」的個體全部並排在 -200*W，
    # 等於重建舊版的平坦地景。改以 GATE_VIOLATION_CAP 保留深層違規的排序。
    return fitness - GATE_PENALTY_W * min(gate_violation(result), GATE_VIOLATION_CAP)


def calculate_exploration_fitness(result) -> float:
    """多目標適應度計算 - 探索期放寬版（與嚴格版共用同一組映射）

    - PnL: 單調獎勵，10% = 0 分，720% = 策略上限 = 滿分
    - MaxDD: C1 連續懲罰，以 8% 為目標，跨門檻時懲罰強度連續
    - Sharpe: 單調獎勵，1.0 = 1.0 分，2.0 封頂
    - Trades: 不納入評分（僅報告 + 730 筆/年 上限檢查）
    - PF: 0.8 = 0 分，2.0 = 滿分，None = 0 分

    2026-09-27 修正：刪除 `if result.gate_annual: fitness += 0.5`。
    舊版該加值佔探索期冠軍 fitness 的 52%，且加法獎金破壞排序
    （平庸過關者 0.30+0.50=0.80 會壓過優秀臨界者 0.55-0.05=0.50）。
    """
    fitness = 0.0

    # PnL 貢獻 (-0.1 ~ 0.8)：連續且單調
    fitness += _pnl_score(result.pnl_365d) * 0.4

    # MaxDD 貢獻：C1 連續懲罰，高回撤個體被持續懲罰但不會壓平
    fitness += dd_penalty_score(result.maxdd_365d) * 0.3

    # Sharpe 貢獻 (0 ~ 0.4)：對齊嚴格門檻 0.5
    if result.sharpe_365d > 0:
        fitness += min(result.sharpe_365d / 1.0, 2.0) * 0.2

    # Trade count（2026-10-07 修正）：舊版「不納入評分」是錯的。
    # 逐基因診斷顯示崩潰的基因全部決定交易頻率，而 Round 7 的 PnL 22.53%
    # 只靠 35 筆交易撐著（擾動後中位 −17.55%）。頻率必須是約束而非自由變數，
    # 否則 GA 會持續往「低頻高 PnL」漂，那是樣本不足不是 edge。
    fitness -= trade_count_penalty(result.trades_365d)

    # PF 貢獻 (0 ~ 0.1)：與嚴格版共用 pf_score，消除 if pf>1.0 階梯與 None 語義相反
    fitness += pf_score(result.pf) * 0.1

    # 2026-11-01 修正：改用 exploration 門檻算違規量。
    # 原本用嚴格期門檻（trades=20），而這個 fitness 只在探索期使用，
    # 實際判生死的是 exploration_validator（trades=60）。結果 19 筆交易的
    # 冠軍在 fitness 只被罰 0.05，卻被 gate 直接否決 —— 兩個目標互相打架。
    fitness -= EXPLORATION_GATE_PENALTY_W * min(
        gate_violation(result, "exploration"), GATE_VIOLATION_CAP
    )

    return fitness
