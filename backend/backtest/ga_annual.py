#!/usr/bin/env python3
"""
Genetic Algorithm Engine for Annual Strategy Optimization

真正的 GA：族群並行、錦標賽選擇、交配、突變、精英保留
"""

import random
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from config.guardrails import single_coin_violation
logger = logging.getLogger(__name__)


# ========== 基因型定義 ==========

@dataclass
class Genome:
    """策略基因組 - 所有可優化參數"""
    # EMA 參數
    ema_fast: int = 20
    ema_slow: int = 50
    ema_tol_buy: float = 0.9985      # 金叉容差
    ema_tol_sell: float = 1.0008     # 死叉容差
    
    # Donchian 參數
    donchian_thr: float = 0.98
    
    # RSI 參數
    use_rsi: bool = True
    rsi_buy_thr: int = 35
    rsi_sell_thr: int = 65
    
    # 風險參數
    tp_pct: float = 0.08
    sl_pct: float = 0.04
    portion: float = 0.25
    leverage: int = 2
    
    # ICT / SMC 參數（2026-10-04 新增）
    # 預設值是「停用」：ote_threshold=0 表示不做 OTE 過濾，
    # 其餘 *_require_* 為 0 表示不要求，ict_weight=0 表示 ICT 只是輔助。
    # 這樣預設基因組的行為與加入 ICT 之前完全一致，舊種子仍然可用。
    ict_ote_threshold: float = 0.0
    ict_require_alignment: int = 0
    ict_require_bull_structure: int = 0
    ict_require_discount: int = 0
    ict_swing_len: int = 5
    ict_weight: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "ema_tol_buy": self.ema_tol_buy,
            "ema_tol_sell": self.ema_tol_sell,
            "donchian_thr": self.donchian_thr,
            "use_rsi": self.use_rsi,
            "rsi_buy_thr": self.rsi_buy_thr,
            "rsi_sell_thr": self.rsi_sell_thr,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "portion": self.portion,
            "leverage": self.leverage,
            "ict_ote_threshold": self.ict_ote_threshold,
            "ict_require_alignment": self.ict_require_alignment,
            "ict_require_bull_structure": self.ict_require_bull_structure,
            "ict_require_discount": self.ict_require_discount,
            "ict_swing_len": self.ict_swing_len,
            "ict_weight": self.ict_weight,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Genome":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})
    
    def hash(self) -> str:
        """基因型雜湊，用於去重"""
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


# ========== 邊界約束 ==========

BOUNDS = {
    "ema_fast": (10, 30),
    "ema_slow": (40, 80),
    "ema_tol_buy": (0.995, 0.9995),
    "ema_tol_sell": (1.0005, 1.002),
    "donchian_thr": (0.97, 0.995),
    "rsi_buy_thr": (25, 45),
    "rsi_sell_thr": (55, 75),
    "tp_pct": (0.04, 0.12),
    "sl_pct": (0.02, 0.08),
    "portion": (0.15, 0.5),
    "leverage": (1, 3),
}

BOOL_GENES = {"use_rsi"}

# ICT 基因併入主 BOUNDS（2026-10-04）。
# 用 BOUNDS.update 而不是分開處理，這樣 clamp / random_genome / mutate
# 全部自動支援新基因，不需要在各處加特例。


# =====================================================================
# ICT / SMC 基因（2026-10-04 新增）
# =====================================================================
# 設計依據
# --------
# 實測敏感度（BTC 500 根 1h，見 ICT_SMC_INDICATORS.md）：
#   - swing_left/right 對 OTE 分數影響小（L=R=3..6 都是 0.45）
#   - fvg min_body_ratio 對缺口數影響小（0.5~1.5 都是 10 個）
#   - sweep wick_ratio 對掃單數無影響（都是 10 個，上限截斷）
#   - OTE 分數本身有區分度（0.45 ~ 0.80，240 樣本中 12.5% 達 0.7+）
#
# 所以 ICT 基因圍繞「OTE 分數門檻」與「結構方向」設計，
# 而不是把 swing_left/fvg_ratio/sweep_ratio 也變成基因 ——
# 那三個參數敏感度太低，搜尋它們只是浪費預算。
#
# 每個 ICT 基因都是「可關閉」的：use_ict_* 為 0 時該特性完全不生效，
# 策略退化成純 EMA+Donchian+RSI。這讓 GA 可以自己決定要不要用 ICT。

ICT_BOUNDS = {
    # OTE confluence 分數門檻。0 = 停用 OTE 過濾。
    # 實測分布：0.45(54%) / 0.55(63%) / 0.65(8%) / 0.75(17%) / 0.80(4%)
    # 門檻設在 0.5 會擋掉一半訊號，0.6 擋掉八成，0.7 只留 12.5%。
    "ict_ote_threshold": (0.0, 0.75),

    # 要求 OTE 方向與趨勢一致（1 = 要求，0 = 不要求）。
    # 0.55 與 0.45 都能出分數，但方向可能相反 —— 這是重要的過濾條件。
    "ict_require_alignment": (0, 1),

    # 要求結構方向為多才進場（1 = 要求 up，0 = 不要求）。
    # ICT 的核心是「順大結構做」，但小結構反向也是策略，所以做成可關。
    "ict_require_bull_structure": (0, 1),

    # premium/discount 區間偏好（1 = 只在 discount 買，0 = 不要求）。
    # ICT 理論認為折價區買入勝率較高，但實測不一定，值得讓 GA 自己判斷。
    "ict_require_discount": (0, 1),

    # swing 確認根數（1~6）。雖對 OTE 分數影響小，但影響 swing 數量，
    # 進而影響 BOS/CHoCH 的觸發頻率。
    "ict_swing_len": (1, 6),

    # ICT 訊號在整體決策中的權重（0~1）。
    # 0 = ICT 只是額外過濾；1 = ICT 必須完全成立才進場。
    # 連續參數讓 GA 可以在「輔助」與「主導」之間插值。
    "ict_weight": (0.0, 1.0),
}

# 併入主 BOUNDS。這必須在 ICT_BOUNDS 定義之後、clamp/random_genome/mutate
# 之前執行，那樣所有基因操作邏輯自動涵蓋 ICT 基因。
BOUNDS.update(ICT_BOUNDS)

# ICT 的整數基因（要 round 的那幾個）
ICT_INT_GENES = {"ict_require_alignment", "ict_require_bull_structure",
                 "ict_require_discount", "ict_swing_len"}


def clamp(genome: Genome) -> Genome:
    """將基因組限制在邊界內"""
    d = genome.to_dict()
    for key, (lo, hi) in BOUNDS.items():
        if key in d:
            d[key] = max(lo, min(hi, d[key]))
    # 整數參數取整
    for key in ["ema_fast", "ema_slow", "rsi_buy_thr", "rsi_sell_thr", "leverage"]:
        if key in d:
            d[key] = int(round(d[key]))
    # 布林參數
    for key in BOOL_GENES:
        if key in d:
            d[key] = bool(d[key])
    return Genome.from_dict(d)


def random_genome() -> Genome:
    """生成隨機基因組"""
    return Genome(
        ema_fast=random.randint(*BOUNDS["ema_fast"]),
        ema_slow=random.randint(*BOUNDS["ema_slow"]),
        ema_tol_buy=random.uniform(*BOUNDS["ema_tol_buy"]),
        ema_tol_sell=random.uniform(*BOUNDS["ema_tol_sell"]),
        donchian_thr=random.uniform(*BOUNDS["donchian_thr"]),
        use_rsi=random.choice([True, False]),
        rsi_buy_thr=random.randint(*BOUNDS["rsi_buy_thr"]),
        rsi_sell_thr=random.randint(*BOUNDS["rsi_sell_thr"]),
        tp_pct=random.uniform(*BOUNDS["tp_pct"]),
        sl_pct=random.uniform(*BOUNDS["sl_pct"]),
        portion=random.uniform(*BOUNDS["portion"]),
        leverage=random.randint(*BOUNDS["leverage"]),
        # ICT 基因（2026-10-04）。分佈刻意偏保守：大部分隨機基因組會
        # 把 ICT 關掉或設成輔助，讓 GA 有機會發現「不用 ICT」也是一個解，
        # 而非被迫承擔 ICT 的複雜度。
        #
        # ict_ote_threshold 用指數分佈而非均勻：均勻會讓 37% 的基因組落在
        # 0.37~0.75 的高門檻區，那區只保留 12.5% 的訊號，會讓族群過早熄火。
        # 指數分佈讓大多數落在低門檻（等於不過濾），少數落在嚴格區。
        ict_ote_threshold=(
            BOUNDS["ict_ote_threshold"][1] * (random.random() ** 2.2)
        ),
        ict_require_alignment=random.choice([0, 0, 0, 1]),
        ict_require_bull_structure=random.choice([0, 0, 1, 1]),
        ict_require_discount=random.choice([0, 0, 0, 1]),
        ict_swing_len=random.randint(*BOUNDS["ict_swing_len"]),
        ict_weight=random.choice([0.0, 0.0, 0.0, 0.5, 1.0]),
    )


def genome_from_baseline(baseline_code: str) -> Genome:
    """從 baseline 代碼提取基因組（簡化版：使用默認值）"""
    return Genome()


# ========== 適應度函數 ==========

@dataclass
class EvalResult:
    """回測評估結果"""
    genome: Genome
    gate_annual: bool
    pnl_365d: float
    maxdd_365d: float
    sharpe_365d: float
    pf: Optional[float]
    trades_365d: int
    win_rate: float
    fitness: float
    elapsed_ms: int
    error: Optional[str] = None


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
# 探索期長度 override（0/None = 使用預設「代數的一半」）。
# 由 iterate_annual_ga.py 的 --exploration-gens 設定，透過
# set_exploration_gens() 注入，設在 import 之後以免循環相依。
EXPLORATION_GENS_OVERRIDE = 0


def set_exploration_gens(n: int) -> None:
    """固定探索期長度；傳 <= 0 恢復預設（代數的一半）。"""
    global EXPLORATION_GENS_OVERRIDE
    EXPLORATION_GENS_OVERRIDE = int(n) if n and int(n) > 0 else 0


def _gate_thresholds() -> Dict[str, float]:
    """從 ab_validators 讀取 gate 門檻，避免兩處常數漂移。"""
    try:
        from backtest.ab_validators import (
            PNL_365D_PCT, MAXDD_365D_PCT, MIN_PF, MIN_SHARPE,
            MIN_WIN_PCT, MIN_TRADES_365D, MAX_TRADES_PER_YEAR, MIN_PF_STRONG,
        )
        return {
            "pnl": float(PNL_365D_PCT), "dd": float(MAXDD_365D_PCT),
            "pf": float(MIN_PF), "sharpe": float(MIN_SHARPE),
            "win": float(MIN_WIN_PCT), "trades": float(MIN_TRADES_365D),
            "max_trades": float(MAX_TRADES_PER_YEAR),
            "pf_strong": float(MIN_PF_STRONG),
        }
    except Exception:
        return {"pnl": 20.0, "dd": 8.0, "pf": 1.5, "sharpe": 0.5,
                "win": 35.0, "trades": 20.0, "max_trades": 730.0,
                "pf_strong": 2.0}


def gate_violation(result: "EvalResult") -> float:
    """相對違規量：0 = 完全通過嚴格 gate；1 = 至少一項差距達門檻的 100%。

    必須與 ab_validators.annual_validator 的 cond 判斷保持一致，
    否則 fitness 與 gate 會各自最佳化不同目標。
    """
    t = _gate_thresholds()
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


def calculate_fitness(result: EvalResult) -> float:
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


def calculate_exploration_fitness(result: EvalResult) -> float:
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

    # 連續 gate 懲罰（2026-09-27 新增，權重為嚴格期一半）：
    # 沒有這一項時，族群會漂向 1~4 筆交易的個體（低 DD、高 PF）而全部無法過 gate。
    # 懲罰量取自與 strict 相同的 gate_violation，確保兩期目標一致、換期不跳變。
    fitness -= EXPLORATION_GATE_PENALTY_W * min(gate_violation(result), GATE_VIOLATION_CAP)

    return fitness


# ========== 選擇操作 ==========

def tournament_select(population: List[Tuple[Genome, float]], tournament_size: int = 3) -> Genome:
    """錦標賽選擇"""
    contestants = random.sample(population, min(tournament_size, len(population)))
    winner = max(contestants, key=lambda x: x[1])
    return deepcopy(winner[0])


# ========== 交配操作 ==========

def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """混合交配 - 參數級別"""
    child_dict = {}
    for key in parent1.to_dict():
        v1 = getattr(parent1, key)
        v2 = getattr(parent2, key)
        
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # 混合交配：線性插值
            alpha = random.random()
            child_dict[key] = v1 * alpha + v2 * (1 - alpha)
        elif isinstance(v1, bool) and isinstance(v2, bool):
            child_dict[key] = random.choice([v1, v2])
        else:
            child_dict[key] = v1
    
    child = Genome.from_dict(child_dict)
    return clamp(child)


# ========== 突變操作 ==========

def mutate(genome: Genome, mutation_rate: float = 0.2, aggressive: bool = False) -> Genome:
    """高斯突變 + 離散跳躍 + 結構性變異 (aggressive 模式)"""
    d = genome.to_dict()
    
    # Aggressive mode: higher mutation rate and structural changes
    if aggressive:
        mutation_rate = min(mutation_rate * 2.5, 0.8)
        # Randomly toggle boolean genes with higher probability
        for key in BOOL_GENES:
            if random.random() < 0.3:
                d[key] = not d.get(key, False)
    
    for key, value in d.items():
        if random.random() < mutation_rate:
            if key in ["tp_pct", "sl_pct", "portion", "ema_tol_buy", "ema_tol_sell", "donchian_thr"]:
                # 連續參數：高斯擾動 ±10% (aggressive: ±20%)
                scale = 0.2 if aggressive else 0.1
                d[key] = value * (1 + random.gauss(0, scale))
            elif key in ICT_INT_GENES:
                # ICT 整數基因用與其他整數基因相同的步進突變
                if aggressive:
                    step = random.choice([-3, -2, -1, 1, 2, 3])
                else:
                    step = random.choice([-1, 1])
                d[key] = int(value) + step
            elif key in ["ema_fast", "ema_slow", "rsi_buy_thr", "rsi_sell_thr"]:
                # 整數參數：步進 ±1~2 (aggressive: ±1~5)
                if aggressive:
                    step = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                else:
                    step = random.choice([-2, -1, 1, 2])
                d[key] = int(value) + step
            elif key == "leverage":
                d[key] = random.randint(*BOUNDS["leverage"])
            elif key in BOOL_GENES:
                d[key] = not value
    
    # Aggressive structural mutations
    if aggressive and random.random() < 0.15:
        d = _structural_mutation(d)
    
    return clamp(Genome.from_dict(d))


def _structural_mutation(d: Dict) -> Dict:
    """結構性變異：隨機改變策略結構（指標開關、邏輯切換）"""
    # Toggle major strategy components
    structural_keys = [
        "use_ema_trend", "use_donchian", "use_rsi_filter", 
        "use_volume_filter", "use_atr_filter", "use_momentum",
        "use_regime_filter", "use_funding_filter"
    ]
    available = [k for k in structural_keys if k in d]
    if available:
        key = random.choice(available)
        d[key] = not d[key]
        # If enabling a component, randomize its params
        if d[key]:
            if key == "use_ema_trend":
                d["ema_fast"] = random.randint(*BOUNDS["ema_fast"])
                d["ema_slow"] = random.randint(*BOUNDS["ema_slow"])
            elif key == "use_donchian":
                d["donchian_thr"] = round(random.uniform(*BOUNDS["donchian_thr"]), 2)
            elif key == "use_rsi_filter":
                d["rsi_buy_thr"] = random.randint(*BOUNDS["rsi_buy_thr"])
                d["rsi_sell_thr"] = random.randint(*BOUNDS["rsi_sell_thr"])
            elif key == "use_atr_filter":
                d["atr_mult"] = round(random.uniform(*BOUNDS["atr_mult"]), 2)
            elif key == "use_momentum":
                d["mom_lookback"] = random.randint(*BOUNDS["mom_lookback"])
                d["mom_thr"] = round(random.uniform(*BOUNDS["mom_thr"]), 4)
    return d


# ========== 多樣性度量 ==========

def population_diversity(population: List[Genome]) -> float:
    """計算族群多樣性（基因型熵）"""
    if len(population) < 2:
        return 0.0
    
    # 計算所有基因的平均標準差
    total_var = 0.0
    count = 0
    
    for key in BOUNDS:
        values = []
        for g in population:
            v = getattr(g, key)
            if isinstance(v, (int, float)):
                values.append(float(v))
        if len(values) > 1:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            total_var += var
            count += 1
    
    return total_var / count if count > 0 else 0.0


# ========== 策略代碼渲染 ==========

# 策略碼的標的清單（render_strategy 的預設值）。
# 2026-10-06 修一個實際存在的缺陷：這份清單原本硬編碼在 render_strategy 裡，
# 而且沒有任何參數可以覆寫它。結果是 --symbols 傳 20 個標的、但回測引擎
# 仍會對這 30 個標的取資料。
# 症狀：GA 在 1000 代 run 上卡死 12 小時 —— ENA/NIL/TAO/PENGU/TRUMP 等
# 在 DB 裡的最後一根 K 線停在 2026-09-24，historical_data_provider 對
# 缺資料的標的無限重試抓 API（實測 run.log 裡 ENA 被重複抓 32 次後卡住）。
#
# 這些標的不是「測試起來慢」，而是會讓整輪 GA 永遠跑不完。
DEFAULT_SYMBOLS = [
    "BTC", "ETH", "ZEC", "SOL", "NEAR", "DOGE", "HYPE", "BNB", "SUI",
    "NIL", "ADA", "TAO", "WLD", "AVAX", "PENGU", "ARB", "TRUMP",
    "FIL", "AAVE", "PUMP",
]


def render_strategy(genome: Genome, symbols: Optional[List[str]] = None) -> str:
    """將基因組渲染為策略代碼

    Args:
        genome: 基因組
        symbols: 策略要掃描的標的清單。None 時用 DEFAULT_SYMBOLS。
                 **必須與回測引擎的 symbols 一致**，否則策略會對引擎
                 沒載入資料的標的發出請求，造成無限重試（見 DEFAULT_SYMBOLS 註解）。
    """
    syms = list(symbols) if symbols else list(DEFAULT_SYMBOLS)
    syms_literal = ", ".join(f'"{s}"' for s in syms)

    # Build strategy code using string formatting
    code_parts = []
    code_parts.append('class Strategy:')
    code_parts.append('    def should_trade(self, data):')
    code_parts.append('        try:')
    code_parts.append('            open_pos = data.positions or {}')
    code_parts.append('        except Exception:')
    code_parts.append('            open_pos = {}')
    code_parts.append('        try:')
    code_parts.append('            n_open = len(open_pos)')
    code_parts.append('        except Exception:')
    code_parts.append('            n_open = 0')
    code_parts.append('        if n_open >= 3:')
    code_parts.append('            return Decision(operation="hold", symbol="BTC", reason="max-3 guard")')
    code_parts.append('        skip = set()')
    code_parts.append('        try:')
    code_parts.append('            skip = set(open_pos.keys())')
    code_parts.append('        except Exception:')
    code_parts.append('            skip = set()')
    code_parts.append('        for symbol in [' + syms_literal + ']:')
    code_parts.append('            if symbol in skip:')
    code_parts.append('                continue')
    code_parts.append('            price = data.get_price(symbol)')
    code_parts.append('            if price == 0:')
    code_parts.append('                continue')
    code_parts.append('            klines = data.get_klines(symbol, "1h", ' + str(genome.ema_slow + 30) + ')')
    code_parts.append('            if len(klines) < ' + str(genome.ema_slow + 10) + ':')
    code_parts.append('                continue')
    code_parts.append('            closes = [k.close for k in klines]')
    code_parts.append('            if len(closes) < ' + str(genome.ema_slow + 5) + ':')
    code_parts.append('                continue')
    code_parts.append('            e_fast = sum(closes[-' + str(genome.ema_fast) + ':]) / ' + str(genome.ema_fast))
    code_parts.append('            e_slow = sum(closes[-' + str(genome.ema_slow) + ':]) / ' + str(genome.ema_slow))
    code_parts.append('            e_fast_p = sum(closes[-' + str(genome.ema_fast + 1) + ':-1]) / ' + str(genome.ema_fast))
    code_parts.append('            e_slow_p = sum(closes[-' + str(genome.ema_slow + 1) + ':-1]) / ' + str(genome.ema_slow))
    code_parts.append('            if e_fast <= e_slow * ' + str(round(genome.ema_tol_buy, 4)) + ':')
    code_parts.append('                continue')
    code_parts.append('            if e_fast_p > e_slow_p * ' + str(round(genome.ema_tol_sell, 4)) + ':')
    code_parts.append('                continue')

    if genome.use_rsi:
        # 2026-09-27 修正兩個死維度問題：
        #   舊碼是 `if rsi >= rsi_buy_thr: continue`，即「只在 RSI 低於門檻時進場」。
        #   但本策略的進場條件是 Donchian 突破（價格接近 ema_slow 根高點），
        #   突破時 RSI 幾乎必然偏高，兩個條件互斥 -> 實測 use_rsi=True 時
        #   交易數為 0（thr<=40 完全無交易，thr=45 僅 3 筆 vs 關閉時 25 筆）。
        #   這等於把族群一半的搜尋空間丟在 0 交易的死區（30 代 smoke 中 8/20 = 40%）。
        #   另外 rsi_sell_thr 從未進入產生碼，是完全無效的搜尋維度。
        # 新語義：動能帶狀濾網，與突破進場相容，並讓兩個門檻都有作用。
        #   - RSI <= rsi_buy_thr  -> 動能不足，跳過
        #   - RSI >= rsi_sell_thr -> 過熱，不追高
        code_parts.append('            rsi_info = data.get_indicator(symbol, "RSI14", "1h")')
        code_parts.append('            rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50')
        code_parts.append('            if rsi is None:')
        code_parts.append('                rsi = 50')
        code_parts.append('            if rsi <= ' + str(genome.rsi_buy_thr) + ':')
        code_parts.append('                continue')
        code_parts.append('            if rsi >= ' + str(genome.rsi_sell_thr) + ':')
        code_parts.append('                continue')

    code_parts.append('            highs = [k.high for k in klines]')
    code_parts.append('            hh = max(highs[-' + str(genome.ema_slow + 1) + ':-1])')
    code_parts.append('            if closes[-1] < hh * ' + str(round(genome.donchian_thr, 3)) + ':')
    code_parts.append('                continue')

    # =================================================================
    # ICT / SMC 過濾（2026-10-04 新增）
    # =================================================================
    # 設計原則：ICT 是「附加的結構確認」，不是獨立的進場邏輯。
    # 原因：ICT 的 OTE 分數實測只有 12.5% 的時點達 0.7+，若讓它獨當一面
    # 會讓大部分基因組變成 0 交易（與 2026-09-27 use_rsi 的死維度問題同構）。
    #
    # ict_weight 語義：
    #   0.0 = ICT 完全不參與決策（等同關閉，行為與加入 ICT 前一致）
    #   1.0 = ICT 必須完全成立才進場
    #   中間值 = ICT 部分成立也可放行，門檻依 weight 線性放寬
    #
    # 門檻插值公式：effective = ote_threshold * (1 - ict_weight)
    #   weight=0   → threshold 原值
    #   weight=1   → threshold 0（只要 score >= 0 就通過）
    #   weight=0.5 → 門檻減半
    # 這樣 ICT 是「軟過濾」，避免硬性把訊號清空。

    ict_active = (genome.ict_ote_threshold > 0 or genome.ict_require_alignment or
                  genome.ict_require_bull_structure or genome.ict_require_discount or
                  genome.ict_weight > 0)

    if ict_active:
        swing = int(genome.ict_swing_len)
        # 用 ICT_<NAME>_<swinglen> 讓確認根數成為可調參數（technical_indicators
        # 會解析名稱尾端的數字）。swing_len=1 時用預設名稱以省一次字串解析。
        sfx = '' if swing == 5 else '_' + str(swing)

        code_parts.append('            # --- ICT/SMC 結構確認 ---')
        code_parts.append('            ote_info = data.get_indicator(symbol, "ICT_OTE' + sfx + '", "1h")')
        code_parts.append('            if not isinstance(ote_info, dict):')
        code_parts.append('                ote_info = {}')
        code_parts.append('            ote_score = ote_info.get("value", 0.0) or 0.0')
        code_parts.append('            ote_dir = ote_info.get("direction")')
        code_parts.append('            ict_pass = True')

        # 條件 1：OTE 分數門檻（依 weight 放寬）
        if genome.ict_ote_threshold > 0:
            eff = round(genome.ict_ote_threshold * (1.0 - genome.ict_weight), 4)
            if eff > 0:
                code_parts.append('            if ote_score < ' + str(eff) + ':')
                code_parts.append('                ict_pass = False')

        # 條件 2：OTE 方向必須為多
        if genome.ict_require_alignment:
            code_parts.append('            if ote_dir != "bullish":')
            code_parts.append('                ict_pass = False')

        # 條件 3：結構方向必須為多
        if genome.ict_require_bull_structure:
            code_parts.append('            st_info = data.get_indicator(symbol, "ICT_STRUCTURE' + sfx + '", "1h")')
            code_parts.append('            st_dir = st_info.get("direction") if isinstance(st_info, dict) else None')
            code_parts.append('            if st_dir != "up":')
            code_parts.append('                ict_pass = False')

        # 條件 4：必須在 discount 區
        if genome.ict_require_discount:
            code_parts.append('            pd_info = data.get_indicator(symbol, "ICT_PREMIUM_DISCOUNT", "1h")')
            # ICT 完整 payload 沒有直接的 zone 欄位，改用 OTE 內的資訊：
            # score 達標且非 premium 語義 —— 用 OTE 的 block 命中狀態近似。
            # 更精確的做法是讀 premium_discount，但那是完整 payload 的一部分，
            # 為了在 preview（5s timeout 已修為 60s）下省 API 呼叫，
            # 這裡用 score >= 0.45（至少命中 CHoCH）作為 discount 區的代理。
            code_parts.append('            if ote_score < 0.45:')
            code_parts.append('                ict_pass = False')

        code_parts.append('            if not ict_pass:')
        code_parts.append('                continue')

    code_parts.append('            return Decision(operation="buy", symbol=symbol, reason="ema-donch-ict-ga", target_portion_of_balance=' + str(round(genome.portion, 2)) + ', leverage=' + str(genome.leverage) + ', max_price=price*1.004, take_profit_price=price*' + str(round(1 + genome.tp_pct, 2)) + ', stop_loss_price=price*' + str(round(1 - genome.sl_pct, 2)) + ', sl_execution="market")')
    code_parts.append('        return Decision(operation="hold", symbol="BTC", reason="no signal")')
    
    return '\n'.join(code_parts)



# ========== 單個體評估（供並行調用） ==========

def evaluate_individual(genome_dict: Dict, symbols: List[str], exchange: str = "hyperliquid", generation: int = 0, total_generations: int = 100, backtest_days: int = 365, end_time_ms: Optional[int] = None, windows: Optional[List[Tuple[int, int, float]]] = None) -> Dict:
    """在子進程中評估單個體 - 必須是頂層函數以支持 pickling
    
    Uses relaxed thresholds for early generations (exploration phase),
    strict thresholds for later generations (exploitation phase).
    """
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    
    from backtest.ab_validators import annual_validator, exploration_validator
    from database.connection import SessionLocal
    
    genome = Genome.from_dict(genome_dict)
    code = render_strategy(genome, symbols=symbols)
    
    # Use relaxed validator for early generations (first 50%)
    # Transition to strict validator for later generations
    # Generation is 1-indexed, so use (generation - 1) for 0-indexed comparison
    # Also allow at least 3 generations of exploration
    # 探索期長度：預設為代數的一半。
    # 2026-09-27 實測：100 代 run 因此得到 50 代探索期，前一半預算花在探索 gate
    # 容易通過的區間，嚴格通過數只有 1/1400；30 代 run 探索期 15 代則有 22/600。
    # 探索期應該是「探索」而不是「主要搜尋區間」，因此提供 override 讓呼叫端
    # 把它壓在總代數的一小部分。
    override = EXPLORATION_GENS_OVERRIDE
    if override is not None and override > 0:
        exploration_gens = int(override)
    else:
        exploration_gens = max(3, int(total_generations * 0.5))
    exploration_gens = min(exploration_gens, max(1, total_generations))
    use_exploration = (generation - 1) < exploration_gens
    
    # 2026-09-27 多視窗一致性評分。
    # 實測問題：單一三年視窗的 fitness 會系統性獎勵「只在強行情賺錢」的策略。
    # 20260927T171500Z 的 5 個候選在三年平均為 22~83% 年化 PnL，但逐年分解後
    #   A: 71.70% -> 30.15% -> -2.55%
    #   B: 89.82% -> 32.97% -> +0.50%
    # 全部單調衰減，最近一年五個候選 100% 未過嚴格 gate。
    # 改為在多個互不重疊的視窗各算一次，並對「未通過的視窗比例」加懲罰。
    # windows: [(end_time_ms, backtest_days, weight), ...]，None 則退回單一視窗。
    if not windows:
        windows = [(end_time_ms, backtest_days, 1.0)]
    validate = exploration_validator if use_exploration else annual_validator
    calc = calculate_exploration_fitness if use_exploration else calculate_fitness

    db = None
    try:
        db = SessionLocal()
        per_window = []
        for w_end, w_days, w_weight in windows:
            wg, winfo = validate(
                db, code, symbols=symbols, exchange=exchange,
                fast_fail=True, max_triggers=50000,
                backtest_days=w_days, end_time_ms=w_end
            )
            per_window.append((w_end, w_days, float(w_weight), wg, winfo))
    except Exception as e:
        return {
            "genome": genome_dict,
            "gate_annual": False,
            "pnl_365d": 0.0,
            "maxdd_365d": 100.0,
            "sharpe_365d": 0.0,
            "pf": None,
            "trades_365d": 0,
            "win_rate": 0.0,
            "fitness": -1000.0,
            "error": str(e)[:500],
            "genome_hash": genome.hash(),
        }
    finally:
        if db:
            db.close()
    
    # 提取指標：主要視窗（權重最高者，最後一個）決定對外回報的個體指標，
    # 這與「候選是否可用」最相關。
    primary = max(range(len(per_window)), key=lambda i: (per_window[i][2], i))
    w_end, w_days, w_weight, ga, info = per_window[primary]
    f365 = info.get("f365", {})
    pnl = float(f365.get("pnl_percent", 0) or 0)
    maxdd = float(f365.get("maxdd_pct", 100) or 100)
    sharpe = float(f365.get("sharpe", 0) or 0)
    # 真實數值 PF 在 f365["pf"]；info["pf_ok"] 只是是否 > 門檻的布林值，不可當數值使用
    raw_pf = f365.get("pf")
    pf = float(raw_pf) if raw_pf is not None else None
    pf_ok = bool(info.get("pf_ok"))
    trades = int(f365.get("trades", 0) or 0)
    win = float(f365.get("win", 0) or 0)

    result = EvalResult(
        genome=genome,
        gate_annual=ga,
        pnl_365d=pnl,
        maxdd_365d=maxdd,
        sharpe_365d=sharpe,
        pf=pf,
        trades_365d=trades,
        win_rate=win,
        fitness=0.0,
        elapsed_ms=0,
        error=None,
    )
    primary_fitness = calc(result)

    # 一致性懲罰：未通過的視窗比例越高，fitness 越低。
    # 探索期不施加此懲罰，否則等於用嚴格 gate 扼殺探索。
    n_pass = 0
    weight_total = 0.0
    fitness_num = 0.0
    for _we, _wd, _ww, _wg, winfo in per_window:
        wf = winfo.get("f365", {})
        wr = EvalResult(
            genome=genome,
            gate_annual=_wg,
            pnl_365d=float(wf.get("pnl_percent", 0) or 0),
            maxdd_365d=float(wf.get("maxdd_pct", 100) or 100),
            sharpe_365d=float(wf.get("sharpe", 0) or 0),
            pf=(float(wf["pf"]) if wf.get("pf") is not None else None),
            trades_365d=int(wf.get("trades", 0) or 0),
            win_rate=float(wf.get("win", 0) or 0),
            fitness=0.0,
            elapsed_ms=0,
        )
        if _wg:
            n_pass += 1
        fitness_num += _ww * calc(wr)
        weight_total += _ww
    consistency = n_pass / float(len(per_window)) if per_window else 0.0
    aggregate = fitness_num / weight_total if weight_total else primary_fitness
    if use_exploration:
        fitness = aggregate
    else:
        fitness = aggregate - WINDOW_CONSISTENCY_PENALTY * (1.0 - consistency)

    return {
        "genome": genome.to_dict(),
        "gate_annual": ga,
        "pnl_365d": pnl,
        "maxdd_365d": maxdd,
        "sharpe_365d": sharpe,
        "pf": pf,
        "pf_ok": pf_ok,
        "trades_365d": trades,
        "win_rate": win,
        "fitness": fitness,
        "genome_hash": genome.hash(),
        # 診斷欄位（2026-09-27）
        "window_consistency": consistency,
        "windows_passed": n_pass,
        "windows_total": len(per_window),
        "window_detail": [
            {
                "days": _wd,
                "weight": _ww,
                "pass": bool(_wg),
                "pnl": round(float((_w.get("f365", {}) or {}).get("pnl_percent", 0) or 0), 2),
                "maxdd": round(float((_w.get("f365", {}) or {}).get("maxdd_pct", 0) or 0), 2),
            }
            for _we, _wd, _ww, _wg, _w in per_window
        ],
    }
