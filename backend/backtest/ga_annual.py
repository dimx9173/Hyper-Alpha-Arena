#!/usr/bin/env python3
"""
Genetic Algorithm Engine for Annual Strategy Optimization

真正的 GA：族群並行、錦標賽選擇、交配、突變、精英保留
"""

import random
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy
from backtest.scoring import (  # noqa: F401  重匯出：既有呼叫端不改
    DD_DERIV_AT_LIMIT,
    DD_LIMIT_PCT,
    DD_OVER_LIN,
    DD_OVER_QUAD,
    DD_SCORE_FLOOR,
    EXPLORATION_GATE_PENALTY_W,
    GATE_PENALTY_W,
    GATE_VIOLATION_CAP,
    PF_SCORE_HI,
    PF_SCORE_LO,
    PF_SCORE_SPAN,
    PNL_SCORE_CEIL_PCT,
    PNL_SCORE_FLOOR_PCT,
    PNL_SCORE_MAX,
    PNL_SCORE_NEG_FLOOR,
    PNL_SCORE_SPAN,
    TRADE_COUNT_MAX,
    TRADE_COUNT_MIN,
    TRADE_COUNT_PENALTY_W,
    WINDOW_CONSISTENCY_PENALTY,
    _gate_thresholds,
    _pnl_score,
    calculate_exploration_fitness,
    calculate_fitness,
    dd_penalty_score,
    gate_violation,
    pf_score,
    trade_count_penalty,
)
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


# 探索期長度 override（0/None = 使用預設「代數的一半」）。
# 由 iterate_annual_ga.py 的 --exploration-gens 設定，透過
# set_exploration_gens() 注入，設在 import 之後以免循環相依。
EXPLORATION_GENS_OVERRIDE = 0


def set_exploration_gens(n: int) -> None:
    """固定探索期長度；傳 <= 0 恢復預設（代數的一半）。"""
    global EXPLORATION_GENS_OVERRIDE
    EXPLORATION_GENS_OVERRIDE = int(n) if n and int(n) > 0 else 0


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


def render_strategy(genome: Genome, symbols: Optional[List[str]] = None,
                    kline_period: str = "1h") -> str:
    """將基因組渲染為策略代碼

    Args:
        genome: 基因組
        symbols: 策略要掃描的標的清單。None 時用 DEFAULT_SYMBOLS。
                 **必須與回測引擎的 symbols 一致**，否則策略會對引擎
                 沒載入資料的標的發出請求，造成無限重試（見 DEFAULT_SYMBOLS 註解）。
        kline_period: 策略讀取的 K 線週期。HL 的 1h 歷史只有約半年，
                 2h/4h/1d 有整年；整年視窗的 refit 用「2h」跑，
                 半年視窗保持「1h」。注意：kline 數量閾值（ema_slow+30 等）
                 是按根數算的，同樣的 EMA 週期在 2h 上覆蓋的時間更長，
                 訊號頻率會比 1h 更低，這是換週期的預期行為不是 bug。
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
    code_parts.append('            klines = data.get_klines(symbol, "{P}", ' + str(genome.ema_slow + 30) + ')')
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
        code_parts.append('            rsi_info = data.get_indicator(symbol, "RSI14", "{P}")')
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
        code_parts.append('            ote_info = data.get_indicator(symbol, "ICT_OTE' + sfx + '", "{P}")')
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
            code_parts.append('            st_info = data.get_indicator(symbol, "ICT_STRUCTURE' + sfx + '", "{P}")')
            code_parts.append('            st_dir = st_info.get("direction") if isinstance(st_info, dict) else None')
            code_parts.append('            if st_dir != "up":')
            code_parts.append('                ict_pass = False')

        # 條件 4：必須在 discount 區
        if genome.ict_require_discount:
            code_parts.append('            pd_info = data.get_indicator(symbol, "ICT_PREMIUM_DISCOUNT", "{P}")')
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
    
    code = "\n".join(code_parts)
    return code.replace("{P}", kline_period)





# ========== 單個體評估（供並行調用） ==========

def evaluate_individual(genome_dict: Dict, symbols: List[str], exchange: str = "hyperliquid", generation: int = 0, total_generations: int = 100, backtest_days: int = 365, end_time_ms: Optional[int] = None, windows: Optional[List[Tuple[int, int, float]]] = None, kline_period: str = "1h") -> Dict:
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
    code = render_strategy(genome, symbols=symbols, kline_period=kline_period)
    
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
                fast_fail=True, max_triggers=8000,
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
