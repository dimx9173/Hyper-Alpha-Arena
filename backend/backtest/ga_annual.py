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


def calculate_fitness(result: EvalResult) -> float:
    """多目標適應度計算"""
    if not result.gate_annual:
        return -1000.0  # 硬性懲罰：未通過年化閘門
    
    # 正規化各指標到 0-2 範圍
    pnl_score = min(result.pnl_365d / 120.0, 2.0)           # 120% = 1.0, 240% = 2.0
    dd_score = max(0.0, 1.0 - result.maxdd_365d / 5.0)      # 0% = 1.0, 5% = 0.0, >5% 負值
    sharpe_score = min(result.sharpe_365d / 1.0, 2.0)       # 1.0 = 1.0, 2.0 = 2.0
    pf_score = min((float(result.pf) if result.pf else 0) / 2.0, 2.0)  # 2.0 = 1.0, 4.0 = 2.0
    
    # 加權總分
    fitness = (
        0.40 * pnl_score +
        0.30 * dd_score +
        0.20 * sharpe_score +
        0.10 * pf_score
    )
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

def mutate(genome: Genome, mutation_rate: float = 0.2) -> Genome:
    """高斯突變 + 離散跳躍"""
    d = genome.to_dict()
    
    for key, value in d.items():
        if random.random() < mutation_rate:
            if key in ["tp_pct", "sl_pct", "portion", "ema_tol_buy", "ema_tol_sell", "donchian_thr"]:
                # 連續參數：高斯擾動 ±10%
                d[key] = value * (1 + random.gauss(0, 0.1))
            elif key in ["ema_fast", "ema_slow", "rsi_buy_thr", "rsi_sell_thr"]:
                # 整數參數：步進 ±1~2
                step = random.choice([-2, -1, 1, 2])
                d[key] = int(value) + step
            elif key == "leverage":
                d[key] = random.randint(*BOUNDS["leverage"])
            elif key in BOOL_GENES:
                d[key] = not value
    
    return clamp(Genome.from_dict(d))


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

def render_strategy(genome: Genome) -> str:
    """將基因組渲染為策略代碼"""
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
    code_parts.append('        for symbol in ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "NEAR", "SUI", "DOGE", "ARB", "AAVE", "TAO"]:')
    code_parts.append('            if symbol in skip:')
    code_parts.append('                continue')
    code_parts.append('            price = data.get_price(symbol)')
    code_parts.append('            if price == 0:')
    code_parts.append('                continue')
    code_parts.append('            klines = data.get_klines(symbol, "4h", ' + str(genome.ema_slow + 30) + ')')
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
        code_parts.append('            rsi_info = data.get_indicator(symbol, "RSI14", "4h")')
        code_parts.append('            rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50')
        code_parts.append('            if rsi is None:')
        code_parts.append('                rsi = 50')
        code_parts.append('            if rsi >= ' + str(genome.rsi_buy_thr) + ':')
        code_parts.append('                continue')

    code_parts.append('            highs = [k.high for k in klines]')
    code_parts.append('            hh = max(highs[-' + str(genome.ema_slow + 1) + ':-1])')
    code_parts.append('            if closes[-1] < hh * ' + str(round(genome.donchian_thr, 3)) + ':')
    code_parts.append('                continue')
    code_parts.append('            return Decision(operation="buy", symbol=symbol, reason="ema-donch-4h-ga", target_portion_of_balance=' + str(round(genome.portion, 2)) + ', leverage=' + str(genome.leverage) + ', max_price=price*1.004, take_profit_price=price*' + str(round(1 + genome.tp_pct, 2)) + ', stop_loss_price=price*' + str(round(1 - genome.sl_pct, 2)) + ', sl_execution="market")')
    code_parts.append('        return Decision(operation="hold", symbol="BTC", reason="no signal")')
    
    return '\n'.join(code_parts)



# ========== 單個體評估（供並行調用） ==========

def evaluate_individual(genome_dict: Dict, symbols: List[str], exchange: str = "hyperliquid") -> Dict:
    """在子進程中評估單個體 - 必須是頂層函數以支持 pickling"""
    import sys
    sys.path.insert(0, "/home/brian/project/Hyper-Alpha-Arena/backend")
    
    from backtest.ab_validators import annual_validator
    from database.connection import SessionLocal
    
    genome = Genome.from_dict(genome_dict)
    code = render_strategy(genome)
    
    db = None
    try:
        db = SessionLocal()
        ga, info = annual_validator(db, code, symbols=symbols, exchange=exchange)
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
    
    # 提取指標
    f365 = info.get("f365", {})
    pnl = float(f365.get("pnl_percent", 0) or 0)
    maxdd = float(f365.get("maxdd_pct", 100) or 100)
    sharpe = float(f365.get("sharpe", 0) or 0)
    pf = info.get("pf_ok")
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
    result.fitness = calculate_fitness(result)
    
    return {
        "genome": genome.to_dict(),
        "gate_annual": ga,
        "pnl_365d": pnl,
        "maxdd_365d": maxdd,
        "sharpe_365d": sharpe,
        "pf": pf,
        "trades_365d": trades,
        "win_rate": win,
        "fitness": result.fitness,
        "genome_hash": genome.hash(),
    }
