"""StrategySpec：訓練 agent 可選的因子結構（取代寫死的 19 基因 Genome）。

設計說明
--------
- 本模組是「型別化模型」：filters + risk + logic 取代各處重複的形狀假設。
- 與 ga_annual.Genome / render_strategy 的關係：Genome 是 StrategySpec 的特例
  （見 genome_to_spec），render_strategy_from_spec 輸出與 ga_annual.render_strategy
  相同 block 形狀的 sandbox 相容策略碼。
- 本檔案是新增檔案，不修改 ga_annual.py 及其它任何檔案。
- mutate 結構（filter 增刪）是訓練 agent 的工作，不在本模組；
  perturb_spec 只做數值型鄰域擾動（鏡像 neighborhood_robustness.perturb 語義）。
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
try:
    from services.factor_registry import FACTOR_BY_NAME
except ImportError:  # pragma: no cover - 非標準路徑下的退路
    from backend.services.factor_registry import FACTOR_BY_NAME  # type: ignore

# ========== 型別 ==========

FilterKind = Literal["ema_cross", "donchian_breakout", "rsi_band", "ict_confirm"]
LogicSpec = Literal["all_pass"]

FILTER_KINDS = ("ema_cross", "donchian_breakout", "rsi_band", "ict_confirm")

# 每種 filter 的參數範圍，鏡像 ga_annual.BOUNDS（含 ICT_BOUNDS）。
FILTER_PARAM_RANGES: Dict[str, Dict[str, tuple]] = {
    "ema_cross": {
        "ema_fast": (10, 30),
        "ema_slow": (40, 80),
        "ema_tol_buy": (0.995, 0.9995),
        "ema_tol_sell": (1.0005, 1.002),
    },
    "donchian_breakout": {
        # genome 等價：lookback = ema_slow + 1，故範圍鏡像 (40, 80)+1
        "lookback": (41, 81),
        # = donchian_thr
        "threshold": (0.97, 0.995),
    },
    "rsi_band": {
        # = rsi_buy_thr / rsi_sell_thr（範圍不重疊，low < high 恆成立）
        "low": (25, 45),
        "high": (55, 75),
    },
    "ict_confirm": {
        # = ict_ote_threshold / ict_require_* / ict_swing_len / ict_weight
        "ote_threshold": (0.0, 0.75),
        "require_alignment": (0, 1),
        "require_bull_structure": (0, 1),
        "require_discount": (0, 1),
        "swing_len": (1, 6),
        "weight": (0.0, 1.0),
    },
}

# 每種 filter 的預設值（= Genome 預設值）。
FILTER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ema_cross": {"ema_fast": 20, "ema_slow": 50,
                  "ema_tol_buy": 0.9985, "ema_tol_sell": 1.0008},
    "donchian_breakout": {"lookback": 51, "threshold": 0.98},
    "rsi_band": {"low": 35, "high": 65},
    "ict_confirm": {"ote_threshold": 0.0, "require_alignment": 0,
                    "require_bull_structure": 0, "require_discount": 0,
                    "swing_len": 5, "weight": 0.0},
}

# 整數參數（擾動後 round；鏡像 neighborhood_robustness.INT_GENES 語義）。
INT_PARAMS = {"ema_fast", "ema_slow", "lookback", "low", "high",
              "swing_len", "leverage",
              "require_alignment", "require_bull_structure", "require_discount"}

# 0/1 旗標：沒有「±rate」語義，以 prob=rate 翻轉。
FLAG_PARAMS = {"require_alignment", "require_bull_structure", "require_discount"}

RISK_RANGES: Dict[str, tuple] = {
    "tp_pct": (0.04, 0.12),
    "sl_pct": (0.02, 0.08),
    "portion": (0.15, 0.5),
    "leverage": (1, 3),
}

RISK_DEFAULTS: Dict[str, Any] = {
    "tp_pct": 0.08, "sl_pct": 0.04, "portion": 0.25,
    "leverage": 2, "max_positions": 3,
}

# 每種 filter 背後的因子名（validate_spec 用它做可解析性檢查）。
_KIND_FACTOR_NAMES: Dict[str, List[str]] = {
    "ema_cross": ["ema_fast", "ema_slow"],
    "donchian_breakout": ["donchian"],
    "rsi_band": ["RSI14"],
    "ict_confirm": ["ICT_OTE", "ICT_STRUCTURE", "ICT_PREMIUM_DISCOUNT"],
}

# closes 迴歸計算用的 lookback 名（非 registry 因子，但合法）。
_LOOKBACK_NAMES = {"ema_fast", "ema_slow", "donchian", "lookback"}

MAX_POSITIONS_FIXED = 3


def _resolvable(name: str) -> bool:
    """因子名是否可解析：registry 命中，或 ICT_* 前綴，或 closes lookback 名。"""
    if name in FACTOR_BY_NAME:
        return True
    if name.startswith("ICT_"):
        return True
    if name in _LOOKBACK_NAMES:
        return True
    return False


# ========== 資料模型 ==========

@dataclass
class FilterSpec:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        merged = dict(FILTER_DEFAULTS.get(self.kind, {}))
        merged.update(self.params or {})
        self.params = merged

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterSpec":
        return cls(kind=d["kind"], params=dict(d.get("params", {})))


@dataclass
class RiskSpec:
    tp_pct: float = 0.08
    sl_pct: float = 0.04
    portion: float = 0.25
    leverage: int = 2
    max_positions: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {"tp_pct": self.tp_pct, "sl_pct": self.sl_pct,
                "portion": self.portion, "leverage": self.leverage,
                "max_positions": self.max_positions}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskSpec":
        return cls(**{k: v for k, v in d.items()
                      if k in ("tp_pct", "sl_pct", "portion",
                               "leverage", "max_positions")})


@dataclass
class StrategySpec:
    filters: List[FilterSpec] = field(default_factory=list)
    risk: RiskSpec = field(default_factory=RiskSpec)
    logic: str = "all_pass"

    def to_dict(self) -> Dict[str, Any]:
        return {"filters": [f.to_dict() for f in self.filters],
                "risk": self.risk.to_dict(),
                "logic": self.logic}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategySpec":
        return cls(filters=[FilterSpec.from_dict(f) for f in d.get("filters", [])],
                   risk=RiskSpec.from_dict(d.get("risk", {})),
                   logic=d.get("logic", "all_pass"))

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)

    def spec_id(self) -> str:
        """穩定識別碼：sha256 hex，鏡像 champion_ledger.genome_id 鍵形狀（前 16 碼）。"""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]


# ========== 驗證 ==========

def _check_numeric(errors: List[str], where: str, name: str,
                   value: Any, lo: float, hi: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{where}: {name}={value!r} 不是數值")
        return
    if not (lo <= value <= hi):
        errors.append(f"{where}: {name}={value} 超出範圍 [{lo}, {hi}]")


def validate_spec(spec: StrategySpec) -> List[str]:
    """驗證 spec；預期的壞輸入回傳錯誤字串（不拋例外），合法回傳 []。"""
    errors: List[str] = []

    if spec.logic != "all_pass":
        errors.append(f"logic={spec.logic!r} 不支援，目前只允許 'all_pass'")

    risk = spec.risk
    for key, (lo, hi) in RISK_RANGES.items():
        _check_numeric(errors, "risk", key, getattr(risk, key, None), lo, hi)
    if risk.max_positions != MAX_POSITIONS_FIXED:
        errors.append(f"risk: max_positions={risk.max_positions!r} 必須固定為 "
                      f"{MAX_POSITIONS_FIXED}")

    filters = spec.filters
    if not isinstance(filters, list):
        return errors + ["filters 不是 list"]
    for i, f in enumerate(filters):
        where = f"filter[{i}]"
        kind = getattr(f, "kind", None)
        if kind not in FILTER_KINDS:
            errors.append(f"{where}: 未知 kind {kind!r}，允許 {list(FILTER_KINDS)}")
            continue
        ranges = FILTER_PARAM_RANGES[kind]
        params = getattr(f, "params", {}) or {}
        for pname in params:
            if pname not in ranges:
                errors.append(f"{where} ({kind}): 未知參數 {pname!r}")
        for pname, (lo, hi) in ranges.items():
            if pname in params:
                _check_numeric(errors, f"{where} ({kind})", pname,
                               params[pname], lo, hi)
        for fname in _KIND_FACTOR_NAMES[kind]:
            if not _resolvable(fname):
                errors.append(f"{where} ({kind}): 因子名 {fname!r} 無法解析")
    return errors


# ========== 渲染 ==========

def _filter_map(spec: StrategySpec) -> Dict[str, FilterSpec]:
    out: Dict[str, FilterSpec] = {}
    for f in spec.filters:
        if f.kind not in out:
            out[f.kind] = f
    return out


def _default_symbols() -> List[str]:
    try:
        from backtest.ga_annual import DEFAULT_SYMBOLS
        return list(DEFAULT_SYMBOLS)
    except Exception:
        return ["BTC", "ETH", "SOL"]


def render_strategy_from_spec(spec: StrategySpec,
                              symbols: Optional[List[str]] = None,
                              kline_period: str = "1h") -> str:
    """把 StrategySpec 渲染成 sandbox 相容的 Strategy 類別碼。

    Block 形狀與 ga_annual.render_strategy 一致（code_parts、max-3 guard、
    per-symbol loop、Decision buy/hold 欄位）。未知 kind 拋 ValueError。
    """
    for f in spec.filters:
        if f.kind not in FILTER_KINDS:
            raise ValueError(f"未知 filter kind: {f.kind!r}")

    fmap = _filter_map(spec)
    ema = dict(FILTER_DEFAULTS["ema_cross"])
    ema.update(fmap["ema_cross"].params if "ema_cross" in fmap else {})
    don = dict(FILTER_DEFAULTS["donchian_breakout"])
    don.update(fmap["donchian_breakout"].params if "donchian_breakout" in fmap else {})
    rsi = fmap.get("rsi_band")
    ict = fmap.get("ict_confirm")
    risk = spec.risk

    ema_fast = int(ema["ema_fast"])
    ema_slow = int(ema["ema_slow"])
    lookback = int(don["lookback"])

    syms = list(symbols) if symbols else _default_symbols()
    syms_literal = ", ".join(f'"{s}"' for s in syms)

    # 與 GA 的等價性：Genome 的 lookback 恆等於 ema_slow + 1，因此在預設映射下
    # 這三個數字就是 GA 的 ema_slow+30 / +10 / +5（已用 byte-diff 驗證）。
    # StrategySpec 允許 lookback 與 ema_slow 解耦（lookback 上限 81），
    # 而 donchian slice 是 highs[-lookback:-1]，需要 lookback+1 根 K；
    # 取 max 後三個門檻仍維持 need_closes <= need_min <= need_bars。
    need_bars = max(ema_slow + 30, lookback + 2)
    need_min = max(ema_slow + 10, lookback + 2)
    need_closes = max(ema_slow + 5, lookback + 2)
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
    code_parts.append(f'        if n_open >= {int(risk.max_positions)}:')
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
    code_parts.append('            klines = data.get_klines(symbol, "{P}", ' + str(need_bars) + ')')
    code_parts.append('            if len(klines) < ' + str(need_min) + ':')
    code_parts.append('                continue')
    code_parts.append('            closes = [k.close for k in klines]')
    code_parts.append('            if len(closes) < ' + str(need_closes) + ':')
    code_parts.append('                continue')
    code_parts.append('            e_fast = sum(closes[-' + str(ema_fast) + ':]) / ' + str(ema_fast))
    code_parts.append('            e_slow = sum(closes[-' + str(ema_slow) + ':]) / ' + str(ema_slow))
    code_parts.append('            e_fast_p = sum(closes[-' + str(ema_fast + 1) + ':-1]) / ' + str(ema_fast))
    code_parts.append('            e_slow_p = sum(closes[-' + str(ema_slow + 1) + ':-1]) / ' + str(ema_slow))
    code_parts.append('            if e_fast <= e_slow * ' + str(round(float(ema["ema_tol_buy"]), 4)) + ':')
    code_parts.append('                continue')
    code_parts.append('            if e_fast_p > e_slow_p * ' + str(round(float(ema["ema_tol_sell"]), 4)) + ':')
    code_parts.append('                continue')

    if rsi is not None:
        # 動能帶狀濾網（與突破進場相容）：RSI 過低跳過、過熱不追高。
        code_parts.append('            rsi_info = data.get_indicator(symbol, "RSI14", "{P}")')
        code_parts.append('            rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50')
        code_parts.append('            if rsi is None:')
        code_parts.append('                rsi = 50')
        code_parts.append('            if rsi <= ' + str(int(rsi.params["low"])) + ':')
        code_parts.append('                continue')
        code_parts.append('            if rsi >= ' + str(int(rsi.params["high"])) + ':')
        code_parts.append('                continue')

    code_parts.append('            highs = [k.high for k in klines]')
    code_parts.append('            hh = max(highs[-' + str(lookback) + ':-1])')
    code_parts.append('            if closes[-1] < hh * ' + str(round(float(don["threshold"]), 3)) + ':')
    code_parts.append('                continue')

    if ict is not None:
        ip = ict.params
        ict_active = (float(ip.get("ote_threshold", 0.0)) > 0
                      or int(ip.get("require_alignment", 0))
                      or int(ip.get("require_bull_structure", 0))
                      or int(ip.get("require_discount", 0))
                      or float(ip.get("weight", 0.0)) > 0)
        if ict_active:
            swing = int(ip.get("swing_len", 5))
            sfx = '' if swing == 5 else '_' + str(swing)

            code_parts.append('            # --- ICT/SMC 結構確認 ---')
            code_parts.append('            ote_info = data.get_indicator(symbol, "ICT_OTE' + sfx + '", "{P}")')
            code_parts.append('            if not isinstance(ote_info, dict):')
            code_parts.append('                ote_info = {}')
            code_parts.append('            ote_score = ote_info.get("value", 0.0) or 0.0')
            code_parts.append('            ote_dir = ote_info.get("direction")')
            code_parts.append('            ict_pass = True')

            if float(ip.get("ote_threshold", 0.0)) > 0:
                eff = round(float(ip["ote_threshold"]) * (1.0 - float(ip.get("weight", 0.0))), 4)
                if eff > 0:
                    code_parts.append('            if ote_score < ' + str(eff) + ':')
                    code_parts.append('                ict_pass = False')

            if int(ip.get("require_alignment", 0)):
                code_parts.append('            if ote_dir != "bullish":')
                code_parts.append('                ict_pass = False')

            if int(ip.get("require_bull_structure", 0)):
                code_parts.append('            st_info = data.get_indicator(symbol, "ICT_STRUCTURE' + sfx + '", "{P}")')
                code_parts.append('            st_dir = st_info.get("direction") if isinstance(st_info, dict) else None')
                code_parts.append('            if st_dir != "up":')
                code_parts.append('                ict_pass = False')

            if int(ip.get("require_discount", 0)):
                code_parts.append('            pd_info = data.get_indicator(symbol, "ICT_PREMIUM_DISCOUNT", "{P}")')
                code_parts.append('            if ote_score < 0.45:')
                code_parts.append('                ict_pass = False')

            code_parts.append('            if not ict_pass:')
            code_parts.append('                continue')

    code_parts.append('            return Decision(operation="buy", symbol=symbol, reason="ema-donch-ict-ga", target_portion_of_balance=' + str(round(float(risk.portion), 2)) + ', leverage=' + str(int(risk.leverage)) + ', max_price=price*1.004, take_profit_price=price*' + str(round(1 + float(risk.tp_pct), 2)) + ', stop_loss_price=price*' + str(round(1 - float(risk.sl_pct), 2)) + ', sl_execution="market")')
    code_parts.append('        return Decision(operation="hold", symbol="BTC", reason="no signal")')

    code = "\n".join(code_parts)
    return code.replace("{P}", kline_period)


# ========== 鄰域擾動 ==========

def _perturb_value(key: str, value: Any, rate: float,
                   rng: random.Random, lo: float, hi: float) -> Any:
    if isinstance(value, bool):
        return (not value) if rng.random() < rate else value
    if key in FLAG_PARAMS:
        if rng.random() < rate:
            return 1 - int(value)
        return int(value)
    if not isinstance(value, (int, float)):
        return value
    newval = value * (1.0 + rng.uniform(-rate, rate))
    newval = max(lo, min(hi, newval))
    if key in INT_PARAMS:
        newval = int(round(newval))
        newval = max(int(lo), min(int(hi), newval))
    return newval


def perturb_spec(spec: StrategySpec, rate: float,
                 rng: Optional[random.Random] = None) -> StrategySpec:
    """數值型鄰域擾動：numeric × (1 ± rate) 後 clamp；旗標以 prob=rate 翻轉。

    不做 filter 增刪（結構變異是訓練 agent 的工作）。回傳必為合法 spec。
    """
    rng = rng if rng is not None else random.Random()
    new_filters = []
    for f in spec.filters:
        ranges = FILTER_PARAM_RANGES.get(f.kind, {})
        new_params = {}
        for k, v in f.params.items():
            lo, hi = ranges.get(k, (None, None))
            if lo is None:
                new_params[k] = v
            else:
                new_params[k] = _perturb_value(k, v, rate, rng, lo, hi)
        new_filters.append(FilterSpec(kind=f.kind, params=new_params))
    r = spec.risk
    new_risk = RiskSpec(
        tp_pct=_perturb_value("tp_pct", r.tp_pct, rate, rng, *RISK_RANGES["tp_pct"]),
        sl_pct=_perturb_value("sl_pct", r.sl_pct, rate, rng, *RISK_RANGES["sl_pct"]),
        portion=_perturb_value("portion", r.portion, rate, rng, *RISK_RANGES["portion"]),
        leverage=_perturb_value("leverage", r.leverage, rate, rng, *RISK_RANGES["leverage"]),
        max_positions=MAX_POSITIONS_FIXED,
    )
    out = StrategySpec(filters=new_filters, risk=new_risk, logic=spec.logic)
    assert validate_spec(out) == [], f"perturb 產生非法 spec: {validate_spec(out)}"
    return out


# ========== Genome 相容 ==========

def genome_to_spec(genome_dict: Dict[str, Any]) -> StrategySpec:
    """把 19 基因 Genome dict 精確映射為 StrategySpec。

    - ema_fast/slow/tol_buy/tol_sell -> ema_cross
    - donchian_thr -> donchian_breakout（lookback = ema_slow + 1）
    - use_rsi + rsi 門檻 -> rsi_band（關閉時省略）
    - ict_* -> ict_confirm（全關閉時省略）
    - tp/sl/portion/leverage -> risk
    """
    g = genome_dict
    filters = [
        FilterSpec("ema_cross", {
            "ema_fast": int(g["ema_fast"]),
            "ema_slow": int(g["ema_slow"]),
            "ema_tol_buy": float(g["ema_tol_buy"]),
            "ema_tol_sell": float(g["ema_tol_sell"]),
        }),
        FilterSpec("donchian_breakout", {
            "lookback": int(g["ema_slow"]) + 1,
            "threshold": float(g["donchian_thr"]),
        }),
    ]
    if g.get("use_rsi"):
        filters.append(FilterSpec("rsi_band", {
            "low": int(g["rsi_buy_thr"]),
            "high": int(g["rsi_sell_thr"]),
        }))
    ict_active = (float(g.get("ict_ote_threshold", 0.0)) > 0
                  or int(g.get("ict_require_alignment", 0))
                  or int(g.get("ict_require_bull_structure", 0))
                  or int(g.get("ict_require_discount", 0))
                  or float(g.get("ict_weight", 0.0)) > 0)
    if ict_active:
        filters.append(FilterSpec("ict_confirm", {
            "ote_threshold": float(g.get("ict_ote_threshold", 0.0)),
            "require_alignment": int(g.get("ict_require_alignment", 0)),
            "require_bull_structure": int(g.get("ict_require_bull_structure", 0)),
            "require_discount": int(g.get("ict_require_discount", 0)),
            "swing_len": int(g.get("ict_swing_len", 5)),
            "weight": float(g.get("ict_weight", 0.0)),
        }))
    risk = RiskSpec(tp_pct=float(g["tp_pct"]), sl_pct=float(g["sl_pct"]),
                    portion=float(g["portion"]), leverage=int(g["leverage"]),
                    max_positions=MAX_POSITIONS_FIXED)
    return StrategySpec(filters=filters, risk=risk, logic="all_pass")

# ========== 評估入口 ==========
#
# 目的：讓訓練 agent 用 StrategySpec 跑完整鏈路（spec -> code -> backtest
# -> fitness/gate -> 鄰域擾動），且對既有 GA 路徑零改動。
#
# 等價性保證：
#   1. render_strategy_from_spec 對 genome_to_spec 的輸出與
#      ga_annual.render_strategy byte-identical（已用 4 個基因組含 ICT/RSI
#      開關組合驗證），因此回測引擎看到的策略碼完全相同。
#   2. fitness 不重寫，直接呼叫 ga_annual.calculate_fitness /
#      calculate_exploration_fitness；gate 懲罰也只透過 EvalResult 進入，
#      所以兩條路徑共享同一份計分程式碼。
#   3. EvalResult.genome 型別是 typed Genome（GA 主路徑相依，不可改），
#      因此 spec 路徑傳入 _SpecGenomeShim（只提供 gate_violation 需要的
#      portion / leverage），metric 欄位則逐項鏡像 evaluate_individual。

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])

# ga_annual / ab_validators 的延遲載入參照（首次評估時填充）。
_GA = None


def _load_ga():
    """延遲載入 ga_annual + ab_validators，回傳共用參照表。

    ga_annual 不 import strategy_spec，因此不存在循環相依；延遲只是讓
    「僅需要型別/渲染」的呼叫端不必付出 ga_annual 的載入成本。
    兩處 import 皆指向同一模組物件，所以 GA 端的模組級狀態
    （EXPLORATION_GENS_OVERRIDE 等）是共用的，不是快照。
    """
    global _GA
    if _GA is not None:
        return _GA
    try:
        from backtest import ab_validators as _ab
        from backtest import ga_annual as _ga
    except ImportError:  # pragma: no cover - 非標準路徑下的退路
        from backend.backtest import ab_validators as _ab
        from backend.backtest import ga_annual as _ga
    _GA = {
        "annual_validator": _ab.annual_validator,
        "exploration_validator": _ab.exploration_validator,
        "calculate_fitness": _ga.calculate_fitness,
        "calculate_exploration_fitness": _ga.calculate_exploration_fitness,
        "EvalResult": _ga.EvalResult,
        "ga": _ga,
    }
    return _GA


class _SpecGenomeShim:
    """gate_violation 需要的 genome 介面最小代理。

    ga_annual.gate_violation 只用 result.genome.portion / .leverage 計算
    實盤可部署性違規項（single_coin_violation）。以屬性代理傳入可讓 spec
    路徑重用同一份計分，而不必把 spec 欄位硬塞進 typed Genome，
    也不必改 EvalResult 的欄位型別。
    """

    def __init__(self, spec):
        self.portion = spec.risk.portion
        self.leverage = spec.risk.leverage


def _make_eval_result(spec, gate, f365):
    """用 f365 指標建 EvalResult（欄位語義與 evaluate_individual 一致）。"""
    EvalResult = _load_ga()["EvalResult"]
    raw_pf = f365.get("pf")
    return EvalResult(
        genome=_SpecGenomeShim(spec),
        gate_annual=bool(gate),
        pnl_365d=float(f365.get("pnl_percent", 0) or 0),
        maxdd_365d=float(f365.get("maxdd_pct", 100) or 100),
        sharpe_365d=float(f365.get("sharpe", 0) or 0),
        pf=(float(raw_pf) if raw_pf is not None else None),
        trades_365d=int(f365.get("trades", 0) or 0),
        win_rate=float(f365.get("win", 0) or 0),
        fitness=0.0,
        elapsed_ms=0,
        error=None,
    )


def _run_windows(code, symbols, exchange, use_exploration, windows, db):
    """逐視窗跑 validator，鏡像 evaluate_individual 的 per_window 迴圈。"""
    refs = _load_ga()
    validate = (refs["exploration_validator"] if use_exploration
                else refs["annual_validator"])
    per_window = []
    for w_end, w_days, w_weight in windows:
        # fast_fail / max_triggers=8000 與 evaluate_individual 完全一致：
        # 不一致會讓同一份策略在兩條路徑得到不同交易數與 PnL。
        wg, winfo = validate(
            db, code, symbols=symbols, exchange=exchange,
            fast_fail=True, max_triggers=8000,
            backtest_days=w_days, end_time_ms=w_end,
        )
        per_window.append((w_end, w_days, float(w_weight), wg, winfo))
    return per_window


def _score_windows(spec, use_exploration, per_window):
    """把 per_window 匯總成 fitness 與對外指標（逐行鏡像 evaluate_individual）。"""
    refs = _load_ga()
    ga_mod = refs["ga"]
    calc = (refs["calculate_exploration_fitness"] if use_exploration
            else refs["calculate_fitness"])
    # 主要視窗 = 權重最高者（同權重取索引最大者），決定對外回報的個體指標。
    primary = max(range(len(per_window)), key=lambda i: (per_window[i][2], i))
    _w_end, _w_days, _w_weight, gate, info = per_window[primary]
    f365 = (info or {}).get("f365", {}) or {}
    pf_ok = bool((info or {}).get("pf_ok"))
    result = _make_eval_result(spec, gate, f365)
    primary_fitness = calc(result)
    # 一致性懲罰：未通過的視窗比例越高，fitness 越低（探索期不施加）。
    n_pass = 0
    weight_total = 0.0
    fitness_num = 0.0
    for _we, _wd, _ww, _wg, winfo in per_window:
        wr = _make_eval_result(spec, _wg, (winfo or {}).get("f365", {}) or {})
        if _wg:
            n_pass += 1
        fitness_num += _ww * calc(wr)
        weight_total += _ww
    consistency = n_pass / float(len(per_window)) if per_window else 0.0
    aggregate = fitness_num / weight_total if weight_total else primary_fitness
    if use_exploration:
        fitness = aggregate
    else:
        fitness = (aggregate
                   - ga_mod.WINDOW_CONSISTENCY_PENALTY * (1.0 - consistency))
    return {
        "spec": spec.to_dict(),
        "gate_annual": bool(gate),
        "pnl_365d": result.pnl_365d,
        "maxdd_365d": result.maxdd_365d,
        "sharpe_365d": result.sharpe_365d,
        "pf": result.pf,
        "pf_ok": pf_ok,
        "trades_365d": result.trades_365d,
        "win_rate": result.win_rate,
        "fitness": fitness,
        "spec_id": spec.spec_id(),
        "window_consistency": consistency,
        "windows_passed": n_pass,
        "windows_total": len(per_window),
        "window_detail": [
            {
                "days": _wd,
                "weight": _ww,
                "pass": bool(_wg),
                "pnl": round(float(((_w or {}).get("f365", {}) or {}).get("pnl_percent", 0) or 0), 2),
                "maxdd": round(float(((_w or {}).get("f365", {}) or {}).get("maxdd_pct", 0) or 0), 2),
            }
            for _we, _wd, _ww, _wg, _w in per_window
        ],
    }


def _evaluate_spec(spec, symbols, exchange, use_exploration, backtest_days, end_time_ms, windows, kline_period):
    """spec 評估主體：render -> 逐視窗 validate -> 匯總。"""
    code = render_strategy_from_spec(spec, symbols=symbols, kline_period=kline_period)
    if not windows:
        windows = [(end_time_ms, backtest_days, 1.0)]
    if _BACKEND_DIR not in sys.path:
        sys.path.insert(0, _BACKEND_DIR)
    from database.connection import SessionLocal  # 延遲：鏡像 evaluate_individual
    db = None
    try:
        db = SessionLocal()
        per_window = _run_windows(code, symbols, exchange, use_exploration, windows, db)
    except Exception as e:
        return {
            "spec": spec.to_dict(),
            "gate_annual": False,
            "pnl_365d": 0.0,
            "maxdd_365d": 100.0,
            "sharpe_365d": 0.0,
            "pf": None,
            "trades_365d": 0,
            "win_rate": 0.0,
            "fitness": -1000.0,
            "error": str(e)[:500],
            "spec_id": spec.spec_id(),
        }
    finally:
        if db:
            db.close()
    return _score_windows(spec, use_exploration, per_window)


def evaluate_spec_individual(spec_dict, symbols, exchange="hyperliquid", generation=0, total_generations=100, backtest_days=365, end_time_ms=None, windows=None, kline_period="1h"):
    """以 StrategySpec 跑 evaluate_individual 的同一套流程。

    spec_dict -> StrategySpec.from_dict -> validate_spec（非法則拋
    ValueError，訊息含每一條錯誤字串）-> render_strategy_from_spec
    -> 逐視窗 validate + fitness + 多視窗一致性懲罰。

    回傳欄位與 evaluate_individual 的結果同名同義（gate_annual / pnl_365d /
    maxdd_365d / sharpe_365d / pf / pf_ok / trades_365d / win_rate / fitness /
    window_consistency / windows_passed / windows_total / window_detail），
    但以 spec / spec_id 取代 genome / genome_hash。

    參數語義完全對齊 evaluate_individual：generation 為 1-indexed，
    探索期長度由 ga_annual.EXPLORATION_GENS_OVERRIDE 或
    total_generations 的一半決定（至少 3 代）。
    """
    spec = StrategySpec.from_dict(spec_dict)
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid spec: " + "; ".join(errors))
    _load_ga()
    override = _GA["ga"].EXPLORATION_GENS_OVERRIDE
    if override is not None and override > 0:
        exploration_gens = int(override)
    else:
        exploration_gens = max(3, int(total_generations * 0.5))
    exploration_gens = min(exploration_gens, max(1, total_generations))
    use_exploration = (generation - 1) < exploration_gens
    return _evaluate_spec(spec, symbols, exchange, use_exploration, backtest_days, end_time_ms, windows, kline_period)


def _neighborhood_stats(base_pnl, samples):
    """單一 rate 的鄰居統計；空樣本回 NaN 而非偽造數值。"""
    pnls = [float(s["pnl_365d"]) for s in samples]
    med = float(statistics.median(pnls)) if pnls else float("nan")
    if base_pnl > 0.0 and med == med:
        retention = med / base_pnl * 100.0
    else:
        retention = 0.0
    return {
        "baseline_pnl": float(base_pnl),
        "median_pnl": float(med),
        "retention": float(retention),
        "delta_pnl": float(med - base_pnl) if med == med else float("nan"),
        "retention_defined": bool(base_pnl > 0.0 and med == med),
        "n": len(samples),
        "gate_pass": sum(1 for s in samples if s.get("gate_annual")),
        "trades_median": (float(statistics.median([int(s["trades_365d"]) for s in samples])) if samples else float("nan")),
    }


def neighborhood_report(spec_dict, symbols, rates=(0.05, 0.10), neighbors=4,
                        seed=42, backtest_days=365, end_time_ms=None,
                        kline_period="1h", exchange="hyperliquid"):
    """spec 版鄰域穩健性報告（小樣本、無 GA 依賴）。

    流程：baseline 一次 + 每個 rate 用 seeded rng 抽 ``neighbors`` 個
    perturb_spec 鄰居各評估一次（嚴格路徑、單一視窗）。相同 seed 完全可重現。

    回傳 dict：
      - "baseline"：原 spec 的指標區塊
      - "baseline_pnl"：float
      - "median_pnl" / "retention"：float，等於 rates[0]（最小半徑＝主要判準）
        的同一組數值，方便直接讀單一判準
      - "rates"：每一圈的完整統計（含 rate 原值），供迭代
      - 每個 rate 一個 f"{rate:g}" 鍵（"0.05" / "0.1"）：每條目含
        baseline_pnl / median_pnl / retention / delta_pnl（皆 float）、
        retention_defined（bool）、n / gate_pass（int）、trades_median（float）

    retention = 鄰居中位 PnL / 基準 PnL * 100；基準 <= 0 時比例失去意義，
    因此回 0.0 並由 delta_pnl / retention_defined 表示真實情況
    （鏡像 neighborhood_robustness.summarize 的基準<=0 語義）。
    """
    spec = StrategySpec.from_dict(spec_dict)
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid spec: " + "; ".join(errors))
    if neighbors <= 0:
        raise ValueError("neighbors 必須 >= 1")
    if not rates:
        raise ValueError("rates 不可為空")
    # 嚴格路徑：鄰居要與「部署判準」同一組門檻比較，否則鄰居用寬鬆 gate
    # 過關、基準用嚴格 gate，保留率沒有意義。
    base = _evaluate_spec(spec, symbols, exchange, False, backtest_days,
                          end_time_ms, None, kline_period)
    base_pnl = float(base.get("pnl_365d", 0.0) or 0.0)
    out = {
        "spec_id": spec.spec_id(),
        "baseline": {
            "pnl_365d": base_pnl,
            "fitness": float(base.get("fitness", 0.0) or 0.0),
            "maxdd_365d": float(base.get("maxdd_365d", 0.0) or 0.0),
            "sharpe_365d": float(base.get("sharpe_365d", 0.0) or 0.0),
            "trades_365d": int(base.get("trades_365d", 0) or 0),
            "gate_annual": bool(base.get("gate_annual", False)),
            "error": base.get("error"),
        },
        "baseline_pnl": base_pnl,
        "neighbors": int(neighbors),
        "seed": int(seed),
        "backtest_days": int(backtest_days),
    }
    per_rate = []
    for rate in rates:
        # 每圈獨立 rng（同 seed）：換 rate 不會改變另一圈抽到的鄰居，
        # 也讓不同 rate 之間可比較（同一組隨機流的 ±rate 版本）。
        rng = random.Random(seed)
        samples = []
        for _ in range(neighbors):
            nb = perturb_spec(spec, float(rate), rng)
            try:
                r = _evaluate_spec(nb, symbols, exchange, False, backtest_days,
                                   end_time_ms, None, kline_period)
            except Exception:
                continue      # 單一鄰居失敗不中斷整批
            if r.get("error"):
                continue
            samples.append(r)
        stats = _neighborhood_stats(base_pnl, samples)
        entry = {"rate": float(rate), **stats}
        per_rate.append(entry)
        out[f"{float(rate):g}"] = stats
    out["rates"] = per_rate
    # 主要判準（最小半徑）的數值拉到頂層，讓單一 rate 呼叫端直接讀。
    out["median_pnl"] = float(per_rate[0]["median_pnl"])
    out["retention"] = float(per_rate[0]["retention"])
    return out

