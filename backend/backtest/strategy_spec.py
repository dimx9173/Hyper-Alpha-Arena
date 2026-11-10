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
from typing import Any, Dict, List, Literal, Optional
try:
    from services.factor_registry import FACTOR_BY_NAME
except ImportError:  # pragma: no cover - 非標準路徑下的退路
    from backend.services.factor_registry import FACTOR_BY_NAME  # type: ignore

# ========== 型別 ==========

FilterKind = Literal["ema_cross", "donchian_breakout", "rsi_band", "ict_confirm", "lgbm_gate"]
LogicSpec = Literal["all_pass"]

FILTER_KINDS = ("ema_cross", "donchian_breakout", "rsi_band", "ict_confirm", "lgbm_gate")

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
    # lgbm_gate 是腦不是規則：x/y/M 走三重障礙（v15 下限由 validate 強制），
    # entry_threshold 是 p_cal 進場線。lag_n 固定 5 不開放擾動。
    "lgbm_gate": {
        "x": (0.004, 0.06),
        "y": (1.0, 4.0),
        "M": (144, 500),
        "entry_threshold": (0.50, 0.75),
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
    "lgbm_gate": {"x": 0.03, "y": 2.0, "M": 144, "entry_threshold": 0.55},
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

_KIND_FACTOR_NAMES: Dict[str, List[str]] = {
    "ema_cross": ["ema_fast", "ema_slow"],
    "donchian_breakout": ["donchian"],
    "rsi_band": ["RSI14"],
    "ict_confirm": ["ICT_OTE", "ICT_STRUCTURE", "ICT_PREMIUM_DISCOUNT"],
    # lgbm_gate 不走 registry 因子：特徵由管線內定，驗證只收 M 與閾值。
    "lgbm_gate": [],
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
        if kind == "lgbm_gate" and "M" in params:
            # v15 執行層約束：M 低於 12h 等價根數時直接判非法，
            # 錯誤字串即判據，不依賴呼叫端記得檢查。
            # 非數值（None/字串）由 _check_numeric 先報，跳過避免 TypeError
            # 掩蓋真正的範圍錯誤。
            try:
                m_val = float(params["M"])
            except (TypeError, ValueError):
                m_val = None
            if m_val is not None:
                try:
                    from backtest.lgbm_meta_labeling import min_hold_bars
                except ImportError:  # pragma: no cover - 非標準路徑退路
                    from backend.backtest.lgbm_meta_labeling import (  # type: ignore
                        min_hold_bars,
                    )
                if m_val < min_hold_bars():
                    errors.append(
                        f"{where} (lgbm_gate): M={params['M']} 低於 v15 下限 "
                        f"{min_hold_bars()} 根（12h / 5m 軸），短持倉不做"
                    )
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
        if f.kind == "lgbm_gate":
            # LGBM 無法在受限 sandbox 內推論（無 lightgbm/numpy）。
            # lgbm_gate 走 _evaluate_lgbm_spec 的 Fuxi 模擬器，不 render。
            raise ValueError("lgbm_gate 不支援 render：請用 evaluate_lgbm_spec")

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
#   2. fitness 不重寫，直接呼叫 scoring.calculate_fitness /
#      scoring.calculate_exploration_fitness（ga_annual 重匯出同一物件）；
#      gate 懲罰也只透過 EvalResult 進入，所以兩條路徑共享同一份計分程式碼。
#   3. EvalResult.genome 型別是 typed Genome（GA 主路徑相依，不可改），
#      因此 spec 路徑傳入 _SpecGenomeShim（只提供 gate_violation 需要的
#      portion / leverage），metric 欄位則逐項鏡像 evaluate_individual。

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])

# ga_annual / ab_validators 的延遲載入參照（首次評估時填充）。
_GA = None


def _load_ga():
    """延遲載入 scoring + ga_annual + ab_validators，回傳共用參照表。

    fitness 與門檻來自 ``backtest.scoring``；``ga_annual`` 只為
    ``EvalResult`` 與 ``EXPLORATION_GENS_OVERRIDE`` 而載入。
    三者皆不 import strategy_spec，因此不存在循環相依；延遲只是讓
    「僅需要型別/渲染」的呼叫端不必付出載入成本。
    兩處 import 皆指向同一模組物件，所以 GA 端的模組級狀態
    （EXPLORATION_GENS_OVERRIDE 等）是共用的，不是快照。
    """
    global _GA
    if _GA is not None:
        return _GA
    try:
        from backtest import ab_validators as _ab
        from backtest import ga_annual as _ga
        from backtest import scoring as _sc
    except ImportError:  # pragma: no cover - 非標準路徑下的退路
        from backend.backtest import ab_validators as _ab
        from backend.backtest import ga_annual as _ga
        from backend.backtest import scoring as _sc
    _GA = {
        "annual_validator": _ab.annual_validator,
        "exploration_validator": _ab.exploration_validator,
        "calculate_fitness": _sc.calculate_fitness,
        "calculate_exploration_fitness": _sc.calculate_exploration_fitness,
        "EvalResult": _ga.EvalResult,
        "ga": _ga,
        "scoring": _sc,
    }
    return _GA


class _SpecGenomeShim:
    """gate_violation 需要的 genome 介面最小代理。

    scoring.gate_violation 只用 result.genome.portion / .leverage 計算
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
    sc_mod = refs["scoring"]
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
                   - sc_mod.WINDOW_CONSISTENCY_PENALTY * (1.0 - consistency))
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


LGBM_GATE_KIND = "lgbm_gate"


def is_lgbm_spec(spec: "StrategySpec") -> bool:
    """spec 是否含 lgbm_gate。含則走 Fuxi 模擬器，不走 render 路徑。"""
    return any(getattr(f, "kind", None) == LGBM_GATE_KIND for f in (spec.filters or []))


def lgbm_gate_params(spec: "StrategySpec") -> dict:
    """取第一個 lgbm_gate 的參數，未填則補預設。"""
    merged = dict(FILTER_DEFAULTS[LGBM_GATE_KIND])
    for f in spec.filters or []:
        if getattr(f, "kind", None) == LGBM_GATE_KIND:
            merged.update(dict(getattr(f, "params", {}) or {}))
            break
    return merged


def _fuxi_result_to_f365(result, initial_capital: float = 1000.0) -> dict:
    """FuxiBacktestResult -> f365 欄位（語義對齊 _extract_gate_fields）。

    pnl_percent 是相對本金的百分比；win 是 0-100；sharpe 用逐筆淨報酬的
    mean/std（單筆或零波動回 0，不偽造）；pf 是總盈利/總虧損。
    """
    import numpy as _np

    trades = list(getattr(result, "trades", []) or [])
    wins = [t for t in trades if float(getattr(t, "net_pnl_u", 0.0) or 0.0) > 0]
    losses = [t for t in trades if float(getattr(t, "net_pnl_u", 0.0) or 0.0) <= 0]
    gross_win = float(sum(float(getattr(t, "net_pnl_u", 0.0) or 0.0) for t in wins))
    gross_loss = float(sum(-float(getattr(t, "net_pnl_u", 0.0) or 0.0) for t in losses))
    rets = _np.asarray([float(getattr(t, "net_ret", 0.0) or 0.0) for t in trades], dtype=float)
    if len(rets) >= 2 and float(_np.std(rets, ddof=1)) > 1e-12:
        # 引擎口徑：逐筆 Sharpe 乘 sqrt(365) 年化（engine.py:1046 同式），
        # 否則 MIN_SHARPE=0.5 這類年化門檻會把正常的逐筆 SR 0.15 判成不達標。
        sharpe = float(_np.mean(rets) / _np.std(rets, ddof=1)) * float(_np.sqrt(365.0))
    else:
        sharpe = 0.0
    base = float(initial_capital) if float(initial_capital or 0.0) > 0 else 1000.0
    return {
        "pnl_percent": float(getattr(result, "total_net_pnl", 0.0) or 0.0) / base * 100.0,
        "pnl": float(getattr(result, "total_net_pnl", 0.0) or 0.0),
        "sharpe": sharpe,
        "pf": (gross_win / gross_loss) if gross_loss > 0 else None,
        "win": float(getattr(result, "win_rate", 0.0) or 0.0) * 100.0,
        "trades": int(getattr(result, "n_trades", 0) or 0),
        "maxdd_pct": float(getattr(result, "max_drawdown", 0.0) or 0.0) * 100.0,
        "maxdd": float(getattr(result, "max_drawdown", 0.0) or 0.0),
    }

def _slice_window(bars, w_end, w_days):
    """按 (end_time_ms, days) 切 5m 視窗。

    bars 是 DatetimeIndex（UTC）的 DataFrame。w_end 為 None 時取整段尾部；
    否則視窗右界為 w_end，左界為右界減 w_days 天。左界早於資料開頭時從
    開頭切起，不回空。
    """
    import pandas as _pd

    if w_days is None or float(w_days) <= 0:
        return bars.iloc[0:0]
    if w_end is None:
        start = bars.index.max() - _pd.Timedelta(days=float(w_days))
        return bars.loc[bars.index > start]
    right = _pd.to_datetime(int(w_end), unit="ms", utc=True)
    start = right - _pd.Timedelta(days=float(w_days))
    frame = bars.loc[(bars.index > start) & (bars.index <= right)]
    if len(frame) == 0 and len(bars) > 0:
        # 右界早於全部資料：回最近的一段，避免視窗靜默全空。
        frame = bars.iloc[-min(len(bars), int(float(w_days) * 288)):]
    return frame


def _promotion_verdict(winfo) -> dict:
    """§7 四關的最終上線判決。

    權威實作是 ``fuxi.validation.PromotionGate``，四關全在裡面且共用同一份
    FeeSchedule。``backtest/handoff_gates.py`` 只保留 QD 對帳用途，不參與
    這裡的決策。

    樣本不足（少於 2 筆交易或缺時間戳）時回 ``available=False``，呼叫端
    必須視為未通過，不得當成已測量。

    ``window_days``（有則取）：被評估的視窗長度（天）。OOS-only 定價只跑
    驗證段，出場時間戳跨度約為視窗的 30%，若直接用戳跨度年化，365d 視窗
    也會得到 years≈0.3 而被硬判 short_sample。有 window_days 時把戳做仿射
    展開到 [w_end - w_days, w_end] 再送門——年化按視窗走，報酬序列本身不動，
    利潤仍只計 OOS。不帶 window_days 的舊呼叫沿用純戳跨度路徑。
    """
    rets = list((winfo or {}).get("trade_returns") or [])
    stamps = list((winfo or {}).get("trade_exit_ms") or [])
    if len(rets) < 2 or len(stamps) != len(rets):
        return {
            "available": False,
            "reason": "insufficientTrades",
            "passed": False,
            "trades": len(rets),
        }
    try:
        from fuxi.validation import PromotionGate
    except ImportError:  # pragma: no cover - 非標準路徑退路
        from backend.fuxi.validation import PromotionGate  # type: ignore
    import numpy as _np

    ann_stamps = _np.asarray(stamps, dtype=_np.int64)
    try:
        w_span = float((winfo or {}).get("window_days") or 0.0)
    except (TypeError, ValueError):
        w_span = 0.0
    if w_span > 0.0:
        # OOS-only 定價把出場戳壓縮到驗證段（約 30% 視窗）；門必須按被評估
        # 的視窗長度年化，利潤仍只計 OOS（報酬序列原樣不動，只展開時間軸）。
        # 右端錨在最後一筆真實出場，不向未來外推。
        t0, t1 = int(_np.min(ann_stamps)), int(_np.max(ann_stamps))
        if t1 > t0:
            w_end = t1
            w_start = w_end - int(w_span * 86_400_000.0)
            frac = (ann_stamps.astype(float) - float(t0)) / float(t1 - t0)
            ann_stamps = (float(w_start) + frac * float(w_end - w_start)).astype(_np.int64)
        # 零跨度（全同刻出場）不展開：沿舊路徑讓 infer 拋錯 → cannotAnnualize。

    try:
        verdict = PromotionGate().evaluate(
            _np.asarray(rets, dtype=float),
            ann_stamps,
        )
    except ValueError as exc:
        # 時間戳跨度為 0（全部同一刻出場）無法年化，明確回報而非假造數值。
        return {
            "available": False,
            "reason": f"cannotAnnualize: {exc}",
            "passed": False,
            "trades": len(rets),
        }
    return {"available": True, **verdict}



def evaluate_lgbm_spec(
    spec_dict,
    bars_5m,
    symbols=("BTC",),
    exchange="hyperliquid",
    generation=0,
    total_generations=100,
    backtest_days=365,
    end_time_ms=None,
    windows=None,
    kline_period="5m",
):
    """用 Fuxi 模擬器評估含 lgbm_gate 的 spec。

    ``bars_5m`` 是 dict[symbol, DataFrame] 或單一 DataFrame（單標的時）。
    回傳鍵與 evaluate_spec_individual 同名同義。
    """
    spec = StrategySpec.from_dict(spec_dict)
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid spec: " + "; ".join(errors))
    if not is_lgbm_spec(spec):
        raise ValueError("evaluate_lgbm_spec 只接受含 lgbm_gate 的 spec")

    _load_ga()
    override = _GA["ga"].EXPLORATION_GENS_OVERRIDE
    if override is not None and override > 0:
        exploration_gens = int(override)
    else:
        exploration_gens = max(3, int(total_generations * 0.5))
    exploration_gens = min(exploration_gens, max(1, total_generations))
    use_exploration = (generation - 1) < exploration_gens

    if isinstance(bars_5m, dict):
        per_symbol = dict(bars_5m)
    else:
        only = list(symbols or ["BTC"])[0]
        per_symbol = {only: bars_5m}
    if not windows:
        windows = [(end_time_ms, backtest_days, 1.0)]

    try:
        from backtest.lgbm_meta_labeling import (
            MetaLabelConfig,
            build_meta_frame,
        )
    except ImportError:  # pragma: no cover - 非標準路徑退路
        from backend.backtest.lgbm_meta_labeling import (  # type: ignore
            MetaLabelConfig,
            build_meta_frame,
        )
    try:
        from fuxi.backtest import FuxiBacktestConfig, FuxiBacktestRunner
        from fuxi.ratchet import AsymmetricCompoundingRatchet
        from fuxi.risk import PerpetualRiskAndSmoothingEngine
        from fuxi.rolling import RollingClosedLoopController
    except ImportError:  # pragma: no cover - 非標準路徑退路
        from backend.fuxi.backtest import (  # type: ignore
            FuxiBacktestConfig,
            FuxiBacktestRunner,
        )
        from backend.fuxi.ratchet import AsymmetricCompoundingRatchet  # type: ignore
        from backend.fuxi.risk import PerpetualRiskAndSmoothingEngine  # type: ignore
        from backend.fuxi.rolling import RollingClosedLoopController  # type: ignore
    try:
        from config.fees import FeeSchedule
    except ImportError:  # pragma: no cover - 非標準路徑退路
        from backend.config.fees import FeeSchedule  # type: ignore

    params = lgbm_gate_params(spec)
    threshold = float(params.get("entry_threshold", 0.55))
    cfg_bt = FuxiBacktestConfig(entry_threshold=threshold)
    try:
        from fuxi.labels import MetaLabelingEngine
    except ImportError:  # pragma: no cover - 非標準路徑退路
        from backend.fuxi.labels import MetaLabelingEngine  # type: ignore
    per_window = []
    for _w_end, w_days, w_weight in windows:
        cfg_lgbm = MetaLabelConfig(
            x=float(params["x"]), y=float(params["y"]), M_bars=int(params["M"]),
            entry_threshold=threshold,
        )
        symbol = list(per_symbol.keys())[0]
        full = per_symbol[symbol]
        # 多視窗按 (end, days) 切段：每個視窗只看自己的資料，
        # 否則各視窗跑出逐字相同的結果，consistency 恆 0/1 且含樣本內利潤。
        bars = _slice_window(full, _w_end, w_days)
        if len(bars) == 0:
            per_window.append((_w_end, w_days, float(w_weight), False, {
                "f365": {"pnl_percent": 0.0, "maxdd_pct": 100.0, "sharpe": 0.0,
                         "trades": 0, "win": 0.0, "pf": None},
                "pf_ok": False,
                "trade_returns": [], "trade_exit_ms": [],
            }))
            continue
        X, y, w, realized, expired = build_meta_frame(bars, cfg_lgbm)
        if len(X) == 0:
            per_window.append((_w_end, w_days, float(w_weight), False, {
                "f365": {"pnl_percent": 0.0, "maxdd_pct": 100.0, "sharpe": 0.0,
                         "trades": 0, "win": 0.0, "pf": None},
                "pf_ok": False,
                "trade_returns": [], "trade_exit_ms": [],
            }))
            continue
        split = max(1, int(len(X) * 0.7))
        ctrl = RollingClosedLoopController(dsr_threshold=0.95)
        duel = ctrl.train_and_evaluate_challenger(
            X.iloc[:split], y.iloc[:split], w[:split],
            X.iloc[split:], y.iloc[split:], realized[split:],
            candidate_params={"x": cfg_lgbm.x, "y": cfg_lgbm.y,
                              "M": float(cfg_lgbm.M_bars)},
            val_expired=expired[split:],
            entry_threshold=threshold,
        )
        if ctrl.champion is None:
            per_window.append((_w_end, w_days, float(w_weight), False, {
                "f365": {"pnl_percent": 0.0, "maxdd_pct": 100.0, "sharpe": 0.0,
                         "trades": 0, "win": 0.0, "pf": None},
                "pf_ok": False, "duel": duel,
                "trade_returns": [], "trade_exit_ms": [],
            }))
            continue
        # OOS 評估只看驗證段：runner 在訓練段也會開單，那段利潤是樣本內，
        # 計入會把 pnl 吹大、consistency 吹成永遠一致。
        oos_start = split
        runner = FuxiBacktestRunner(
            ctrl, AsymmetricCompoundingRatchet(cfg_bt.initial_capital),
            PerpetualRiskAndSmoothingEngine(), cfg_bt, FeeSchedule(),
        )
        primary_signals = MetaLabelingEngine(
            x=cfg_lgbm.x, y=cfg_lgbm.y, M=cfg_lgbm.M_bars,
        ).generate_primary_signals(bars)
        fuxi_out = runner.run(bars.iloc[oos_start:], X.iloc[oos_start:],
                              primary_signals.loc[X.index[oos_start:]])
        f365 = _fuxi_result_to_f365(fuxi_out, cfg_bt.initial_capital)
        pf = f365.get("pf")
        pf_ok = (pf is not None) and (float(pf) > 1.5)
        # duel 的部署狀態只是 Loop2 機械結果，不代表候選通過門檻。
        # 首部署無條件建倉，所以只看回測四項硬指標。
        gate_pass = bool(
            f365["pnl_percent"] >= 20.0
            and f365["maxdd_pct"] <= 8.0
            and pf_ok
            and float(duel.get("chal_dsr", 0.0) or 0.0) >= 0.95
        )
        per_window.append((_w_end, w_days, float(w_weight), gate_pass, {
            "f365": f365, "pf_ok": pf_ok, "duel": duel,
            # PromotionGate 的輸入：逐筆淨報酬（名目口徑）與出場時間戳。
            # 不在這裡先判，留給 primary 視窗的四關做最終門。
            "trade_returns": [float(v) for v in fuxi_out.trade_returns()],
            "trade_exit_ms": [int(v) for v in fuxi_out.exit_timestamps_ms()],
            # 被評估的視窗長度：OOS-only 定價壓縮戳跨度，門按視窗年化；
            # 報酬序列原樣不動，利潤仍只計 OOS。見 _promotion_verdict。
            "window_days": float(w_days),
        }))

    refs = _load_ga()
    sc_mod = refs["scoring"]
    calc = (refs["calculate_exploration_fitness"] if use_exploration
            else refs["calculate_fitness"])
    primary = max(range(len(per_window)), key=lambda i: (per_window[i][2], i))
    _w_end, _w_days, _w_weight, gate, info = per_window[primary]
    f365 = (info or {}).get("f365", {}) or {}
    pf_ok = bool((info or {}).get("pf_ok"))
    # §7 四關只在 promote 決策上生效，不進 fitness：fitness 是搜尋訊號，
    # gate 是上線資格，兩者混用會讓 GA 用嚴格門檻扼殺探索。
    promotion = _promotion_verdict(info)
    result = _make_eval_result(spec, gate, f365)
    primary_fitness = calc(result)
    n_pass, fitness_num, weight_total = 0, 0.0, 0.0
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
        fitness = aggregate - sc_mod.WINDOW_CONSISTENCY_PENALTY * (1.0 - consistency)
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
        # 上線資格：四關全過才算。與 gate_annual（回測門檻）分開回報，
        # 呼叫端不得只看 gate_annual 就 promote。
        "promotion": promotion,
        "promote_ok": bool(promotion.get("passed")),
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

